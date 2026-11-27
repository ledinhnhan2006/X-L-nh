"""In-memory image sessions, processing stages, reviewed geometry and portable reports."""
import base64
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import io
import json
from pathlib import Path
from threading import Lock,Timer
import time
import uuid
from zipfile import ZipFile,ZIP_DEFLATED
import cv2
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from image_features import extract_features,geometry
from scene_scoring import prepare_review,score_scene,DEFAULT_CONFIG
from placement_score import CLASSES,ZONES,canonical_class

OPEN_IMAGE=Image.open
SESSIONS={}
SESSION_LOCK=Lock()
MAX_SIDE=1280
SESSION_SECONDS=3600


def drop_session(key):
    with SESSION_LOCK:
        old=SESSIONS.pop(key,None)
        if old and old.get('timer'):old['timer'].cancel()


def encode_image(image,png=False):
    extension='.png' if png or image.ndim==2 else '.jpg'
    ok,buffer=cv2.imencode(extension,image,[cv2.IMWRITE_JPEG_QUALITY,85] if extension=='.jpg' else [])
    if not ok:raise ValueError('Không mã hóa được ảnh kết quả.')
    mime='image/png' if extension=='.png' else 'image/jpeg'
    return 'data:'+mime+';base64,'+base64.b64encode(buffer).decode()


def decode_image(data):
    if not isinstance(data,dict):raise ValueError('Yêu cầu cần là một bản ghi JSON.')
    try:
        raw=base64.b64decode(data['image'],validate=True)
        if len(raw)>14*1024*1024:raise ValueError('Ảnh tối đa 14 MB.')
        with OPEN_IMAGE(io.BytesIO(raw),formats=('JPEG','PNG','WEBP','BMP')) as source:
            if source.width*source.height>25_000_000:raise ValueError('Ảnh tối đa 25 triệu pixel.')
            rgb=np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    except (KeyError,TypeError,base64.binascii.Error,UnidentifiedImageError,OSError) as exc:
        raise ValueError('Hãy chọn ảnh JPG, PNG, WebP hoặc BMP hợp lệ.') from exc
    h,w=rgb.shape[:2]
    if min(h,w)<8:raise ValueError('Ảnh cần ít nhất 8 pixel mỗi chiều.')
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    factor=min(1.,MAX_SIDE/max(h,w))
    if factor<1:image=cv2.resize(image,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA)
    return image,{'original_width':w,'original_height':h,'analysis_width':image.shape[1],
                  'analysis_height':image.shape[0],'sha256':hashlib.sha256(raw).hexdigest(),
                  'scaled':factor<1,'area_units':'pixel² trên ảnh phân tích; tỷ lệ là phép chiếu 2D'}


def get_session(key):
    if not isinstance(key,str):raise ValueError('Chưa có phiên ảnh. Hãy phân tích ảnh trước.')
    with SESSION_LOCK:
        session=SESSIONS.get(key)
        if session is None or time.monotonic()-session['created']>SESSION_SECONDS:
            SESSIONS.pop(key,None)
            raise ValueError('Phiên ảnh đã hết hạn hoặc không còn trên máy. Hãy phân tích lại ảnh.')
        return session


def stage_record(stage):
    key,title,img,formula,function=stage
    return {'key':key,'title':title,'image':encode_image(img,png=key=='original'),'formula':formula,'function':function}


def create_session(data,detector):
    image,meta=decode_image(data);confidence=data.get('confidence',.25)
    if not isinstance(confidence,(float,int)) or isinstance(confidence,bool) or not .1<=confidence<=.9:
        raise ValueError('Ngưỡng tin cậy từ 0,10 đến 0,90.')
    use_yolo=data.get('yolo',True)
    if not isinstance(use_yolo,bool):raise ValueError('Tùy chọn YOLO cần là có/không.')
    features=extract_features(image);detections=[];model={};error=None
    if use_yolo:
        try:detections,model=detector(image,confidence)
        except Exception as exc:error=f'Chưa chạy được YOLO ({type(exc).__name__}). Bạn vẫn có thể thêm vật thủ công.'
    objects=[]
    for index,d in enumerate(detections,1):
        key=canonical_class(d['name'])
        objects.append({'id':f'auto-{index}','source':'model','name':d['name'],
            'model_name':d['name'],'class_key':key if key in CLASSES else 'other',
            'model_confidence':float(d['confidence']),'box':list(map(float,d['box'])),
            'object_contours':d.get('object_contours',[]),'area_method':d.get('area_method','unavailable'),
            'status':'auto','quantity':1})
    key=uuid.uuid4().hex;now=time.monotonic()
    session={'created':now,'image':image,'meta':meta,'features':features,'objects':objects,
             'model':model,'yolo_error':error,'confidence':confidence}
    with SESSION_LOCK:
        for old in [k for k,s in SESSIONS.items() if now-s['created']>SESSION_SECONDS]:
            expired=SESSIONS.pop(old,None)
            if expired and expired.get('timer'):expired['timer'].cancel()
        while len(SESSIONS)>=4:
            evicted=SESSIONS.pop(min(SESSIONS,key=lambda k:SESSIONS[k]['created']))
            if evicted.get('timer'):evicted['timer'].cancel()
        timer=Timer(SESSION_SECONDS,drop_session,args=(key,));timer.daemon=True;session['timer']=timer
        SESSIONS[key]=session
    timer.start()
    prepared=prepare_review(objects,[],image.shape,features['edges'])
    if prepared:
        proposed=audit_overlay(image,{'objects':[{**o,'state':'unknown'} for o in prepared]},[],None)
        features['stages'].append(('proposals','Gợi ý vật thể ban đầu',proposed,
            'Đường bao và điểm đặt còn là gợi ý; người dùng có thể sửa hoặc bỏ qua.',
            'YOLO.predict + masks/GrabCut + findContours'))
    return {'session_id':key,'image':meta,'image_features':features['metrics'],'cv_contours':features['contours'],
       'objects':prepared,'model':model,'yolo_error':error,'confidence':confidence,
       'steps':[stage_record(s) for s in features['stages']],
       'session_lifetime_minutes':SESSION_SECONDS//60,
       'note':'Ảnh chỉ tạm giữ trong RAM tối đa một giờ hoặc đến khi phiên bị thay thế. Contour OpenCV không phải số vật thể. Chỉ những vật đã đủ thông tin mới được đưa vào điểm.'}


def audit_overlay(image,report,zones,scope):
    result=image.copy();h,w=image.shape[:2]
    if scope and scope.get('points'):
        pts=np.round(np.asarray(scope['points'])*[w-1,h-1]).astype(np.int32)
        cv2.polylines(result,[pts],True,(200,110,60),3)
    for z in zones:
        pts=np.round(np.asarray(z['points'])*[w-1,h-1]).astype(np.int32)
        cv2.polylines(result,[pts],True,(170,125,100),1)
    colors={'correct':(70,155,55),'wrong':(60,65,220),'clutter':(35,120,230),
            'unknown':(20,185,230),'ignored':(155,155,155),'outside':(185,170,150)}
    for i,row in enumerate(report['objects'],1):
        color=colors[row['state']]
        for ring in row['object_contours']:
            pts=np.round(np.asarray(ring['points'])).astype(np.int32)
            cv2.polylines(result,[pts],True,color,2)
        box=row.get('box')
        if box:
            x,y=max(0,int(box[0])),max(16,int(box[1])-4)
            cv2.putText(result,f"#{i} {row['class_key']} {row['state']}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
        if row.get('anchor'):
            point=tuple(np.clip(np.round(np.asarray(row['anchor'])*[w-1,h-1]),[0,0],[w-1,h-1]).astype(int))
            cv2.circle(result,point,4,color,-1)
    return result


def scoring_result(data,with_images=True):
    session=get_session(data.get('session_id'));image=session['image'];features=session['features']
    rows=prepare_review(session['objects'],data.get('reviews',[]),image.shape,features['edges'])
    rules=data.get('rules',{k:list(v[1]) for k,v in CLASSES.items()})
    zones=data.get('zones',[]);scope=data.get('scope');limits=data.get('limits',[]);config=data.get('config',{})
    report=score_scene(rows,image.shape,zones,rules,scope,limits,config)
    masks={key:report.pop(key) for key in ['_scope_mask','_bad_union','_all_union','_clear_mask']}
    timestamp=datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    result={'schema':'goc_gon_reviewed_scene_v5','created_at':timestamp,'image':session['meta'],
      'model':session['model'],'yolo_error':session['yolo_error'],'image_features':features['metrics'],
      'cv_contours':features['contours'],'scoring':report,
      'annotation_input':{'reviews':data.get('reviews',[]),'zones':zones,'scope':scope,'rules':rules,'limits':limits,'config':report['config']},
      'definitions':{'image_edge_density':'Pixel Canny / diện tích ảnh phân tích; chỉ tham khảo.',
         'region_area':'Contour OpenCV bao vùng cạnh; không đồng nghĩa diện tích đồ vật.',
         'known_occupancy':'Hợp mask vật đã đánh giá / phạm vi trên ảnh; không phải diện tích sàn theo m².',
         'object_count':'Mỗi dòng là một vật/nhóm; số lượng do mô hình gợi ý hoặc người dùng nhập.',
         'missing_or_ignored':'Giữ trong báo cáo, không bị phạt; điểm chỉ có ý nghĩa trên phần đã đánh giá.'}}
    if with_images:
        boundary=cv2.morphologyEx(masks['_bad_union'],cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        overlay=audit_overlay(image,report,zones,scope)
        geo_image=image.copy()
        for i,row in enumerate(report['objects'],1):
            g=row.get('geometry')
            if not g:continue
            rect=np.round(np.asarray(g['rotated_rect']['corners'])).astype(np.int32)
            cv2.polylines(geo_image,[rect],True,(155,110,45),2)
            center=tuple(np.round(g['centroid']).astype(int));cv2.circle(geo_image,center,4,(45,125,220),-1)
            cv2.putText(geo_image,f'#{i}',center,cv2.FONT_HERSHEY_SIMPLEX,.5,(40,80,160),1,cv2.LINE_AA)
        result['steps']=[stage_record(s) for s in features['stages']]+[
          {'key':'object_masks','title':'Hợp mask vật đã đánh giá','image':encode_image(masks['_all_union'],True),
           'formula':'Diện tích contour của hợp mask; phần chồng chỉ tính một lần.','function':'cv2.fillPoly + bitwise_or + contourArea'},
          {'key':'bad_masks','title':'Vùng vật cần sắp xếp','image':encode_image(masks['_bad_union'],True),
           'formula':'Chỉ gồm vật sai chỗ hoặc do người dùng đánh dấu cần sắp xếp.','function':'Hợp mask đã rà soát, cắt theo phạm vi đánh giá'},
          {'key':'object_boundaries','title':'Biên vùng vật cần sắp xếp','image':encode_image(boundary,True),
           'formula':'Biên dùng chấm lấy từ mask vật; các nét hoa văn bên trong không tự gây phạt.','function':'findContours + tổng độ dài đoạn nằm trong phạm vi'},
          {'key':'geometry','title':'Tâm và hình chữ nhật xoay','image':encode_image(geo_image),
           'formula':'Tâm từ moments; khung xoay từ minAreaRect. Khung chỉ giúp xem hình học, không dùng thay diện tích mask.',
           'function':'cv2.moments + cv2.minAreaRect + cv2.boxPoints'},
          {'key':'reviewed','title':'Ảnh đối chiếu sau rà soát','image':encode_image(overlay),
           'formula':'Xanh: đúng · đỏ: sai · cam: cần sắp xếp · vàng: chưa rõ · xám: bỏ qua.','function':'Đường bao mask, điểm đặt và trạng thái của từng vật'}]
    return result


def export_result(data):
    result=scoring_result(data);buffer=io.BytesIO()
    with ZipFile(buffer,'w',ZIP_DEFLATED) as archive:
        report={k:v for k,v in result.items() if k!='steps'}
        archive.writestr('bao_cao.json',json.dumps(report,ensure_ascii=False,indent=2))
        for i,step in enumerate(result['steps']):
            header,encoded=step['image'].split(',',1);ext='png' if 'image/png' in header else 'jpg'
            archive.writestr(f'cac_buoc/{i+1:02d}_{step["key"]}.{ext}',base64.b64decode(encoded))
        text=io.StringIO();writer=csv.writer(text,lineterminator='\n')
        writer.writerow(['id','ten','lop','trang_thai','nguon','tin_cay_mo_hinh','so_luong','dien_tich_pixel2','ty_le_trong_pham_vi','chu_vi_pixel','do_tron','do_dac','ghi_chu'])
        for r in result['scoring']['objects']:
            g=r.get('geometry') or {}
            # Prefix spreadsheet formula-like user input; JSON keeps the exact original text.
            def cell(value):return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value
            writer.writerow([cell(v) for v in [r['id'],r['name'],r['class_key'],r['state'],r['source'],r['model_confidence'],r['quantity'],g.get('area_pixels'),r['area_in_scope_ratio'],g.get('perimeter_pixels'),g.get('circularity'),g.get('solidity'),r['note']]])
        archive.writestr('vat_the.csv','\ufeff'+text.getvalue())
        archive.writestr('HUONG_DAN.txt','Mở bao_cao.json để xem công thức, từng khoản trừ, thông tin thủ công và vật bỏ qua.\nẢnh từng bước nằm trong cac_buoc.\nĐây là đánh giá thử nghiệm trên phép chiếu 2D; không phải phép đo diện tích sàn thật hoặc xác nhận phòng sạch.\n')
    return buffer.getvalue()
