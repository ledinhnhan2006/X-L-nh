"""Ước lượng mặt nạ bằng GrabCut trong ROI YOLO, đo diện tích contour; không dùng diện tích box thay thế."""
import cv2
import numpy as np


def contour_rings(mask, offset=(0,0), scale=(1,1)):
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    rings=[]
    for i,contour in enumerate(contours):
        if len(contour)<3 or cv2.contourArea(contour)<=0:
            continue
        depth=0;parent=int(hierarchy[0,i,3])
        while parent!=-1:
            depth+=1;parent=int(hierarchy[0,parent,3])
        points=contour.reshape(-1,2).astype(float)*np.array(scale)+np.array(offset)
        rings.append({'points':points.tolist(),'hole':bool(depth%2),'depth':depth})
    return rings


def geometric_area(rings):
    return max(0.,sum((-1 if r['hole'] else 1)*cv2.contourArea(np.array(r['points'],np.float32)) for r in rings))


def extract_object_contour(image,box,max_side=320):
    h,w=image.shape[:2]
    x1,y1,x2,y2=box
    x1,y1=max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
    x2,y2=min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
    failure={'area_method':'grabcut_contour','area_status':'unavailable','area_pixels':None,
             'area_ratio':None,'object_contours':[]}
    if x2-x1<8 or y2-y1<8:
        return {**failure,'area_reason':'Vật quá nhỏ để tách contour ổn định.'}
    pad=max(5,int(max(x2-x1,y2-y1)*.12))
    left,top=max(0,x1-pad),max(0,y1-pad)
    right,bottom=min(w,x2+pad),min(h,y2+pad)
    roi=image[top:bottom,left:right]
    factor=min(1.,max_side/max(roi.shape[:2]))
    rw,rh=max(3,round(roi.shape[1]*factor)),max(3,round(roi.shape[0]*factor))
    small=cv2.resize(roi,(rw,rh),interpolation=cv2.INTER_AREA)
    sx,sy=rw/roi.shape[1],rh/roi.shape[0]
    bx1,by1=int((x1-left)*sx),int((y1-top)*sy)
    bx2,by2=min(rw,int(np.ceil((x2-left)*sx))),min(rh,int(np.ceil((y2-top)*sy)))
    region=np.zeros((rh,rw),np.uint8);region[by1:by2,bx1:bx2]=1
    if not np.any(region==0):
        return {**failure,'area_reason':'Khung bao phủ toàn ảnh, thiếu vùng nền để tách vật.'}
    blur=cv2.GaussianBlur(small,(3,3),0)
    gray=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,60,140)
    if float(np.std(blur.reshape(-1,3),axis=0).max())<2:
        return {**failure,'area_reason':'Ảnh gần như đồng màu, không đủ thông tin tách nền.'}
    gc=np.where(region,cv2.GC_PR_FGD,cv2.GC_BGD).astype(np.uint8)
    try:
        cv2.grabCut(blur,gc,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return {**failure,'area_reason':'GrabCut không tách được vật khỏi nền.'}
    fg=np.where(((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD))&(region==1),255,0).astype(np.uint8)
    # Loại mảnh nhiễu rất nhỏ nhưng giữ nhiều phần rời của cùng vật và các lỗ bên trong.
    n,labels,stats,_=cv2.connectedComponentsWithStats(fg,connectivity=8)
    for label in range(1,n):
        if stats[label,cv2.CC_STAT_AREA]<max(4,region.sum()*.003):
            fg[labels==label]=0
    fraction=cv2.countNonZero(fg)/max(1,int(region.sum()))
    if fraction<.01 or fraction>.98:
        return {**failure,'area_reason':'Không tìm thấy mặt nạ đáng dùng hoặc mặt nạ gần phủ nguyên khung bao.'}
    rings=contour_rings(fg,(left,top),(1/sx,1/sy))
    area=geometric_area(rings)
    if area<=0:
        return {**failure,'area_reason':'Không tìm được contour kín có diện tích dương.'}
    return {'area_method':'grabcut_contour','area_status':'estimated','area_reason':'Contour ước lượng, cần kiểm tra trên ảnh.',
            'area_pixels':round(area,2),'area_ratio':area/(h*w),'object_contours':rings,
            'roi_edge_pixels':int(cv2.countNonZero(edges))}


def measure_detections(image,detections):
    result=[]
    for det in detections:
        if det.get('confidence',0)<.5:
            area={'area_method':'grabcut_contour','area_status':'unavailable','area_reason':'Độ tin cậy dưới 50%, không đo diện tích để phạt.',
                  'area_pixels':None,'area_ratio':None,'object_contours':[]}
        else:
            area=extract_object_contour(image,det['box'])
        result.append({**det,**area})
    return result


def union_contour_ratio(detections,image_shape,size=512):
    h,w=image_shape[:2];union=np.zeros((size,size),np.uint8)
    for det in detections:
        mask=np.zeros_like(union)
        for ring in sorted(det.get('object_contours',[]),key=lambda r:r.get('depth',0)):
            points=np.round(np.array(ring['points'])*[size/w,size/h]).astype(np.int32)
            cv2.fillPoly(mask,[points],0 if ring['hole'] else 255)
        union=cv2.bitwise_or(union,mask)
    return geometric_area(contour_rings(union))/(size*size)
