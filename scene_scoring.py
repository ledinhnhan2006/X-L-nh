"""Auditable scoring from reviewed scene objects; texture alone never creates a violation."""
import math
from collections import defaultdict
import cv2
import numpy as np
from placement_score import CLASSES,ZONES,CLASS_PENALTIES,canonical_class,validate_settings
from object_contours import contour_rings,geometric_area
from image_features import geometry

GRID=512
CAPS={'position':12.,'quantity':12.,'area':16.,'contours':10.,'boundary':10.}
DEFAULT_CONFIG={'area_allowance':.10,'clear_allowance':.02,'component_min_ratio':.001,'boundary_allowance':.10}


def finite(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def polygon(points,label='Đường bao',minimum=.000001):
    if not isinstance(points,list) or not 3<=len(points)<=200:
        raise ValueError(f'{label}: cần từ 3 đến 200 đỉnh.')
    if any(not isinstance(p,list) or len(p)!=2 or any(not finite(v) or not 0<=v<=1 for v in p) for p in points):
        raise ValueError(f'{label}: tọa độ phải là số trong [0,1].')
    # Reuse the existing simple-polygon intersection checks with a larger coordinate scale
    # so small object polygons are not rejected as tiny room zones.
    p=np.asarray(points,np.float32)
    if len({tuple(v) for v in points})!=len(points) or abs(cv2.contourArea(p))<minimum:
        raise ValueError(f'{label}: vùng quá nhỏ hoặc có đỉnh trùng.')
    def cross(a,b,c):
        u,v=b-a,c-a
        return float(u[0]*v[1]-u[1]*v[0])
    def hit(a,b,c,d):
        v=[cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)]
        if v[0]*v[1]<0 and v[2]*v[3]<0:return True
        for value,q,u,t in [(v[0],c,a,b),(v[1],d,a,b),(v[2],a,c,d),(v[3],b,c,d)]:
            if abs(value)<1e-10 and np.all(q>=np.minimum(u,t)-1e-9) and np.all(q<=np.maximum(u,t)+1e-9):return True
        return False
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1):continue
            if hit(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]):
                raise ValueError(f'{label}: các cạnh tự cắt; hãy vẽ theo thứ tự quanh vật.')
    return p


def poly_mask(points,grid_shape=(GRID,GRID)):
    p=polygon(points)
    gh,gw=grid_shape
    mask=np.zeros(grid_shape,np.uint8)
    cv2.fillPoly(mask,[np.round(p*[gw-1,gh-1]).astype(np.int32)],255)
    return mask


def object_mask(row,shape,grid_shape=(GRID,GRID)):
    h,w=shape[:2];gh,gw=grid_shape;mask=np.zeros(grid_shape,np.uint8)
    for r in sorted(row.get('object_contours',[]),key=lambda r:r.get('depth',int(r.get('hole',False)))):
        points=np.asarray(r['points'],np.float32)*[(gw-1)/w,(gh-1)/h]
        cv2.fillPoly(mask,[np.round(points).astype(np.int32)],0 if r.get('hole') else 255)
    return mask


def area(mask):return geometric_area(contour_rings(mask))


def prepare_review(base_objects,reviews,shape,edges=None):
    if not isinstance(reviews,list) or len(reviews)>200:raise ValueError('Tối đa 200 bản ghi vật thể.')
    h,w=shape[:2];base={r['id']:r for r in base_objects};patches={}
    for patch in reviews:
        if not isinstance(patch,dict) or not isinstance(patch.get('id'),str):raise ValueError('Bản ghi vật thể không hợp lệ.')
        key=patch['id']
        if len(key)>100 or key in patches:raise ValueError('ID vật thể bị trùng hoặc quá dài.')
        if key not in base and not key.startswith('manual-'):raise ValueError('Vật thể không thuộc phiên ảnh này.')
        patches[key]=patch
    rows=[]
    for key in list(base)+[k for k in patches if k not in base]:
        original=base.get(key,{});p=patches.get(key,{})
        status=p.get('status',original.get('status','pending'))
        if status not in ['auto','confirmed','pending','ignored']:raise ValueError('Trạng thái vật thể không hợp lệ.')
        if not original and status=='auto':raise ValueError('Vật thêm thủ công cần xác nhận, để chờ hoặc bỏ qua.')
        name=p.get('name',original.get('name','Vật chưa rõ'))
        class_key=canonical_class(p.get('class_key',original.get('class_key','other')))
        if not isinstance(name,str) or len(name)>120 or class_key not in [*CLASSES,'other']:
            raise ValueError('Tên hoặc lớp vật thể không hợp lệ.')
        quantity=p.get('quantity',1)
        if not isinstance(quantity,int) or isinstance(quantity,bool) or not 1<=quantity<=100:raise ValueError('Số lượng mỗi nhóm phải từ 1 đến 100.')
        decision=p.get('position','auto');assigned=p.get('zone_id')
        if decision not in ['auto','correct','wrong','unknown']:raise ValueError('Nhận xét vị trí không hợp lệ.')
        if assigned is not None and not isinstance(assigned,str):raise ValueError('Vùng đặt vật không hợp lệ.')
        clutter=p.get('clutter',False)
        if not isinstance(clutter,bool):raise ValueError('Cờ cần sắp xếp không hợp lệ.')
        note=p.get('note','')
        if not isinstance(note,str) or len(note)>500:raise ValueError('Ghi chú tối đa 500 ký tự.')
        rings=original.get('object_contours',[]);method=original.get('area_method','unavailable')
        manual_points=p.get('points')
        if manual_points is not None:
            coords=polygon(manual_points,'Đường bao vật thể')*[w,h]
            rings=[{'points':coords.astype(float).tolist(),'hole':False,'depth':0}];method='manual_polygon'
        verified=p.get('geometry_verified',False)
        if not isinstance(verified,bool):raise ValueError('Xác nhận đường bao phải là có/không.')
        if status=='auto' and (class_key!=original.get('class_key') or manual_points is not None or quantity!=1 or decision!='auto' or assigned or clutter):
            status='pending'
        row={'id':key,'source':'model' if original else 'manual','model_name':original.get('model_name'),
             'model_confidence':original.get('model_confidence'),'name':name.strip() or 'Vật chưa rõ',
             'class_key':class_key,'status':status,'quantity':quantity,'position_decision':decision,
             'assigned_zone_id':assigned,'clutter':clutter,'note':note,
             'object_contours':rings,'area_method':method,'geometry_verified':verified or manual_points is not None}
        geo=geometry(rings,shape,edges) if rings else None
        row['geometry']=geo;row['area_pixels']=geo['area_pixels'] if geo else None
        row['area_ratio']=geo['area_ratio'] if geo else None
        row['box']=geo['bbox'] if geo else original.get('box')
        row['anchor']=geo['contact_normalized'] if geo else ([float((row['box'][0]+row['box'][2])/2/w),float(min(1,row['box'][3]/h))] if row['box'] else None)
        rows.append(row)
    return rows


def validate_scene(zones,rules,scope,limits,config,grid_shape=(GRID,GRID)):
    if not isinstance(zones,list) or len(zones)>40:raise ValueError('Tối đa 40 vùng.')
    validate_settings(zones,rules)
    seen=set();clean=[]
    for i,z in enumerate(zones):
        key=z.get('id',f'zone-{i+1}')
        if not isinstance(key,str) or not key or len(key)>100 or key in seen:raise ValueError('ID vùng cần duy nhất.')
        seen.add(key)
        clear=z.get('clear',z['kind']=='walkway')
        if not isinstance(clear,bool):raise ValueError('Thuộc tính vùng cần trống không hợp lệ.')
        clean.append({**z,'id':key,'clear':clear,'name':ZONES[z['kind']]})
    scope_points=scope.get('points') if isinstance(scope,dict) else None
    if scope is not None and not isinstance(scope,dict):raise ValueError('Phạm vi ảnh không hợp lệ.')
    scope_mask=poly_mask(scope_points,grid_shape) if scope_points else np.full(grid_shape,255,np.uint8)
    if area(scope_mask)<1:raise ValueError('Phạm vi đánh giá quá nhỏ.')
    if not isinstance(limits,list) or len(limits)>120:raise ValueError('Tối đa 120 giới hạn số lượng.')
    unique=set()
    for item in limits:
        if not isinstance(item,dict) or item.get('zone_id') not in seen or item.get('class_key') not in [*CLASSES,'other','all']:
            raise ValueError('Giới hạn cần có vùng và loại vật hợp lệ.')
        n=item.get('max_count');key=(item['zone_id'],item['class_key'])
        if not isinstance(n,int) or isinstance(n,bool) or not 0<=n<=1000 or key in unique:
            raise ValueError('Giới hạn số lượng từ 0 đến 1000, không được trùng vùng/loại.')
        unique.add(key)
    if not isinstance(config,dict) or set(config)-set(DEFAULT_CONFIG):raise ValueError('Ngưỡng chấm điểm không hợp lệ.')
    cfg={**DEFAULT_CONFIG,**config}
    for key,value in cfg.items():
        high=.5 if key!='component_min_ratio' else .1
        if not finite(value) or not 0<=value<=high:raise ValueError('Ngưỡng diện tích/biên không hợp lệ.')
    return clean,scope_mask,cfg


def score_scene(objects,shape,zones,rules,scope=None,limits=None,config=None):
    # Normalize scale for repeatability while preserving aspect ratio for perimeter.
    h,w=shape[:2];scale=GRID/max(h,w)
    grid_shape=(max(8,round(h*scale)),max(8,round(w*scale)));gh,gw=grid_shape
    zones,scope_mask,cfg=validate_scene(zones,rules,scope,limits if limits is not None else [],config if config is not None else {},grid_shape)
    scope_area=area(scope_mask);zone_map={z['id']:z for z in zones}
    # Smaller surfaces have priority over a surrounding floor polygon.
    ordered=sorted(zones,key=lambda z:(round(abs(cv2.contourArea(np.asarray(z['points'],np.float32))),8),z['id']))
    covered=np.zeros_like(scope_mask);clear_mask=np.zeros_like(scope_mask)
    groups=defaultdict(list)
    for z in ordered:groups[round(abs(cv2.contourArea(np.asarray(z['points'],np.float32))),8)].append(z)
    for group in groups.values():
        masks=[cv2.bitwise_and(poly_mask(z['points'],grid_shape),scope_mask) for z in group]
        ambiguous_mask=np.zeros_like(scope_mask);group_union=np.zeros_like(scope_mask)
        for i,z in enumerate(group):
            group_union=cv2.bitwise_or(group_union,masks[i])
            for j in range(i):
                if (z['kind'],z['clear'])!=(group[j]['kind'],group[j]['clear']):
                    ambiguous_mask=cv2.bitwise_or(ambiguous_mask,cv2.bitwise_and(masks[i],masks[j]))
        for z,mask in zip(group,masks):
            effective=cv2.bitwise_and(mask,cv2.bitwise_not(cv2.bitwise_or(covered,ambiguous_mask)))
            if z['clear']:clear_mask=cv2.bitwise_or(clear_mask,effective)
        covered=cv2.bitwise_or(covered,group_union)
    rows=[];eligible=[];bad=[];position_bad=[];all_union=np.zeros_like(scope_mask);bad_union=np.zeros_like(scope_mask)
    for original in objects:
        row={**original};mask=object_mask(row,shape,grid_shape);clipped=cv2.bitwise_and(mask,scope_mask)
        anchor=row['anchor'];inside=False
        if anchor:
            px,py=np.clip(np.round(np.asarray(anchor)*[gw-1,gh-1]).astype(int),[0,0],[gw-1,gh-1])
            inside=bool(scope_mask[py,px])
        has_shape=bool(row.get('geometry'))
        if has_shape:inside=inside or cv2.countNonZero(clipped)>=max(1,.5*cv2.countNonZero(mask))
        matches=[]
        if anchor:
            for z in zones:
                p=np.asarray(z['points'],np.float32)
                if cv2.pointPolygonTest(p,tuple(map(float,anchor)),False)>=0:matches.append((abs(cv2.contourArea(p)),z))
        matches.sort(key=lambda x:(round(x[0],8),x[1]['id']));assigned=row['assigned_zone_id']
        if assigned and assigned not in zone_map:raise ValueError('Vùng gán cho vật đã bị xóa; chọn lại vị trí.')
        region=zone_map.get(assigned) if assigned else matches[0][1] if matches else None
        if not has_shape and assigned:
            inside=bool(cv2.countNonZero(cv2.bitwise_and(poly_mask(region['points'],grid_shape),scope_mask)))
        elif not has_shape and not anchor and not (scope and scope.get('points')) and row['status']=='confirmed' and row['position_decision'] in ['correct','wrong']:
            inside=True
        ambiguous=not assigned and len(matches)>1 and abs(matches[0][0]-matches[1][0])<1e-8 and (matches[0][1]['kind'],matches[0][1]['clear'])!=(matches[1][1]['kind'],matches[1][1]['clear'])
        row.update(state='unknown',reason='Cần xác nhận lớp/vị trí.',zone_id=region['id'] if region else None,
                   region=region['kind'] if region else None,area_in_scope_ratio=area(clipped)/scope_area if has_shape else None,
                   bad=False,position_wrong=False,position_severity=0.)
        if row['status']=='ignored':row.update(state='ignored',reason='Người dùng bỏ qua; không cộng vào số lượng, diện tích hoặc điểm phạt.')
        elif not inside:row.update(state='outside',reason='Ngoài phạm vi ảnh đang đánh giá, hoặc chưa có vị trí/đường bao.')
        elif row['status']=='pending':row['reason']='Chưa được xác nhận; không trừ điểm.'
        elif row['status']=='auto' and (row.get('model_confidence') is None or row['model_confidence']<.5):
            row['reason']='Mô hình tin cậy dưới 50%; hãy xác nhận bằng mắt, sửa hoặc bỏ qua.'
        else:
            human=row['status']=='confirmed';decision=row['position_decision'] if human else 'auto'
            if decision in ['correct','wrong']:
                row.update(state=decision,reason='Vị trí do người dùng xác nhận.')
            elif decision=='unknown':row['reason']='Người dùng chưa xác định được vị trí đúng/sai.'
            elif ambiguous:row['reason']='Hai vùng khác loại/quy tắc cùng diện tích chồng tại điểm đặt vật; hãy sửa hoặc gán vùng thủ công.'
            elif region and row['class_key'] in CLASSES:
                wrong=region['clear'] or region['kind'] not in rules[row['class_key']]
                row.update(state='wrong' if wrong else 'correct',reason=('Vùng cần trống: ' if region['clear'] else 'Đối chiếu quy tắc tại: ')+region['name'])
            elif row['class_key']=='other':row['reason']='Vật ngoài bộ lớp: cần người dùng chọn đúng/sai vị trí.'
            else:row['reason']='Chưa có vùng chứa điểm đặt vật; hãy khoanh vùng hoặc xác nhận vị trí thủ công.'
            if row['state'] in ['correct','wrong']:
                row['position_wrong']=row['state']=='wrong'
                row['bad']=row['position_wrong'] or (human and row['clutter'])
                if human and row['clutter'] and not row['position_wrong']:
                    row['state']='clutter';row['reason']='Đúng vị trí nhưng người dùng đánh dấu cần sắp xếp.'
                eligible.append(row)
                if has_shape:all_union=cv2.bitwise_or(all_union,clipped)
                if row['bad']:
                    bad.append(row)
                    if has_shape:bad_union=cv2.bitwise_or(bad_union,clipped)
                if row['position_wrong']:
                    severity=CLASS_PENALTIES.get(row['class_key'],1)*(2 if region and region['clear'] else 1)
                    row['position_severity']=severity;position_bad.append(row)
        rows.append(row)
    bad_qty=sum(r['quantity'] for r in bad)
    limit_rows=[];extra_by_zone=defaultdict(float)
    for limit in limits or []:
        members=[r for r in eligible if r['zone_id']==limit['zone_id'] and (limit['class_key']=='all' or r['class_key']==limit['class_key'])]
        count=sum(r['quantity'] for r in members);excess=max(0,count-limit['max_count'])
        overlap=sum(r['quantity'] for r in members if r['bad'])
        extra_by_zone[limit['zone_id']]=max(extra_by_zone[limit['zone_id']],max(0,excess-overlap))
        limit_rows.append({**limit,'count':count,'excess':excess})
    quantity_units=bad_qty+sum(extra_by_zone.values())
    known_bad=[r for r in bad if r.get('geometry')];missing_bad=len(bad)-len(known_bad)
    bad_area=area(bad_union);clear_area=area(clear_mask);blocked=cv2.bitwise_and(bad_union,clear_mask)
    ratio=bad_area/scope_area;clear_ratio=area(blocked)/clear_area if clear_area else 0.
    count,_,stats,_=cv2.connectedComponentsWithStats(bad_union)
    components=int(sum(s[cv2.CC_STAT_AREA]>=max(4,cfg['component_min_ratio']*scope_area) for s in stats[1:])) if count>1 else 0
    inner=cv2.erode(scope_mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    contours,_=cv2.findContours(bad_union,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE);perimeter=0.
    for contour in contours:
        p=contour.reshape(-1,2);q=np.roll(p,-1,axis=0)
        valid=(inner[p[:,1],p[:,0]]>0)&(inner[q[:,1],q[:,0]]>0)
        perimeter+=float(np.linalg.norm(q-p,axis=1)[valid].sum())
    boundary_index=perimeter/(2*math.sqrt(math.pi*scope_area))
    quantities={'position':2*max((r['position_severity'] for r in position_bad),default=0.),
                'quantity':quantity_units,'area':40*max(0,ratio-cfg['area_allowance'],clear_ratio-cfg['clear_allowance']),
                'contours':2*max(0,components-1),'boundary':2*max(0,boundary_index-cfg['boundary_allowance'])}
    labels={'position':'Vị trí','quantity':'Số lượng cần sắp xếp','area':'Diện tích chiếm sai / cản lối','contours':'Phân tán các mảng contour','boundary':'Đường biên vật cần sắp xếp'}
    formulas={'position':'min(12, 2 × mức nghiêm trọng lớn nhất); vùng cần trống nhân 2',
      'quantity':'min(12, số vật sai/cần sắp xếp + phần vượt giới hạn chưa tính)',
      'area':'min(16, 40 × max(0, r_sai − ngưỡng miễn, r_cản − ngưỡng miễn vùng trống))',
      'contours':'min(10, 2 × max(0, số mảng rời đủ lớn − 1))',
      'boundary':'min(10, 2 × max(0, L / (2√(πA_phạm_vi)) − ngưỡng miễn))'}
    breakdown=[]
    for key in CAPS:
        applicable=not bad or bool(known_bad) or key in ['position','quantity']
        breakdown.append({'key':key,'label':labels[key],'raw':float(quantities[key]),'cap':CAPS[key],
                          'deduction':round(min(CAPS[key],quantities[key]),4) if applicable else 0.,
                          'applicable':applicable,'partial':bool(missing_bad) and key in ['area','contours','boundary'],
                          'formula':formulas[key]})
    deduction=sum(r['deduction'] for r in breakdown)
    assessed=len(eligible);total_in_scope=sum(r['state']!='outside' for r in rows)
    score=round(max(40.,100-deduction),2) if assessed else None
    return {'method':'reviewed_scene_v5','score':score,'baseline':100,'floor':40,'deduction':round(deduction,4),
      'formula':'S = max(40, 100 − P_vị_trí − P_số_lượng − P_diện_tích − P_contour − P_biên)',
      'breakdown':breakdown,'objects':rows,'evaluated':assessed,'total':len(rows),
      'ignored':sum(r['state']=='ignored' for r in rows),'unknown':sum(r['state']=='unknown' for r in rows),
      'outside':sum(r['state']=='outside' for r in rows),'human_confirmed':sum(r['status']=='confirmed' and r['state'] not in ['outside','ignored'] for r in rows),
      'coverage':assessed/total_in_scope if total_in_scope else 0.,'wrong':len(position_bad),
      'bad_quantity':bad_qty,'quantity_units':quantity_units,'limit_details':limit_rows,
      'bad_area_ratio':ratio,'known_occupancy_ratio':area(all_union)/scope_area,'clear_blocked_ratio':clear_ratio,
      'bad_components':int(components),'bad_boundary_index':boundary_index,'bad_boundary_length_grid':perimeter,
      'missing_bad_geometry':missing_bad,'scope_kind':'user_polygon' if scope and scope.get('points') else 'whole_image',
      'geometry_grid':{'width':gw,'height':gh,'max_side':GRID,'preserve_aspect_ratio':True},'config':cfg,'provisional':True,
      'note':'Điểm theo quy tắc trên phần vật thể đã đánh giá. Vật bỏ qua/chưa rõ không bị phạt và không chứng minh phòng gọn. Diện tích là phép chiếu trên ảnh, không phải m². Mật độ cạnh và contour toàn ảnh chỉ để tham khảo; hoa văn không tự gây trừ điểm. Tỷ lệ đã đánh giá chỉ tính trên danh sách vật; mô hình có thể còn bỏ sót vật ngoài danh sách.',
      '_scope_mask':scope_mask,'_bad_union':bad_union,'_all_union':all_union,'_clear_mask':clear_mask}
