"""Repeatable image diagnostics and contour geometry; these are not object labels."""
import math
import cv2
import numpy as np
from object_contours import contour_rings, geometric_area


def rings_mask(rings, shape):
    mask=np.zeros(shape[:2],np.uint8)
    for ring in sorted(rings,key=lambda r:r.get('depth',int(r.get('hole',False)))):
        points=np.asarray(ring['points'],np.float32).reshape(-1,2)
        if len(points)>=3:
            cv2.fillPoly(mask,[np.round(points).astype(np.int32)],0 if ring.get('hole') else 255)
    return mask


def geometry(rings, shape, edges=None):
    h,w=shape[:2]
    outer=[np.asarray(r['points'],np.float32).reshape(-1,2) for r in rings if not r.get('hole')]
    if not outer or geometric_area(rings)<=0:
        return None
    points=np.concatenate(outer)
    area=geometric_area(rings)
    perimeter=sum(float(cv2.arcLength(np.asarray(r['points'],np.float32),True)) for r in rings)
    hull=cv2.convexHull(points);hull_area=float(cv2.contourArea(hull))
    x1,y1=points.min(axis=0);x2,y2=points.max(axis=0)
    bw,bh=float(x2-x1),float(y2-y1)
    moments=[((-1 if ring.get('hole') else 1),cv2.moments(np.asarray(ring['points'],np.float32))) for ring in rings]
    m00=sum(sign*m['m00'] for sign,m in moments)
    center=[sum(sign*m[key] for sign,m in moments)/m00 for key in ['m10','m01']] if m00>0 else points.mean(axis=0).tolist()
    rect=cv2.minAreaRect(points);corners=cv2.boxPoints(rect)
    mask=rings_mask(rings,(h,w));n,_,stats,_=cv2.connectedComponentsWithStats(mask)
    rows=np.where(mask>0)
    bottom=float(np.quantile(rows[0],.95))
    contact=[float(np.median(rows[1][rows[0]>=bottom])),float(rows[0].max())]
    result=dict(area_pixels=round(area,3),area_ratio=area/(h*w),perimeter_pixels=round(perimeter,3),
        centroid=[round(v,3) for v in center],centroid_normalized=[float(center[0]/w),float(center[1]/h)],
        contact_normalized=[float(contact[0]/w),float(contact[1]/h)],
        bbox=[float(x1),float(y1),float(x2),float(y2)],bbox_width=bw,bbox_height=bh,
        rotated_rect={'center':list(map(float,rect[0])),'size':list(map(float,rect[1])),'angle':float(rect[2]),'corners':corners.astype(float).tolist()},
        convex_hull_area=hull_area,solidity=min(1.,area/hull_area) if hull_area else None,
        extent=min(1.,area/(bw*bh)) if bw*bh else None,aspect_ratio=bw/bh if bh else None,
        circularity=min(1.,4*math.pi*area/(perimeter*perimeter)) if perimeter else None,
        equivalent_diameter=2*math.sqrt(area/math.pi),
        contour_count=len(outer),hole_count=sum(bool(r.get('hole')) for r in rings),
        connected_components=n-1,vertices=sum(len(r['points']) for r in rings),
        mask_pixels=int(cv2.countNonZero(mask)))
    if edges is not None:
        result['internal_edge_density']=float(np.count_nonzero((edges>0)&(mask>0))/max(1,cv2.countNonZero(mask)))
    return result


def extract_features(image):
    h,w=image.shape[:2];total=h*w
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,80,160)
    binary=cv2.dilate(cv2.threshold(edges,50,255,cv2.THRESH_BINARY)[1],np.ones((5,5),np.uint8),iterations=2)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    selected=[c for c in contours if cv2.contourArea(c)>total*.001]
    raw=image.copy();filtered=image.copy();cv2.drawContours(raw,contours,-1,(70,160,220),1)
    cv2.drawContours(filtered,selected,-1,(70,170,60),2)
    union=np.zeros((h,w),np.uint8)
    cv2.drawContours(union,selected,-1,255,cv2.FILLED)
    contour_rows=[]
    for i,c in enumerate(sorted(selected,key=cv2.contourArea,reverse=True),1):
        rings=[{'points':c.reshape(-1,2).astype(float).tolist(),'hole':False,'depth':0}]
        row=geometry(rings,image.shape,edges)
        if row:contour_rows.append({'id':f'cv-{i}',**row})
    metrics={'width':w,'height':h,'pixel_count':total,'edge_pixels':int(cv2.countNonZero(edges)),
        'edge_density':float(cv2.countNonZero(edges)/total),'raw_contour_count':len(contours),'contour_count':len(selected),
        'region_area_pixels':geometric_area(contour_rings(union)),
        'region_area_ratio':geometric_area(contour_rings(union))/total,
        'contour_area_sum':sum(float(cv2.contourArea(c)) for c in selected),
        'contours_per_megapixel':len(selected)/(total/1_000_000),
        'mean_contour_area':float(np.mean([r['area_pixels'] for r in contour_rows])) if contour_rows else 0,
        'parameters':{'gaussian_kernel':[5,5],'canny_low':80,'canny_high':160,'dilation_kernel':[5,5],
                      'dilation_iterations':2,'minimum_contour_area_ratio':.001}}
    stages=[
      ('original','Ảnh đầu vào',image,'Ảnh màu ở kích thước phân tích; giữ tỷ lệ khung hình.','Ảnh màu BGR'),
      ('gray','Ảnh xám',gray,'Y = 0,299R + 0,587G + 0,114B','cv2.cvtColor(..., COLOR_BGR2GRAY)'),
      ('blur','Giảm nhiễu Gaussian',blur,'I′(x,y) = Σ G(u,v)I(x−u,y−v); kernel 5×5','cv2.GaussianBlur(gray, (5,5), 0)'),
      ('edges','Đường biên Canny',edges,'Edge density = số pixel cạnh / số pixel ảnh','cv2.Canny(blur, 80, 160)'),
      ('binary','Giãn biên / ảnh nhị phân',binary,'Ghép các nét cạnh gần nhau để tạo vùng; đây chưa phải mask vật thể.','cv2.threshold + cv2.dilate(kernel 5×5, 2 lần)'),
      ('raw_contours','Tất cả contour OpenCV',raw,'Contour mô tả vùng cạnh, không tương đương số đồ vật.','cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)'),
      ('contours','Contour sau lọc diện tích',filtered,'Giữ A > 0,001 × W × H; A = cv2.contourArea(contour)','cv2.contourArea + cv2.drawContours')]
    return {'metrics':metrics,'contours':contour_rows,'stages':stages,'edges':edges}
