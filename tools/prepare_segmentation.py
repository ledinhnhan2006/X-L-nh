"""Tạo COCO polygon hỗ trợ gán nhãn, KHÔNG coi nhãn tự động là ground truth."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
import torch
from ultralytics import YOLO,SAM

CLASSES=['bag','clothes','blanket','pillow','footwear','trash','handbag','laptop','book','bottle','chair','backpack']
COCO_CLASSES={24:'backpack',26:'handbag',39:'bottle',56:'chair',63:'laptop',73:'book'}


def make_annotation(image_id,category,polygons,source,confidence=None,holes=False):
    points=np.concatenate([np.array(p,dtype=float).reshape(-1,2) for p in polygons])
    x1,y1=points.min(axis=0);x2,y2=points.max(axis=0)
    return dict(image_id=image_id,category_id=CLASSES.index(category)+1,segmentation=polygons,
                bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                area=sum(float(cv2.contourArea(np.array(p,np.float32).reshape(-1,2))) for p in polygons),iscrowd=0,
                annotation_source=source,needs_review=True,detector_confidence=confidence,holes_need_review=holes)


def prepare(folder,archive,model_path,sam_path):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    records=json.loads((folder/'manifest.json').read_text())
    detector=YOLO(str(model_path));segmenter=SAM(str(sam_path))
    dataset={'info':{'description':'DRAFT annotation assistance: not exhaustive, not a train-ready 12-class dataset',
                     'review_status':'pending_all_images','roboflow_43_images_included':False},
             'images':[],'annotations':[],
             'categories':[{'id':i+1,'name':name,'supercategory':'room_object'} for i,name in enumerate(CLASSES)]}
    review=[];counts=Counter();skipped=0
    with ZipFile(archive) as labels:
        filenames=set(labels.namelist())
        for idx,row in enumerate(records,1):
            path=folder/row['file'];image=cv2.imread(str(path))
            if image is None:raise ValueError(f'Cannot read {path.name}')
            h,w=image.shape[:2]
            dataset['images'].append(dict(id=idx,file_name=row['file'],width=w,height=h,
                source_coco_id=row['coco_id'],collection=row['collection'],review_status='pending'))
            annotation_count={'source':0,'automatic':0}
            label=f"coco/labels/val2017/{row['coco_id']:012d}.txt"
            if label in filenames:
                for line in labels.read(label).decode().splitlines():
                    parts=line.split();class_id=int(parts[0])
                    if class_id not in COCO_CLASSES:continue
                    coords=np.array(list(map(float,parts[1:])),dtype=float).reshape(-1,2)
                    if len(coords)<3:continue
                    coords=np.clip(coords,0,1)*[w,h]
                    annotation=make_annotation(idx,COCO_CLASSES[class_id],[coords.reshape(-1).tolist()],'coco_polygon_ultralytics_conversion')
                    annotation['id']=len(dataset['annotations'])+1;dataset['annotations'].append(annotation)
                    annotation_count['source']+=1;counts[('source',COCO_CLASSES[class_id])]+=1
            predictions=detector.predict(image,device='cpu',imgsz=960,conf=.45,max_det=100,verbose=False)[0]
            candidates=[]
            for box in predictions.boxes:
                name=str(detector.names[int(box.cls.item())]).strip().lower()
                if name in CLASSES and name not in COCO_CLASSES.values():
                    candidates.append((name,float(box.conf.item()),box.xyxy[0].cpu().numpy()))
            if candidates:
                result=segmenter.predict(image,bboxes=[b.tolist() for _,_,b in candidates],device='cpu',verbose=False)[0]
                if result.masks is None or len(result.masks.data)!=len(candidates):
                    raise RuntimeError('SAM prompt/mask count mismatch')
                for (name,confidence,box),mask in zip(candidates,result.masks.data):
                    binary=(mask.cpu().numpy()>.5).astype(np.uint8)*255
                    if binary.shape!=(h,w):
                        binary=cv2.resize(binary,(w,h),interpolation=cv2.INTER_NEAREST)
                    x1,y1,x2,y2=np.round(box).astype(int);x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
                    area=cv2.countNonZero(binary);inside=cv2.countNonZero(binary[y1:y2,x1:x2])
                    if area<20 or area>.9*w*h or inside/max(1,area)<.5:
                        skipped+=1;continue
                    contours,hierarchy=cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    polygons=[];holes=False
                    for j,contour in enumerate(contours):
                        if hierarchy[0,j,3]!=-1:holes=True;continue
                        if cv2.contourArea(contour)<max(8,area*.005):continue
                        approx=cv2.approxPolyDP(contour,max(.5,.001*cv2.arcLength(contour,True)),True)
                        if len(approx)>=3:polygons.append(approx.reshape(-1).astype(float).tolist())
                    if not polygons:skipped+=1;continue
                    annotation=make_annotation(idx,name,polygons,'automatic_detector_plus_sam_draft',confidence,holes)
                    annotation['id']=len(dataset['annotations'])+1;dataset['annotations'].append(annotation)
                    annotation_count['automatic']+=1;counts[('automatic',name)]+=1
            review.append(dict(image=row['file'],collection=row['collection'],source_polygons=annotation_count['source'],
                               draft_polygons=annotation_count['automatic'],review_status='pending_all_12_classes'))
            if idx%5==0 or idx==len(records):
                (folder/'annotations.draft.coco.json').write_text(json.dumps(dataset,ensure_ascii=False))
                print(f'Prepared {idx}/{len(records)} images; {len(dataset["annotations"])} polygons',flush=True)
    (folder/'review_manifest.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
    stats={'image_count':len(records),'old_room_images':sum(r['collection']=='previous_rooms100' for r in records),
           'new_images':sum(r['collection']=='new_30' for r in records),'roboflow_43_included':False,'reviewed_images':0,
           'ready_to_train_12_classes':False,'skipped_automatic_masks':skipped,
           'classes':{name:{'source_polygons':counts[('source',name)],'automatic_draft_polygons':counts[('automatic',name)]} for name in CLASSES},
           'segmentation_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'detector_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
           'sam_sha256':hashlib.sha256(sam_path.read_bytes()).hexdigest()}
    (folder/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--coco-segments',type=Path,required=True)
    parser.add_argument('--detector',type=Path,default=Path('weights.pt'))
    parser.add_argument('--sam',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.folder.resolve(),args.coco_segments.resolve(),args.detector.resolve(),args.sam.resolve())
