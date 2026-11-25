"""Loại nhãn tự động sai rõ trong lần xem ảnh và mặt nạ trùng; vẫn cần rà soát đủ lớp."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np


def clean(folder):
    path=folder/'annotations.draft.coco.json';dataset=json.loads(path.read_text())
    decisions=json.loads((folder/'manual_rejections.json').read_text())
    rejected_path=folder/'rejected_annotations.json'
    rejected=json.loads(rejected_path.read_text()) if rejected_path.exists() else []
    previous_ids={row['annotation']['id'] for row in rejected}
    images={im['id']:im for im in dataset['images']};kept=[];masks={}
    for a in sorted(dataset['annotations'],key=lambda a: (not a['annotation_source'].startswith('coco'),-(a.get('detector_confidence') or 0))):
        reason=None
        if str(a['id']) in decisions:
            reason=decisions[str(a['id'])]
        if a['annotation_source'].startswith('automatic') and not reason:
            im=images[a['image_id']];mask=np.zeros((256,256),np.uint8)
            for polygon in a['segmentation']:
                pts=np.array(polygon).reshape(-1,2)*[256/im['width'],256/im['height']]
                cv2.fillPoly(mask,[np.round(pts).astype(np.int32)],1)
            key=(a['image_id'],a['category_id'])
            for prior in masks.get(key,[]):
                union=np.count_nonzero(mask|prior)
                if union and np.count_nonzero(mask&prior)/union>.90:
                    reason='Automatic mask duplicates another suggestion of the same class (IoU > 0.90).';break
            if not reason:masks.setdefault(key,[]).append(mask)
        if reason:
            if a['id'] not in previous_ids:
                rejected.append({'annotation':a,'reason':reason});previous_ids.add(a['id'])
        else:kept.append(a)
    dataset['annotations']=sorted(kept,key=lambda a:a['id'])
    dataset['info']['thumbnail_screened']=True
    dataset['info']['fully_reviewed_images']=0
    path.write_text(json.dumps(dataset,ensure_ascii=False))
    rejected_path.write_text(json.dumps(rejected,ensure_ascii=False,indent=2))
    names={c['id']:c['name'] for c in dataset['categories']};counts=Counter();per_image={}
    for a in kept:
        source='source' if a['annotation_source'].startswith('coco') else 'automatic_draft'
        counts[(source,names[a['category_id']])]+=1
        per_image.setdefault(a['image_id'],Counter())[source]+=1
    records=json.loads((folder/'manifest.json').read_text())
    review=[dict(image=im['file_name'],collection=im['collection'],source_polygons=per_image.get(im['id'],{}).get('source',0),
                 draft_polygons=per_image.get(im['id'],{}).get('automatic_draft',0),review_status='pending_all_12_classes') for im in dataset['images']]
    (folder/'review_manifest.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
    stats=json.loads((folder/'stats.json').read_text())
    for name in stats['classes']:
        stats['classes'][name]={'source_polygons':counts[('source',name)],'automatic_draft_polygons':counts[('automatic_draft',name)]}
    stats['rejected_drafts']=len(rejected);stats['object_annotations']=len(kept)
    stats['polygon_count']=sum(len(a['segmentation']) for a in kept)
    (folder/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print('Remaining object annotations',len(kept),'rejected drafts',len(rejected))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    clean(parser.parse_args().folder)
