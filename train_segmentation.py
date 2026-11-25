"""Huấn luyện YOLO segmentation từ bộ YOLO polygon đã được kiểm tra."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile

TARGET_CLASSES=['bag','clothes','blanket','pillow','footwear','trash','handbag','laptop','book','bottle','chair','backpack']


def validate_polygon_label(path,class_count):
    counts=Counter()
    for row,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        parts=line.split()
        if len(parts)<7 or (len(parts)-1)%2:
            raise ValueError(f'{path.name}:{row}: cần nhãn polygon, không phải nhãn box 5 cột.')
        cls=int(parts[0]);values=list(map(float,parts[1:]))
        if not 0<=cls<class_count or any(not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError(f'{path.name}:{row}: ID lớp hoặc tọa độ không hợp lệ.')
        points=list(zip(values[0::2],values[1::2]))
        area=abs(sum(x*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*y for i,(x,y) in enumerate(points)))/2
        if area<1e-8:raise ValueError(f'{path.name}:{row}: polygon không có diện tích.')
        counts[cls]+=1
    return counts


def validate_dataset(data_path):
    from ultralytics.utils import YAML
    config=YAML.load(str(data_path))
    if not isinstance(config,dict) or 'names' not in config:
        raise ValueError('Cần data.yaml từ bộ YOLO segmentation đã kiểm tra, không dùng COCO nháp trực tiếp.')
    names=config['names']
    if isinstance(names,dict):
        keys=sorted(names,key=int)
        if [int(k) for k in keys]!=list(range(len(keys))):
            raise ValueError('ID lớp trong names phải liên tiếp từ 0.')
        names=[names[k] for k in keys]
    if len(names)!=len(set(names)) or set(names)!=set(TARGET_CLASSES):
        raise ValueError('Bộ này cần đủ 12 tên lớp mục tiêu; giữ ID đúng như nhãn xuất, không tự đảo thứ tự.')
    base=Path(config.get('path') or data_path.parent)
    if not base.is_absolute():base=data_path.parent/base
    normalized={'path':str(base.resolve()),'names':names};summary={}
    for split in ['train','val','test']:
        if split=='test' and not config.get(split):continue
        value=config.get(split)
        if not isinstance(value,str):raise ValueError(f'Cần đường dẫn thư mục {split} trong data.yaml.')
        images=Path(value)
        if not images.is_absolute():images=base/images
        images=images.resolve()
        if not images.is_dir() or images.name!='images':
            raise ValueError(f'{split}: không tìm thấy thư mục ảnh {images}; kiểm tra path/train/val/test.')
        files=sorted(p for p in images.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp'])
        if not files:raise ValueError(f'{split}: không có ảnh.')
        counts=Counter()
        for image in files:
            label=images.parent/'labels'/image.relative_to(images).with_suffix('.txt')
            if not label.exists():
                raise ValueError(f'Thiếu nhãn {label}. Nếu ảnh thật sự không có vật mục tiêu, cần tệp nhãn rỗng đã kiểm tra.')
            counts.update(validate_polygon_label(label,len(names)))
        missing=[name for i,name in enumerate(names) if not counts[i]]
        if split=='train' and missing:
            raise ValueError('Train chưa có ví dụ cho: '+', '.join(missing))
        summary[split]={'images':len(files),'instances':{names[i]:counts[i] for i in range(len(names))},'missing_classes':missing}
        normalized[split]=str(images)
    return normalized,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',default='yolo11n-seg.pt')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.batch,args.imgsz)<=0:parser.error('epochs, batch và imgsz phải dương.')
    try:
        config,summary=validate_dataset(args.data.expanduser().resolve())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.validate_only:return
    from ultralytics import YOLO
    model=YOLO(args.model)
    if model.task!='segment':parser.error('Cần mô hình segmentation, ví dụ yolo11n-seg.pt; không dùng weights.pt detection cũ.')
    with tempfile.TemporaryDirectory(prefix='room-segmentation-') as tmp:
        config_path=Path(tmp)/'data.yaml'
        config_path.write_text(json.dumps(config,ensure_ascii=False),encoding='utf-8')
        model.train(data=str(config_path),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
                    device=args.device,workers=0,seed=42,project=str(Path(__file__).resolve().parent/'runs'/'segment'),name='room_objects_seg')
    print('Checkpoint tốt nhất:',model.trainer.best)
    print('Chưa thay thế weights.pt của app. App cần hỗ trợ đọc masks từ mô hình segmentation để dùng trực tiếp.')

if __name__=='__main__':main()
