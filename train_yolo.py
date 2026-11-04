"""Fine-tune YOLO trên dữ liệu đã gán nhãn; không tự tạo nhãn giả."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', help='Đường dẫn data.yaml xuất từ công cụ gán nhãn')
    source.add_argument('--starter100', action='store_true', help='Dùng bộ 100 ảnh có sẵn (6 lớp)')
    parser.add_argument('--model', default=str(Path(__file__).with_name('yolo11n.pt')))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu', help='cpu hoặc 0 nếu có GPU CUDA')
    parser.add_argument('--workers', type=int, default=0, help='0 tương thích Windows')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='room_objects')
    args = parser.parse_args()
    if args.starter100:
        dataset = Path(__file__).resolve().parent / 'training' / 'starter100'
        data = dataset / 'data-local.yaml'
        template = (dataset / 'data.yaml').read_text()
        data.write_text(template.replace('path: .', 'path: ' + json.dumps(dataset.as_posix())), encoding='utf-8')
    else:
        data = Path(args.data).expanduser().resolve()
    if not data.is_file():
        parser.error(f'Không tìm thấy dữ liệu: {data}')
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        parser.error('epochs, imgsz, batch phải dương; workers không được âm.')
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                project=args.project, name=args.name, seed=42)
    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise RuntimeError('Huấn luyện kết thúc nhưng không tìm thấy best.pt.')
    print(f'Mô hình tốt nhất: {best}')
    print('Đặt YOLO_MODEL_PATH trỏ đến tệp này trước khi khởi động lại web_app.py.')


if __name__ == '__main__':
    main()
