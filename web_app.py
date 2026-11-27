"""Giao diện web: python web_app.py (mặc định cổng 8000)."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Timer

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from chamdiem import calculate_score
from object_contours import measure_detections, measure_mask_contour, extract_object_contour
from placement_score import score_placement, CLASSES, ZONES, CLASS_PENALTIES, canonical_class

OPEN_IMAGE = Image.open  # Giữ bộ đọc Pillow trước khi Ultralytics thay đổi Image.open.
from image_workbench import create_session,scoring_result,export_result
ROOT = Path(__file__).resolve().parent
MODEL = None
MODEL_FILE = None
MODEL_LOCK = Lock()
LABELS = ['Phòng bừa bộn', 'Phòng hơi bừa bộn', 'Phòng khá gọn', 'Phòng gọn gàng']


def configured_model_path():
    # Chỉ dùng mô hình được chọn; không tự rơi về YOLO nền khi weights.pt lỗi.
    path = Path(os.environ.get('YOLO_MODEL_PATH') or (ROOT / 'weights.pt')).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_model():
    global MODEL, MODEL_FILE
    path = configured_model_path()
    if not path.is_file():
        raise FileNotFoundError('Chưa có mô hình: chép weights.pt từ Roboflow vào cùng thư mục web_app.py rồi mở lại app.')
    if MODEL is None or MODEL_FILE != path:
        from ultralytics import YOLO
        MODEL = YOLO(str(path))
        MODEL_FILE = path
    return MODEL


def model_info():
    try:
        with MODEL_LOCK:
            model = load_model()
            classes = list(model.names.values())
        known = {canonical_class(n) for n in classes}
        return dict(ready=True, file=configured_model_path().name, classes=classes,
                    task=model.task, missing_classes=[k for k in CLASSES if k not in known])
    except Exception as exc:
        return dict(ready=False, file=configured_model_path().name,
                    error=f'{type(exc).__name__}: {str(exc)[:240]}')


def image_url(image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Không thể tạo ảnh kết quả.')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()


def segmentation_masks(prediction,image_shape):
    """Undo YOLO letterbox padding before measuring masks in original-image pixels."""
    from ultralytics.utils.ops import scale_masks
    if prediction.masks is None or len(prediction.masks.data)!=len(prediction.boxes):
        yield from (None for _ in prediction.boxes)
        return
    for mask in prediction.masks.data:
        scaled=scale_masks(mask[None,None],image_shape[:2])[0,0]
        yield scaled.cpu().numpy()


def detect_for_workbench(image,confidence):
    """Keep low-confidence geometry for human review without pretending it is confirmed."""
    with MODEL_LOCK:
        model=load_model()
        prediction=model.predict(image,device='cpu',imgsz=640,conf=confidence,max_det=100,verbose=False)[0]
        detections=[dict(name=model.names[int(box.cls.item())],confidence=float(box.conf.item()),
                         box=box.xyxy[0].tolist()) for box in prediction.boxes]
        if model.task=='segment':
            for det,mask in zip(detections,segmentation_masks(prediction,image.shape)):
                det.update(measure_mask_contour(mask,image.shape))
        else:
            for det in detections:det.update(extract_object_contour(image,det['box']))
        info={'file':configured_model_path().name,'task':model.task,'classes':list(model.names.values())}
    return detections,info


def analyze(data):
    try:
        raw = base64.b64decode(data['image'], validate=True)
        with OPEN_IMAGE(io.BytesIO(raw), formats=('JPEG', 'PNG', 'BMP', 'WEBP')) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError('Ảnh quá lớn. Vui lòng chọn ảnh dưới 25 triệu pixel.')
            rgb = np.array(ImageOps.exif_transpose(source).convert('RGB'))
    except (KeyError, UnidentifiedImageError, OSError, base64.binascii.Error) as exc:
        raise ValueError('Tệp không phải ảnh hợp lệ. Hãy chọn JPG, PNG, WebP hoặc BMP.') from exc
    if min(rgb.shape[:2]) < 2:
        raise ValueError('Ảnh cần có chiều rộng và chiều cao ít nhất 2 pixel.')
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    score, count, density, area, clutter, contours = calculate_score(image)
    penalties = [min(density * 300, 100), min(count * 3, 100), clutter, min(area * 200, 100)]
    weights = [.25, .30, .25, .20]
    annotated = image.copy()
    cv2.drawContours(annotated, contours, -1, (43, 190, 90), 2)
    output = dict(score=score, label=LABELS[3 if score >= 80 else 2 if score >= 60 else 1 if score >= 40 else 0],
                  regions=count, density=density, area=area,
                  penalties=penalties, deductions=[p*w for p,w in zip(penalties, weights)],
                  contour_image=image_url(annotated), detections=[], yolo_image=None, yolo_error=None)
    placement = data.get('scoring_mode') == 'placement'
    if placement:
        from placement_score import validate_settings
        validate_settings(data.get('zones', []), data.get('rules', {}))
        if not data.get('zones'):
            raise ValueError('Hãy khoanh ít nhất một vùng trước khi chấm theo vị trí.')
        if not data.get('yolo'):
            raise ValueError('Chấm theo vị trí cần bật YOLO.')
    confidence = data.get('confidence', .25)
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not .1 <= confidence <= .9:
        raise ValueError('Ngưỡng tin cậy phải từ 0,10 đến 0,90.')
    output['confidence_threshold'] = confidence
    masks = None
    if data.get('yolo', False):
        try:
            with MODEL_LOCK:
                model = load_model()
                prediction = model.predict(image, device='cpu', imgsz=640, conf=confidence, verbose=False)[0]
                output['model_file'] = configured_model_path().name
                output['model_task'] = model.task
                output['yolo_image'] = image_url(prediction.plot())
                output['detections'] = [dict(name=MODEL.names[int(box.cls.item())],
                    confidence=float(box.conf.item()), box=box.xyxy[0].tolist()) for box in prediction.boxes]
                if model.task == 'segment':
                    masks = segmentation_masks(prediction,image.shape)
        except Exception as exc:
            output['yolo_error'] = f'Không nạp/chạy được mô hình {configured_model_path().name}: {type(exc).__name__}: {str(exc)[:240]}. Kiểm tra weights.pt và phiên bản ultralytics; không dùng mô hình nền thay thế.'
    if output['detections']:
        output['detections'] = measure_detections(image,output['detections'],masks)
    if placement:
        if output['yolo_error']:
            output['score'] = None
            output['label'] = 'Không chấm được: YOLO gặp lỗi'
        else:
            output['placement'] = score_placement(output['detections'], image.shape, data['zones'], data['rules'])
            audit = image.copy()
            colors = {'correct':(55,170,55), 'wrong':(30,30,220), 'unknown':(0,170,230)}
            for zone in data['zones']:
                polygon = (np.array(zone['points']) * [image.shape[1]-1,image.shape[0]-1]).astype(np.int32)
                cv2.polylines(audit,[polygon],True,(180,130,70),2)
            for i, item in enumerate(output['placement']['detections'],1):
                x1,y1,x2,y2 = map(int,item['box'])
                color = colors[item['state']]
                for ring in item.get('object_contours',[]):
                    polygon = np.round(ring['points']).astype(np.int32)
                    cv2.polylines(audit,[polygon],True,color,2)
                cv2.circle(audit,((x1+x2)//2,y2),4,color,-1)
                text = f"#{i} {item['name']} {item['state']} -{item['position_penalty']}"
                cv2.putText(audit,text,(max(0,x1),max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
            output['yolo_image'] = image_url(audit)
            output['score'] = output['placement']['score']
            value = output['score']
            output['label'] = ('Chưa đủ dữ liệu để chấm' if value is None else LABELS[3 if value >= 90 else 2 if value >= 75 else 1 if value >= 55 else 0])
    if placement and (output.get('placement', {}).get('unknown', 0) or output.get('placement', {}).get('area_unknown_wrong',0)) and output['score'] is not None:
        output['label'] = 'Điểm tạm tính trên phần đã đánh giá'
    output['scoring_mode'] = 'placement' if placement else 'opencv'
    if MODEL is not None:
        output['model_classes'] = list(MODEL.names.values())
    return output


class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, content_type='application/json; charset=utf-8'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (ROOT / 'web' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/model':
            self.send(200, model_info())
        elif self.path == '/settings':
            self.send(200, {'classes': {k:{'label':v[0], 'allowed':v[1], 'penalty':CLASS_PENALTIES[k]} for k,v in CLASSES.items()}, 'zones':ZONES})
        elif self.path == '/placement.js':
            self.send(200, (ROOT / 'web' / 'placement.js').read_bytes(), 'text/javascript; charset=utf-8')
        elif self.path == '/app.js':
            self.send(200,(ROOT/'web'/'app.js').read_bytes(),'text/javascript; charset=utf-8')
        elif self.path == '/styles.css':
            self.send(200,(ROOT/'web'/'styles.css').read_bytes(),'text/css; charset=utf-8')
        elif self.path == '/legacy':
            self.send(200,(ROOT/'web'/'legacy.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path == '/health':
            self.send(200, {'status': 'ok'})
        elif self.path == '/sample':
            self.send(200, (ROOT / 'room.jpg').read_bytes(), 'image/jpeg')
        else:
            self.send(404, {'error': 'Không tìm thấy trang.'})

    def do_POST(self):
        endpoints=['/api/analyze','/api/workbench/analyze','/api/workbench/score','/api/workbench/export']
        if self.path not in endpoints:
            return self.send(404, {'error': 'Không tìm thấy chức năng.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20_000_000:
                return self.send(413, {'error': 'Vui lòng chọn ảnh có dung lượng dưới 14 MB.'})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Yêu cầu không hợp lệ.')
            if self.path=='/api/analyze':self.send(200,analyze(data))
            else:
                if self.path.endswith('/analyze'):self.send(200,create_session(data,detect_for_workbench))
                elif self.path.endswith('/score'):self.send(200,scoring_result(data))
                else:self.send(200,export_result(data),'application/zip')
        except (ValueError, TypeError, Image.DecompressionBombError) as exc:
            self.send(400, {'error': str(exc)})
        except Exception:
            self.send(500, {'error': 'Không xử lý được ảnh. Vui lòng thử ảnh khác.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    print(f'Ứng dụng đang chạy ở cổng {args.port}. Nhấn Ctrl+C để dừng.', flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.open_browser:
        import webbrowser
        Timer(1, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    server.serve_forever()
