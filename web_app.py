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
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from chamdiem import calculate_score

ROOT = Path(__file__).resolve().parent
MODEL = None
MODEL_LOCK = Lock()
LABELS = ['Phòng bừa bộn', 'Phòng hơi bừa bộn', 'Phòng khá gọn', 'Phòng gọn gàng']


def image_url(image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Không thể tạo ảnh kết quả.')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()


def analyze(data):
    try:
        raw = base64.b64decode(data['image'], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
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
    if data.get('yolo', False):
        try:
            global MODEL
            with MODEL_LOCK:
                if MODEL is None:
                    from ultralytics import YOLO
                    MODEL = YOLO(str(ROOT / 'yolo11n.pt'))
                prediction = MODEL.predict(image, device='cpu', conf=.25, verbose=False)[0]
                output['yolo_image'] = image_url(prediction.plot())
                output['detections'] = [dict(name=MODEL.names[int(box.cls.item())],
                    confidence=float(box.conf.item()), box=box.xyxy[0].tolist()) for box in prediction.boxes]
        except Exception:
            output['yolo_error'] = 'Không chạy được YOLO. Kiểm tra thư viện ultralytics và tệp yolo11n.pt. Điểm OpenCV vẫn có kết quả.'
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
        elif self.path == '/health':
            self.send(200, {'status': 'ok'})
        elif self.path == '/sample':
            self.send(200, (ROOT / 'room.jpg').read_bytes(), 'image/jpeg')
        else:
            self.send(404, {'error': 'Không tìm thấy trang.'})

    def do_POST(self):
        if self.path != '/api/analyze':
            return self.send(404, {'error': 'Không tìm thấy chức năng.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 20_000_000:
                return self.send(413, {'error': 'Vui lòng chọn ảnh có dung lượng dưới 14 MB.'})
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Yêu cầu không hợp lệ.')
            self.send(200, analyze(data))
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
