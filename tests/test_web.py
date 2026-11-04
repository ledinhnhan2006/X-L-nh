import base64
import json
from pathlib import Path
import unittest
import urllib.request
import urllib.error

BASE = 'http://127.0.0.1:8000'
ROOT = Path(__file__).resolve().parents[1]

class WebTests(unittest.TestCase):
    def request(self, payload):
        req = urllib.request.Request(BASE + '/api/analyze', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=90))

    def test_page(self):
        self.assertIn('Chấm điểm ảnh', urllib.request.urlopen(BASE).read().decode())
        self.assertEqual(json.load(urllib.request.urlopen(BASE + '/health'))['status'], 'ok')

    def test_samples_and_yolo(self):
        for image in ROOT.glob('*.jpg'):
            with self.subTest(image=image.name):
                result = self.request({'image':base64.b64encode(image.read_bytes()).decode(), 'yolo':True})
                expected = 65.3 if image.name == 'room.jpg' else 56.45
                self.assertEqual(result['score'], expected)
                self.assertAlmostEqual(result['score'], 100 - sum(result['deductions']), delta=.011)
                self.assertIsNone(result['yolo_error'])
                self.assertTrue(result['yolo_image'].startswith('data:image/jpeg;base64,'))
                print(image.name, result['score'], len(result['detections']), 'YOLO detections')

    def test_invalid_upload(self):
        for payload in [{'image':'not base64'}, {'image':base64.b64encode(b'hello').decode()}, []]:
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(payload)
            self.assertEqual(caught.exception.code, 400)

    def test_without_yolo(self):
        from PIL import Image
        import io
        image = io.BytesIO()
        Image.new('RGB', (40,40), 'white').save(image, 'PNG')
        result = self.request({'image':base64.b64encode(image.getvalue()).decode(), 'yolo':False})
        self.assertEqual(result['score'],100)
        self.assertEqual(result['regions'],0)
        self.assertIsNone(result['yolo_image'])

if __name__ == '__main__':
    unittest.main()
