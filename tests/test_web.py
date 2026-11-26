import base64
import json
from pathlib import Path
import unittest
import urllib.request
import urllib.error

import os
BASE = os.environ.get('TEST_BASE_URL', 'http://127.0.0.1:8000')
ROOT = Path(__file__).resolve().parents[1]

class WebTests(unittest.TestCase):
    def request(self, payload):
        req = urllib.request.Request(BASE + '/api/analyze', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=90))

    def test_model_status(self):
        info=json.load(urllib.request.urlopen(BASE+'/model'))
        self.assertTrue(info['ready'],info)
        self.assertEqual(info['file'],os.environ.get('TEST_MODEL_FILE','weights.pt'))
        self.assertGreater(len(info['classes']),0)

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
                expected_area_method='yolo_segmentation_contour' if result.get('model_task')=='segment' else 'grabcut_contour'
                for obj in result['detections']:
                    self.assertEqual(obj['area_method'],expected_area_method)
                    self.assertIn(obj['area_status'],['estimated','unavailable'])
                    if obj['area_status']=='unavailable':
                        self.assertIsNone(obj['area_ratio'])
                    else:
                        self.assertGreater(obj['area_pixels'],0)
                        self.assertTrue(obj['object_contours'])
                self.assertTrue(result['yolo_image'].startswith('data:image/jpeg;base64,'))
                print(image.name, result['score'], len(result['detections']), 'YOLO detections')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request({'image':base64.b64encode(b'not an image').decode()})
        self.assertEqual(caught.exception.code,400)

    def test_invalid_upload(self):
        for payload in [{'image':'not base64'}, {'image':base64.b64encode(b'hello').decode()}, []]:
            with self.subTest(payload=payload), self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(payload)
            self.assertEqual(caught.exception.code, 400)

    def test_placement_mode(self):
        from placement_score import CLASSES
        payload = {'image':base64.b64encode((ROOT/'room.jpg').read_bytes()).decode(),
                   'yolo':True, 'scoring_mode':'placement',
                   'zones':[{'kind':'floor', 'points':[[0,0],[1,0],[1,1],[0,1]]}],
                   'rules':{k:['floor'] for k in CLASSES}}
        result=self.request(payload)
        self.assertGreater(result['placement']['evaluated'],0)
        self.assertEqual(result['placement']['wrong'],0)
        self.assertEqual(result['score'],100)
        # Exercise the rule change for a class this checkpoint actually assessed.
        assessed=[row['class_key'] for row in result['placement']['detections'] if row['state']=='correct']
        self.assertTrue(assessed)
        payload['rules'][assessed[0]]=[]
        forbidden=self.request(payload)
        self.assertGreater(forbidden['placement']['wrong'],0)
        self.assertLess(forbidden['score'],result['score'])
        self.assertEqual(forbidden['placement']['method'],'placement_v4_contour')
        payload['zones']=[]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(payload)
        self.assertEqual(caught.exception.code,400)

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
