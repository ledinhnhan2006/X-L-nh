"""HTTP contract for the reviewed workflow; run against TEST_BASE_URL."""
import base64
import io
import json
import os
import unittest
import urllib.error
import urllib.request
from zipfile import ZipFile
from PIL import Image

BASE=os.environ.get('TEST_BASE_URL','http://127.0.0.1:8000')


class WorkbenchHttpTests(unittest.TestCase):
    def post(self,endpoint,payload,raw=False):
        request=urllib.request.Request(BASE+'/api/workbench/'+endpoint,
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=90) as response:
            return response.read() if raw else json.load(response)

    def setUp(self):
        buffer=io.BytesIO();Image.new('RGB',(160,100),'white').save(buffer,'PNG')
        self.image=base64.b64encode(buffer.getvalue()).decode()
        self.session=self.post('analyze',{'image':self.image,'yolo':False})

    def test_manual_review_ignore_and_export_via_http(self):
        self.assertEqual(self.session['image_features']['edge_density'],0)
        self.assertEqual(self.session['objects'],[])
        self.assertGreaterEqual(len(self.session['steps']),7)
        body={'session_id':self.session['session_id'],'reviews':[]}
        self.assertIsNone(self.post('score',body)['scoring']['score'])
        body['reviews']=[{'id':'manual-http','name':'Sách nhìn thấy bằng mắt','class_key':'book',
            'quantity':2,'status':'confirmed','position':'wrong',
            'points':[[.1,.1],[.3,.1],[.3,.3],[.1,.3]]}]
        result=self.post('score',body)
        self.assertEqual(result['scoring']['bad_quantity'],2)
        self.assertLess(result['scoring']['score'],100)
        self.assertIsNone(result['scoring']['objects'][0]['model_confidence'])
        self.assertGreater(result['scoring']['objects'][0]['geometry']['area_pixels'],0)
        body['reviews'][0]['status']='ignored'
        ignored=self.post('score',body)
        self.assertIsNone(ignored['scoring']['score'])
        self.assertEqual(ignored['scoring']['bad_area_ratio'],0)
        with ZipFile(io.BytesIO(self.post('export',body,raw=True))) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read('bao_cao.json'))['scoring']['ignored'],1)
            self.assertIn('vat_the.csv',archive.namelist())
            self.assertTrue(any(n.endswith('_original.png') for n in archive.namelist()))

    def test_invalid_session_review_and_polygon_return_explained_errors(self):
        session=self.session['session_id']
        invalid=[{'session_id':'expired-or-unknown'},
            {'session_id':session,'reviews':[{'id':'auto-unrelated'}]},
            {'session_id':session,'reviews':[{'id':'manual-invalid','points':[[0,0],[2,0],[0,1]]}]}]
        for body in invalid:
            with self.subTest(body=body),self.assertRaises(urllib.error.HTTPError) as caught:
                self.post('score',body)
            self.assertEqual(caught.exception.code,400)
            self.assertTrue(json.load(caught.exception)['error'])


if __name__=='__main__':unittest.main()
