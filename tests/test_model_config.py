import base64
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import web_app
from placement_score import CLASSES

class ModelConfigTests(unittest.TestCase):
    def test_default_is_trained_weights(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(web_app.configured_model_path(),web_app.ROOT/'weights.pt')

    def test_missing_model_never_falls_back(self):
        with patch.dict(os.environ, {'YOLO_MODEL_PATH':'not_a_checkpoint.pt'}):
            self.assertFalse(web_app.model_info()['ready'])
            result=web_app.analyze({'image':base64.b64encode((web_app.ROOT/'room.jpg').read_bytes()).decode(),
                  'yolo':True,'scoring_mode':'placement',
                  'zones':[{'kind':'floor','points':[[0,0],[1,0],[1,1],[0,1]]}],
                  'rules':{k:list(v[1]) for k,v in CLASSES.items()}})
            self.assertIsNone(result['score'])
            self.assertIn('not_a_checkpoint.pt',result['yolo_error'])

    def test_invalid_confidence(self):
        image=base64.b64encode((web_app.ROOT/'room.jpg').read_bytes()).decode()
        for confidence in [-1,0,1,True,'0.25',float('nan')]:
            with self.subTest(confidence=confidence),self.assertRaises(ValueError):
                web_app.analyze({'image':image,'confidence':confidence})
