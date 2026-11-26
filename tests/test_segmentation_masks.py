import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import torch
from object_contours import measure_mask_contour,measure_detections
from web_app import segmentation_masks


class SegmentationMaskTests(unittest.TestCase):
    def test_measures_instance_shape_and_subtracts_hole(self):
        mask=np.zeros((100,160),np.float32)
        cv2.circle(mask,(80,50),30,1,-1)
        cv2.circle(mask,(80,50),10,0,-1)
        result=measure_mask_contour(mask,(100,160,3))
        self.assertEqual(result['area_method'],'yolo_segmentation_contour')
        self.assertTrue(any(r['hole'] for r in result['object_contours']))
        self.assertAlmostEqual(result['area_pixels'],np.pi*(30**2-10**2),delta=150)
        self.assertAlmostEqual(result['area_ratio'],result['area_pixels']/16000)

    def test_letterbox_padding_is_removed_for_non_square_image(self):
        # A 100x200 image was fitted inside a 320x320 input: 80 padding rows per side.
        masks=torch.zeros((1,320,320))
        masks[0,112:208,80:240]=1
        prediction=SimpleNamespace(masks=SimpleNamespace(data=masks),boxes=[object()])
        mask=next(segmentation_masks(prediction,(100,200,3)))
        self.assertEqual(mask.shape,(100,200))
        ys,xs=np.where(mask>.5)
        self.assertEqual((xs.min(),xs.max(),ys.min(),ys.max()),(50,149,20,79))
        self.assertAlmostEqual(measure_mask_contour(mask,(100,200,3))['area_pixels'],99*59)

    def test_low_confidence_does_not_shift_next_instances_mask(self):
        small=np.zeros((100,100));small[10:21,10:21]=1
        large=np.zeros((100,100));large[30:71,30:71]=1
        detections=[{'confidence':.3,'box':[0,0,90,90]},{'confidence':.9,'box':[0,0,90,90]}]
        with patch('object_contours.extract_object_contour',side_effect=AssertionError('Segmentation must use its mask')):
            results=measure_detections(np.zeros((100,100,3),np.uint8),detections,[small,large])
        self.assertIsNone(results[0]['area_pixels'])
        self.assertEqual(results[1]['area_pixels'],1600)

    def test_invalid_masks_never_use_rectangle_area(self):
        for mask in [None,np.zeros((100,100)),np.ones((50,100)),np.full((100,100),np.nan)]:
            with self.subTest(mask_type=type(mask).__name__):
                result=measure_mask_contour(mask,(100,100,3))
                self.assertIsNone(result['area_pixels'])
                self.assertEqual(result['area_status'],'unavailable')

    def test_missing_or_mismatched_masks_stay_unavailable(self):
        for data in [None,SimpleNamespace(data=torch.zeros((2,100,100)))]:
            prediction=SimpleNamespace(masks=data,boxes=[object()])
            self.assertEqual(list(segmentation_masks(prediction,(100,100,3))),[None])
        with patch('object_contours.extract_object_contour',side_effect=AssertionError('No mask fallback')):
            result=measure_detections(np.zeros((100,100,3),np.uint8),[{'confidence':.9,'box':[0,0,100,100]}],[])[0]
        self.assertEqual(result['area_method'],'yolo_segmentation_contour')
        self.assertIsNone(result['area_pixels'])


if __name__=='__main__':unittest.main()
