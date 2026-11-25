import unittest
import cv2
import numpy as np
from object_contours import extract_object_contour,union_contour_ratio,contour_rings,geometric_area
from placement_score import score_placement,CLASSES

class ContourTests(unittest.TestCase):
    def test_circle_smaller_than_detection_box(self):
        image=np.zeros((100,100,3),np.uint8)
        cv2.circle(image,(50,50),24,(230,230,230),-1)
        measured=extract_object_contour(image,[20,20,80,80])
        self.assertEqual(measured['area_status'],'estimated')
        self.assertAlmostEqual(measured['area_pixels'],np.pi*24**2,delta=180)
        self.assertLess(measured['area_pixels'],.65*60*60)

    def test_holes_are_subtracted(self):
        mask=np.zeros((100,100),np.uint8)
        cv2.circle(mask,(50,50),30,255,-1)
        cv2.circle(mask,(50,50),10,0,-1)
        rings=contour_rings(mask)
        self.assertTrue(any(r['hole'] for r in rings))
        self.assertAlmostEqual(geometric_area(rings),np.pi*(30**2-10**2),delta=150)

    def test_failure_never_uses_box_area(self):
        measured=extract_object_contour(np.full((100,100,3),128,np.uint8),[20,20,80,80])
        self.assertEqual(measured['area_status'],'unavailable')
        self.assertIsNone(measured['area_pixels'])
        self.assertIsNone(measured['area_ratio'])
        self.assertEqual(measured['object_contours'],[])
        detection={'name':'trash','confidence':.9,'box':[20,20,80,80],**measured}
        result=score_placement([detection],(100,100,3),[{'kind':'floor','points':[[0,0],[1,0],[1,1],[0,1]]}],{k:list(v[1]) for k,v in CLASSES.items()})
        self.assertEqual(result['deductions'],[3,0])
        self.assertEqual(result['area_unknown_wrong'],1)

    def test_union_does_not_double_count_or_erase_other_object_inside_hole(self):
        outer={'points':[[10,10],[90,10],[90,90],[10,90]],'hole':False,'depth':0}
        hole={'points':[[30,30],[70,30],[70,70],[30,70]],'hole':True,'depth':1}
        ring={'object_contours':[outer,hole]}
        inner={'object_contours':[{'points':[[40,40],[60,40],[60,60],[40,60]],'hole':False,'depth':0}]}
        one=union_contour_ratio([ring],(100,100,3))
        duplicate=union_contour_ratio([ring,ring],(100,100,3))
        combined=union_contour_ratio([inner,ring],(100,100,3))
        self.assertEqual(one,duplicate)
        self.assertGreater(combined,one)
        self.assertAlmostEqual(one,.48,delta=.015)
        self.assertAlmostEqual(combined,.52,delta=.015)

    def test_small_or_full_frame_roi_unavailable(self):
        image=np.zeros((100,100,3),np.uint8)
        for box in [[10,10,14,14],[0,0,100,100]]:
            self.assertEqual(extract_object_contour(image,box)['area_status'],'unavailable')

if __name__=='__main__':unittest.main()
