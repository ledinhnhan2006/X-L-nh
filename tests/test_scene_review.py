import base64
import io
import json
import unittest
from zipfile import ZipFile
import cv2
import numpy as np
from image_features import extract_features,geometry
from image_workbench import create_session,scoring_result,export_result,drop_session
from scene_scoring import prepare_review,score_scene
from placement_score import CLASSES

RULES={k:list(v[1]) for k,v in CLASSES.items()}
SHAPE=(200,200,3)
SQUARE=[[.1,.1],[.3,.1],[.3,.3],[.1,.3]]
def manual(key='manual-1',**kw):
    return {'id':key,'name':'Sách','class_key':'book','status':'confirmed','position':'wrong','quantity':1,'points':SQUARE,**kw}
def evaluate(patches,zones=None,shape=SHAPE,rules=None,scope=None,limits=None):
    return score_scene(prepare_review([],patches,shape),shape,zones or [],rules or RULES,scope,limits)


class FeatureTests(unittest.TestCase):
    def test_uniform_image_has_no_edges_or_contours(self):
        f=extract_features(np.full((80,100,3),160,np.uint8))
        self.assertEqual(f['metrics']['edge_density'],0)
        self.assertEqual(f['metrics']['contour_count'],0)
        self.assertEqual(f['metrics']['region_area_ratio'],0)
        self.assertGreaterEqual(len(f['stages']),7)

    def test_geometry_area_perimeter_and_holes(self):
        rings=[{'points':[[10,10],[90,10],[90,90],[10,90]],'hole':False,'depth':0},
               {'points':[[30,30],[70,30],[70,70],[30,70]],'hole':True,'depth':1}]
        g=geometry(rings,(100,100,3))
        self.assertEqual(g['area_pixels'],4800)
        self.assertEqual(g['perimeter_pixels'],480)
        self.assertEqual(g['centroid'],[50,50])
        self.assertEqual(g['hole_count'],1)
        self.assertAlmostEqual(g['solidity'],.75)


class SceneReviewTests(unittest.TestCase):
    def test_manual_confirmation_does_not_fabricate_confidence(self):
        base=[{'id':'auto-1','name':'book','class_key':'book','model_name':'book','model_confidence':.2,'box':[20,20,60,60],'status':'auto','object_contours':[]}]
        low=prepare_review(base,[],SHAPE)
        self.assertIsNone(score_scene(low,SHAPE,[{'id':'desk','kind':'desk','points':[[0,0],[1,0],[1,1],[0,1]]}],RULES)['score'])
        confirmed=prepare_review(base,[{'id':'auto-1','status':'confirmed','position':'correct'}],SHAPE)
        report=score_scene(confirmed,SHAPE,[],RULES)
        self.assertEqual(report['score'],100)
        self.assertEqual(report['objects'][0]['model_confidence'],.2)
        self.assertEqual(report['human_confirmed'],1)

    def test_class_change_cannot_reuse_old_model_confidence(self):
        base=[{'id':'auto-1','name':'book','class_key':'book','model_confidence':.9,'box':[20,20,60,60],'status':'auto'}]
        rows=prepare_review(base,[{'id':'auto-1','class_key':'trash','status':'auto'}],SHAPE)
        self.assertEqual(rows[0]['status'],'pending')

    def test_ignore_excludes_all_penalties_and_coverage_stays_partial(self):
        correct=manual('manual-good',position='correct')
        ignored=manual('manual-ignore',class_key='trash',status='ignored',quantity=50)
        report=evaluate([correct,ignored])
        self.assertEqual(report['score'],100)
        self.assertEqual(report['ignored'],1)
        self.assertEqual(report['coverage'],.5)
        self.assertEqual(report['bad_area_ratio'],0)
        self.assertEqual(report['bad_quantity'],0)
        self.assertTrue(report['provisional'])
        self.assertIsNone(evaluate([ignored])['score'])
        self.assertIsNone(evaluate([])['score'])

    def test_no_geometry_still_allows_manual_count_and_position(self):
        report=evaluate([manual(points=None,quantity=3)])
        self.assertEqual(report['evaluated'],1)
        self.assertEqual(report['missing_bad_geometry'],1)
        self.assertIsNone(report['objects'][0]['area_pixels'])
        self.assertTrue(all(not r['applicable'] for r in report['breakdown'] if r['key'] in ['area','contours','boundary']))
        self.assertLess(report['score'],100)

    def test_correct_placement_is_not_penalized_for_quantity_or_shape(self):
        objects=[manual('manual-'+str(i),class_key='book',position='correct',quantity=30) for i in range(8)]
        report=evaluate(objects)
        self.assertEqual(report['score'],100)
        self.assertTrue(all(r['deduction']==0 for r in report['breakdown']))

    def test_right_position_but_confirmed_clutter_uses_geometry_not_position(self):
        report=evaluate([manual(position='correct',clutter=True)])
        self.assertEqual(report['wrong'],0)
        self.assertEqual(report['objects'][0]['state'],'clutter')
        self.assertEqual(report['breakdown'][0]['deduction'],0)
        self.assertLess(report['score'],100)

    def test_union_area_is_not_multiplied_by_quantity_or_overlap(self):
        one=evaluate([manual()]);two=evaluate([manual(),manual('manual-2',quantity=10)])
        self.assertEqual(one['bad_area_ratio'],two['bad_area_ratio'])
        self.assertEqual(one['bad_components'],two['bad_components'])
        self.assertEqual(one['bad_boundary_index'],two['bad_boundary_index'])
        self.assertGreater(two['quantity_units'],one['quantity_units'])

    def test_scope_denominator_and_outside_exclusion(self):
        small={'points':[[0,0],[.5,0],[.5,.5],[0,.5]]}
        full=evaluate([manual()]);clipped=evaluate([manual()],scope=small)
        self.assertAlmostEqual(clipped['bad_area_ratio'],4*full['bad_area_ratio'],delta=.004)
        outside=evaluate([manual(points=[[.6,.6],[.8,.6],[.8,.8],[.6,.8]])],scope=small)
        self.assertEqual(outside['outside'],1)
        self.assertIsNone(outside['score'])

    def test_geometry_score_is_resolution_independent(self):
        one=evaluate([manual()],shape=(200,300,3));two=evaluate([manual()],shape=(400,600,3))
        self.assertAlmostEqual(one['score'],two['score'],delta=.01)
        self.assertAlmostEqual(one['bad_area_ratio'],two['bad_area_ratio'],delta=.001)

    def test_perimeter_preserves_image_aspect_ratio(self):
        # An 80×80 square on a 200×400 image stays square on the scoring grid.
        points=[[.1,.1],[.5,.1],[.5,.3],[.1,.3]]
        report=evaluate([manual(points=points)],shape=(400,200,3))
        expected=320/(2*np.sqrt(np.pi*80000))
        self.assertAlmostEqual(report['bad_boundary_index'],expected,delta=.005)
        self.assertEqual(report['geometry_grid']['height'],512)
        self.assertEqual(report['geometry_grid']['width'],256)

    def test_image_boundary_does_not_create_artificial_object_edges(self):
        report=evaluate([manual(points=[[0,0],[1,0],[1,1],[0,1]])])
        self.assertEqual(report['bad_boundary_length_grid'],0)
        self.assertEqual(next(r['deduction'] for r in report['breakdown'] if r['key']=='boundary'),0)

    def test_capacity_uses_explicit_limits_without_area_guess(self):
        zones=[{'id':'cab','kind':'cabinet','points':[[0,0],[1,0],[1,1],[0,1]]}]
        report=evaluate([manual(position='auto',quantity=12)],zones=zones,limits=[{'zone_id':'cab','class_key':'book','max_count':10}])
        self.assertEqual(report['quantity_units'],2)
        self.assertEqual(report['bad_area_ratio'],0)
        self.assertEqual(report['score'],98)

    def test_equal_area_zone_conflicts_are_independent_of_input_order(self):
        points=[[0,0],[1,0],[1,1],[0,1]]
        zones=[{'id':'floor','kind':'floor','clear':True,'points':points},
               {'id':'bed','kind':'bed','clear':False,'points':points}]
        one=evaluate([manual(position='auto')],zones=zones)
        two=evaluate([manual(position='auto')],zones=list(reversed(zones)))
        self.assertIsNone(one['score']);self.assertIsNone(two['score'])
        self.assertEqual(one['clear_blocked_ratio'],two['clear_blocked_ratio'])

    def test_scattered_bad_masks_increase_fragmentation_only_for_bad_objects(self):
        separate=[manual(),manual('manual-2',points=[[.6,.6],[.8,.6],[.8,.8],[.6,.8]])]
        overlap=[manual(),manual('manual-2')]
        a=evaluate(separate);b=evaluate(overlap)
        c1=next(r['deduction'] for r in a['breakdown'] if r['key']=='contours')
        c2=next(r['deduction'] for r in b['breakdown'] if r['key']=='contours')
        self.assertGreater(c1,c2)
        good=evaluate([{**r,'position':'correct'} for r in separate])
        self.assertEqual(good['score'],100)

    def test_confirmed_custom_object_can_be_scored_without_known_class(self):
        report=evaluate([manual(class_key='other',name='Điện thoại')])
        self.assertEqual(report['evaluated'],1)
        self.assertIsNone(report['objects'][0]['model_confidence'])

    def test_reject_bad_polygons_and_unknown_ids(self):
        for points in [[[0,0],[1,1],[0,1],[1,0]],[[0,0],[2,0],[1,1]],[[.2,.2],[.2,.2],[.3,.4]]]:
            with self.subTest(points=points),self.assertRaises(ValueError):evaluate([manual(points=points)])
        with self.assertRaises(ValueError):prepare_review([],[{'id':'auto-nonexistent'}],SHAPE)
        with self.assertRaises(ValueError):prepare_review([],[manual(),manual()],SHAPE)


class WorkbenchExportTests(unittest.TestCase):
    def setUp(self):
        image=np.full((100,160,3),210,np.uint8);cv2.circle(image,(50,50),20,(30,80,40),-1)
        ok,png=cv2.imencode('.png',image);assert ok
        self.session=create_session({'image':base64.b64encode(png).decode(),'yolo':False},lambda *_:self.fail('YOLO must be optional'))
    def tearDown(self):drop_session(self.session['session_id'])
    def test_manual_workflow_works_without_model_and_exports_steps(self):
        payload={'session_id':self.session['session_id'],'reviews':[manual(name='=1+1')],'zones':[],'rules':RULES}
        report=scoring_result(payload)
        self.assertEqual(report['scoring']['evaluated'],1)
        self.assertGreaterEqual(len(report['steps']),11)
        exported=export_result(payload)
        with ZipFile(io.BytesIO(exported)) as z:
            self.assertIsNone(z.testzip())
            rows=json.loads(z.read('bao_cao.json'))
            self.assertEqual(rows['scoring']['objects'][0]['name'],'=1+1')
            self.assertIn("'=1+1",z.read('vat_the.csv').decode('utf-8-sig'))
            self.assertGreaterEqual(len([n for n in z.namelist() if n.startswith('cac_buoc/')]),11)
    def test_rescoring_does_not_mutate_original_detection_or_accumulate(self):
        payload={'session_id':self.session['session_id'],'reviews':[manual()],'zones':[],'rules':RULES}
        a=scoring_result(payload);b=scoring_result(payload)
        self.assertEqual(a['scoring']['score'],b['scoring']['score'])
        payload['reviews']=[manual(status='ignored')]
        self.assertIsNone(scoring_result(payload)['scoring']['score'])
        payload['reviews']=[manual()]
        self.assertEqual(scoring_result(payload)['scoring']['score'],a['scoring']['score'])

if __name__=='__main__':unittest.main()
