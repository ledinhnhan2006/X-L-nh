import unittest
from placement_score import CLASSES, score_placement

RULES={k:list(v[1]) for k,v in CLASSES.items()}
def zone(kind, points=None):
    return dict(kind=kind, points=points or [[0,0],[1,0],[1,1],[0,1]])
def detection(name='book',box=None):
    return dict(name=name, confidence=.9, box=box or [10,10,30,30])

class PlacementTests(unittest.TestCase):
    def score(self, detections,zones):
        return score_placement(detections,(100,100,3),zones,RULES)
    def test_book_on_desk_correct_but_floor_wrong(self):
        self.assertEqual(self.score([detection()],[zone('desk')])['score'],100)
        result=self.score([detection()],[zone('floor')])
        self.assertEqual(result['wrong'],1)
        self.assertLess(result['score'],100)
    def test_no_detections_or_uncovered_not_perfect(self):
        self.assertIsNone(self.score([],[zone('floor')])['score'])
        self.assertIsNone(self.score([detection()],[zone('desk',[[.5,.5],[1,.5],[1,1],[.5,1]])])['score'])
        self.assertIsNone(self.score([detection('bed')],[zone('floor')])['score'])
    def test_area_union_not_double_counted(self):
        one=self.score([detection()],[zone('floor')])
        two=self.score([detection(),detection()],[zone('floor')])
        self.assertEqual(one['wrong_area_ratio'],two['wrong_area_ratio'])
        self.assertEqual(two['deductions'][0],1)
    def test_small_surface_overrides_floor(self):
        result=self.score([detection()],[zone('floor'),zone('desk',[[0,0],[.4,0],[.4,.4],[0,.4]])])
        self.assertEqual(result['score'],100)
    def test_equal_overlapping_surfaces_ambiguous(self):
        self.assertIsNone(self.score([detection()],[zone('floor'),zone('desk')])['score'])
    def test_reject_crossed_polygon(self):
        with self.assertRaises(ValueError):
            self.score([], [zone('desk',[[0,0],[1,1],[0,1],[1,0]])])
    def test_reject_malformed_coordinates(self):
        with self.assertRaises(ValueError):
            self.score([], [zone('desk',[[0,0],[2,0],[1,1]])])
    def test_custom_rule_and_trash(self):
        result=self.score([detection('trash')],[zone('desk')])
        self.assertEqual(result['wrong'],1)
        changed={**RULES,'book':['floor']}
        result=score_placement([detection()],(100,100,3),[zone('floor')],changed)
        self.assertEqual(result['score'],100)


class WeightedScoreTests(unittest.TestCase):
    def score(self, detections, kind):
        return score_placement(detections,(100,100,3),[zone(kind)],RULES)

    def test_class_severity_and_breakdown(self):
        book=self.score([detection('book')],'floor')
        trash=self.score([detection('trash')],'floor')
        self.assertEqual(book['deductions'][0],.5)
        self.assertEqual(trash['deductions'][0],3)
        self.assertLess(trash['score'],book['score'])
        self.assertAlmostEqual(trash['score'],100-sum(trash['deductions']),delta=.01)
        self.assertEqual(trash['detections'][0]['position_penalty'],3)

    def test_trash_bin_and_shoe_storage(self):
        self.assertEqual(self.score([detection('trash')],'bin')['score'],100)
        self.assertEqual(self.score([detection('footwear')],'shoe_area')['score'],100)
        self.assertLess(self.score([detection('footwear')],'floor')['score'],100)
        self.assertLess(self.score([detection('chair')],'walkway')['score'],100)

    def test_partial_coverage_and_class_normalization(self):
        result=self.score([detection(' BOOK '),detection('unrecognized_class')],'desk')
        self.assertEqual(result['evaluated'],1)
        self.assertEqual(result['coverage'],.5)
        self.assertTrue(result['provisional'])
        self.assertEqual(result['detections'][1]['position_penalty'],0)

    def test_penalty_caps(self):
        result=self.score([detection('trash',[0,0,100,100]) for _ in range(12)],'floor')
        self.assertEqual(result['deductions'],[8,15])
        self.assertEqual(result['score'],77)

class GentleScoreTests(unittest.TestCase):
    def test_small_violation_keeps_base_score(self):
        result=score_placement([detection('book')],(100,100,3),[zone('floor')],RULES)
        self.assertEqual(result['score'],99.5)
        self.assertEqual(result['deductions'],[.5,0])

    def test_low_confidence_not_penalized_or_called_clean(self):
        weak={**detection('trash'), 'confidence':.49}
        result=score_placement([weak],(100,100,3),[zone('floor')],RULES)
        self.assertIsNone(result['score'])
        self.assertEqual(result['unknown'],1)
        self.assertEqual(result['deductions'],[0,0])
        self.assertEqual(result['wrong_area_ratio'],0)
        mixed=score_placement([weak,detection('book')],(100,100,3),[zone('desk')],RULES)
        self.assertEqual(mixed['score'],100)
        self.assertEqual(mixed['coverage'],.5)

    def test_floor_and_caps_for_many_types(self):
        rules={name:[] for name in CLASSES}
        objects=[detection(name,[0,0,100,100]) for name in CLASSES for _ in range(20)]
        result=score_placement(objects,(100,100,3),[zone('floor')],rules)
        self.assertEqual(result['deductions'],[45,15])
        self.assertEqual(result['score'],40)
        self.assertTrue(all(v<=8 for v in result['class_penalties'].values()))

    def test_area_allowance_and_excess(self):
        small=score_placement([detection('trash',[0,0,20,20])],(100,100,3),[zone('floor')],RULES)
        large=score_placement([detection('trash',[0,0,50,50])],(100,100,3),[zone('floor')],RULES)
        self.assertEqual(small['deductions'][1],0)
        self.assertAlmostEqual(large['deductions'][1],4.5)
        self.assertEqual(large['score'],92.5)

if __name__=='__main__': unittest.main()
