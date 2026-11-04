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
        self.assertEqual(two['deductions'][0],12)
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

if __name__=='__main__': unittest.main()
