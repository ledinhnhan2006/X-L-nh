import tempfile
from pathlib import Path
import unittest
from train_segmentation import validate_polygon_label

class SegmentationLabelTests(unittest.TestCase):
    def label(self,text):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        path=Path(tmp.name)/'sample.txt';path.write_text(text);return path
    def test_polygon_is_accepted(self):
        result=validate_polygon_label(self.label('2 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n'),12)
        self.assertEqual(result[2],1)
    def test_detection_label_rejected(self):
        with self.assertRaisesRegex(ValueError,'polygon'):
            validate_polygon_label(self.label('2 0.5 0.5 0.3 0.4'),12)
    def test_invalid_coordinate_rejected(self):
        with self.assertRaises(ValueError):
            validate_polygon_label(self.label('2 0 0 2 0 1 1'),12)
    def test_zero_area_rejected(self):
        with self.assertRaises(ValueError):
            validate_polygon_label(self.label('2 0 0 0.1 0.1 0.2 0.2'),12)
    def test_negative_image_label_can_be_empty(self):
        self.assertEqual(dict(validate_polygon_label(self.label(''),12)),{})


class SegmentationDatasetTests(unittest.TestCase):
    def dataset(self,missing_last=False):
        import json
        from train_segmentation import TARGET_CLASSES
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        # Đảo thứ tự names để xác minh giữ ID của bộ xuất, không tự sắp tên.
        names=list(reversed(TARGET_CLASSES))
        for split in ['train','valid']:
            (root/split/'images').mkdir(parents=True);(root/split/'labels').mkdir()
            from PIL import Image
            Image.new('RGB',(20,20),'white').save(root/split/'images'/'sample.png')
            count=11 if missing_last and split=='train' else 12
            (root/split/'labels'/'sample.txt').write_text('\n'.join(f'{i} 0.1 0.1 0.8 0.1 0.8 0.8' for i in range(count)))
        path=root/'data.yaml';path.write_text(json.dumps({'train':'train/images','val':'valid/images','names':names}))
        return path,names
    def test_dataset_names_preserved(self):
        from train_segmentation import validate_dataset
        path,names=self.dataset();config,summary=validate_dataset(path)
        self.assertEqual(config['names'],names)
        self.assertEqual(summary['train']['instances']['trash'],1)
        self.assertEqual(summary['train']['missing_classes'],[])
    def test_missing_training_class_rejected(self):
        from train_segmentation import validate_dataset
        path,_=self.dataset(missing_last=True)
        with self.assertRaisesRegex(ValueError,'Train chưa có ví dụ'):
            validate_dataset(path)

if __name__=='__main__':unittest.main()
