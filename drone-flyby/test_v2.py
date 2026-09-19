import unittest

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from data_v2 import DiverseScenes, HOLDOUT_FRAMES, refine_alpha
from dtos import OBJECT_CLASSES
from gpu_detector import FlipDetector, TorchDetector, fuse_views
from submission import FlybyPredictor
from test_solution import FakeDetector, detection, request_for
from train_v2 import create_model
from training_data import Sprite


class DataTests(unittest.TestCase):
    def test_background_mask_is_not_used_as_foreground(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:] = [40, 140, 40]
        image[22:42, 22:42] = [230, 230, 230]
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[12:52, 12:52] = 1
        alpha[22:42, 22:42] = 0
        result = refine_alpha(image, alpha, np.array([12, 12, 52, 52], dtype=np.float32))
        self.assertGreater(result[32, 32], 0.9)
        self.assertLess(result[16, 16], 0.1)

    def test_synthetic_labels_preserve_physical_scale(self):
        data = DiverseScenes.__new__(DiverseScenes)
        data.size = 256
        data.canonical_scale = 0.5
        data.rng = np.random.default_rng(801)
        image = np.full((44, 44, 3), 180, dtype=np.uint8)
        alpha = np.ones((44, 44), dtype=np.float32)
        data.sprites = [[Sprite(image, alpha, label, np.array([6, 6, 38, 38], dtype=np.float32))] for label in range(len(OBJECT_CLASSES))]
        data._background = lambda: np.full((256, 256, 3), 70, dtype=np.uint8)
        count = 0
        for _ in range(20):
            image, boxes, labels = data._synthetic()
            self.assertEqual(image.shape, (256, 256, 3))
            self.assertEqual(len(boxes), len(labels))
            if len(boxes):
                dimensions = boxes[:, 2:] - boxes[:, :2]
                self.assertGreater(dimensions.min(), 11)
                self.assertLess(dimensions.max(), 30)
                self.assertTrue((boxes >= 0).all() and (boxes <= 256).all())
            count += len(boxes)
        self.assertGreater(count, 100)

    def test_holdout_split_is_fixed_and_nonempty(self):
        self.assertEqual(HOLDOUT_FRAMES, {1, 8, 15, 19, 24})
        self.assertFalse(HOLDOUT_FRAMES & (set(range(25)) - HOLDOUT_FRAMES))


class ModelTests(unittest.TestCase):
    def test_p2_model_has_a_stride_four_head_and_valid_loss(self):
        torch.set_num_threads(2)
        source = YOLO('yolo11n.yaml').model
        model = create_model(source, p2=True)
        self.assertEqual(model.stride.tolist(), [4, 8, 16, 32])
        self.assertEqual(list(model.names.values()), list(OBJECT_CLASSES))
        model.train()
        batch = {'img': torch.rand(2, 3, 128, 128), 'cls': torch.tensor([[0.0], [3.0]]), 'bboxes': torch.tensor([[0.3, 0.3, 0.05, 0.05], [0.7, 0.7, 0.1, 0.1]]), 'batch_idx': torch.tensor([0, 1])}
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertIsNotNone(model.model[0].conv.weight.grad)

    def test_zoom_changes_sampling_not_physical_scale(self):
        detector = TorchDetector.__new__(TorchDetector)
        detector.canonical = True
        detector.full_width = 1920
        detector.threshold = 0.05
        detector.infer = lambda tensor: np.zeros((1, 4 + len(OBJECT_CLASSES), 1), dtype=np.float32)
        for level, expected in enumerate([(1920, 1088), (960, 544), (480, 288)]):
            detector.detect(np.zeros((540, 960, 3), dtype=np.uint8), request_for(level=level))
            self.assertEqual((detector.width, detector.height), expected)


class InferenceTests(unittest.TestCase):
    def test_flip_restores_crop_global_coordinates(self):
        request = request_for(level=1, center=(2200, 1080))
        views = iter([[detection((1400, 700, 1440, 740))], [detection((2960, 700, 3000, 740))]])
        backend = FakeDetector()
        backend.detect = lambda image, request: next(views)
        found = FlipDetector(backend).detect(np.zeros((540, 960, 3), dtype=np.uint8), request)
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0].box, [1400, 700, 1440, 740])
        self.assertAlmostEqual(found[0].confidence, 0.8)

    def test_fusion_keeps_distinct_instances(self):
        first = [detection((100, 100, 140, 140)), detection((300, 100, 340, 140))]
        second = [detection((101, 100, 141, 140)), detection((301, 100, 341, 140))]
        found = fuse_views([first, second])
        self.assertEqual(len(found), 2)
        np.testing.assert_allclose(found[0].box, [100.5, 100, 140.5, 140])

    def test_single_view_evidence_has_lower_confidence(self):
        found = fuse_views([[detection()], []])
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].confidence, 0.4)

    def test_full_policy_never_zooms_away(self):
        engine = FlybyPredictor(FakeDetector(), policy='full')
        response = engine.predict(request_for())
        self.assertEqual(response.requested_view.resolution_level, 0)
        response = engine.predict(request_for(index=1, level=2, center=(480, 270)))
        self.assertEqual(response.requested_view.resolution_level, 1)

    def test_long_gap_does_not_reuse_a_previous_objects_class(self):
        backend = FakeDetector([detection(label=0)])
        engine = FlybyPredictor(backend)
        for index in range(3):
            engine.predict(request_for(index=index))
        backend.detections = [detection(label=1)]
        result = engine.predict(request_for(index=20))
        self.assertEqual([item.object_id for item in result.annotations], [OBJECT_CLASSES[1]])

    def test_overview_policy_returns_after_one_zoom(self):
        engine = FlybyPredictor(FakeDetector(), policy='overview')
        self.assertEqual(engine.predict(request_for()).requested_view.resolution_level, 1)
        self.assertEqual(engine.predict(request_for(index=1, level=1)).requested_view.resolution_level, 0)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
