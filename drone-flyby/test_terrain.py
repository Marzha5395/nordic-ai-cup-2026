import unittest

import numpy as np

from terrain_data import TrainingScenes, clip_labels, rotate_boxes90


class TerrainDataGeometryTests(unittest.TestCase):
    def test_real_crops_are_downsampled_without_aliasing(self):
        data = TrainingScenes.__new__(TrainingScenes)
        data.size = 8
        texture = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
        data.frames = [(np.repeat(texture[..., None], 3, axis=2), np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int64))]
        for seed in range(6):
            data.rng = np.random.default_rng(seed)
            result, _, _ = data._real()
            self.assertAlmostEqual(float(result.mean()), 127.5, delta=2)

    def test_quarter_turn_boxes_follow_the_image(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[10:40, 5:20] = 1
        boxes = np.array([[5, 10, 20, 40]], dtype=np.float32)
        for turns in range(4):
            ys, xs = np.nonzero(np.rot90(image, turns))
            np.testing.assert_array_equal(rotate_boxes90(boxes, 64, turns), [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]])

    def test_cut_out_label_reproduces_annotation_and_swaps_at_quarter_turn(self):
        import sprites

        alpha = np.zeros((40, 60), dtype=np.float32)
        alpha[15:25, 20:40] = 1
        sprite = {'image': np.full((40, 60, 3), 200, dtype=np.uint8), 'alpha': alpha, 'box': np.array([16, 12, 44, 28], dtype=np.float32)}
        _, _, box = sprites.render(sprite, 0, 1.0)
        np.testing.assert_allclose(box[2:] - box[:2], [28, 16], atol=1e-4)
        _, _, box = sprites.render(sprite, 90, 1.0)
        np.testing.assert_allclose(box[2:] - box[:2], [16, 28], atol=1e-4)

    def test_clipping_removes_degenerate_labels(self):
        boxes, classes = clip_labels([[-20, -20, -10, -10], [5, 5, 15, 15], [90, 90, 110, 110]], [0, 1, 2], 100, 100)
        np.testing.assert_array_equal(classes, [1])
        np.testing.assert_array_equal(boxes, [[5, 5, 15, 15]])

    def test_only_the_reference_scene_is_trained_on(self):
        from terrain_data import TRAINING_SCENES

        self.assertEqual(TRAINING_SCENES, ('helsinki',))


if __name__ == '__main__':
    unittest.main()

