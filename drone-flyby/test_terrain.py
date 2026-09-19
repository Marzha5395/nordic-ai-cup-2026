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



class PendingCommandTests(unittest.TestCase):
    def test_command_not_yet_applied_is_resent_and_a_rejection_is_not(self):
        from dtos import RequestedViewDto
        from solution import Sequence
        from submission import FlybyPredictor
        from test_solution import FakeDetector, request_for

        predictor = FlybyPredictor(FakeDetector(), policy='adaptive', lag='resend')
        state = Sequence(3840, 2160)
        state.pending_command = (1, 2880, 540)
        state.last_view = (0, 1920, 1080)
        predictor.choose = lambda request, state, motion_ok: RequestedViewDto(resolution_level=1, center_x=960, center_y=540)
        # The view still shows the old position and nothing was rejected: the command is pending, so repeat it.
        command = predictor.select_view(request_for(level=0), state, False)
        self.assertEqual((command.resolution_level, command.center_x, command.center_y), (1, 2880, 540))
        # Once the view shows the command, decide afresh.
        command = predictor.select_view(request_for(level=1, center=(2880, 540)), state, False)
        self.assertEqual((command.center_x, command.center_y), (960, 540))
        # A rejected command is not pending: decide afresh from the view.
        state.pending_command = (1, 2880, 540)
        state.last_view = (0, 1920, 1080)
        request = request_for(level=0)
        request.camera_command_feedback = {'frame': 1, 'requested_view': {'resolution_level': 1, 'center_x': 2880, 'center_y': 540}, 'reason': 'test'}
        request = type(request).model_validate(request.model_dump())
        command = predictor.select_view(request, state, False)
        self.assertEqual((command.center_x, command.center_y), (960, 540))
        # A view that moved somewhere else is not a late command: decide afresh.
        state.pending_command = (1, 2880, 540)
        state.last_view = (1, 1920, 1080)
        command = predictor.select_view(request_for(level=0), state, False)
        self.assertEqual((command.center_x, command.center_y), (960, 540))
        # Never resend a command that would be illegal from this view (level 2 cannot reach level 0).
        state.pending_command = (0, 1920, 1080)
        state.last_view = (2, 2000, 900)
        command = predictor.select_view(request_for(level=2, center=(2000, 900)), state, False)
        self.assertEqual((command.center_x, command.center_y), (960, 540))
        # Without the option the policy always decides afresh.
        predictor.lag = 'none'
        state.pending_command = (1, 2880, 540)
        state.last_view = (0, 1920, 1080)
        command = predictor.select_view(request_for(level=0), state, False)
        self.assertEqual((command.center_x, command.center_y), (960, 540))


class PlanAheadTests(unittest.TestCase):
    def test_plans_from_the_pending_view_and_only_sends_moves_legal_from_both(self):
        from dtos import RequestedViewDto
        from solution import Sequence
        from submission import FlybyPredictor
        from test_solution import FakeDetector, request_for

        predictor = FlybyPredictor(FakeDetector(), policy='adaptive', lag='ahead')
        seen = []

        def choose(request, state, motion_ok):
            seen.append((request.view.resolution_level, request.view.center_x, request.view.center_y, request.camera_constraints.maximum_center_delta))
            return RequestedViewDto(resolution_level=1, center_x=target[0], center_y=target[1])

        predictor.choose = choose
        state = Sequence(3840, 2160)
        # Camera still at the full view; our command to L1 (2880, 540) has not landed yet.
        target = (2880, 1100)
        state.pending_command, state.last_view = (1, 2880, 540), (0, 1920, 1080)
        command = predictor.select_view(request_for(level=0), state, False)
        self.assertEqual(seen[-1], (1, 2880, 540, 1102.0))  # planned as if the camera were already there
        self.assertEqual((command.center_x, command.center_y), target)  # legal from L1 (2880, 540) and from L0
        # From the pending L1 view a move to the far corner is fine, but seen from the current L2 view it is too
        # far, so the pending command is resent instead.
        target = (2880, 1620)
        state.pending_command, state.last_view = (1, 2400, 800), (2, 2300, 700)
        command = predictor.select_view(request_for(level=2, center=(2300, 700)), state, False)
        self.assertEqual((command.resolution_level, command.center_x, command.center_y), (1, 2400, 800))
