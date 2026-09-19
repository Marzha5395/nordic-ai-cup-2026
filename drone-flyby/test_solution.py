from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detector import Detection, Detector, suppress
from dtos import ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS, OBJECT_CLASSES, DroneFlybyPredictRequestDto
from motion import IDENTITY, estimate_motion, move_box
from solution import Predictor, Sequence, Track, annotations_for, choose_view, legal_view, update_tracks
from training_data import clip_labels, transform_boxes
from utils import center_bounds_for_level, encode_image, source_region_for_view, validate_response


def request_for(index=0, level=0, center=None, sequence='synthetic', image=None, limit=None):
    center = center or (1920, 1080)
    image = image if image is not None else np.zeros((540, 960, 3), dtype=np.uint8)
    bounds = []
    for allowed in ALLOWED_RESOLUTION_LEVELS[level]:
        x1, x2, y1, y2 = center_bounds_for_level(allowed)
        bounds.append(dict(resolution_level=allowed, width=960, height=540, minimum_center_x=x1, maximum_center_x=x2, minimum_center_y=y1, maximum_center_y=y2))
    return DroneFlybyPredictRequestDto(
        sequence_id=sequence, frame=index + 10, frame_index=index,
        request_id=f'{sequence}:{index}:{level}', frame_interval_ms=333, response_timeout_ms=3333,
        original_width=3840, original_height=2160,
        view=dict(resolution_level=level, center_x=center[0], center_y=center[1], view_id='view', image=encode_image(image), image_media_type='image/png', width=960, height=540, source_region_xyxy=list(source_region_for_view(level, *center))),
        camera_constraints=dict(maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[level] if limit is None else limit, allowed_resolution_levels=list(ALLOWED_RESOLUTION_LEVELS[level]), center_bounds=bounds, full_view_reset_exempt_from_delta=True),
    )


def detection(box=(100, 100, 140, 140), label=0, confidence=0.8):
    scores = np.zeros(len(OBJECT_CLASSES), dtype=np.float32)
    scores[label] = confidence
    return Detection(np.asarray(box, dtype=np.float32), scores)


class FakeDetector:
    def __init__(self, detections=()):
        self.detections = detections
        self.calls = 0

    def detect(self, image, request):
        self.calls += 1
        return list(self.detections)


class CameraTests(unittest.TestCase):
    def assert_legal(self, request, command):
        self.assertIsNotNone(command)
        constraints = request.camera_constraints
        self.assertIn(command.resolution_level, constraints.allowed_resolution_levels)
        bounds = constraints.bounds_for_level(command.resolution_level)
        self.assertTrue(bounds.minimum_center_x <= command.center_x <= bounds.maximum_center_x)
        self.assertTrue(bounds.minimum_center_y <= command.center_y <= bounds.maximum_center_y)
        self.assertIs(type(command.center_x), int)
        self.assertIs(type(command.center_y), int)
        if command.resolution_level != 0 or not constraints.full_view_reset_exempt_from_delta:
            self.assertLessEqual(np.hypot(command.center_x - request.view.center_x, command.center_y - request.view.center_y), constraints.maximum_center_delta)

    def test_random_moves_respect_actual_constraints(self):
        rng = np.random.default_rng(48)
        for level in range(3):
            base = request_for(level=level)
            low_x, high_x, low_y, high_y = center_bounds_for_level(level)
            for _ in range(80):
                center = (int(rng.integers(low_x, high_x + 1)), int(rng.integers(low_y, high_y + 1)))
                request = base.model_copy(deep=True)
                request.view.center_x, request.view.center_y = center
                for target_level in ALLOWED_RESOLUTION_LEVELS[level]:
                    command = legal_view(request, target_level, rng.uniform([-2000, -2000], [6000, 5000]))
                    self.assert_legal(request, command)

    def test_zero_delta_is_not_replaced_with_default(self):
        request = request_for(level=1, center=(1000, 600), limit=0)
        command = legal_view(request, 2, (3000, 1500))
        self.assert_legal(request, command)
        self.assertEqual((command.center_x, command.center_y), (1000, 600))

    def test_reset_exemption_is_read_from_request(self):
        request = request_for(level=1, center=(960, 540), limit=0)
        self.assert_legal(request, legal_view(request, 0, (1920, 1080)))
        request.camera_constraints.full_view_reset_exempt_from_delta = False
        self.assertIsNone(legal_view(request, 0, (1920, 1080)))

    def test_corner_zoom_out_is_reachable(self):
        request = request_for(level=2, center=(480, 270))
        self.assert_legal(request, legal_view(request, 1, (1920, 1080)))

    def test_unreachable_level_is_not_requested(self):
        self.assertIsNone(legal_view(request_for(), 2, (1920, 1080)))
        self.assertIsNone(legal_view(request_for(level=2), 0, (1920, 1080)))

    def test_policy_returns_to_full_view(self):
        state = Sequence(3840, 2160, last_full_frame=0, previous_level=2)
        request = request_for(index=3, level=1)
        command = choose_view(request, state, True)
        self.assertEqual(command.resolution_level, 0)
        self.assert_legal(request, command)

    def test_policy_explores_without_detections(self):
        state = Sequence(3840, 2160)
        positions = set()
        for index in range(4):
            request = request_for(index=index)
            command = choose_view(request, state, False)
            self.assert_legal(request, command)
            positions.add((command.center_x, command.center_y))
        self.assertEqual(len(positions), 4)


class DetectorTests(unittest.TestCase):
    def test_crop_lifting_and_padding(self):
        for level in range(3):
            request = request_for(level=level)
            row = np.zeros((1, 4 + len(OBJECT_CLASSES), 1), dtype=np.float32)
            row[0, :4, 0] = [110, 212, 20, 20]
            row[0, 4 + 3, 0] = 0.9
            detector = Detector.__new__(Detector)
            detector.width, detector.height = 960, 544
            detector.input_name = 'images'
            detector.threshold = 0.1
            detector.session = SimpleNamespace(run=lambda *args: [row])
            found = detector.detect(np.zeros((540, 960, 3), dtype=np.uint8), request)
            self.assertEqual(len(found), 1)
            region = np.asarray(request.view.source_region_xyxy)
            scale = np.tile((region[2:] - region[:2]) / [960, 540], 2)
            expected = np.array([100, 200, 120, 220]) * scale + np.tile(region[:2], 2)
            np.testing.assert_allclose(found[0].box, expected)
            self.assertEqual(found[0].scores.argmax(), 3)

    def test_nms_handles_competing_classes_and_multiple_instances(self):
        output = suppress([detection(), detection(label=1, confidence=0.7), detection(box=(250, 250, 290, 290))])
        self.assertEqual(len(output), 2)
        self.assertTrue(all(item.scores.argmax() == 0 for item in output))

    def test_nonfinite_outputs_are_filtered(self):
        detector = Detector.__new__(Detector)
        detector.width, detector.height = 960, 544
        detector.input_name, detector.threshold = 'images', 0.1
        output = np.full((1, 20, 2), np.nan, dtype=np.float32)
        detector.session = SimpleNamespace(run=lambda *args: [output])
        self.assertEqual(detector.detect(np.zeros((540, 960, 3), dtype=np.uint8), request_for()), [])


class MotionTests(unittest.TestCase):
    def test_affine_box_transform_uses_all_corners(self):
        matrix = np.array([[0, -1, 100], [1, 0, 10]], dtype=np.float32)
        np.testing.assert_allclose(move_box([10, 20, 30, 40], matrix), [60, 20, 80, 40])

    def test_textureless_images_do_not_invent_motion(self):
        image = np.zeros((540, 960), dtype=np.uint8)
        matrix, reliable = estimate_motion(image, image, [0, 0, 3840, 2160], [0, 0, 3840, 2160])
        self.assertFalse(reliable)
        np.testing.assert_array_equal(matrix, IDENTITY)

    def test_motion_across_zoom_and_pan(self):
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (540, 960), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 1)
        source = cv2.resize(texture, (3840, 2160))
        moved = cv2.warpAffine(source, np.array([[1, 0, 16], [0, 1, -12]], dtype=np.float32), (3840, 2160))
        current_region = [1240, 440, 3160, 1520]
        current = cv2.resize(moved[440:1520, 1240:3160], (960, 540), interpolation=cv2.INTER_AREA)
        matrix, reliable = estimate_motion(texture, current, [0, 0, 3840, 2160], current_region)
        self.assertTrue(reliable)
        np.testing.assert_allclose(matrix[:, :2], IDENTITY[:, :2], atol=0.002)
        np.testing.assert_allclose(matrix[:, 2], [16, -12], atol=2)


class TrackingTests(unittest.TestCase):
    def test_tracks_outside_crop_move_in_source_coordinates(self):
        state = Sequence(3840, 2160, tracks=[Track(np.array([100, 100, 140, 140], dtype=np.float32), detection().scores, 0.8, 0, hits=3)], frame_index=0)
        request = request_for(index=1, level=2)
        matrix = np.array([[1, 0, 15], [0, 1, 20]], dtype=np.float32)
        update_tracks(state, [], request, matrix, True)
        result = annotations_for(state, 1)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].bbox, np.array([115, 120, 155, 160]) / [3840, 2160, 3840, 2160])

    def test_long_gaps_expire_old_predictions(self):
        state = Sequence(3840, 2160, tracks=[Track(np.array([100, 100, 140, 140], dtype=np.float32), detection().scores, 0.8, 0, hits=3)], frame_index=0)
        update_tracks(state, [], request_for(index=10, level=2), IDENTITY, False)
        self.assertEqual(state.tracks, [])

    def test_two_instances_of_same_class_are_kept(self):
        state = Sequence(3840, 2160)
        update_tracks(state, [detection(), detection(box=(200, 100, 240, 140))], request_for(), IDENTITY, False)
        self.assertEqual(len(annotations_for(state, 0)), 2)

    def test_retry_is_idempotent_and_response_is_not_shared(self):
        detector = FakeDetector([detection()])
        engine = Predictor(detector)
        request = request_for()
        result = engine.predict(request)
        validate_response(result)
        result.annotations.clear()
        retry = engine.predict(request)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(len(retry.annotations), 1)
        self.assertEqual(retry.request_id, request.request_id)
        self.assertEqual(retry.frame, request.frame)

    def test_sequences_are_isolated_and_bounded(self):
        detector = FakeDetector([detection()])
        engine = Predictor(detector, max_sequences=2)
        engine.predict(request_for(sequence='first'))
        detector.detections = []
        self.assertEqual(engine.predict(request_for(sequence='second')).annotations, [])
        engine.predict(request_for(sequence='third'))
        self.assertNotIn('first', engine.sequences)
        self.assertEqual(len(engine.sequences), 2)

    def test_rewind_cannot_reuse_future_detections(self):
        detector = FakeDetector([detection()])
        engine = Predictor(detector)
        engine.predict(request_for(index=8))
        detector.detections = []
        self.assertEqual(engine.predict(request_for(index=0)).annotations, [])

    def test_fully_offscreen_boxes_are_dropped(self):
        state = Sequence(3840, 2160)
        update_tracks(state, [detection(box=(-100, 100, -50, 150))], request_for(), IDENTITY, False)
        self.assertEqual(annotations_for(state, 0), [])


class TrainingGeometryTests(unittest.TestCase):
    def test_training_downsampling_does_not_alias_texture(self):
        from unittest.mock import Mock
        from training_data import TrainingScenes

        data = TrainingScenes.__new__(TrainingScenes)
        data.size = 8
        texture = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
        image = np.repeat(texture[..., None], 3, axis=2)
        data.frames = [(image, np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int64))]
        data.rng = SimpleNamespace(integers=lambda *args: 0, choice=lambda *args: 0.25, uniform=Mock(side_effect=[1.0, np.array([32.0, 32.0]), 0.0]))
        result, _, _ = data._real()
        self.assertAlmostEqual(float(result.mean()), 127.5, delta=2)

    def test_augmentation_box_transform(self):
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        matrix = np.array([[0, -1, 100], [1, 0, 10]], dtype=np.float32)
        np.testing.assert_array_equal(transform_boxes(boxes, matrix), [[60, 20, 80, 40]])

    def test_clipping_removes_degenerate_labels(self):
        boxes, classes = clip_labels([[-20, -20, -10, -10], [5, 5, 15, 15], [90, 90, 110, 110]], [0, 1, 2], 100, 100)
        np.testing.assert_array_equal(classes, [1])
        np.testing.assert_array_equal(boxes, [[5, 5, 15, 15]])


class ApiTests(unittest.TestCase):
    def test_endpoint_contract_with_fake_detector(self):
        from fastapi.testclient import TestClient
        import api

        engine = Predictor(FakeDetector([detection()]))
        with patch('api.initialize', return_value=engine), patch('api.predict', side_effect=engine.predict):
            with TestClient(api.app) as client:
                request = request_for()
                response = client.post('/predict', json=request.model_dump())
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(set(body), {'request_id', 'frame', 'annotations', 'requested_view'})
                self.assertEqual(body['request_id'], request.request_id)
                self.assertEqual(body['frame'], request.frame)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
