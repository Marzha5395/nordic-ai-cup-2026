import json
from pathlib import Path
import time
import unittest

import numpy as np

from detector import Detector
from dtos import DroneFlybyPredictResponseDto
from test_solution import request_for
from utils import validate_response


ROOT = Path(__file__).resolve().parent


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = Detector(ROOT / 'weights' / 'detector.onnx')

    def test_training_schedule_completed_without_scoring(self):
        metadata = json.loads((ROOT / 'weights' / 'detector.json').read_text())
        self.assertEqual(metadata['completed_steps'], metadata['planned_steps'])
        self.assertEqual(metadata['evaluations_run'], 0)
        self.assertTrue(metadata['antialiased_resampling'])

    def test_onnx_matches_torch_on_generated_input(self):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(4)
        model = YOLO(str(ROOT / 'weights' / 'detector.pt')).model.float().fuse().eval()
        tensor = np.random.default_rng(9).random((1, 3, 544, 960), dtype=np.float32)
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor))[0].numpy()
        actual = self.detector.session.run(None, {self.detector.input_name: tensor})[0]
        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.0002)

    def test_real_endpoint_on_generated_images_only(self):
        from fastapi.testclient import TestClient
        import api

        rng = np.random.default_rng(10)
        durations = []
        with TestClient(api.app) as client:
            for level in range(3):
                image = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
                request = request_for(index=level, level=level, image=image)
                started = time.perf_counter()
                reply = client.post('/predict', json=request.model_dump())
                durations.append(1000 * (time.perf_counter() - started))
                self.assertEqual(reply.status_code, 200)
                response = DroneFlybyPredictResponseDto.model_validate(reply.json())
                validate_response(response)
                self.assertEqual(response.frame, request.frame)
                self.assertEqual(response.request_id, request.request_id)
        print(f'Generated-input request durations (not a detection benchmark): {[round(value, 1) for value in durations]} ms')


if __name__ == '__main__':
    unittest.main()
