import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from data_v2 import HOLDOUT_FRAMES
from dtos import DroneFlybyPredictResponseDto, OBJECT_CLASSES
from test_solution import request_for
from utils import validate_response


ROOT = Path(__file__).resolve().parent


class ArtifactTests(unittest.TestCase):
    def test_packaged_weights_and_runtime_manifest_match(self):
        weights = ROOT / 'weights/flyby_v2.pt'
        manifest = json.loads(weights.with_suffix('.json').read_text())
        runtime = json.loads(weights.with_suffix('.runtime.json').read_text())
        with weights.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        self.assertEqual(runtime['weights_sha256'], digest)
        self.assertEqual(manifest['weights_sha256'], digest)
        self.assertEqual(manifest['training']['classes'], list(OBJECT_CLASSES))
        self.assertEqual(runtime['camera_policy'], 'full')
        self.assertNotIn('training', runtime)
        self.assertNotIn('measurements', runtime)

    def test_recorded_training_excludes_heldout_frames(self):
        manifest = json.loads((ROOT / 'weights/flyby_v2.json').read_text())
        self.assertFalse(set(manifest['training']['training_frames']) & HOLDOUT_FRAMES)
        self.assertGreaterEqual(manifest['training']['completed_steps'], 4000)
        self.assertTrue(manifest['training']['p2'])
        self.assertEqual(manifest['training']['training_backgrounds'], 1040)
        self.assertTrue(manifest['measurements'])
        for measurement in manifest['measurements']:
            self.assertTrue(0 <= measurement['map50'] <= 1)

    def test_real_endpoint_without_external_http_calls(self):
        from fastapi.testclient import TestClient
        import api
        import submission

        rng = np.random.default_rng(10)
        with patch.dict('os.environ', {'DRONE_DEVICE': 'cpu'}), patch.object(submission, '_predictor', None):
            with patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected external HTTP call')):
                with TestClient(api.app) as client:
                    for level in range(3):
                        image = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
                        request = request_for(index=level, level=level, image=image)
                        reply = client.post('/predict', json=request.model_dump())
                        self.assertEqual(reply.status_code, 200)
                        response = DroneFlybyPredictResponseDto.model_validate(reply.json())
                        validate_response(response)
                        self.assertEqual(response.frame, request.frame)
                        self.assertEqual(response.request_id, request.request_id)


if __name__ == '__main__':
    unittest.main()
