import hashlib
import json
import os
from pathlib import Path
import threading

import cv2

from solution import Predictor, legal_view


ROOT = Path(__file__).resolve().parent


class FlybyPredictor(Predictor):
    def __init__(self, detector, policy='full', max_sequences=8):
        if policy not in ('full', 'overview', 'adaptive'):
            raise ValueError('Camera policy must be full, overview, or adaptive')
        super().__init__(detector, max_sequences=max_sequences)
        self.policy = policy

    def select_view(self, request, state, motion_ok):
        if self.policy == 'full' or (self.policy == 'overview' and request.view.resolution_level > 0):
            level = 1 if request.view.resolution_level == 2 else 0
            return legal_view(request, level, (request.original_width / 2, request.original_height / 2))
        return super().select_view(request, state, motion_ok)


def create_predictor(weights, device='cuda:0', width=1920, threshold=0.05, tta='none', policy='full'):
    from gpu_detector import FlipDetector, TorchDetector

    if tta not in ('none', 'flip'):
        raise ValueError('Test-time augmentation must be none or flip')
    detector = TorchDetector(weights, device=device, width=width, threshold=threshold, canonical=True)
    if tta == 'flip':
        detector = FlipDetector(detector)
    return FlybyPredictor(detector, policy=policy)


_predictor = None
_lock = threading.Lock()


def initialize():
    global _predictor

    with _lock:
        if _predictor is None:
            cv2.setNumThreads(1)
            weights = Path(os.environ.get('DRONE_MODEL_PATH', ROOT / 'weights' / 'flyby_v2.pt'))
            settings_path = weights.with_suffix('.runtime.json')
            settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
            if settings.get('weights_sha256'):
                with weights.open('rb') as handle:
                    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                if actual != settings['weights_sha256']:
                    raise ValueError('Model weights do not match their deployment manifest')
            _predictor = create_predictor(
                weights,
                device=os.environ.get('DRONE_DEVICE', settings.get('device', 'cuda:0')),
                width=int(os.environ.get('DRONE_INPUT_WIDTH', settings.get('input_width', 1920))),
                threshold=float(os.environ.get('DRONE_CONFIDENCE', settings.get('confidence', 0.05))),
                tta=os.environ.get('DRONE_TTA', settings.get('tta', 'none')),
                policy=os.environ.get('DRONE_CAMERA_POLICY', settings.get('camera_policy', 'full')),
            )
    return _predictor


def predict(request):
    return initialize().predict(request)
