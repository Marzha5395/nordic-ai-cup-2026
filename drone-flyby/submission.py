import hashlib
import json
import os
from pathlib import Path
import threading

import cv2

from recorder import record
from dtos import ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS, RequestedViewDto
from solution import Predictor, legal_view
from utils import center_bounds_for_level, describe_camera_rejection, source_region_for_view


ROOT = Path(__file__).resolve().parent


class FlybyPredictor(Predictor):
    def __init__(self, detector, policy='full', max_sequences=8, lag='ahead'):
        if policy not in ('full', 'overview', 'adaptive'):
            raise ValueError('Camera policy must be full, overview, or adaptive')
        if lag not in ('none', 'resend', 'ahead'):
            raise ValueError('Lag handling must be none, resend, or ahead')
        super().__init__(detector, max_sequences=max_sequences)
        self.policy = policy
        self.lag = lag

    def select_view(self, request, state, motion_ok):
        """Choose the next view, but resend a command the service has not applied yet.

        On validation, 104 of 237 views still showed the view from before our previous command: when a
        response arrives just after the next frame is rendered, the service applies the command one frame
        late. Deciding again from that stale view gave targets judged against the already-moved camera, and
        they were rejected. So when the view is exactly the previous request's view (the camera has not moved),
        differs from the last command, and no rejection came back, that command is still pending: repeat it.
        It is only repeated if it is also legal from this view, so it is legal whether it has landed yet or not.
        """
        view = (request.view.resolution_level, request.view.center_x, request.view.center_y)
        pending = getattr(state, 'pending_command', None)
        unmoved = getattr(state, 'last_view', None) == view
        state.last_view = view
        waiting = pending is not None and pending != view and request.camera_command_feedback is None
        legal_here = lambda level, x, y: level in request.camera_constraints.allowed_resolution_levels and describe_camera_rejection(view[0], view[1:], level, (x, y)) is None
        command = None
        if self.lag == 'ahead' and waiting:
            # Commands land in order, so an unrejected last command that is not the view is still to come.
            # Plan the next move from there, so the camera keeps moving every frame, and send it only if it
            # is also legal from where the camera is now; otherwise resend the pending command if that is legal.
            planned = self.choose(self.from_view(request, pending), state, motion_ok)
            if planned is not None and legal_here(planned.resolution_level, planned.center_x, planned.center_y):
                command = planned
            elif legal_here(*pending):
                command = RequestedViewDto(resolution_level=pending[0], center_x=pending[1], center_y=pending[2])
        elif self.lag == 'resend' and waiting and unmoved and legal_here(*pending):
            command = RequestedViewDto(resolution_level=pending[0], center_x=pending[1], center_y=pending[2])
        if command is None:
            command = self.choose(request, state, motion_ok)
        state.pending_command = None if command is None else (command.resolution_level, command.center_x, command.center_y)
        return command

    @staticmethod
    def from_view(request, view):
        """The same request, but as if the camera were already at ``view`` (level, x, y)."""
        level, x, y = view
        bounds = []
        for allowed in ALLOWED_RESOLUTION_LEVELS[level]:
            x1, x2, y1, y2 = center_bounds_for_level(allowed)
            bounds.append({'resolution_level': allowed, 'width': request.view.width, 'height': request.view.height,
                           'minimum_center_x': x1, 'maximum_center_x': x2, 'minimum_center_y': y1, 'maximum_center_y': y2})
        constraints = {'maximum_center_delta': MAXIMUM_CENTER_DELTA_PIXELS[level], 'allowed_resolution_levels': list(ALLOWED_RESOLUTION_LEVELS[level]),
                       'center_bounds': bounds, 'full_view_reset_exempt_from_delta': request.camera_constraints.full_view_reset_exempt_from_delta}
        shifted = request.view.model_copy(update={'resolution_level': level, 'center_x': x, 'center_y': y,
                                                  'source_region_xyxy': list(source_region_for_view(level, x, y))})
        return request.model_copy(update={'view': shifted, 'camera_constraints': type(request.camera_constraints).model_validate(constraints)})

    def choose(self, request, state, motion_ok):
        if self.policy == 'full' or (self.policy == 'overview' and request.view.resolution_level > 0):
            level = 1 if request.view.resolution_level == 2 else 0
            return legal_view(request, level, (request.original_width / 2, request.original_height / 2))
        return super().select_view(request, state, motion_ok)


def create_predictor(weights, device='cuda:0', width=1920, threshold=0.05, tta='none', policy='full', canonical=True):
    from gpu_detector import FlipDetector, TorchDetector

    if tta not in ('none', 'flip'):
        raise ValueError('Test-time augmentation must be none or flip')
    detector = TorchDetector(weights, device=device, width=width, threshold=threshold, canonical=canonical)
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
            weights = Path(os.environ.get('DRONE_MODEL_PATH', ROOT / 'weights' / 'flyby_v6.pt'))
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
                # V2 resizes every view to one physical scale; the terrain-trained models run each view at native size.
                canonical=bool(settings.get('canonical', True)),
            )
    return _predictor


def predict(request):
    response = initialize().predict(request)
    record(request, response)
    return response
