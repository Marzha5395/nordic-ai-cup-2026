"""The endpoint's model: decode the view, answer for the whole frame, steer.

The work lives in ``solution.py``. This file is the adapter between the wire
protocol and that code:

* decode the transmitted PNG;
* hand it to the solver together with the geometry it needs;
* turn the solver's source-pixel boxes into the frame-global normalized boxes a
  response uses;
* never raise, because an exception costs the whole frame.

The solver is constructed once, at import, so the model is loaded and warmed up
before the first request arrives rather than during it.

The original edge-detection baseline that shipped with the use case is kept at
``lab/example_baseline_reference.py``.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from fv import recorder
from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, source_bbox_to_global

logger = logging.getLogger(__name__)
EXPERIMENT = os.environ.get('DRONE_EXPERIMENT', 'v19frames')
REPORT_CONF = float(os.environ.get('DRONE_REPORT_CONF', '0.0'))
if not 0.0 <= REPORT_CONF <= 1.0:
    raise ValueError('DRONE_REPORT_CONF must be finite and between 0 and 1')
MODEL_STATUS = {'experiment': EXPERIMENT, 'detector_sha256': None, 'load_error': None,
                'verifier': {'enabled': EXPERIMENT in ('verifier-v1', 'dino-verify'),
                             'loaded': False}}


def _build_solver():
    """Load the detector once. A failure here must not take the server down."""
    try:
        from fv.solution import DETECT_CONF, Solver, WEIGHTS
        MODEL_STATUS['detector_sha256'] = hashlib.sha256(Path(WEIGHTS).read_bytes()).hexdigest()
        if os.environ.get('DRONE_DETECTOR') == 'rfdetr':
            from lab.rfdetr_detector import RFDetrDetector

            shape = tuple(int(value) for value in
                          os.environ.get('DRONE_RFDETR_SHAPE', '960,544').split(','))
            detector = RFDetrDetector(
                WEIGHTS, size=os.environ.get('DRONE_RFDETR_SIZE', 'nano'),
                shape=shape, confidence=DETECT_CONF)
            solver = Solver(detector=detector)
            MODEL_STATUS['detector_type'] = 'rfdetr'
            MODEL_STATUS['rfdetr_shape'] = list(shape)
        else:
            solver = Solver()
            MODEL_STATUS['detector_type'] = 'yolo'
        if EXPERIMENT == 'verifier-v1':
            from lab.crop_verifier import VerifiedDetector

            checkpoint = Path(__file__).resolve().parent / 'lab' / 'out' / 'verifier_v1' / 'verifier.pt'
            threshold = os.environ.get('DRONE_VERIFIER_CONF') or None
            solver.detector = VerifiedDetector(solver.detector, checkpoint,
                                               threshold=float(threshold) if threshold is not None else None)
        if EXPERIMENT == 'dino-verify':
            from lab.reference_verifier import ReferenceRescoredDetector

            bank = os.environ.get('DRONE_REFBANK') or str(
                Path(__file__).resolve().parent / 'lab' / 'out' / 'refbank_v1')
            margin = os.environ.get('DRONE_REFBANK_MARGIN') or None
            solver.detector = ReferenceRescoredDetector(
                solver.detector, bank,
                margin=float(margin) if margin is not None else None)
        camera_ckpt = os.environ.get('DRONE_CAMERA_CKPT')
        if camera_ckpt:
            # Swap the fixed loop for the fitted linear scorer. Fresh instance
            # per sequence (state_for calls the factory once per attempt), which
            # also resets its coverage-age grid. from_checkpoint enforces the
            # detector-sha guard, so this only loads against the detector the
            # policy was fitted on.
            from camera_policy import LearnedCameraPolicy
            LearnedCameraPolicy.from_checkpoint(camera_ckpt, WEIGHTS)  # validate now, fail loud
            solver.policy_factory = lambda: LearnedCameraPolicy.from_checkpoint(camera_ckpt, WEIGHTS)
            MODEL_STATUS['camera_ckpt'] = str(camera_ckpt)
            logger.info('Camera policy: learned scorer from %s', camera_ckpt)
        logger.info('Solver ready')
        return solver
    except Exception as error:
        MODEL_STATUS['load_error'] = type(error).__name__
        logger.exception('Could not load the selected experiment; API will report degraded status')
        return None


SOLVER = _build_solver()


def model_status():
    status = dict(MODEL_STATUS, solver_loaded=SOLVER is not None, report_conf=REPORT_CONF)
    if SOLVER is not None and EXPERIMENT in ('verifier-v1', 'dino-verify'):
        status['verifier'] = SOLVER.detector.health()
    return status


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: report detections and pick the next camera position."""
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning('Camera command from frame %s was ignored: %s',
                       feedback.frame, feedback.reason)

    annotations: List[DroneFlybyPredictionDto] = []
    requested_view: Optional[RequestedViewDto] = None
    image = None

    if SOLVER is not None:
        try:
            image = decode_view(request.view)
            detections, requested = SOLVER(
                sequence_id=request.sequence_id,
                frame=request.frame,
                view=image,
                region=request.view.source_region_xyxy,
                level=request.view.resolution_level,
                center_x=request.view.center_x,
                center_y=request.view.center_y,
                allowed_levels=request.camera_constraints.allowed_resolution_levels,
                maximum_delta=request.camera_constraints.maximum_center_delta,
            )
            annotations = _to_annotations(detections, request)
            requested_view = _to_requested_view(requested, request)
        except Exception:
            # An empty list still scores the frame; an exception loses it.
            logger.exception('Prediction failed on frame %s', request.frame)

    if image is not None:
        # Off unless DRONE_RECORD_DIR is set, and never on the hot path.
        recorder.record(request, image, annotations)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )


def _to_annotations(detections, request) -> List[DroneFlybyPredictionDto]:
    """Source pixels to the frame-global normalized boxes a response uses."""
    out: List[DroneFlybyPredictionDto] = []
    for detection in detections:
        confidence = float(detection['confidence'])
        if not np.isfinite(confidence) or confidence < REPORT_CONF:
            continue
        bbox = clip_bbox_to_frame(
            source_bbox_to_global(detection['bbox'],
                                  request.original_width,
                                  request.original_height))
        # clip_bbox_to_frame returns None for a box with no area left. One of
        # those would invalidate the whole response.
        if bbox is None:
            continue
        out.append(DroneFlybyPredictionDto(
            object_id=detection['object_id'],
            bbox=[float(c) for c in bbox],
            confidence=float(np.clip(confidence, 1e-4, 1.0)),
        ))
        if len(out) >= 500:
            break
    return out


def _to_requested_view(requested, request) -> Optional[RequestedViewDto]:
    """Only send a command that this request's constraints already allow."""
    if requested is None:
        return None
    level, center_x, center_y = (int(v) for v in requested)
    constraints = request.camera_constraints
    if level not in constraints.allowed_resolution_levels:
        return None
    bounds = constraints.bounds_for_level(level)
    if bounds is None:
        return None
    center_x = min(max(center_x, bounds.minimum_center_x), bounds.maximum_center_x)
    center_y = min(max(center_y, bounds.minimum_center_y), bounds.maximum_center_y)
    if level != 0:
        distance = float(np.hypot(center_x - request.view.center_x,
                                  center_y - request.view.center_y))
        if distance > constraints.maximum_center_delta:
            return None
    return RequestedViewDto(resolution_level=level, center_x=center_x,
                            center_y=center_y)
