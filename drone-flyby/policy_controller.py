"""Drone flyby controller: YOLO detections fused into a flow-propagated
memory, emitted as whole-frame predictions, plus a flow-following camera
scheduler. The heavy lifting lives in the ``policy`` package.

Run ``python local_evaluator.py`` (optionally ``--realtime``) with
``api.py`` serving this module's ``predict``.
"""

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from policy.detector import Detector
from policy.flow import Flow
from policy.geometry import edge_reliability, view_to_source_box
from policy.memory import Memory
from policy.scheduler import Scheduler
from utils import clip_bbox_to_frame, decode_view

logger = logging.getLogger(__name__)

# Set DRONE_RECORD_DIR to keep every request (view PNG + geometry) as JSON,
# e.g. to archive the official validation sequence for later analysis.
RECORD_DIR = os.environ.get('DRONE_RECORD_DIR')


def _record(request: DroneFlybyPredictRequestDto) -> None:
    if not RECORD_DIR:
        return
    try:
        d = Path(RECORD_DIR) / request.sequence_id.replace('/', '_')
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{request.frame_index:06d}.json').write_text(request.model_dump_json())
    except Exception:
        logger.exception('recording failed on frame %s', request.frame)


# Loaded once, shared by every sequence. If it fails we still import and
# answer frames with memory-only (empty) predictions rather than crash.
try:
    DETECTOR: Optional[Detector] = Detector()
except Exception:
    logger.exception('Detector init failed')
    DETECTOR = None


class Controller:
    """Per-sequence state: flow map, object memory, camera scheduler."""

    def __init__(self):
        self.flow = Flow()
        self.memory = Memory(self.flow)
        self.scheduler = Scheduler()
        self.prev_gray = None
        self.prev_region = None
        self.prev_frame = None
        self.prev_level = None
        self.hypothesis_done = False
        self.flow_residual = 0.0
        self.last_annotations = None
        self.last_requested_view = None


_STATE: 'OrderedDict[str, Controller]' = OrderedDict()


def _controller(sequence_id: str, frame: int) -> Controller:
    ctrl = _STATE.get(sequence_id)
    if ctrl is not None and ctrl.prev_frame is not None \
            and frame < ctrl.prev_frame:
        ctrl = None  # clock rewound: fresh sequence
    if ctrl is None:
        ctrl = Controller()
        _STATE[sequence_id] = ctrl
    _STATE.move_to_end(sequence_id)
    while len(_STATE) > 4:
        _STATE.popitem(last=False)
    return ctrl


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: whole-frame predictions plus the next camera move."""
    _record(request)
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            'Camera command from frame %s was ignored: %s',
            feedback.frame,
            feedback.reason,
        )

    try:
        image = decode_view(request.view)
    except Exception:
        logger.exception('Could not decode view on frame %s', request.frame)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[], requested_view=None)

    ctrl = _controller(request.sequence_id, request.frame)

    # duplicate delivery of a frame we already answered: replay the cached
    # answer, no flow/ingest work
    if (ctrl.prev_frame == request.frame
            and ctrl.last_annotations is not None):
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=ctrl.last_annotations,
            requested_view=ctrl.last_requested_view)

    annotations = []
    requested_view: Optional[RequestedViewDto] = None
    t0 = time.monotonic()
    n_dets = 0

    try:
        view = request.view
        level = int(view.resolution_level)
        region = np.asarray(view.source_region_xyxy, dtype=np.float64)
        frame = int(request.frame)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # --- flow update ---
        if ctrl.prev_gray is not None:
            dframe = frame - ctrl.prev_frame
            if dframe > 0:
                try:
                    if (ctrl.prev_level == 0 and level == 1
                            and not ctrl.hypothesis_done):
                        ctrl.hypothesis_done = True
                        ctrl.flow.hypothesis_check(
                            ctrl.prev_gray, ctrl.prev_region, gray, region)
                    stats = ctrl.flow.observe(ctrl.prev_gray, ctrl.prev_region,
                                              gray, region, dframe)
                    ctrl.flow_residual = stats['residual_px']
                except Exception:
                    logger.exception('flow update failed on frame %s', frame)

        # --- detect + ingest ---
        dets_src = []
        if DETECTOR is not None:
            try:
                for box_v, conf, cls in DETECTOR.detect(image, level):
                    box_s = view_to_source_box(box_v, region)
                    dets_src.append((box_s, conf, cls,
                                     edge_reliability(box_s, region)))
            except Exception:
                logger.exception('Detector failed on frame %s', frame)
        n_dets = len(dets_src)
        ctrl.memory.ingest(dets_src, frame, level, region)

        # --- emit whole-frame predictions ---
        for cls_name, box, conf in ctrl.memory.emit(frame):
            bbox = clip_bbox_to_frame(
                [box[0] / 3840.0, box[1] / 2160.0,
                 box[2] / 3840.0, box[3] / 2160.0])
            if bbox is None:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=cls_name, bbox=list(bbox),
                confidence=round(float(conf), 4)))

        # --- camera ---
        try:
            requested_view = ctrl.scheduler.next_view(request, ctrl.flow,
                                                      ctrl.memory)
        except Exception:
            logger.exception('scheduler failed on frame %s', frame)
            requested_view = None

        ctrl.prev_gray = gray
        ctrl.prev_region = region
        ctrl.prev_frame = frame
        ctrl.prev_level = level
        ctrl.last_annotations = annotations
        ctrl.last_requested_view = requested_view
    except Exception:
        logger.exception('predict failed on frame %s', request.frame)
        try:
            annotations = []
            for c, b, cf in ctrl.memory.emit(request.frame):
                bb = clip_bbox_to_frame([b[0] / 3840.0, b[1] / 2160.0,
                                         b[2] / 3840.0, b[3] / 2160.0])
                if bb is None:
                    continue
                annotations.append(DroneFlybyPredictionDto(
                    object_id=c, bbox=list(bb),
                    confidence=round(float(cf), 4)))
        except Exception:
            annotations = []

    logger.info(
        'frame %d L%d (%d,%d) dets=%d tracks=%d ann=%d flow_res=%.2f %.0fms',
        request.frame, request.view.resolution_level,
        request.view.center_x, request.view.center_y, n_dets,
        len(ctrl.memory.tracks), len(annotations), ctrl.flow_residual,
        (time.monotonic() - t0) * 1000)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
