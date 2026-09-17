"""One request in, one response out.

The protocol is asymmetric: the request carries a 960x540 crop of wherever the
camera is pointed, and the response has to describe the whole 3840x2160 source
frame. So a frame is answered from the world model, not from the image -- the
image only updates the world model and decides where to look next.

All the state is per sequence and keyed by ``sequence_id``, so a fresh attempt
starts from an empty world without the server having to be restarted.
"""

import base64
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    OBJECT_CLASSES,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, describe_camera_rejection

from . import config
from .detector import Detector
from .motion import FlowModel, flow_samples
from .planner import CameraPlanner
from .tracking import World

logger = logging.getLogger(__name__)

_detector: Optional[Detector] = None
_detector_error: Optional[BaseException] = None
_detector_lock = threading.Lock()


def get_detector() -> Optional[Detector]:
    """Load the model once. A failure here must not take the server down."""
    global _detector, _detector_error
    if _detector is not None or _detector_error is not None:
        return _detector
    with _detector_lock:
        if _detector is None and _detector_error is None:
            try:
                _detector = Detector()
                logger.info('detector ready on device %s', _detector.device)
            except BaseException as error:           # noqa: BLE001 - never fatal
                _detector_error = error
                logger.exception('detector unavailable, answering with no detections')
    return _detector


class SequenceState:
    """Everything remembered across the frames of one attempt."""

    def __init__(self, frame_width: int, frame_height: int):
        self.world = World(frame_width, frame_height)
        self.planner = CameraPlanner()
        self.flow = FlowModel()
        self.previous_image: Optional[np.ndarray] = None
        self.previous_region: Optional[Tuple[float, float, float, float]] = None
        # The frame the kept image came from, which is not the last frame
        # answered when a view failed to decode.
        self.previous_image_frame: Optional[int] = None
        self.previous_frame: Optional[int] = None
        self.frames_handled = 0


_states: Dict[str, SequenceState] = {}
_state_lock = threading.Lock()
# The world model is mutable state shared across requests, and uvicorn will
# happily run two of them at once on different threads. Frames arrive one at a
# time in an attempt, so serialising costs nothing and rules out the race.
_answer_lock = threading.Lock()


def _state_for(request: DroneFlybyPredictRequestDto) -> SequenceState:
    with _state_lock:
        state = _states.get(request.sequence_id)
        restart = (
            state is None
            or state.previous_frame is None
            or request.frame < state.previous_frame
        )
        if restart:
            state = SequenceState(request.original_width, request.original_height)
            # One sequence at a time, and an attempt can be retried with a new
            # identifier, so there is no reason to keep the old worlds around.
            _states.clear()
            _states[request.sequence_id] = state
        return state


def warm_up() -> None:
    """Run one complete synthetic request through the whole path.

    Warming the model alone is not enough. The first real request also pays for
    pydantic model construction, the PNG decode, the first cuDNN autotune at
    the real input size and the first pass through the tracker, and on the
    reference harness that first answer took 1.7 seconds -- four emitted frames
    -- while every later one took 37 ms. Doing it here means the attempt does
    not.
    """
    try:
        detector = get_detector()
        if detector is None:
            return
        import cv2

        blank = np.zeros((540, 960, 3), np.uint8)
        encoded, buffer = cv2.imencode('.png', blank, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not encoded:
            return
        payload = {
            'sequence_id': '__warmup__',
            'frame': 0,
            'frame_index': 0,
            'request_id': '__warmup__',
            'frame_interval_ms': 333,
            'response_timeout_ms': 3333,
            'original_width': 3840,
            'original_height': 2160,
            'view': {
                'resolution_level': 0,
                'center_x': 1920,
                'center_y': 1080,
                'view_id': '__warmup__',
                'image': base64.b64encode(buffer.tobytes()).decode('ascii'),
                'image_media_type': 'image/png',
                'width': 960,
                'height': 540,
                'source_region_xyxy': [0, 0, 3840, 2160],
            },
            'camera_constraints': {
                'maximum_center_delta': 2203.0,
                'allowed_resolution_levels': [0, 1],
                'center_bounds': [
                    {
                        'resolution_level': 0, 'width': 960, 'height': 540,
                        'minimum_center_x': 1920, 'maximum_center_x': 1920,
                        'minimum_center_y': 1080, 'maximum_center_y': 1080,
                    },
                    {
                        'resolution_level': 1, 'width': 960, 'height': 540,
                        'minimum_center_x': 960, 'maximum_center_x': 2880,
                        'minimum_center_y': 540, 'maximum_center_y': 1620,
                    },
                ],
                'full_view_reset_exempt_from_delta': True,
            },
            'camera_command_feedback': None,
        }
        for _ in range(2):
            predict(DroneFlybyPredictRequestDto.model_validate(payload))
    except BaseException:                            # noqa: BLE001 - never fatal
        logger.exception('warm-up failed; the first frame will be slower')
    finally:
        reset()


def reset(drop_detector: bool = False) -> None:
    """Forget every sequence. Used by the offline harnesses.

    ``drop_detector`` also unloads the model, which is only useful to a sweep
    that changes the detector's own settings between runs.
    """
    global _detector, _detector_error
    with _state_lock:
        _states.clear()
    if drop_detector:
        with _detector_lock:
            _detector = None
            _detector_error = None


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame."""
    started = time.perf_counter()
    annotations: List[DroneFlybyPredictionDto] = []
    requested_view: Optional[RequestedViewDto] = None
    try:
        with _answer_lock:
            annotations, requested_view = _answer(request)
    except BaseException:                            # noqa: BLE001 - never fatal
        logger.exception('failed on frame %s, answering empty', request.frame)

    if request.camera_command_feedback is not None:
        logger.warning(
            'camera command from frame %s was ignored: %s',
            request.camera_command_feedback.frame,
            request.camera_command_feedback.reason,
        )

    logger.debug(
        'frame %s answered in %.1f ms with %d annotations',
        request.frame,
        1000.0 * (time.perf_counter() - started),
        len(annotations),
    )
    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )


def _answer(request: DroneFlybyPredictRequestDto):
    state = _state_for(request)
    view = request.view
    region = tuple(float(c) for c in view.source_region_xyxy)
    try:
        image = decode_view(view)
    except Exception:                                # noqa: BLE001 - never fatal
        # A view that will not decode costs the sighting, not the answer: the
        # world model still knows where everything was and how fast it moves.
        logger.exception('could not decode the view for frame %s', request.frame)
        image = None

    elapsed = 0
    if state.previous_frame is not None:
        elapsed = max(0, request.frame - state.previous_frame)

    # 1. How far has the ground moved since the last frame we saw? Measured
    #    across a grid of tiles, because the field is not uniform.
    if image is not None and state.previous_image is not None:
        since_image = request.frame - state.previous_image_frame
        if since_image > 0:
            state.flow.add(
                flow_samples(
                    state.previous_image, state.previous_region, image, region
                ),
                since_image,
            )
    velocity = state.flow.centre_velocity

    # 2. Move the world forward to this frame.
    if elapsed > 0:
        state.world.advance(elapsed, state.flow)
        state.planner.advance(velocity, elapsed)

    # 3. Look at what arrived, and fold it in.
    if image is not None:
        detector = get_detector()
        detections = detector.detect(image, region) if detector is not None else []
        state.world.update(detections, request.frame, region)
        state.planner.observe(region, view.resolution_level)

    # 4. Answer for the whole source frame.
    annotations: List[DroneFlybyPredictionDto] = []
    for prediction in state.world.predictions(request.frame):
        bbox = clip_bbox_to_frame(
            (
                prediction.box[0] / request.original_width,
                prediction.box[1] / request.original_height,
                prediction.box[2] / request.original_width,
                prediction.box[3] / request.original_height,
            )
        )
        if bbox is None:
            continue
        # Round first, then check: a box narrower than a millionth survives
        # clipping and then rounds to zero width, and a zero-width box fails
        # validation and takes every other detection in the response with it.
        x1, y1, x2, y2 = (round(float(c), 6) for c in bbox)
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        confidence = min(1.0, max(0.0, round(float(prediction.confidence), 6)))
        annotations.append(
            DroneFlybyPredictionDto(
                object_id=OBJECT_CLASSES[prediction.class_index],
                bbox=[x1, y1, x2, y2],
                confidence=confidence,
            )
        )
        if len(annotations) >= config.MAXIMUM_ANNOTATIONS:
            break

    # 5. Decide where to point next.
    requested_view = _choose_view(state, request, velocity)

    if image is not None:
        state.previous_image = image
        state.previous_region = region
        state.previous_image_frame = request.frame
    state.previous_frame = request.frame
    state.frames_handled += 1
    return annotations, requested_view


def _choose_view(
    state: SequenceState,
    request: DroneFlybyPredictRequestDto,
    velocity: Tuple[float, float],
) -> Optional[RequestedViewDto]:
    view = request.view
    chosen = state.planner.choose(
        current_level=view.resolution_level,
        current_centre=(view.center_x, view.center_y),
        constraints=request.camera_constraints,
        tracks=state.world.tracks,
        frame=request.frame,
        drift=velocity,
    )
    if chosen is None:
        return None
    level, centre_x, centre_y = chosen
    if (level, centre_x, centre_y) == (
        view.resolution_level, view.center_x, view.center_y
    ):
        return None                                   # already there, hold

    # The same three checks the evaluator runs. A command it refuses costs a
    # frame of camera movement, so never send one.
    rejection = describe_camera_rejection(
        view.resolution_level,
        (view.center_x, view.center_y),
        level,
        (centre_x, centre_y),
    )
    if rejection is not None:
        logger.warning('dropping illegal camera command: %s', rejection)
        return None
    return RequestedViewDto(
        resolution_level=int(level),
        center_x=int(centre_x),
        center_y=int(centre_y),
    )
