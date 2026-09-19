from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import math
import threading
import time

import cv2
import numpy as np

from detector import Detection, Detector, overlap, suppress
from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictResponseDto, RequestedViewDto
from motion import IDENTITY, estimate_motion, move_box
from utils import clip_bbox_to_frame, decode_view, source_bbox_to_global


logger = logging.getLogger(__name__)


@dataclass
class Track:
    box: np.ndarray
    evidence: np.ndarray
    confidence: float
    last_seen: int
    hits: int = 1
    missed: int = 0
    last_detail: int = -1000

    @property
    def label(self):
        return int(self.evidence.argmax())


@dataclass
class Sequence:
    width: int
    height: int
    tracks: list = field(default_factory=list)
    previous_image: object = None
    previous_region: object = None
    frame_index: int = -1
    previous_level: int = 0
    last_full_frame: int = -1
    exploration: int = 0
    response: object = None
    request_id: str = ''
    touched: float = field(default_factory=time.monotonic)


def legal_view(request, level, center):
    constraints = request.camera_constraints
    bounds = constraints.bounds_for_level(level)
    if level not in constraints.allowed_resolution_levels or bounds is None:
        return None
    low = np.array([bounds.minimum_center_x, bounds.minimum_center_y], dtype=float)
    high = np.array([bounds.maximum_center_x, bounds.maximum_center_y], dtype=float)
    current = np.array([request.view.center_x, request.view.center_y], dtype=float)
    target = np.clip(np.asarray(center, dtype=float), low, high)
    if not np.isfinite(target).all():
        return None
    exempt = level == 0 and constraints.full_view_reset_exempt_from_delta
    limit = constraints.maximum_center_delta
    if not math.isfinite(limit) or limit < 0:
        return None
    closest = np.clip(current, low, high)
    if not exempt and np.linalg.norm(closest - current) > limit:
        return None
    if not exempt and np.linalg.norm(target - current) > limit:
        start, end = 0.0, 1.0
        for _ in range(40):
            middle = (start + end) / 2
            point = closest + middle * (target - closest)
            if np.linalg.norm(point - current) <= limit:
                start = middle
            else:
                end = middle
        target = closest + start * (target - closest)
    choices = []
    for x in range(math.floor(target[0]) - 1, math.ceil(target[0]) + 2):
        for y in range(math.floor(target[1]) - 1, math.ceil(target[1]) + 2):
            point = np.array([x, y])
            if (point >= low).all() and (point <= high).all() and (exempt or np.linalg.norm(point - current) <= limit):
                choices.append((float(np.linalg.norm(point - target)), x, y))
    if not choices:
        return None
    _, x, y = min(choices)
    return RequestedViewDto(resolution_level=int(level), center_x=int(x), center_y=int(y))


def choose_view(request, state, motion_ok):
    level = request.view.resolution_level
    center = (request.original_width / 2, request.original_height / 2)
    if level == 2:
        return legal_view(request, 1, center)
    if level == 1:
        reliable_map = bool(state.tracks) and sum(track.hits > 1 for track in state.tracks) >= 0.6 * len(state.tracks)
        if motion_ok and reliable_map and state.previous_level != 2 and request.frame_index - state.last_full_frame <= 1:
            region = np.asarray(request.view.source_region_xyxy)
            candidates = []
            for track in state.tracks:
                point = (track.box[:2] + track.box[2:]) / 2
                extent = max(track.box[2:] - track.box[:2])
                if (point >= region[:2]).all() and (point <= region[2:]).all() and extent < 65 and track.confidence < 0.7:
                    candidates.append(((1 - track.confidence) / max(extent, 8), point))
            if candidates:
                command = legal_view(request, 2, max(candidates, key=lambda item: item[0])[1])
                if command is not None:
                    return command
        return legal_view(request, 0, center)
    bounds = request.camera_constraints.bounds_for_level(1)
    if bounds is None:
        return None
    corners = [
        (bounds.minimum_center_x, bounds.minimum_center_y),
        (bounds.maximum_center_x, bounds.minimum_center_y),
        (bounds.maximum_center_x, bounds.maximum_center_y),
        (bounds.minimum_center_x, bounds.maximum_center_y),
    ]
    exploration = corners[state.exploration % len(corners)]
    state.exploration += 1
    if state.exploration % 3 == 0 or not state.tracks:
        return legal_view(request, 1, exploration)
    candidates = []
    for track in state.tracks:
        age = request.frame_index - track.last_detail
        if age < 5 or request.frame_index - track.last_seen > 2:
            continue
        extent = max(track.box[2:] - track.box[:2])
        utility = (0.4 + 1 - track.confidence) * min(3, 80 / max(extent, 8))
        candidates.append((utility, (track.box[:2] + track.box[2:]) / 2))
    if not candidates:
        return legal_view(request, 1, exploration)
    target = max(candidates, key=lambda item: item[0])[1]
    return legal_view(request, 1, target)


def update_tracks(state, detections, request, matrix, motion_ok):
    index = request.frame_index
    gap = max(1, index - state.frame_index)
    for track in state.tracks:
        track.box = move_box(track.box, matrix)
    unmatched = set(range(len(state.tracks)))
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        best, best_score = None, 0.0
        for candidate in unmatched:
            track = state.tracks[candidate]
            iou = float(overlap(detection.box, [track.box])[0])
            same_class = track.label == int(detection.scores.argmax())
            threshold = 0.12 if same_class and motion_ok else 0.35
            if iou < threshold:
                continue
            score = iou + 0.15 * same_class
            if score > best_score:
                best, best_score = candidate, score
        weight = detection.quality * (1 + request.view.resolution_level)
        if best is None:
            state.tracks.append(Track(
                detection.box.copy(), detection.scores.copy() * weight,
                detection.confidence, index,
                last_detail=index if request.view.resolution_level else -1000,
            ))
            continue
        unmatched.remove(best)
        track = state.tracks[best]
        if detection.quality < 1 and motion_ok:
            track.box = track.box * 0.75 + detection.box * 0.25
        else:
            track.box = detection.box.copy()
        track.evidence = track.evidence * 0.9 + detection.scores * weight
        track.confidence = 0.5 * track.confidence + 0.5 * detection.confidence
        track.hits += 1
        track.missed = 0
        track.last_seen = index
        if request.view.resolution_level:
            track.last_detail = index
    region = np.asarray(request.view.source_region_xyxy)
    survivors = []
    for track_index, track in enumerate(state.tracks):
        age = index - track.last_seen
        if track_index in unmatched:
            center = (track.box[:2] + track.box[2:]) / 2
            in_view = (center >= region[:2]).all() and (center <= region[2:]).all()
            track.missed += gap if in_view else 0
            track.confidence *= 0.8 ** gap if in_view else 0.94 ** gap
        limit = 6 if motion_ok and track.hits > 1 else 1
        if age > limit or track.missed > 2 or track.confidence < 0.07:
            continue
        if clip_bbox_to_frame(source_bbox_to_global(track.box, state.width, state.height)) is None:
            continue
        survivors.append(track)
    state.tracks = sorted(survivors, key=lambda item: item.confidence, reverse=True)[:250]


def annotations_for(state, index):
    detections = []
    for track in state.tracks:
        box = clip_bbox_to_frame(source_bbox_to_global(track.box, state.width, state.height))
        if box is None or not np.isfinite(track.box).all():
            continue
        age = index - track.last_seen
        confidence = min(0.98, track.confidence * (1 + 0.04 * min(track.hits - 1, 4))) * 0.88 ** age
        scores = np.zeros(len(OBJECT_CLASSES), dtype=np.float32)
        scores[track.label] = confidence
        detections.append(Detection(np.asarray(box, dtype=np.float32), scores))
    return [DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[int(item.scores.argmax())], bbox=[float(value) for value in item.box], confidence=item.confidence) for item in suppress(detections)]


class Predictor:
    def __init__(self, detector, max_sequences=8):
        self.detector = detector
        self.max_sequences = max_sequences
        self.sequences = OrderedDict()
        self.lock = threading.Lock()

    def select_view(self, request, state, motion_ok):
        return choose_view(request, state, motion_ok)

    def predict(self, request):
        with self.lock:
            now = time.monotonic()
            for key in list(self.sequences):
                if now - self.sequences[key].touched > 600:
                    del self.sequences[key]
            state = self.sequences.get(request.sequence_id)
            if state is not None and state.request_id == request.request_id:
                state.touched = now
                self.sequences.move_to_end(request.sequence_id)
                return state.response.model_copy(deep=True)
            if state is None or request.frame_index <= state.frame_index or (state.width, state.height) != (request.original_width, request.original_height):
                state = Sequence(request.original_width, request.original_height)
                self.sequences[request.sequence_id] = state
            state.touched = now
            self.sequences.move_to_end(request.sequence_id)
            while len(self.sequences) > self.max_sequences:
                self.sequences.popitem(last=False)
            image = decode_view(request.view)
            if image.shape[:2] != (request.view.height, request.view.width):
                raise ValueError('Image dimensions disagree with the transmitted view')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            matrix, motion_ok = IDENTITY.copy(), False
            gap = request.frame_index - state.frame_index
            if gap > 6:
                state.tracks.clear()
            if state.previous_image is not None and 0 < gap <= 6:
                matrix, motion_ok = estimate_motion(state.previous_image, gray, state.previous_region, request.view.source_region_xyxy)
            detections = self.detector.detect(image, request)
            update_tracks(state, detections, request, matrix, motion_ok)
            if request.view.resolution_level == 0:
                state.last_full_frame = request.frame_index
            if request.camera_command_feedback is not None:
                logger.warning('Camera command rejected: %s', request.camera_command_feedback.reason)
            response = DroneFlybyPredictResponseDto(
                request_id=request.request_id,
                frame=request.frame,
                annotations=annotations_for(state, request.frame_index),
                requested_view=self.select_view(request, state, motion_ok),
            )
            state.previous_image = gray
            state.previous_region = tuple(request.view.source_region_xyxy)
            state.previous_level = request.view.resolution_level
            state.frame_index = request.frame_index
            state.request_id = request.request_id
            state.response = response
            return response.model_copy(deep=True)


_predictor = None
_initialization_lock = threading.Lock()


def initialize():
    global _predictor

    with _initialization_lock:
        if _predictor is None:
            cv2.setNumThreads(1)
            _predictor = Predictor(Detector())
    return _predictor


def predict(request):
    return initialize().predict(request)
