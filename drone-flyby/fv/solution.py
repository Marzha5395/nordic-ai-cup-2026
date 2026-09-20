"""The solution: a detector, a world model and a camera policy.

The three pieces map onto the three things the protocol asks for. See
``EXPERIMENTS.md`` for how each was arrived at and what the alternatives
scored.

**Detector.** A YOLO trained on synthetic views composited from the supplied
scene (see ``lab/``). It runs on the 960x540 image that arrived with the
request and returns boxes in source-frame pixels.

**World model.** Every response has to cover the whole frame, but the camera
sees a quarter of it at Level 1 and a sixteenth at Level 2. The flight is a
constant translation over planar ground, so consecutive frames are related by a
single fixed homography. Every tracked object is warped forward by that
homography each frame, whether or not the camera is looking at it, and is
reported until it leaves the frame. Measured worth: +0.02 mAP with the camera
parked on the full view, +0.34 when it zooms.

**Camera policy.** Mostly Level 0. The challenge is framed around zooming, but
the detector turned out to read the small classes from the 4x-downsampled full
view well enough that answering for the entire frame every frame beats
observing a quarter of it in detail. The policy still dips to Level 1 every
fourth frame, as insurance: the detector's Level-0 accuracy was measured on the
scene its training sprites were cut from, so it is optimistic there in a way a
genuine 2x resolution advantage is not.

All state is per ``sequence_id``, and is also dropped if the frame number ever
goes backwards, so a fresh attempt starts clean.
"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
VIEW_WIDTH = 960
VIEW_HEIGHT = 540
SOURCE_REGION_SIZES = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}

# The inter-frame homography, measured in lab/motion_model.py over all 24
# consecutive pairs of the supplied scene. It maps a point in frame i to the
# same ground point in frame i+1. The standard deviation of its translation
# terms across those pairs is under 0.8 px, so it is treated as a constant.
DEFAULT_HOMOGRAPHY = np.array([
    [1.006333e+00, -1.266000e-03, -1.2454043e+01],
    [1.980000e-04, 1.011600e+00, 5.2717630e+01],
    [0.0, -1.0e-06, 1.0],
], dtype=np.float64)

WEIGHTS = os.environ.get('DRONE_WEIGHTS', str(HERE.parent / 'weights' / 'fv_v19frames.pt'))
DEVICE = os.environ.get('DRONE_DEVICE', '0')
# The transmitted image is 960x540 and the detector was trained at that size.
# Upsampling before inference was measured and is worse as well as slower, so
# the view is passed through at native resolution.
INFER_IMGSZ = int(os.environ.get('DRONE_IMGSZ', '960'))
# A low floor: extra low-ranked detections cost little in a metric that ranks
# predictions globally, and buy recall if the detector is less sure on an
# unfamiliar scene than it is on the one it was trained from.
DETECT_CONF = float(os.environ.get('DRONE_CONF', '0.05'))

# Extra class guesses per track, emitted below the leading one. Zero restores
# single-class reporting.
ALTERNATE_CLASSES = int(os.environ.get('DRONE_ALTERNATES', '2'))
ALTERNATE_DAMPING = float(os.environ.get('DRONE_ALTERNATE_DAMPING', '0.5'))
# YOLO suppresses per class by default, so one object can carry a box for every
# class it resembles. Section 21 judged those extras cheap, but that rested on
# the leading box usually being wrong; at 0.65 it usually is not.
AGNOSTIC_NMS = os.environ.get('DRONE_AGNOSTIC_NMS', '0') == '1'
# The drone holds 600 m in every capture and the flight homographies agree to
# about a pixel, so an asset covers nearly the same ground area in any scene.
# Mean annotated size in source pixels, from the supplied Helsinki labels.
CLASS_SIZES = {
    'ta-ta': (32, 17), 'small_launcher': (22, 30), 'jammer': (33, 42),
    'medium_launcher': (40, 43), 'spacecraft': (45, 47), 'small_plane': (42, 50),
    'tank': (50, 46), 'medium_plane': (56, 45), 'mine_roller': (53, 60),
    'small_tower': (58, 57), 'large_tower': (62, 62), 'jet_plane': (77, 79),
    'helicopter': (117, 92), 'large_launcher': (145, 110), 'hangar': (186, 94),
    'condor': (174, 152),
}
SIZE_PRIOR = os.environ.get('DRONE_SIZE_PRIOR', '0') == '1'
SAME_CLASS_CONTAINMENT = os.environ.get('DRONE_SAME_CLASS_CONTAINMENT', '0') == '1'
# Beyond this the box cannot be that class at a fixed altitude; between the two
# the confidence is damped rather than cut, because rotation and clipping move
# an honest box around.
SIZE_TOLERATED = float(os.environ.get('DRONE_SIZE_TOLERATED', '1.6'))
SIZE_REJECT = float(os.environ.get('DRONE_SIZE_REJECT', '3.0'))


def size_agreement(object_id: str, width: float, height: float) -> float:
    """How far a box is from its class's known size, by longest axis.

    Only the long axis is used. Rotating an elongated object leaves its longest
    extent unchanged but inflates the short side of an axis-aligned box - the
    hangar measures 189 px long against an expected 186, while its short side
    reads 1.8x too wide purely from orientation.
    """
    expected = CLASS_SIZES.get(object_id)
    if not expected or width <= 0 or height <= 0:
        return 1.0
    longest = max(width, height)
    want = max(expected)
    return max(longest / want, want / longest)


def contained(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of the inner box that lies inside the outer one."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix * iy / area) if area > 0 else 0.0


def suppress_same_class_containment(detections: List[Dict],
                                    threshold: float = 0.70) -> List[Dict]:
    """For nested boxes with the same label, retain the largest box."""
    by_area = sorted(
        detections,
        key=lambda item: -((item['bbox'][2] - item['bbox'][0])
                           * (item['bbox'][3] - item['bbox'][1])))
    kept: List[Dict] = []
    for item in by_area:
        if any(item['object_id'] == other['object_id']
               and contained(item['bbox'], other['bbox']) >= threshold
               for other in kept):
            continue
        kept.append(item)
    return kept


def apply_size_prior(detections: List[Dict]) -> List[Dict]:
    """Damp or drop boxes the wrong size, and boxes swallowed by a peer.

    Two objects of similar real size cannot contain one another, so a
    `medium_launcher` sitting inside a `tank` is one object wearing two labels.
    A `jet_plane` parked inside a `hangar` box is left alone, because a hangar
    really is large enough to contain one.
    """
    sized = []
    for item in detections:
        box = item['bbox']
        ratio = size_agreement(item['object_id'], box[2] - box[0], box[3] - box[1])
        if ratio >= SIZE_REJECT:
            continue
        if ratio > SIZE_TOLERATED:
            span = max(1e-6, SIZE_REJECT - SIZE_TOLERATED)
            item = dict(item, confidence=item['confidence']
                        * (1.0 - (ratio - SIZE_TOLERATED) / span))
        sized.append(item)

    kept = []
    candidates = suppress_same_class_containment(sized) if SAME_CLASS_CONTAINMENT else sized
    for item in sorted(candidates,
                       key=lambda detection: -detection['confidence']):
        box = item['bbox']
        swallowed = False
        for other in kept:
            if other['object_id'] == item['object_id']:
                continue
            if contained(box, other['bbox']) < 0.7:
                continue
            mine = CLASS_SIZES.get(item['object_id'])
            theirs = CLASS_SIZES.get(other['object_id'])
            if not mine or not theirs:
                continue
            if mine[0] * mine[1] >= 0.6 * theirs[0] * theirs[1]:
                swallowed = True
                break
        if not swallowed:
            kept.append(item)
    return kept

# Measured on the recorded competition flight: output grows from 110 to 367
# boxes per frame while Helsinki holds ~10 real objects per frame. These bound
# how long unconfirmed belief survives and how much of it is emitted.
MAX_AGE = int(os.environ.get('DRONE_MAX_AGE', '45'))
MIN_HITS = int(os.environ.get('DRONE_MIN_HITS', '1'))
REPORT_MAX = int(os.environ.get('DRONE_REPORT_MAX', '400'))

# lab/domain_gap.py measured the training scene at saturation 129, brightness
# 88 and median local texture 3570, against 104 / 113 / 1975 on the real
# flight: softer, brighter and flatter than anything the detector was trained
# on. Matching each view to those targets is a no-op on the training scene, so
# it can be regression-tested there. Sharpening is off by default because it
# measurably cost confident detections on the real flight (2.18 -> 1.76 per
# frame) while the photometric match alone was near neutral at 2.07.
DOMAIN_MATCH = int(os.environ.get('DRONE_DOMAIN', '0'))
DOMAIN_SATURATION = float(os.environ.get('DRONE_DOMAIN_SAT', '129.0'))
DOMAIN_VALUE = float(os.environ.get('DRONE_DOMAIN_VAL', '88.0'))
DOMAIN_TEXTURE = float(os.environ.get('DRONE_DOMAIN_TEX', '3570.0'))
DOMAIN_MAX_SHARPEN = float(os.environ.get('DRONE_DOMAIN_SHARPEN', '0.0'))


def match_training_domain(view: np.ndarray) -> np.ndarray:
    """Rescale saturation and brightness, then sharpen toward the training scene."""
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    mean_saturation, mean_value = float(saturation.mean()), float(value.mean())
    if mean_saturation > 1.0:
        hsv[:, :, 1] = np.clip(saturation * (DOMAIN_SATURATION / mean_saturation), 0, 255)
    if mean_value > 1.0:
        hsv[:, :, 2] = np.clip(value * (DOMAIN_VALUE / mean_value), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    energy = cv2.Laplacian(grey, cv2.CV_32F)
    texture = float(np.median(cv2.blur(energy * energy, (17, 17))))
    if texture > 1.0 and texture < DOMAIN_TEXTURE:
        amount = min(DOMAIN_MAX_SHARPEN, np.sqrt(DOMAIN_TEXTURE / texture) - 1.0)
        if amount > 0.01:
            blurred = cv2.GaussianBlur(out, (0, 0), 1.1)
            out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)
    return out

# The scene holds exactly one instance of each class (run_metadata.json), so
# class assignment is a global matching problem, not an independent choice per
# track. 0 disables it.
ASSIGN_CLASSES = int(os.environ.get('DRONE_ASSIGN', '0'))
ASSIGN_BOOST = float(os.environ.get('DRONE_ASSIGN_BOOST', '1.25'))


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def warp_bbox(homography: np.ndarray, bbox: Sequence[float]) -> List[float]:
    """Push a source-pixel box through a homography, as an axis-aligned box."""
    x1, y1, x2, y2 = bbox
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return [float(warped[:, 0].min()), float(warped[:, 1].min()),
            float(warped[:, 0].max()), float(warped[:, 1].max())]


def clip_to_frame(bbox: Sequence[float], minimum_side: float = 2.0):
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(IMAGE_WIDTH, x1)), max(0.0, min(IMAGE_WIDTH, x2))
    y1, y2 = max(0.0, min(IMAGE_HEIGHT, y1)), max(0.0, min(IMAGE_HEIGHT, y2))
    if x2 - x1 < minimum_side or y2 - y1 < minimum_side:
        return None
    return [x1, y1, x2, y2]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_bounds(level: int) -> Tuple[int, int, int, int]:
    width, height = SOURCE_REGION_SIZES[level]
    return width // 2, IMAGE_WIDTH - width // 2, height // 2, IMAGE_HEIGHT - height // 2


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class Detector:
    """YOLO on the transmitted view, answering in source-frame pixels.

    Detections touching the edge of the *view* are dropped, because a box cut
    off by the crop is not the box the evaluator is scoring - the ground truth
    covers the whole object. The exception is an edge that is also an edge of
    the source frame: there the ground truth really is clipped, and an object
    entering at the top of the frame has to be reported clipped too.

    Inference runs on a single dedicated thread that is warmed up when this
    object is constructed. Warming up on the importing thread is not enough:
    the server answers requests on a worker thread, and the first CUDA call on
    a new thread costs about two seconds. At 3 fps that is seven lost frames
    and a request that can blow the 3333 ms budget, on the one attempt that
    counts. Pinning inference to one warm thread also serialises access to the
    single GPU.
    """

    # How close to the view edge (in transmitted pixels) counts as touching.
    EDGE_TOLERANCE = 3.0

    def __init__(self, weights: str = WEIGHTS, imgsz: int = INFER_IMGSZ,
                 confidence: float = DETECT_CONF, device: str = DEVICE,
                 drop_view_edge: bool = True):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = int(device) if str(device).isdigit() else device
        self.drop_view_edge = drop_view_edge
        self.names = self.model.names
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix='drone-detector')
        self._pool.submit(self.warm_up).result()

    def warm_up(self) -> None:
        """The first inference is the slowest and the attempt does not wait.

        A blank frame yields no detections, so it never touches NMS with real
        boxes, the tensor copy back to host, or the work that follows - and
        frame 2 was dropped on 7 of 12 recorded runs because the first real
        view paid for all of it. Warm up on imagery that actually detects
        something, and drive the whole path.

        This already runs on the inference thread, so it calls the stages
        directly; going through ``__call__`` would queue behind itself on the
        single-worker pool and deadlock.
        """
        for view, region in self._warm_up_views():
            for _ in range(2):
                self._to_detections(self._infer(view), view, region)

    def _warm_up_views(self):
        """Real imagery if the supplied scene is present, else a blank frame."""
        views = []
        try:
            from utils import frame_numbers, load_frame
            frames = frame_numbers()
            for number in (frames[0], frames[len(frames) // 2]):
                source = load_frame(number)
                views.append((cv2.resize(source, (VIEW_WIDTH, VIEW_HEIGHT),
                                         interpolation=cv2.INTER_AREA),
                              (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)))
        except Exception:
            pass
        if not views:
            views = [(np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), np.uint8),
                      (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT))]
        return views

    def _infer(self, view: np.ndarray):
        if DOMAIN_MATCH:
            view = match_training_domain(view)
        return self.model.predict(
            view, imgsz=self.imgsz, conf=self.confidence, iou=0.55,
            agnostic_nms=AGNOSTIC_NMS,
            device=self.device, half=True, verbose=False, max_det=120)[0]

    def __call__(self, view: np.ndarray, region: Sequence[int]):
        result = self._pool.submit(self._infer, view).result()
        return self._to_detections(result, view, region)

    def _to_detections(self, result, view: np.ndarray, region: Sequence[int]):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        sx1, sy1, sx2, sy2 = region
        height, width = view.shape[:2]
        fx = (sx2 - sx1) / width
        fy = (sy2 - sy1) / height
        tolerance = self.EDGE_TOLERANCE
        # Which view edges are real frame edges, where clipping is legitimate.
        frame_edge = (sx1 <= 0, sy1 <= 0, sx2 >= IMAGE_WIDTH, sy2 >= IMAGE_HEIGHT)

        out = []
        for (x1, y1, x2, y2), score, index in zip(xyxy, scores, classes):
            if self.drop_view_edge and (
                    (x1 <= tolerance and not frame_edge[0]) or
                    (y1 <= tolerance and not frame_edge[1]) or
                    (x2 >= width - tolerance and not frame_edge[2]) or
                    (y2 >= height - tolerance and not frame_edge[3])):
                continue
            out.append({
                'object_id': self.names[int(index)],
                'bbox': [sx1 + x1 * fx, sy1 + y1 * fy,
                         sx1 + x2 * fx, sy1 + y2 * fy],
                'confidence': float(score),
            })
        if SIZE_PRIOR:
            out = apply_size_prior(out)
        return out


# A second detector trained on a different distribution (the team's policy
# model: Helsinki sprites over Inria orthophotos). Its boxes are pooled with
# the primary detector's before tracking; the world model merges duplicate
# tracks and pools their class votes. Off unless DRONE_ENSEMBLE_WEIGHTS is set.
ENSEMBLE_WEIGHTS = os.environ.get('DRONE_ENSEMBLE_WEIGHTS', '')
ENSEMBLE_SCALE = float(os.environ.get('DRONE_ENSEMBLE_SCALE', '1.0'))
ENSEMBLE_CONF = float(os.environ.get('DRONE_ENSEMBLE_CONF', str(DETECT_CONF)))


class EnsembleDetector:
    """Run two detectors on the same view and concatenate their boxes."""

    def __init__(self, primary, weights: str, scale: float = 1.0,
                 confidence: float = DETECT_CONF):
        self.primary = primary
        self.secondary = Detector(weights=weights, confidence=confidence,
                                  device=DEVICE, drop_view_edge=True)
        self.scale = scale
        self.names = primary.names

    def __call__(self, view, region):
        out = list(self.primary(view, region))
        for item in self.secondary(view, region):
            out.append(dict(item, confidence=min(1.0, item['confidence'] * self.scale)))
        return out


# --------------------------------------------------------------------------- #
# World model
# --------------------------------------------------------------------------- #

class Track:
    __slots__ = ('object_id', 'bbox', 'score', 'votes', 'hits', 'misses',
                 'age_since_seen')

    def __init__(self, object_id: str, bbox: List[float], score: float):
        self.object_id = object_id
        self.bbox = bbox
        self.score = score
        self.votes: Dict[str, float] = {object_id: score}
        self.hits = 1
        self.misses = 0
        self.age_since_seen = 0

    def vote(self, object_id: str, score: float) -> None:
        """Accumulate class evidence rather than trusting the latest frame.

        The same object seen twenty times should not flip class because one
        downsampled view was ambiguous.
        """
        self.votes[object_id] = self.votes.get(object_id, 0.0) + score
        self.object_id = max(self.votes, key=self.votes.get)


class WorldModel:
    """Objects persist, and are carried forward by the flight homography."""

    # Association gate. Detections are matched to tracks by IoU first, then by
    # centre distance relative to object size, which rescues the small classes
    # where a few pixels of drift destroys IoU.
    MATCH_IOU = 0.30
    MATCH_DISTANCE = 1.20          # in units of mean box side

    # A track the camera is looking at but does not detect is probably gone.
    # Three rather than one: on the supplied scene one scores marginally
    # better, but the difference is inside the noise of 16 instances, and a
    # tolerant setting is the safer one if the detector blinks more often on an
    # unfamiliar scene.
    MAX_MISSES_IN_VIEW = int(os.environ.get('DRONE_MAX_MISSES', '3'))
    # A track the camera has not revisited is kept much longer: the homography
    # is accurate enough to carry it, and recall is what the metric rewards.
    MAX_AGE_UNSEEN = MAX_AGE

    def __init__(self, homography: np.ndarray = DEFAULT_HOMOGRAPHY):
        self.homography = homography
        self.tracks: List[Track] = []
        self.last_frame: Optional[int] = None

    # -- propagation ------------------------------------------------------- #

    def advance(self, frame: int) -> None:
        if self.last_frame is None:
            self.last_frame = frame
            return
        steps = frame - self.last_frame
        self.last_frame = frame
        if steps < 0:
            # The frame number went backwards, so this is a different attempt
            # reusing the sequence id. Nothing carried over is valid.
            self.tracks = []
            return
        if steps == 0:
            return
        # Frames can be skipped when the server is slow, so chain the matrix by
        # however many source frames actually elapsed.
        homography = (self.homography if steps == 1
                      else np.linalg.matrix_power(self.homography, steps))
        surviving = []
        for track in self.tracks:
            track.bbox = warp_bbox(homography, track.bbox)
            track.age_since_seen += steps
            if track.age_since_seen > self.MAX_AGE_UNSEEN:
                continue
            if clip_to_frame(track.bbox) is None:
                continue
            surviving.append(track)
        self.tracks = surviving

    # -- association ------------------------------------------------------- #

    def update(self, detections: List[dict], region: Sequence[int]) -> None:
        existing = len(self.tracks)
        used = set()
        pairs = []
        for d_index, detection in enumerate(detections):
            for t_index, track in enumerate(self.tracks):
                overlap = iou(detection['bbox'], track.bbox)
                if overlap >= self.MATCH_IOU:
                    pairs.append((overlap, d_index, t_index))
                    continue
                # Fallback: close centres and a compatible size.
                dcx = (detection['bbox'][0] + detection['bbox'][2]) / 2
                dcy = (detection['bbox'][1] + detection['bbox'][3]) / 2
                tcx = (track.bbox[0] + track.bbox[2]) / 2
                tcy = (track.bbox[1] + track.bbox[3]) / 2
                size = max(8.0, 0.25 * (track.bbox[2] - track.bbox[0] +
                                        track.bbox[3] - track.bbox[1] +
                                        detection['bbox'][2] - detection['bbox'][0] +
                                        detection['bbox'][3] - detection['bbox'][1]))
                distance = math.hypot(dcx - tcx, dcy - tcy)
                if distance < self.MATCH_DISTANCE * size:
                    pairs.append((0.29 * (1 - distance / (self.MATCH_DISTANCE * size)),
                                  d_index, t_index))

        pairs.sort(reverse=True)
        matched_tracks = set()
        for _, d_index, t_index in pairs:
            if d_index in used or t_index in matched_tracks:
                continue
            used.add(d_index)
            matched_tracks.add(t_index)
            track = self.tracks[t_index]
            detection = detections[d_index]
            track.bbox = detection['bbox']
            track.vote(detection['object_id'], detection['confidence'])
            track.score = max(track.score * 0.6, detection['confidence'])
            track.hits += 1
            track.misses = 0
            track.age_since_seen = 0

        # Tracks the camera was pointed at and still did not see. This has to
        # run before the new tracks are appended, or every brand-new detection
        # is immediately charged with a miss.
        for t_index in range(existing):
            track = self.tracks[t_index]
            if t_index in matched_tracks:
                continue
            if not self._mostly_inside(track.bbox, region):
                continue
            track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses < self.MAX_MISSES_IN_VIEW]

        for d_index, detection in enumerate(detections):
            if d_index in used:
                continue
            self.tracks.append(Track(detection['object_id'],
                                     list(detection['bbox']),
                                     detection['confidence']))
        self._merge_overlapping()

    def _merge_overlapping(self, threshold: float = 0.60) -> None:
        """Fuse tracks sitting on the same object, whatever they call it.

        Association is per class-agnostic geometry but nothing stops two tracks
        forming on one object, and report-time deduplication only removes
        duplicates *within* a class. Two survivors with different labels then
        compete, which is how one object ends up named inconsistently from
        frame to frame. Merging pools their class evidence instead.
        """
        merged: List[Track] = []
        for track in sorted(self.tracks, key=lambda t: -t.hits):
            for keeper in merged:
                if iou(track.bbox, keeper.bbox) < threshold:
                    continue
                for name, weight in track.votes.items():
                    keeper.votes[name] = keeper.votes.get(name, 0.0) + weight
                keeper.object_id = max(keeper.votes, key=keeper.votes.get)
                keeper.hits += track.hits
                keeper.score = max(keeper.score, track.score)
                keeper.age_since_seen = min(keeper.age_since_seen,
                                            track.age_since_seen)
                break
            else:
                merged.append(track)
        self.tracks = merged

    @staticmethod
    def _mostly_inside(bbox, region, minimum: float = 0.7) -> bool:
        x1, y1, x2, y2 = region
        ix1, iy1 = max(bbox[0], x1), max(bbox[1], y1)
        ix2, iy2 = min(bbox[2], x2), min(bbox[3], y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        return inter / area >= minimum

    # -- output ------------------------------------------------------------ #

    def assign_classes(self) -> Dict[int, str]:
        """One class per track, exploiting one-instance-per-class.

        Taking each track's argmax independently lets two tracks claim the same
        class, which is the failure measured on the evaluation scene: the
        helicopter and the medium_plane were both called medium_plane, so one
        class scored zero and the other was polluted. A global assignment
        forbids that. Returns track index -> class name.
        """
        if not self.tracks:
            return {}
        from dtos import OBJECT_CLASSES
        from scipy.optimize import linear_sum_assignment

        scores = np.zeros((len(OBJECT_CLASSES), len(self.tracks)))
        for column, track in enumerate(self.tracks):
            total = sum(track.votes.values()) or 1.0
            evidence = min(1.0, track.hits / 4.0)
            for row, name in enumerate(OBJECT_CLASSES):
                scores[row, column] = track.votes.get(name, 0.0) / total * evidence
        rows, columns = linear_sum_assignment(-scores)
        return {int(c): OBJECT_CLASSES[int(r)]
                for r, c in zip(rows, columns) if scores[r, c] > 0}

    def report(self, maximum: int = REPORT_MAX) -> List[dict]:
        """Current belief about the whole frame, deduplicated and ranked.

        Confidence decays with how long it has been since the object was last
        observed, because the metric ranks every prediction against every other
        one: a fresh detection has to outrank a thirty-frame-old extrapolation.
        """
        out = []
        assignment = self.assign_classes() if ASSIGN_CLASSES else {}
        for index, track in enumerate(self.tracks):
            bbox = clip_to_frame(track.bbox)
            if bbox is None:
                continue
            if track.hits < MIN_HITS:
                continue
            staleness = 1.0 / (1.0 + 0.045 * track.age_since_seen)
            support = min(1.0, 0.55 + 0.15 * track.hits)
            confidence = float(np.clip(track.score * staleness * support, 1e-4, 1.0))
            out.append({'object_id': track.object_id, 'bbox': bbox,
                        'confidence': confidence})

            # The globally assigned class leads its class's ranking, which is
            # what AP rewards; the per-track argmax stays as a fallback.
            assigned = assignment.get(index)
            if assigned is not None and assigned != track.object_id:
                out.append({'object_id': assigned, 'bbox': bbox,
                            'confidence': float(np.clip(confidence * ASSIGN_BOOST,
                                                        1e-4, 1.0))})

            # The score is a mean over the classes present, so a class we never
            # label right is a flat zero for 1/16th of the total, while a
            # low-confidence wrong box ranks below the true positives and costs
            # almost nothing. When the vote is close, name the runners-up too.
            if ALTERNATE_CLASSES > 0 and len(track.votes) > 1:
                ranked = sorted(track.votes.items(), key=lambda kv: -kv[1])
                leader = ranked[0][1]
                for object_id, weight in ranked[1:1 + ALTERNATE_CLASSES]:
                    share = weight / leader if leader > 0 else 0.0
                    alternate = confidence * share * ALTERNATE_DAMPING
                    if alternate < 1e-4:
                        continue
                    out.append({'object_id': object_id, 'bbox': bbox,
                                'confidence': float(min(alternate,
                                                        confidence * 0.99))})

        if SAME_CLASS_CONTAINMENT:
            out = suppress_same_class_containment(out)
        out.sort(key=lambda d: -d['confidence'])
        # The scorer applies no NMS of its own: a duplicate is a false positive.
        kept: List[dict] = []
        for candidate in out:
            if any(candidate['object_id'] == other['object_id'] and
                   iou(candidate['bbox'], other['bbox']) > 0.55 for other in kept):
                continue
            kept.append(candidate)
            if len(kept) >= maximum:
                break
        return kept


# --------------------------------------------------------------------------- #
# Online motion estimation
# --------------------------------------------------------------------------- #

class MotionEstimator:
    """Correct the inter-frame homography from the views as they arrive.

    The matrix in ``DEFAULT_HOMOGRAPHY`` was measured on the supplied scene.
    The evaluation sequence is a different location, and while the capture
    settings look identical (600 m, 3 fps, straight line), betting the whole
    world model on that is unnecessary: consecutive views overlap, so the real
    ground motion can be measured.

    What is measured is a **residual translation** on top of the prior, not a
    new homography. Two consecutive views overlap in only part of the frame,
    and a full eight-parameter homography fitted to a narrow strip extrapolates
    badly across the rest of it - which is worse than no correction at all,
    because a bad matrix drags every track off its object at once. Two
    parameters fitted to the same strip extrapolate safely, and translation is
    where a difference in ground speed or heading would actually show up.
    """

    MINIMUM_MATCHES = 40
    # Reject a residual larger than this: it is a mismatch, not a real drift.
    MAXIMUM_RESIDUAL = 50.0
    BLEND = 0.30
    # ORB runs on a half-size copy of the view. The residual is a single
    # translation estimated from hundreds of matches, so half the keypoint
    # precision costs nothing measurable and the detection is four times
    # cheaper - which matters, because this runs inside the frame budget.
    DOWNSCALE = 2

    def __init__(self, prior: np.ndarray = DEFAULT_HOMOGRAPHY, enabled: bool = True):
        self.prior = prior.copy()
        self.current = prior.copy()
        self.enabled = enabled
        self.offset = np.zeros(2, np.float64)
        self._previous = None          # (frame, grey view, region)
        self._detector = None
        self._matcher = None
        self.accepted = 0
        self.rejected = 0

    def _orb(self):
        if self._detector is None:
            self._detector = cv2.ORB_create(nfeatures=700)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        return self._detector, self._matcher

    @staticmethod
    def _to_source(points: np.ndarray, region, shape) -> np.ndarray:
        x1, y1, x2, y2 = region
        fx = (x2 - x1) / shape[1]
        fy = (y2 - y1) / shape[0]
        return np.stack([x1 + points[:, 0] * fx, y1 + points[:, 1] * fy], 1)

    def update(self, frame: int, view: np.ndarray, region: Sequence[int]) -> np.ndarray:
        """Return the homography to use for this frame's propagation step."""
        if not self.enabled:
            return self.current
        grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        if self.DOWNSCALE > 1:
            grey = cv2.resize(grey, None, fx=1 / self.DOWNSCALE,
                              fy=1 / self.DOWNSCALE, interpolation=cv2.INTER_AREA)
        previous, self._previous = self._previous, (frame, grey, tuple(region))
        # A new attempt can reuse the sequence id; a frame number that went
        # backwards means nothing carried over is valid.
        if previous is not None and frame < previous[0]:
            self.offset = np.zeros(2, np.float64)
            self.current = self.prior.copy()
            return self.current
        # Only a single-frame gap gives a per-frame motion directly.
        if previous is None or frame - previous[0] != 1:
            return self.current
        try:
            residual = self._residual(previous, (frame, grey, tuple(region)))
        except cv2.error:
            residual = None
        if residual is None:
            self.rejected += 1
            return self.current
        self.accepted += 1
        self.offset = (1 - self.BLEND) * self.offset + self.BLEND * residual
        self.current = self.prior.copy()
        self.current[0, 2] += self.offset[0]
        self.current[1, 2] += self.offset[1]
        return self.current

    def _residual(self, previous, current):
        """Median leftover translation after applying the prior homography."""
        orb, matcher = self._orb()
        _, grey_a, region_a = previous
        _, grey_b, region_b = current
        kp_a, des_a = orb.detectAndCompute(grey_a, None)
        kp_b, des_b = orb.detectAndCompute(grey_b, None)
        if des_a is None or des_b is None or len(kp_a) < 40 or len(kp_b) < 40:
            return None
        raw = matcher.knnMatch(des_a, des_b, k=2)
        good = [pair[0] for pair in raw
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < self.MINIMUM_MATCHES:
            return None
        src = self._to_source(np.float32([kp_a[m.queryIdx].pt for m in good]),
                              region_a, grey_a.shape)
        dst = self._to_source(np.float32([kp_b[m.trainIdx].pt for m in good]),
                              region_b, grey_b.shape)
        predicted = cv2.perspectiveTransform(
            src.astype(np.float64).reshape(-1, 1, 2), self.current).reshape(-1, 2)
        residuals = dst - predicted
        # The median rejects the mismatched pairs; the MAD says whether the
        # agreement is good enough to believe at all.
        centre = np.median(residuals, axis=0)
        spread = float(np.median(np.abs(residuals - centre)))
        if spread > 4.0 or float(np.hypot(*centre)) > self.MAXIMUM_RESIDUAL:
            return None
        # Express the correction relative to the prior, not to the last estimate.
        return centre + self.offset


# --------------------------------------------------------------------------- #
# Camera policy
# --------------------------------------------------------------------------- #

class CameraPolicy:
    """Walk a fixed loop of camera positions, one legal hop per frame.

    A loop entry is ``(resolution_level, center_x, center_y)``. The policy only
    ever asks for something the rules allow: it changes level one step at a
    time, clamps the centre into that level's bounds, and shortens any hop that
    would exceed the movement limit, walking toward a far target over several
    frames instead.

    The default loop alternates one Level-0 frame with one Level-1 quadrant.
    Level 0 answers for the whole frame and measures best on the supplied
    scene, but that scene is the one the training sprites were cut from, so its
    Level-0 accuracy is optimistic. Measured with a model held out from half
    the terrain - the regime the evaluation actually puts us in - this even
    alternation scores 0.888 against 0.862 for three Level-0 frames to one and
    0.848 for Level 0 alone. Objects are about 13 px across at Level 0 and the
    class has to be read from those pixels, so the 2x linear resolution buys
    accuracy that the world model then carries across the frames spent
    elsewhere. Level 0 is reachable from Level 1 in one move and is exempt from
    the distance limit, so the alternation costs no extra frames.
    """

    FULL = (0, IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
    QUADRANTS = ((1, 960, 540), (1, 2880, 540), (1, 2880, 1620), (1, 960, 1620))
    LOOP = (
        FULL, (1, 960, 540),
        FULL, (1, 2880, 540),
        FULL, (1, 2880, 1620),
        FULL, (1, 960, 1620),
    )
    # Competition validation settles this against the local experiments.
    # Holding Level 0 every frame scored 0.1157 against 0.153 for the even
    # alternation, so Level-1 resolution is worth real score even though
    # exp11 and exp12 found no Level-1 benefit on the supplied scene. Both of
    # those were measured in-domain on a scene the detector was trained from.
    # The gradient runs toward more resolution, so these spend more of the
    # flight at Level 1. Ground drifts about 66 px per frame downward, so new
    # terrain arrives at the top: 'l1top' pans the top band where objects
    # first appear, giving the longest possible track life per object.
    _MODES = {
        'l0': (FULL,),
        'l1': QUADRANTS,
        'l1top': ((1, 960, 540), (1, 2880, 540)),
        'l1heavy': (FULL,) + QUADRANTS,
        # Data-collection only: raster the whole frame at Level 2 (native
        # resolution) so a recorded run captures real L2 pixels of every
        # object. Whole-frame score is poor (L2 sees 1/16 of the frame), which
        # is expected - this mode exists to gather real L2 training crops, not
        # to score. 16 tiles tiling L2 bounds x[480,3360] y[270,1890].
        'l2raster': tuple((2, cx, cy) for cy in (270, 810, 1350, 1890)
                          for cx in (480, 1440, 2400, 3360)),
        # Also data-collection only, and strictly better at it: exp20 measures
        # 87.2% of the flight's ground observed against 46.2% for l2raster. A
        # 16-tile raster cannot cycle before the ground scrolls away, so it
        # samples the frame close to randomly. Every object instead enters
        # through the top, so sweeping the top band back and forth catches each
        # one on arrival - the section 2 top-band argument, at Level 2.
        'l2top': tuple((2, cx, 270) for cx in (480, 1440, 2400, 3360, 2400, 1440)),
    }
    LOOP = _MODES.get(os.environ.get('DRONE_CAMERA', 'loop'), LOOP)

    def __init__(self, loop=None):
        self.loop = tuple(loop) if loop is not None else self.LOOP
        self.index = 0

    def __call__(self, level: int, center_x: int, center_y: int,
                 allowed_levels: Sequence[int], maximum_delta: float):
        target_level, target_x, target_y = self.loop[self.index]

        if self._arrived(level, center_x, center_y, target_level, target_x, target_y):
            self.index = (self.index + 1) % len(self.loop)
            target_level, target_x, target_y = self.loop[self.index]

        # Level changes are one step at a time; L0 and L2 cannot reach each
        # other directly.
        step = target_level
        if abs(target_level - level) > 1:
            step = level + (1 if target_level > level else -1)
        if step not in allowed_levels:
            return None
        if step == 0:
            # The full view has exactly one legal centre, and getting there is
            # exempt from the movement limit.
            return self.FULL
        return self._move_toward(step, center_x, center_y,
                                 (target_x, target_y), maximum_delta)

    @staticmethod
    def _arrived(level, center_x, center_y, target_level, target_x, target_y) -> bool:
        return (level == target_level and
                abs(center_x - target_x) <= 4 and abs(center_y - target_y) <= 4)

    @staticmethod
    def _move_toward(level, center_x, center_y, target, maximum_delta):
        min_x, max_x, min_y, max_y = center_bounds(level)
        tx = min(max(float(target[0]), min_x), max_x)
        ty = min(max(float(target[1]), min_y), max_y)
        dx, dy = tx - center_x, ty - center_y
        distance = math.hypot(dx, dy)
        limit = max(1.0, maximum_delta * 0.97)
        if distance > limit:
            tx = center_x + dx / distance * limit
            ty = center_y + dy / distance * limit
            tx = min(max(tx, min_x), max_x)
            ty = min(max(ty, min_y), max_y)
        return (int(level), int(round(tx)), int(round(ty)))


class AttentionPolicy(CameraPolicy):
    """Alternate the full view with a Level-1 look at the least certain track.

    The fixed loop sweeps quadrants blind. But the score is a mean over the
    classes present, so the thing worth spending a zoom frame on is whichever
    object we cannot yet name - not whichever quadrant is next in line. This
    picks the in-frame track with the closest vote between its leading two
    classes, and falls back to the fixed raster when nothing is contested.
    """

    # A track is worth revisiting until it has been seen this many times.
    SETTLED_HITS = 6
    # Hold a chosen target this long before picking another.
    LOCK_FRAMES = 6

    def __init__(self, loop=None):
        super().__init__(loop)
        self.tick = 0
        self.locked: Optional[tuple] = None
        self.locked_for = 0

    def __call__(self, level, center_x, center_y, allowed_levels,
                 maximum_delta, world=None):
        self.tick += 1
        if world is None or self.tick % 2 == 1:
            return super().__call__(level, center_x, center_y,
                                    allowed_levels, maximum_delta)

        # Re-choosing every zoom frame leaves the camera permanently in
        # transit, so hold a target until it is reached or goes stale.
        self.locked_for += 1
        if self.locked is not None and self.locked_for <= self.LOCK_FRAMES:
            arrived = (level == 1 and
                       abs(center_x - self.locked[0]) <= 8 and
                       abs(center_y - self.locked[1]) <= 8)
            if arrived:
                self.locked = None
        else:
            self.locked = None

        if self.locked is None:
            self.locked = self._most_contested(world)
            self.locked_for = 0
        target = self.locked
        if target is None:
            return super().__call__(level, center_x, center_y,
                                    allowed_levels, maximum_delta)

        step = 1
        if abs(step - level) > 1:
            step = level + (1 if step > level else -1)
        if step not in allowed_levels:
            return super().__call__(level, center_x, center_y,
                                    allowed_levels, maximum_delta)
        return self._move_toward(step, center_x, center_y, target, maximum_delta)

    def _most_contested(self, world):
        best = None
        best_score = -1.0
        for track in getattr(world, 'tracks', ()):
            if track.hits >= self.SETTLED_HITS and len(track.votes) <= 1:
                continue
            ranked = sorted(track.votes.values(), reverse=True)
            leader = ranked[0] if ranked else 0.0
            runner_up = ranked[1] if len(ranked) > 1 else 0.0
            # Contested means the runner-up is close; unseen means few hits.
            contest = (runner_up / leader) if leader > 0 else 0.0
            scarcity = 1.0 / (1.0 + track.hits)
            score = contest + scarcity
            if score > best_score:
                best_score = score
                best = track
        if best is None:
            return None
        return ((best.bbox[0] + best.bbox[2]) / 2.0,
                (best.bbox[1] + best.bbox[3]) / 2.0)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

class Solver:
    """One attempt: detector, and one world model / policy / motion estimator
    per sequence."""

    def __init__(self, detector=None, policy_factory=CameraPolicy,
                 homography: np.ndarray = DEFAULT_HOMOGRAPHY,
                 world_factory=WorldModel, online_motion: bool = True):
        self.detector = detector if detector is not None else Detector()
        if ENSEMBLE_WEIGHTS:
            self.detector = EnsembleDetector(self.detector, ENSEMBLE_WEIGHTS,
                                             ENSEMBLE_SCALE, ENSEMBLE_CONF)
        self.policy_factory = policy_factory
        self.world_factory = world_factory
        self.homography = homography
        self.online_motion = online_motion
        self.states: Dict[str, tuple] = {}

    def state_for(self, sequence_id: str):
        if sequence_id not in self.states:
            # One sequence at a time, so an unseen id means a new attempt.
            self.states = {sequence_id: (
                self.world_factory(self.homography),
                self.policy_factory(),
                MotionEstimator(self.homography, enabled=self.online_motion),
            )}
        return self.states[sequence_id]

    def __call__(self, sequence_id: str, frame: int, view: np.ndarray,
                 region: Sequence[int], level: int, center_x: int, center_y: int,
                 allowed_levels: Sequence[int], maximum_delta: float):
        world, policy, motion = self.state_for(sequence_id)
        try:
            world.homography = motion.update(frame, view, region)
        except Exception:
            logger.exception('Motion estimation failed on frame %s', frame)
        world.advance(frame)
        try:
            detections = self.detector(view, region)
        except Exception:
            logger.exception('Detector failed on frame %s', frame)
            detections = []
        world.update(detections, region)
        annotations = world.report()
        try:
            requested = policy(level, center_x, center_y, allowed_levels,
                               maximum_delta, world=world)
        except TypeError:
            requested = policy(level, center_x, center_y, allowed_levels,
                               maximum_delta)
        return annotations, requested
