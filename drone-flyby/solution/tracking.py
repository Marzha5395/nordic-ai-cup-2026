"""The world model: what is on the ground, and where it is right now.

A response is scored against every object in the source frame, not only the
ones the camera happens to be pointed at, and the camera can only look at a
quarter of the frame at a time. So the only way to answer a frame properly is
to remember: detect an object once, then carry it, in source coordinates, for
as long as it stays in the frame.

Carrying it is the part that has to be right. A 32 pixel object needs its
centre within about 6 pixels to still clear IoU 0.50, and the ground drifts
about 65 pixels a frame, so a track is moved by the affine flow field from
``solution.motion`` -- which is accurate to about a pixel -- plus a small
per-track bias for what the field cannot know: how tall the object is and how
high the ground under it sits.
"""

import math
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np

from . import config
from .detector import Detection

NUMBER_OF_CLASSES = 16


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _centre(box: Sequence[float]) -> Tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


class Prediction(NamedTuple):
    class_index: int
    box: Tuple[float, float, float, float]   # source pixels
    confidence: float


class Track:
    """One object, remembered between sightings."""

    __slots__ = (
        'track_id', 'x1', 'y1', 'x2', 'y2', 'bias_x', 'bias_y', 'class_scores',
        'scores', 'observations', 'last_seen_frame', 'created_frame',
        'misses', 'looks',
    )

    def __init__(self, track_id, detection: Detection, frame: int):
        self.track_id = track_id
        self.x1, self.y1, self.x2, self.y2 = detection.box
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.class_scores = np.zeros(NUMBER_OF_CLASSES, dtype=np.float64)
        self.class_scores[detection.class_index] = detection.score
        # The individual detection scores, best first. Kept rather than
        # combined, because how they are combined turns out to matter: a
        # noisy-or over ten sightings at 0.2 comes out at 0.89, which is how a
        # rock that reads faintly as a tank every time the camera passes ends up
        # ranked above real objects, and mAP is decided by that ranking.
        self.scores = [detection.score]
        self.observations = 1
        self.last_seen_frame = frame
        self.created_frame = frame
        self.misses = 0
        # Every time this track was inside the view and should have been seen,
        # whether or not it was. A real object is found on nearly every look; a
        # patch of terrain that sometimes reads as an object is not.
        self.looks = 1

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def centre(self) -> Tuple[float, float]:
        return 0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2)

    @property
    def size(self) -> Tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def class_index(self) -> int:
        return int(np.argmax(self.class_scores))

    def advance(self, frames: int, flow) -> None:
        # Position only. The flow field's gradient looks like an expansion, and
        # objects in the reference scene do grow as they cross the frame, but
        # only at about half the rate the gradient implies -- so applying it to
        # the box makes things worse rather than better. The size follows the
        # detector instead, in ``correct``.
        drift_x, drift_y = flow.displacement(*self.centre)
        shift_x = (drift_x + self.bias_x) * frames
        shift_y = (drift_y + self.bias_y) * frames
        self.x1 += shift_x
        self.x2 += shift_x
        self.y1 += shift_y
        self.y2 += shift_y

    def correct(self, detection: Detection, frame: int) -> None:
        """Fold one sighting into the track."""
        elapsed = max(1, frame - self.last_seen_frame)
        predicted_x, predicted_y = self.centre
        observed_x, observed_y = _centre(detection.box)
        residual_x = observed_x - predicted_x
        residual_y = observed_y - predicted_y

        centre_x = predicted_x + config.TRACK_POSITION_GAIN * residual_x
        centre_y = predicted_y + config.TRACK_POSITION_GAIN * residual_y

        # Whatever the flow field could not account for, attributed to this
        # object and kept small: the field is right to about a pixel, so a
        # large bias is a bad match rather than a tall building.
        limit = config.TRACK_MAXIMUM_BIAS
        self.bias_x += config.TRACK_BIAS_GAIN * residual_x / elapsed
        self.bias_y += config.TRACK_BIAS_GAIN * residual_y / elapsed
        self.bias_x = min(max(self.bias_x, -limit), limit)
        self.bias_y = min(max(self.bias_y, -limit), limit)

        width, height = self.size
        observed_width = detection.box[2] - detection.box[0]
        observed_height = detection.box[3] - detection.box[1]
        blend = config.SIZE_SMOOTHING
        width = (1.0 - blend) * width + blend * observed_width
        height = (1.0 - blend) * height + blend * observed_height

        self.x1 = centre_x - width / 2.0
        self.x2 = centre_x + width / 2.0
        self.y1 = centre_y - height / 2.0
        self.y2 = centre_y + height / 2.0

        self.class_scores[detection.class_index] += detection.score
        self.scores.append(detection.score)
        if len(self.scores) > config.SCORE_HISTORY:
            self.scores.sort(reverse=True)
            del self.scores[config.SCORE_HISTORY:]
        self.observations += 1
        self.last_seen_frame = frame
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        return min(1.0, self.observations / max(1, self.looks))

    def confidence(self, frame: int) -> float:
        """How likely this track is to be a real object, for ranking.

        mAP is decided entirely by the order predictions are considered in, so
        this has to separate the three ways a track can be wrong: it was never
        a strong detection, it was not the same thing twice, or it has not been
        looked at recently enough to still be where it says it is.
        """
        best = sorted(self.scores, reverse=True)[:config.STRENGTH_SAMPLES]
        strength = sum(best) / len(best)

        total = float(self.class_scores.sum())
        purity = float(self.class_scores.max()) / total if total > 0 else 0.0

        floor = config.SUPPORT_FLOOR
        support = floor + (1.0 - floor) * self.hit_rate

        staleness = max(0, frame - self.last_seen_frame)
        decay = math.exp(-config.CONFIDENCE_STALENESS_DECAY * staleness)

        return float(min(0.999, max(0.0, strength * purity * support * decay)))


class World:
    """Every track in the current source frame, and the drift that moves them."""

    def __init__(self, frame_width: int = 3840, frame_height: int = 2160):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracks: List[Track] = []
        self._next_id = 1

    # ------------------------------------------------------------------ #

    def advance(self, frames: int, flow) -> None:
        if frames <= 0:
            return
        for track in self.tracks:
            track.advance(frames, flow)
        self._drop_departed()

    def _drop_departed(self) -> None:
        """Forget tracks that have left the frame, or gone non-finite.

        Nothing should ever produce a non-finite box, but a track that did
        would stay in the world for the rest of the attempt and poison every
        response it appeared in, so it is checked here rather than trusted.
        """
        kept = []
        for track in self.tracks:
            if not all(math.isfinite(c) for c in track.box):
                continue
            if track.x2 <= 0.0 or track.y2 <= 0.0:
                continue
            if track.x1 >= self.frame_width or track.y1 >= self.frame_height:
                continue
            kept.append(track)
        self.tracks = kept

    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: Sequence[Detection],
        frame: int,
        view_region: Sequence[float],
    ) -> None:
        """Fold this frame's detections into the world."""
        # Anything that touches the view may be matched, so that a track on
        # its way out of the crop is updated instead of duplicated. Only a
        # track that is properly inside is expected to be seen, so only those
        # are charged with a miss when they are not.
        reachable = [t for t in self.tracks if self._inside(t.box, view_region, 0.05)]
        expected = [t for t in reachable if self._inside(t.box, view_region, 0.55)]

        candidates = []
        for detection_index, detection in enumerate(detections):
            detection_centre = _centre(detection.box)
            detection_size = max(
                detection.box[2] - detection.box[0], detection.box[3] - detection.box[1]
            )
            for track in reachable:
                overlap = _iou(detection.box, track.box)
                track_size = max(track.size)
                distance = math.hypot(
                    detection_centre[0] - track.centre[0],
                    detection_centre[1] - track.centre[1],
                )
                gate = config.MATCH_DISTANCE_RATIO * 0.5 * (detection_size + track_size)
                if overlap < config.MATCH_IOU and distance > gate:
                    continue
                affinity = overlap + (1.0 - min(1.0, distance / max(1.0, gate))) * 0.5
                if track.class_index == detection.class_index:
                    affinity += config.SAME_CLASS_MATCH_BONUS
                candidates.append((affinity, detection_index, track))

        candidates.sort(key=lambda item: -item[0])
        used_detections = set()
        used_tracks = set()
        for _, detection_index, track in candidates:
            if detection_index in used_detections or id(track) in used_tracks:
                continue
            used_detections.add(detection_index)
            used_tracks.add(id(track))
            track.correct(detections[detection_index], frame)

        # A look is any frame in which the track was either expected or found:
        # a track matched while only partly inside the view was still seen.
        expected_ids = {id(track) for track in expected}
        for track in reachable:
            if id(track) in expected_ids or id(track) in used_tracks:
                track.looks += 1
            if id(track) in expected_ids and id(track) not in used_tracks:
                track.misses += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            self.tracks.append(Track(self._next_id, detection, frame))
            self._next_id += 1

        self._retire()
        self._merge_duplicates()

    def _inside(self, box, region, required: float) -> bool:
        """True when enough of a box lies in the region to expect a sighting."""
        left = max(box[0], region[0])
        top = max(box[1], region[1])
        right = min(box[2], region[2])
        bottom = min(box[3], region[3])
        if right <= left or bottom <= top:
            return False
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        return ((right - left) * (bottom - top)) / area >= required

    def _retire(self) -> None:
        """Drop tracks the camera keeps looking at and keeps not finding."""
        kept = []
        for track in self.tracks:
            limit = (
                config.UNCONFIRMED_MISSES if track.observations <= 1
                else config.MAXIMUM_MISSES
            )
            if track.misses >= limit:
                continue
            if (track.looks >= config.HIT_RATE_MINIMUM_LOOKS
                    and track.hit_rate < config.MINIMUM_HIT_RATE):
                continue
            kept.append(track)
        self.tracks = kept

    def _merge_duplicates(self) -> None:
        """Two tracks on one object mean two boxes, and one of them is wrong."""
        ordered = sorted(self.tracks, key=lambda track: -track.observations)
        kept: List[Track] = []
        for track in ordered:
            duplicate = None
            for other in kept:
                if _iou(track.box, other.box) > config.OUTPUT_NMS_IOU_ANY_CLASS:
                    duplicate = other
                    break
            if duplicate is None:
                kept.append(track)
                continue
            duplicate.class_scores += track.class_scores
            duplicate.scores = sorted(
                duplicate.scores + track.scores, reverse=True
            )[:config.SCORE_HISTORY]
            duplicate.observations += track.observations
            duplicate.looks += track.looks
            duplicate.last_seen_frame = max(duplicate.last_seen_frame, track.last_seen_frame)
        self.tracks = kept

    # ------------------------------------------------------------------ #

    def predictions(self, frame: int) -> List[Prediction]:
        """The answer for the whole source frame."""
        scored: List[Prediction] = []
        for track in self.tracks:
            confidence = track.confidence(frame)
            if confidence < config.MINIMUM_OUTPUT_CONFIDENCE:
                continue
            order = np.argsort(track.class_scores)[::-1]
            best = int(order[0])
            scored.append(Prediction(best, track.box, confidence))

            total = float(track.class_scores.sum())
            runner_up = int(order[1])
            if total <= 0:
                continue
            share = float(track.class_scores[runner_up]) / total
            if share >= config.SECOND_CLASS_MINIMUM_SHARE:
                scored.append(
                    Prediction(
                        runner_up,
                        track.box,
                        confidence * config.SECOND_CLASS_CONFIDENCE_FACTOR * share,
                    )
                )

        # Suppress within a class only: the scorer runs one class at a time,
        # and two classes on the same pixels is the deliberate second guess,
        # not a duplicate. Tracks that genuinely overlap were already merged.
        scored.sort(key=lambda prediction: -prediction.confidence)
        kept: List[Prediction] = []
        for prediction in scored:
            if any(
                other.class_index == prediction.class_index
                and _iou(prediction.box, other.box) > config.OUTPUT_NMS_IOU_SAME_CLASS
                for other in kept
            ):
                continue
            kept.append(prediction)
            if len(kept) >= config.MAXIMUM_ANNOTATIONS:
                break
        return kept
