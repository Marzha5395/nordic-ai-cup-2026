"""How the ground moves under the camera.

The obvious model -- one translation per frame -- is wrong, and measurably so.
Phase correlating tiles of the reference scene against each other gives a field
that is linear in image position and varies by a third across the frame:

    frame 12 -> 13, source pixels per frame

        x =  480      1440      2400      3360
    y=360  (-9.9,58)  (-3.5,56)  (3.2,57)  (10.0,57)
    y=1080 (-9.9,65)  (-3.6,66)  (3.4,66)  (10.7,65)
    y=1800 (-11.8,72) (-3.3,76)  (3.3,75)  (10.7,76)

Horizontal drift grows with x, vertical drift grows with y, and both are
straight lines through the frame centre. Fitting

    dx = a0 + a1 x + a2 y
    dy = b0 + b1 x + b2 y

reproduces the annotated object displacements to under a pixel, while the
single best translation is off by up to ten pixels a frame at the edges. Over
the five or six frames between two sightings of the same object that is the
difference between holding IoU 0.50 on a 32 pixel object and losing it.

So: phase correlate a grid of tiles rather than the whole overlap, collect the
samples over the sequence, and fit the six parameters by ridge-regularised
least squares. The ridge pulls towards a pure translation, which is what the
first few frames deserve, and releases as the camera sweeps and the samples
spread across the frame.
"""

import math
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config


def _overlap(first: Sequence[float], second: Sequence[float]):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right - left <= 0 or bottom - top <= 0:
        return None
    return left, top, right, bottom


def _crop_source_region(
    image: np.ndarray,
    view_region: Sequence[float],
    wanted: Sequence[float],
):
    """Cut the part of a transmitted view covering a source rectangle.

    Returns the crop and the source rectangle it actually covers, which is the
    wanted one snapped to whole view pixels.
    """
    height, width = image.shape[:2]
    region_x1, region_y1, region_x2, region_y2 = (float(c) for c in view_region)
    scale_x = width / (region_x2 - region_x1)
    scale_y = height / (region_y2 - region_y1)
    x1 = max(0, int(round((wanted[0] - region_x1) * scale_x)))
    y1 = max(0, int(round((wanted[1] - region_y1) * scale_y)))
    x2 = min(width, int(round((wanted[2] - region_x1) * scale_x)))
    y2 = min(height, int(round((wanted[3] - region_y1) * scale_y)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None, None
    covered = (
        region_x1 + x1 / scale_x, region_y1 + y1 / scale_y,
        region_x1 + x2 / scale_x, region_y1 + y2 / scale_y,
    )
    return image[y1:y2, x1:x2], covered


_HANNING_WINDOWS: dict = {}


def _hanning(shape: Tuple[int, int]) -> np.ndarray:
    """Cache the window: building it per tile costs more than the correlation."""
    window = _HANNING_WINDOWS.get(shape)
    if window is None:
        window = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
        _HANNING_WINDOWS[shape] = window
    return window


def _tile_correlate(first: np.ndarray, second: np.ndarray, tile: int, stride: int):
    """Phase correlate a grid of tiles, in tile-local pixels."""
    height, width = first.shape[:2]
    results = []
    ys = list(range(0, max(1, height - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, width - tile + 1), stride)) or [0]
    if ys[-1] + tile < height:
        ys.append(height - tile)
    if xs[-1] + tile < width:
        xs.append(width - tile)
    for y in ys:
        for x in xs:
            a = first[y:y + tile, x:x + tile]
            b = second[y:y + tile, x:x + tile]
            if a.shape != b.shape or a.shape[0] < 32 or a.shape[1] < 32:
                continue
            (shift_x, shift_y), response = cv2.phaseCorrelate(
                a, b, _hanning(a.shape[:2])
            )
            if not math.isfinite(response):
                continue
            results.append(
                (x + a.shape[1] / 2.0, y + a.shape[0] / 2.0, shift_x, shift_y, response)
            )
    return results


def flow_samples(
    previous_image: np.ndarray,
    previous_region: Sequence[float],
    current_image: np.ndarray,
    current_region: Sequence[float],
) -> List[Tuple[float, float, float, float, float]]:
    """Measure the ground drift at several places at once.

    Returns ``(source_x, source_y, dx, dy, response)`` per tile, in source
    pixels, for the part of the frame both views cover.
    """
    shared = _overlap(previous_region, current_region)
    if shared is None:
        return []
    if (shared[2] - shared[0]) < config.PHASE_CORRELATION_MINIMUM_OVERLAP:
        return []
    if (shared[3] - shared[1]) < config.PHASE_CORRELATION_MINIMUM_OVERLAP:
        return []

    current_crop, covered = _crop_source_region(current_image, current_region, shared)
    if current_crop is None:
        return []
    previous_crop, _ = _crop_source_region(previous_image, previous_region, covered)
    if previous_crop is None:
        return []
    if previous_crop.shape[:2] != current_crop.shape[:2]:
        previous_crop = cv2.resize(
            previous_crop,
            (current_crop.shape[1], current_crop.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    first = cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # The crop is in the current view's pixels; convert tile centres and
    # shifts back into source pixels.
    scale_x = (covered[2] - covered[0]) / first.shape[1]
    scale_y = (covered[3] - covered[1]) / first.shape[0]

    tile = config.FLOW_TILE_PIXELS
    tile = min(tile, first.shape[0], first.shape[1])
    stride = max(16, tile // 2)
    samples = []
    for local_x, local_y, shift_x, shift_y, response in _tile_correlate(
        first, second, tile, stride
    ):
        if response < config.PHASE_CORRELATION_MINIMUM_RESPONSE:
            continue
        drift_x = shift_x * scale_x
        drift_y = shift_y * scale_y
        if not all(math.isfinite(value) for value in (drift_x, drift_y)):
            continue
        magnitude = math.hypot(drift_x, drift_y)
        if magnitude > config.MAXIMUM_DRIFT_PER_FRAME:
            continue
        # The Hanning window does not move with the ground, so a tile with
        # little texture along the flight line -- crop rows, a road parallel
        # to the track -- correlates best with itself at zero shift, and says
        # so with a high response. The drone never hovers, so a shift of
        # nothing is always that and never the ground.
        if magnitude < config.MINIMUM_DRIFT_PER_FRAME:
            continue
        samples.append(
            (
                covered[0] + local_x * scale_x,
                covered[1] + local_y * scale_y,
                drift_x,
                drift_y,
                float(response),
            )
        )
    return samples


class FlowModel:
    """The affine drift field, fitted to everything measured so far.

    Two timescales, because the field has two. The gradients -- how much the
    drift grows across the frame -- are a property of the geometry and hold
    steady, so they are fitted to the whole history. The offset moves: over the
    reference scene the drift at the frame centre climbs from 62.8 to 69.7
    pixels a frame as the ground beneath the drone falls away. So the offset is
    re-measured from the last few frames only.

    Measured against the annotated object displacements of the reference scene,
    this predicts the next frame's drift to about one pixel.
    """

    FRAME_CENTRE = (1920.0, 1080.0)
    SCALE = 1000.0

    def __init__(self, history: Optional[int] = None):
        self.samples: Deque[Tuple[float, float, float, float, float]] = deque(
            maxlen=history if history is not None else config.FLOW_SAMPLE_HISTORY
        )
        self.recent: Deque[List[Tuple[float, float, float, float, float]]] = deque(
            maxlen=config.FLOW_OFFSET_FRAMES
        )
        # [a0, a1, a2] and [b0, b1, b2] against centred, scaled coordinates.
        self.x_parameters = np.zeros(3)
        self.y_parameters = np.zeros(3)
        self.measurements = 0

    def add(
        self,
        samples: Sequence[Tuple[float, float, float, float, float]],
        frames: int,
    ) -> None:
        if frames <= 0 or not samples:
            return
        normalized = [
            (source_x, source_y, drift_x / frames, drift_y / frames, response)
            for source_x, source_y, drift_x, drift_y, response in samples
        ]
        self.samples.extend(normalized)
        self.recent.append(normalized)
        self.measurements += 1
        self._fit()

    def _design(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.ones_like(xs),
                (xs - self.FRAME_CENTRE[0]) / self.SCALE,
                (ys - self.FRAME_CENTRE[1]) / self.SCALE,
            ],
            axis=1,
        )

    def _fit(self) -> None:
        data = np.asarray(self.samples, dtype=np.float64)
        if len(data) < 4:
            if len(data):
                self.x_parameters = np.array([float(np.median(data[:, 2])), 0.0, 0.0])
                self.y_parameters = np.array([float(np.median(data[:, 3])), 0.0, 0.0])
            return

        xs, ys, dxs, dys, responses = (data[:, index] for index in range(5))
        design = self._design(xs, ys)
        base_weights = np.clip(responses, 1e-3, 1.0)
        weights = base_weights
        ridge = np.diag([0.0, config.FLOW_RIDGE, config.FLOW_RIDGE])

        x_parameters = self.x_parameters
        y_parameters = self.y_parameters
        for _ in range(3):
            x_parameters = self._solve(design, dxs, weights, ridge)
            y_parameters = self._solve(design, dys, weights, ridge)
            residual = np.hypot(
                dxs - design @ x_parameters, dys - design @ y_parameters
            )
            scale = 1.4826 * max(1e-3, float(np.median(residual)))
            # Tukey-style down-weighting: a tile that locked onto the wrong
            # peak over water or snow answers confidently and wrongly.
            weights = base_weights / (1.0 + (residual / (3.0 * scale)) ** 2)

        limit = config.FLOW_MAXIMUM_GRADIENT
        x_parameters[1:] = np.clip(x_parameters[1:], -limit, limit)
        y_parameters[1:] = np.clip(y_parameters[1:], -limit, limit)

        # The offset is whatever the last few frames say, once the gradients
        # are taken out. A median, so one bad frame cannot move it.
        recent = [sample for frame in self.recent for sample in frame]
        if len(recent) >= 4:
            block = np.asarray(recent, dtype=np.float64)
            gradients = self._design(block[:, 0], block[:, 1])[:, 1:]
            x_parameters[0] = float(np.median(block[:, 2] - gradients @ x_parameters[1:]))
            y_parameters[0] = float(np.median(block[:, 3] - gradients @ y_parameters[1:]))

        self.x_parameters = x_parameters
        self.y_parameters = y_parameters

    @staticmethod
    def _solve(design, target, weights, ridge) -> np.ndarray:
        weighted = design * weights[:, None]
        normal = design.T @ weighted + ridge
        try:
            return np.linalg.solve(normal, weighted.T @ target)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(normal, weighted.T @ target, rcond=None)[0]

    # ------------------------------------------------------------------ #

    def displacement(self, source_x: float, source_y: float) -> Tuple[float, float]:
        """Ground drift per frame at one point of the source frame."""
        row = np.array(
            [
                1.0,
                (source_x - self.FRAME_CENTRE[0]) / self.SCALE,
                (source_y - self.FRAME_CENTRE[1]) / self.SCALE,
            ]
        )
        return float(row @ self.x_parameters), float(row @ self.y_parameters)

    @property
    def centre_velocity(self) -> Tuple[float, float]:
        """Drift at the middle of the frame, for whole-frame bookkeeping."""
        return float(self.x_parameters[0]), float(self.y_parameters[0])

    @property
    def settled(self) -> bool:
        return self.measurements >= 2
