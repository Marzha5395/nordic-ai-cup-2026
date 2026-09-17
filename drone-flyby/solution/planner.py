"""Where to point the camera next.

The arithmetic that decides this is worth stating, because it settles the
question before any policy is written.

New ground arrives at the top of the frame at the drift rate, about 65 px per
frame across the full 3840 px width, so roughly 250 000 px^2 of the source
frame is new every frame. What each level can cover in a frame is the width of
its view times how far its centre may move:

    level 0   the whole frame, but reduced by four in each dimension
    level 1   1920x1080 seen at half scale, centre may move 1102 px/frame
              -> 1102 * 1080 = 1.19 Mpx^2 of fresh ground per frame
    level 2   960x540 seen at full scale, centre may move 551 px/frame
              -> 551 * 540  = 0.30 Mpx^2 of fresh ground per frame

Level 1 covers new ground four times faster than it arrives, which leaves room
to revisit and to turn around. Level 2 covers it 1.2 times faster, which does
not survive a single turn at the end of a sweep, and it would still have to
find every object on the first and only look. So the camera lives at level 1.

Given the level, the choice of centre is a staleness problem: keep a map of
how long ago each patch of ground was looked at, drift it with the scene, and
point the camera wherever the most staleness -- plus the tracks most in need of
a second opinion -- sits inside one view.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import config


class CoverageMap:
    """How long ago each patch of the current source frame was observed.

    The map is held in source coordinates and shifted with the ground every
    frame, so the cells that scroll in at the edge are new ground that has
    never been looked at, which is exactly how they should be scored.
    """

    def __init__(self, width: int = 3840, height: int = 2160,
                 cell: Optional[int] = None):
        self.cell = cell if cell is not None else config.COVERAGE_CELL
        cell = self.cell
        self.columns = int(math.ceil(width / cell))
        self.rows = int(math.ceil(height / cell))
        self.age = np.full((self.rows, self.columns), config.COVERAGE_MAXIMUM_AGE, np.float32)
        # At the start every cell is equally unknown, so the first move is
        # decided by whichever candidate the scan reaches first. Breaking the
        # tie at random instead costs nothing and makes the policy's quality
        # measurable: run the same scene with several seeds and the spread is
        # the trajectory noise rather than a property of the settings.
        if config.COVERAGE_INITIAL_JITTER > 0.0:
            generator = np.random.default_rng(config.COVERAGE_TIE_BREAK_SEED)
            self.age -= generator.random(self.age.shape).astype(np.float32) * (
                config.COVERAGE_INITIAL_JITTER * config.COVERAGE_MAXIMUM_AGE
            )
        self._residual_x = 0.0
        self._residual_y = 0.0

    def advance(self, drift: Tuple[float, float], frames: int) -> None:
        if frames <= 0:
            return
        self.age = np.minimum(self.age + frames, config.COVERAGE_MAXIMUM_AGE)
        self._residual_x += drift[0] * frames
        self._residual_y += drift[1] * frames
        shift_x = int(self._residual_x // self.cell)
        shift_y = int(self._residual_y // self.cell)
        if shift_x == 0 and shift_y == 0:
            return
        self._residual_x -= shift_x * self.cell
        self._residual_y -= shift_y * self.cell
        self.age = np.roll(self.age, (shift_y, shift_x), axis=(0, 1))
        fresh = config.COVERAGE_MAXIMUM_AGE
        if shift_y > 0:
            self.age[:min(shift_y, self.rows)] = fresh
        elif shift_y < 0:
            self.age[max(0, self.rows + shift_y):] = fresh
        if shift_x > 0:
            self.age[:, :min(shift_x, self.columns)] = fresh
        elif shift_x < 0:
            self.age[:, max(0, self.columns + shift_x):] = fresh

    def observe(self, region: Sequence[float], quality: float) -> None:
        x1 = max(0, int(math.floor(region[0] / self.cell)))
        y1 = max(0, int(math.floor(region[1] / self.cell)))
        x2 = min(self.columns, int(math.ceil(region[2] / self.cell)))
        y2 = min(self.rows, int(math.ceil(region[3] / self.cell)))
        if x2 <= x1 or y2 <= y1:
            return
        self.age[y1:y2, x1:x2] *= (1.0 - min(1.0, max(0.0, quality)))

    def integral(self) -> np.ndarray:
        """Summed-area table, padded, so a region's total is four lookups."""
        padded = np.zeros((self.rows + 1, self.columns + 1), np.float64)
        padded[1:, 1:] = self.age.cumsum(axis=0).cumsum(axis=1)
        return padded

    def region_total(self, integral: np.ndarray, region: Sequence[float]) -> float:
        """Sum of ages over a source rectangle, counting off-map as untouched.

        A candidate view is evaluated against where the ground will be next
        frame, so part of it can fall above the top of the map. That part is
        ground that has not arrived yet, and it is the most valuable ground
        there is.
        """
        raw_x1 = region[0] / self.cell
        raw_y1 = region[1] / self.cell
        raw_x2 = region[2] / self.cell
        raw_y2 = region[3] / self.cell
        x1 = int(round(min(max(raw_x1, 0), self.columns)))
        y1 = int(round(min(max(raw_y1, 0), self.rows)))
        x2 = int(round(min(max(raw_x2, 0), self.columns)))
        y2 = int(round(min(max(raw_y2, 0), self.rows)))
        inside = 0.0
        if x2 > x1 and y2 > y1:
            inside = float(
                integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            )
        total_cells = max(0.0, raw_x2 - raw_x1) * max(0.0, raw_y2 - raw_y1)
        outside_cells = max(0.0, total_cells - (x2 - x1) * (y2 - y1))
        return inside + outside_cells * config.COVERAGE_MAXIMUM_AGE


SOURCE_REGION_SIZES = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}


def source_region(level: int, centre_x: float, centre_y: float):
    width, height = SOURCE_REGION_SIZES[level]
    return (
        centre_x - width / 2.0, centre_y - height / 2.0,
        centre_x + width / 2.0, centre_y + height / 2.0,
    )


class CameraPlanner:
    """Pick the reachable view that buys the most information."""

    def __init__(self, preferred_level: Optional[int] = None):
        self.preferred_level = (
            preferred_level if preferred_level is not None else config.PREFERRED_LEVEL
        )
        self.coverage = CoverageMap()

    def observe(self, region: Sequence[float], level: int) -> None:
        self.coverage.observe(region, config.LEVEL_DETECTION_QUALITY.get(level, 0.9))

    def advance(self, drift: Tuple[float, float], frames: int) -> None:
        self.coverage.advance(drift, frames)

    def choose(
        self,
        current_level: int,
        current_centre: Tuple[int, int],
        constraints,
        tracks,
        frame: int,
        drift: Tuple[float, float],
    ) -> Optional[Tuple[int, int, int]]:
        """Return (resolution_level, center_x, center_y) or None to hold.

        Everything comes from ``constraints`` -- the reachable levels, the
        legal centre window and the movement limit -- so a command built here
        cannot be one the evaluator refuses.
        """
        allowed = [
            level for level in constraints.allowed_resolution_levels
            if level == self.preferred_level
        ]
        if not allowed:
            # The preferred level is one step away: level 0 and level 2 cannot
            # reach each other, so take the step that gets closer to it.
            reachable = sorted(constraints.allowed_resolution_levels)
            if not reachable:
                return None
            target = min(reachable, key=lambda level: abs(level - self.preferred_level))
            allowed = [target]
        level = allowed[0]

        bounds = constraints.bounds_for_level(level)
        if bounds is None:
            return None

        if level == 0:
            return 0, 1920, 1080

        limit = float(constraints.maximum_center_delta)
        step = config.CANDIDATE_STEP
        xs = list(range(bounds.minimum_center_x, bounds.maximum_center_x + 1, step))
        ys = list(range(bounds.minimum_center_y, bounds.maximum_center_y + 1, step))
        if xs[-1] != bounds.maximum_center_x:
            xs.append(bounds.maximum_center_x)
        if ys[-1] != bounds.maximum_center_y:
            ys.append(bounds.maximum_center_y)

        integral = self.coverage.integral()
        quality = config.LEVEL_DETECTION_QUALITY.get(level, 0.9)

        # Tracks are worth revisiting in proportion to how long it has been
        # since anyone looked, and to how little is known about them.
        track_values: List[Tuple[float, float, float]] = []
        for track in tracks:
            staleness = max(0, frame - track.last_seen_frame)
            if staleness <= 0:
                continue
            weight = staleness * (
                config.UNCONFIRMED_TRACK_WEIGHT if track.observations <= 2 else 1.0
            )
            centre_x, centre_y = track.centre
            track_values.append((centre_x + drift[0], centre_y + drift[1], weight))

        best_score = float("-inf")
        best: Optional[Tuple[int, int, int]] = None
        for centre_x in xs:
            for centre_y in ys:
                if math.hypot(centre_x - current_centre[0],
                              centre_y - current_centre[1]) > limit:
                    continue
                region = source_region(level, centre_x, centre_y)
                # Score against where the ground will be when this view is
                # actually taken, one frame from now.
                shifted = (
                    region[0] - drift[0], region[1] - drift[1],
                    region[2] - drift[0], region[3] - drift[1],
                )
                score = quality * self.coverage.region_total(integral, shifted)
                for track_x, track_y, weight in track_values:
                    if region[0] <= track_x <= region[2] and region[1] <= track_y <= region[3]:
                        score += config.TRACK_REVISIT_WEIGHT * weight
                if config.MOVE_COST > 0.0:
                    score -= config.MOVE_COST * math.hypot(
                        centre_x - current_centre[0], centre_y - current_centre[1]
                    )
                if score > best_score:
                    best_score = score
                    best = (level, int(centre_x), int(centre_y))
        return best
