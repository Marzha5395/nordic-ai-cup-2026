"""Camera scheduler: a short L1 opening tour, then an L2 boustrophedon over
the entry strip that follows the ground flow, so each strip of ground is
scanned once at native resolution while it is near the edge it entered on."""
import logging
import math
import os

import numpy as np

from dtos import (IMAGE_HEIGHT, IMAGE_WIDTH, MAXIMUM_CENTER_DELTA_PIXELS,
                  RequestedViewDto)
from utils import describe_camera_rejection

logger = logging.getLogger(__name__)

CENTRE = np.array([1920.0, 1080.0])

# (min, max) legal centres per level
L1_XB, L1_YB = (960, 2880), (540, 1620)
L2_XB, L2_YB = (480, 3360), (270, 1890)


class Scheduler:
    def __init__(self):
        self.phase = 'opening'
        self.tour = []          # remaining L1 centres
        self.direction = 1      # along the sweep axis
        self._cfg = None        # (horizontal, entry_positive) the plan fits
        self._plan = None
        self._refresh = None    # (strip_start_xy, new_direction) pending
        self._dip_state = None  # (leave_frame, axis_coord, strip_coord)
        self._last_dip = -10
        self._last_return = -10

    # ---------------- plan geometry ----------------

    def _config(self, flow):
        """(horizontal, entry_positive): flow direction axis + sign."""
        d = flow.advance_points(CENTRE[None, :], 1)[0] - CENTRE
        if abs(d[0]) > abs(d[1]):
            return True, d[0] > 0      # entry left if objects move +x
        return False, d[1] > 0         # entry top if objects move +y

    def _build_plan(self, cfg):
        horizontal, entry_pos = cfg
        sweep_level = (1 if os.environ.get('DRONE_SWEEP_LEVEL', '1') == '1'
                       else 2)
        if not horizontal:
            if entry_pos:      # enter at top
                tour = [(960, 1620), (960, 540), (1920, 540),
                        (1920, 1620), (2880, 1620), (2880, 540)]
                start = (3360, 270)
                strip_home = 540 if sweep_level == 1 else 270
                step = 960 if sweep_level == 1 else 480
                axis_lo, axis_hi = (L1_XB if sweep_level == 1 else L2_XB)
                flow_lo, flow_hi = (L1_YB if sweep_level == 1 else L2_YB)
            else:              # enter at bottom
                tour = [(960, 540), (960, 1620), (1920, 1620),
                        (1920, 540), (2880, 540), (2880, 1620)]
                start = (3360, 1890)
                strip_home = 1620 if sweep_level == 1 else 1890
                step = 960 if sweep_level == 1 else 480
                axis_lo, axis_hi = (L1_XB if sweep_level == 1 else L2_XB)
                flow_lo, flow_hi = (L1_YB if sweep_level == 1 else L2_YB)
        else:
            # transposed: sweep axis is y, flow axis is x
            if entry_pos:      # enter at left
                tour = [(2880, 540), (960, 540), (960, 1080),
                        (2880, 1080), (2880, 1620), (960, 1620)]
                start = (480, 1890)
                strip_home = 960 if sweep_level == 1 else 480
            else:              # enter at right
                tour = [(960, 540), (2880, 540), (2880, 1080),
                        (960, 1080), (960, 1620), (2880, 1620)]
                start = (3360, 1890)
                strip_home = 2880 if sweep_level == 1 else 3360
            step = 960 if sweep_level == 1 else 540
            axis_lo, axis_hi = (L1_YB if sweep_level == 1 else L2_YB)
            flow_lo, flow_hi = (L1_XB if sweep_level == 1 else L2_XB)
        return {'level': sweep_level, 'tour': tour, 'start': start,
                'strip_home': strip_home, 'step': step,
                'axis_lo': axis_lo, 'axis_hi': axis_hi,
                'flow_lo': flow_lo, 'flow_hi': flow_hi,
                'refresh': (sweep_level == 2 and
                            os.environ.get('DRONE_L1_REFRESH', '1') != '0'),
                'dips': (sweep_level == 1 and
                         os.environ.get('DRONE_L2_DIPS', '0') != '0')}

    # ---------------- main ----------------

    def next_view(self, request, flow, memory=None):
        view = request.view
        level = int(view.resolution_level)
        cx, cy = int(view.center_x), int(view.center_y)
        frame = int(request.frame)
        cfg = self._config(flow)
        if cfg != self._cfg:
            self._cfg = cfg
            self._plan = self._build_plan(cfg)
            if self.phase == 'opening':
                self.tour = list(self._plan['tour'])
        plan = self._plan
        horizontal = cfg[0]

        candidate = None
        if self.phase == 'opening':
            if level == 0:
                candidate = (1,) + (self.tour[0] if self.tour
                                    else (1920, 540))
            elif level == 1:
                if self.tour and (cx, cy) == self.tour[0]:
                    self.tour.pop(0)
                if self.tour:
                    candidate = (1,) + self._step_toward(
                        (cx, cy), self.tour[0], 1102)
                else:
                    self.phase = 'sweep'
                    if plan['level'] == 1:
                        # the tour already ends on an L1 sweep column at the
                        # entry strip: sweep away from that edge right away
                        axis = cx if not horizontal else cy
                        mid = 1920 if not horizontal else 1080
                        self.direction = -1 if axis >= mid else 1
                        candidate = self._sweep_step(
                            (cx, cy), plan, horizontal, flow)
                    else:
                        candidate = (plan['level'],) + self._step_toward(
                            (cx, cy), plan['start'], 1102)
            else:  # already at sweep level somehow
                self.phase = 'sweep'
        if self.phase == 'sweep' and candidate is None:
            sl = plan['level']
            limit = MAXIMUM_CENTER_DELTA_PIXELS[level]
            if (level == 1 and sl == 2 and plan['refresh']):
                # the L1 refresh view between strips (or any other L1 view
                # reached mid-sweep): issue the new strip start at L2
                if self._refresh is not None:
                    (sx, sy), nd = self._refresh
                    self.direction = nd
                    self._refresh = None
                    candidate = (2, sx, sy)
                else:
                    starts = self._strip_starts(plan, horizontal)
                    target = min(starts, key=lambda s: math.hypot(
                        s[0] - cx, s[1] - cy))
                    candidate = (2,) + self._step_toward(
                        (cx, cy), target, limit)
            elif level < sl:
                # climb one level toward the sweep start
                if level == 0:
                    candidate = (1, 1920, 540)
                else:
                    candidate = (sl,) + self._step_toward(
                        (cx, cy), plan['start'], limit)
            elif level > sl:
                if (plan['dips'] and level == 2
                        and self._dip_state is not None):
                    # coming back from an L2 dip: rejoin the strip where it
                    # has flowed to, on the nearest sweep column; the command
                    # lands one frame later, so advance elapsed + 1
                    leave_f, axis_v, strip_v = self._dip_state
                    self._dip_state = None
                    self._last_return = frame
                    r = self._dip_return(plan, horizontal, flow, axis_v,
                                         strip_v, frame - leave_f + 1)
                    if describe_camera_rejection(
                            level, (cx, cy), sl, r) is not None:
                        # drifted past the hop limit: go as far as legal
                        r = self._step_toward((cx, cy), r, limit)
                    candidate = (sl,) + r
                elif sl == 1:
                    # at L2 without dip state (e.g. rejected return): head
                    # for the nearest column at the flowed strip coordinate
                    ax = (cx if not horizontal else cy)
                    cols = np.arange(plan['axis_lo'],
                                     plan['axis_hi'] + 0.5, plan['step'])
                    axis_v = float(cols[np.argmin(np.abs(cols - ax))])
                    strip_v = cy if not horizontal else cx
                    r = self._dip_return(plan, horizontal, flow, axis_v,
                                         strip_v, 1)
                    candidate = (sl,) + self._step_toward((cx, cy), r, limit)
                else:
                    candidate = (sl,) + self._step_toward(
                        (cx, cy), plan['start'], limit)
            else:
                if level == sl:
                    self._dip_state = None  # any pending dip was not taken
                dip = None
                if (plan['dips'] and sl == 1 and memory is not None
                        and frame - self._last_dip >= 2
                        and frame - self._last_return >= 2):
                    dip = self._try_dip(
                        (cx, cy), plan, horizontal, flow, memory, frame)
                candidate = (dip if dip is not None else
                             self._sweep_step(
                                 (cx, cy), plan, horizontal, flow))

        if candidate is None:
            return None
        lvl, tx, ty = int(candidate[0]), int(candidate[1]), int(candidate[2])
        reason = describe_camera_rejection(level, (cx, cy), lvl, (tx, ty))
        if reason is not None and self._refresh is not None and lvl == 1:
            # refresh hop too far (strip drifted): take the plain turnaround
            logger.info('scheduler: L1 refresh rejected (%s) — plain '
                        'turnaround instead', reason)
            (sx, sy), nd = self._refresh
            self._refresh = None
            self.direction = nd
            lvl, tx, ty = plan['level'], int(sx), int(sy)
            reason = describe_camera_rejection(level, (cx, cy), lvl, (tx, ty))
        if reason is not None:
            logger.warning('scheduler: command (%d,%d,%d) rejected: %s — '
                           'holding', lvl, tx, ty, reason)
            return None
        return RequestedViewDto(resolution_level=lvl, center_x=tx,
                                center_y=ty)

    @staticmethod
    def _step_toward(cur, target, limit):
        dx, dy = target[0] - cur[0], target[1] - cur[1]
        dist = math.hypot(dx, dy)
        if dist > limit:
            dx, dy = dx * limit / dist, dy * limit / dist
        return int(round(cur[0] + dx)), int(round(cur[1] + dy))

    @staticmethod
    def _strip_starts(plan, horizontal):
        if not horizontal:
            return [(plan['axis_lo'], plan['strip_home']),
                    (plan['axis_hi'], plan['strip_home'])]
        return [(plan['strip_home'], plan['axis_lo']),
                (plan['strip_home'], plan['axis_hi'])]

    @staticmethod
    def _refresh_view(plan, horizontal, axis):
        """The single L1 view inserted at a sweep turnaround."""
        if not horizontal:
            x = 2880 if axis > 1920 else 960
            y = 540 if plan['strip_home'] < 1080 else 1620
            return (1, x, y)
        x = 960 if plan['strip_home'] < 1920 else 2880
        y = 540 if axis < 1080 else 1620
        return (1, x, y)

    @staticmethod
    def _dip_return(plan, horizontal, flow, axis_v, strip_v, elapsed):
        """L1 position to rejoin the strip: nearest column axis_v, flow-
        advanced strip coordinate strip_v pushed `elapsed` frames ahead."""
        if not horizontal:
            adv = flow.advance_points(
                np.array([[axis_v, strip_v]]), elapsed)[0]
            return (int(axis_v),
                    int(round(min(max(adv[1], L1_YB[0]), L1_YB[1]))))
        adv = flow.advance_points(
            np.array([[strip_v, axis_v]]), elapsed)[0]
        return (int(round(min(max(adv[0], L1_XB[0]), L1_XB[1]))),
                int(axis_v))

    def _try_dip(self, centre, plan, horizontal, flow, memory, frame):
        """Pick a small/uncertain L1-only track worth one L2 look, or None."""
        cx, cy = centre
        cols = np.arange(plan['axis_lo'], plan['axis_hi'] + 0.5,
                         plan['step'])
        min_total = float(os.environ.get('DRONE_DIP_MIN_TOTAL', '0'))
        best = None
        for t in memory.tracks:
            if t.best_level >= 2 or t.dips >= 1 or t.n_hits < 1:
                continue
            total = float(t.class_scores.sum())
            if total < min_total:
                continue
            pred = np.asarray(memory.predict_box(t, frame + 1), float)
            if (pred[0] < 0 or pred[1] < 0 or
                    pred[2] > IMAGE_WIDTH or pred[3] > IMAGE_HEIGHT):
                continue
            w, h = pred[2] - pred[0], pred[3] - pred[1]
            purity = float(t.class_scores.max() / total) if total else 0.0
            if not (max(w, h) < 80 or purity < 0.75 or t.n_hits == 1):
                continue
            ox, oy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
            tx = min(max(ox, L2_XB[0]), L2_XB[1])
            ty = min(max(oy, L2_YB[0]), L2_YB[1])
            if abs(ox - tx) > 400 or abs(oy - ty) > 200:
                continue
            if math.hypot(cx - tx, cy - ty) > \
                    MAXIMUM_CENTER_DELTA_PIXELS[1]:
                continue
            # where we rejoin the strip two frames later (dip + return)
            axis_v = float(cols[np.argmin(np.abs(
                cols - (tx if not horizontal else ty)))])
            strip_v = cy if not horizontal else cx
            r = self._dip_return(plan, horizontal, flow, axis_v, strip_v, 2)
            if math.hypot(tx - r[0], ty - r[1]) > \
                    MAXIMUM_CENTER_DELTA_PIXELS[2]:
                continue
            if describe_camera_rejection(
                    1, (int(cx), int(cy)), 2, (int(tx), int(ty))) is not None:
                continue
            key = (max(w, h), purity)
            if best is None or key < best[0]:
                best = (key, (2, int(tx), int(ty)),
                        (frame, axis_v, strip_v), t)
        if best is None:
            return None
        _, cand, state, track = best
        track.dips += 1
        self._dip_state = state
        self._last_dip = frame
        return cand

    def _sweep_step(self, centre, plan, horizontal, flow):
        cx, cy = centre
        axis = cx if not horizontal else cy
        nxt = axis + self.direction * plan['step']
        if nxt > plan['axis_hi'] or nxt < plan['axis_lo']:
            new_dir = -self.direction
            # new strip back at the entry edge
            start = ((axis, plan['strip_home']) if not horizontal
                     else (plan['strip_home'], axis))
            if plan['refresh']:
                # go through one L1 view before starting the new strip
                self._refresh = (start, new_dir)
                return self._refresh_view(plan, horizontal, axis)
            self.direction = new_dir
            return (plan['level'],) + start
        moved = flow.advance_points(np.array([[cx, cy]]), 1)[0]
        flowv = moved[1] if not horizontal else moved[0]
        flowv = min(max(flowv, plan['flow_lo']), plan['flow_hi'])
        if not horizontal:
            return (plan['level'], int(nxt), int(round(flowv)))
        return (plan['level'], int(round(flowv)), int(nxt))
