"""Ground-motion model: a constant homography of the source frame, fitted
offline from Helsinki GT (synth/helsinki_camera.json) and refined online by
phase-correlation between consecutive views."""
import json
import logging
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from policy.geometry import region_scale

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CAMERA_JSON = ROOT / 'synth' / 'helsinki_camera.json'
FRAME_CENTRE = np.array([1920.0, 1080.0])

MIN_OVERLAP_AREA = 120_000.0
MIN_RESPONSE = 0.08
MAX_SHIFT_SRC = 40.0
MIN_STD = 4.0


def _rot(k):
    t = math.radians(90 * k)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


class Flow:
    def __init__(self, camera_json=None):
        path = Path(camera_json) if camera_json else CAMERA_JSON
        try:
            self.H = np.array(
                json.loads(path.read_text())['frame_map_homography'],
                dtype=np.float64)
        except Exception:
            logger.warning('No camera homography at %s; using identity + '
                           '(0,66) translation', path)
            self.H = np.eye(3)
            self.H[0, 2] = 0.0
            self.H[1, 2] = 66.0
        self.prior = self.H.copy()
        self._powers = {}
        self._meas = deque(maxlen=12)

    # ---------------- map algebra ----------------

    def power(self, n):
        n = int(n)
        if n == 0:
            return np.eye(3)
        if n not in self._powers:
            self._powers[n] = np.linalg.matrix_power(self.H, n)
        return self._powers[n]

    def advance_points(self, pts, n):
        P = self.power(n)
        pts = np.asarray(pts, dtype=np.float64)
        h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ P.T
        return h[:, :2] / h[:, 2:3]

    def advance_box(self, box, n):
        if n == 0:
            return np.asarray(box, dtype=np.float64).copy()
        corners = np.array([[box[0], box[1]], [box[2], box[1]],
                            [box[2], box[3]], [box[0], box[3]]])
        tc = self.advance_points(corners, n)
        return np.array([tc[:, 0].min(), tc[:, 1].min(),
                         tc[:, 0].max(), tc[:, 1].max()])

    # ---------------- patch measurement ----------------

    def _measure(self, Hd, prev_gray, prev_region, cur_gray, cur_region):
        """Return (pred_pts, meas_pts, responses, n_patches) for the overlap
        between prev_region advanced by Hd and cur_region."""
        corners = np.array([[prev_region[0], prev_region[1]],
                            [prev_region[2], prev_region[1]],
                            [prev_region[2], prev_region[3]],
                            [prev_region[0], prev_region[3]]], np.float64)
        h = np.concatenate([corners, np.ones((4, 1))], axis=1) @ Hd.T
        pc = h[:, :2] / h[:, 2:3]
        pred_region = np.array([pc[:, 0].min(), pc[:, 1].min(),
                                pc[:, 0].max(), pc[:, 1].max()])
        ox1 = max(pred_region[0], cur_region[0])
        oy1 = max(pred_region[1], cur_region[1])
        ox2 = min(pred_region[2], cur_region[2])
        oy2 = min(pred_region[3], cur_region[3])
        if (ox2 - ox1) * (oy2 - oy1) < MIN_OVERLAP_AREA:
            return None

        w = max(region_scale(prev_region), region_scale(cur_region))
        ps = 128 * w  # patch side in source px

        def axis_count(span):
            n = max(1, int(span / ps))
            if span >= ps and span - n * ps > ps / 2:
                n += 1  # overlap the last row/column rather than waste it
            return n

        nx, ny = axis_count(ox2 - ox1), axis_count(oy2 - oy1)

        def centres(o1, o2, n):
            lo, hi = o1 + ps / 2, o2 - ps / 2
            if n <= 1:
                return [(o1 + o2) / 2.0]
            return np.linspace(lo, hi, n)

        try:
            Hinv = np.linalg.inv(Hd)
        except np.linalg.LinAlgError:
            return None
        sp = region_scale(prev_region)
        sc = region_scale(cur_region)
        hanning = cv2.createHanningWindow((128, 128), cv2.CV_32F)

        pred_pts, meas_pts, resps = [], [], []
        for cy in centres(oy1, oy2, ny):
            for cx in centres(ox1, ox2, nx):
                # current-view patch: dst px (i,j) -> cur view px
                Mc = np.array([
                    [w / sc, 0, (cx - ps / 2 - cur_region[0]) / sc],
                    [0, w / sc, (cy - ps / 2 - cur_region[1]) / sc],
                ], np.float64)
                cur_patch = cv2.warpAffine(
                    cur_gray, Mc, (128, 128),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                if cur_patch.std() < MIN_STD:
                    continue
                # previous-view patch: dst px -> cur-frame src -> prev-frame
                # src (locally affine via 3 corners) -> prev view px
                loc = np.array([[0, 0], [128, 0], [0, 128]], np.float64)
                src = np.column_stack([
                    cx - ps / 2 + loc[:, 0] * w,
                    cy - ps / 2 + loc[:, 1] * w,
                ])
                hh = np.concatenate([src, np.ones((3, 1))], axis=1) @ Hinv.T
                prev_src = hh[:, :2] / hh[:, 2:3]
                dst = np.column_stack([
                    (prev_src[:, 0] - prev_region[0]) / sp,
                    (prev_src[:, 1] - prev_region[1]) / sp,
                ]).astype(np.float32)
                Mp = cv2.getAffineTransform(loc.astype(np.float32), dst)
                prev_patch = cv2.warpAffine(
                    prev_gray, Mp, (128, 128),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                (dx, dy), resp = cv2.phaseCorrelate(
                    prev_patch * hanning, cur_patch * hanning)
                shift = np.array([dx * w, dy * w])
                if resp < MIN_RESPONSE or np.hypot(*shift) > MAX_SHIFT_SRC:
                    continue
                pred_pts.append([cx, cy])
                meas_pts.append([cx + shift[0], cy + shift[1]])
                resps.append(resp)
        return (np.array(pred_pts), np.array(meas_pts),
                np.array(resps), nx * ny)

    # ---------------- online estimation ----------------

    def observe(self, prev_gray, prev_region, cur_gray, cur_region, dframe):
        dframe = int(dframe)
        stats = {'n_patches': 0, 'n_good': 0, 'residual_px': 0.0}
        if dframe == 0:
            return stats
        Hd = self.power(dframe)
        out = self._measure(Hd, prev_gray, prev_region, cur_gray, cur_region)
        if out is None:
            return stats
        pred, meas, resps, n_patches = out
        stats['n_patches'] = n_patches
        n_good = len(pred)
        stats['n_good'] = n_good
        if n_good == 0:
            return stats
        shifts = meas - pred
        stats['residual_px'] = float(np.median(np.hypot(*shifts.T)))

        if n_good >= 4:
            C, _ = cv2.estimateAffine2D(pred, meas, method=cv2.RANSAC,
                                        ransacReprojThreshold=3)
            if C is not None:
                C_h = np.eye(3)
                C_h[:2] = C
                # refinement passes: re-measure around the corrected map so
                # residual shifts are near zero and the estimate is unbiased
                for _ in range(2):
                    out2 = self._measure(C_h @ Hd, prev_gray, prev_region,
                                         cur_gray, cur_region)
                    if out2 is None or len(out2[0]) < 4:
                        break
                    C2, _ = cv2.estimateAffine2D(
                        out2[0], out2[1], method=cv2.RANSAC,
                        ransacReprojThreshold=3)
                    if C2 is None:
                        break
                    C2_h = np.eye(3)
                    C2_h[:2] = C2
                    C_h = C2_h @ C_h
                H_meas = C_h @ Hd
            else:
                H_meas = None
        else:
            C = None
            H_meas = None
        if n_good < 4 or C is None:
            med = np.median(shifts, axis=0)
            H_meas = Hd.copy()
            H_meas[:2, 2] += med

        if dframe > 1:
            per = self.H.copy()
            per[:2, 2] += (H_meas[:2, 2] - Hd[:2, 2]) / dframe
            self._meas.append(per)
        else:
            self._meas.append(H_meas)

        if len(self._meas) >= 4:
            med_map = np.median(np.stack(self._meas), axis=0)
            dt = med_map[:2, 2] - self.prior[:2, 2]
            if np.hypot(*dt) > 10:
                logger.warning('flow: measured translation differs from prior '
                               'by %.1f px — adapting to new geometry',
                               np.hypot(*dt))
            # The map is constant, so the deque median is already the robust
            # steady-state estimate of the 0.5 blend; snapping to it
            # converges in one step instead of ~1 bit/frame.
            self.H = med_map
            self._powers.clear()
        return stats

    # ---------------- one-shot hypothesis check ----------------

    def _variants(self):
        """Prior H plus 3 variants whose effective centre motion is rotated
        by 90/180/270 deg."""
        A = self.prior[:2, :2]
        t = self.prior[:2, 2]
        motion = A @ FRAME_CENTRE + t - FRAME_CENTRE
        variants = []
        for k in range(4):
            Hv = self.prior.copy()
            Hv[:2, 2] = FRAME_CENTRE + _rot(k) @ motion - A @ FRAME_CENTRE
            variants.append(Hv)
        return variants

    def _score(self, Hv, prev_gray, prev_region, cur_gray, cur_region):
        out = self._measure(Hv, prev_gray, prev_region, cur_gray, cur_region)
        if out is None:
            return 0.0
        _, _, resps, _ = out
        if len(resps) == 0:
            return 0.0
        return float(resps.mean())

    def hypothesis_check(self, prev_gray, prev_region, cur_gray, cur_region):
        scores = [self._score(v, prev_gray, prev_region,
                              cur_gray, cur_region)
                  for v in self._variants()]
        best = int(np.argmax(scores))
        logger.info('flow hypothesis scores (0/90/180/270): %s',
                    [round(s, 3) for s in scores])
        if best != 0 and scores[best] > scores[0] + 0.05:
            logger.warning('flow: prior hypothesis wrong — switching '
                           'translation to %d-degree variant (%.3f vs %.3f)',
                           best * 90, scores[best], scores[0])
            self.H = self._variants()[best]
            self._powers.clear()
        return best
