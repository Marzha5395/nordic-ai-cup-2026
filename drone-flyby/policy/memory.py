"""Object memory: tracks in source-frame coordinates, advanced by the flow
map, fused per-edge against incoming detections."""
import math
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES
from policy.geometry import (box_iou, boxes_iou_matrix, clip_to_frame,
                             intersects)

W_LEVEL = {0: 0.35, 1: 0.7, 2: 1.0}
MISS_DECAY = {0: 0.95, 1: 0.8,
              2: float(os.environ.get('DRONE_MISS_DECAY_L2', '0.5'))}
DROP_BELOW = float(os.environ.get('DRONE_DROP_BELOW', '0.15'))
TAU = float(os.environ.get('DRONE_CONF_TAU', '1.0'))
SECOND_P = float(os.environ.get('DRONE_SECOND_P', '0.25'))
NEW_TRACK_CONF = dict(zip(
    (0, 1, 2),
    map(float,
        os.environ.get('DRONE_NEW_TRACK_CONF', '0.05,0.05,0.05').split(','))))
FRAME_BOX = np.array([0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT)])
N_CLASSES = len(OBJECT_CLASSES)


class Track:
    __slots__ = ('tid', 'box', 'ref_frame', 'class_scores', 'edge_reliable',
                 'n_hits', 'n_looks', 'first_frame', 'last_seen', 'best_level',
                 'dips')

    def __init__(self, tid, box, ref_frame, cls, conf, edge_reliable, level):
        self.tid = tid
        self.box = np.asarray(box, dtype=np.float64)
        self.ref_frame = ref_frame
        self.class_scores = np.zeros(N_CLASSES)
        self.class_scores[cls] = conf * W_LEVEL[level]
        self.edge_reliable = np.asarray(edge_reliable, dtype=bool).copy()
        self.n_hits = 1
        self.n_looks = 1
        self.first_frame = ref_frame
        self.last_seen = ref_frame
        self.best_level = level
        self.dips = 0


class Memory:
    def __init__(self, flow):
        self.flow = flow
        self.tracks = []
        self._next_tid = 1
        self._pred_cache = {}

    def predict_box(self, track, frame):
        key = (track.tid, frame)
        box = self._pred_cache.get(key)
        if box is None:
            box = self.flow.advance_box(track.box, frame - track.ref_frame)
            self._pred_cache[key] = box
        return box

    def ingest(self, detections_source, frame, level, region):
        """detections_source: list of (box_source np4, conf, cls,
        edge_reliable np4 bool)."""
        self._pred_cache.clear()
        region = np.asarray(region, dtype=np.float64)
        expanded = region + np.array([-24.0, -24.0, 24.0, 24.0])
        shrunk = region + np.array([6.0, 6.0, -6.0, -6.0])

        candidates = [t for t in self.tracks
                      if intersects(self.predict_box(t, frame), expanded)]
        matched_t, matched_d = set(), set()

        if candidates and detections_source:
            preds = np.stack([self.predict_box(t, frame) for t in candidates])
            dets = np.stack([np.asarray(d[0]) for d in detections_source])
            ious = boxes_iou_matrix(preds, dets)
            cost = np.full(ious.shape, 1e6)
            pc = np.column_stack([(preds[:, 0] + preds[:, 2]) / 2,
                                  (preds[:, 1] + preds[:, 3]) / 2])
            dc = np.column_stack([(dets[:, 0] + dets[:, 2]) / 2,
                                  (dets[:, 1] + dets[:, 3]) / 2])
            dist = np.hypot(pc[:, None, 0] - dc[None, :, 0],
                            pc[:, None, 1] - dc[None, :, 1])
            dsize = np.maximum(dets[:, 2] - dets[:, 0],
                               dets[:, 3] - dets[:, 1])
            for i in range(len(candidates)):
                for j in range(len(dets)):
                    if ious[i, j] >= 0.15:
                        cost[i, j] = 1.0 - ious[i, j]
                    elif dist[i, j] <= 0.6 * dsize[j] + 6:
                        cost[i, j] = 0.85 + dist[i, j] / 4000.0
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e6:
                    continue
                matched_t.add(i)
                matched_d.add(j)
                self._fuse(candidates[i], detections_source[j], frame, level)

        for j, det in enumerate(detections_source):
            if j in matched_d:
                continue
            box, conf, cls, dr = det
            if conf < NEW_TRACK_CONF[level]:
                continue  # lone weak detection: not enough to open a track
            self.tracks.append(Track(self._next_tid, box, frame, cls, conf,
                                     dr, level))
            self._next_tid += 1

        dead = []
        for i, t in enumerate(candidates):
            if i in matched_t:
                t.n_looks += 1
                continue
            p = self.predict_box(t, frame)
            inside = (p[0] >= shrunk[0] and p[1] >= shrunk[1] and
                      p[2] <= shrunk[2] and p[3] <= shrunk[3])
            if inside:
                t.n_looks += 1
                t.class_scores *= MISS_DECAY[level]
                if t.class_scores.sum() < DROP_BELOW:
                    dead.append(t)
        for t in dead:
            self.tracks.remove(t)

        # merge duplicates whose predicted boxes overlap heavily
        keep = []
        for t in self.tracks:
            dup = None
            for k in keep:
                if box_iou(self.predict_box(t, frame),
                           self.predict_box(k, frame)) > 0.6:
                    dup = k
                    break
            if dup is None:
                keep.append(t)
            else:
                if t.n_hits > dup.n_hits:
                    dup, t = t, dup
                    keep[keep.index(t)] = dup
                dup.class_scores += t.class_scores
                dup.n_hits += t.n_hits
                dup.n_looks += t.n_looks
        self.tracks = keep

        lim = FRAME_BOX + np.array([-64.0, -64.0, 64.0, 64.0])
        self.tracks = [t for t in self.tracks
                       if intersects(self.predict_box(t, frame), lim)]

    def _fuse(self, track, det, frame, level):
        box, conf, cls, dr = det
        dr = np.asarray(dr, dtype=bool)
        prev_best = track.best_level
        track.class_scores[cls] += conf * W_LEVEL[level]
        track.n_hits += 1
        track.last_seen = frame
        track.best_level = max(track.best_level, level)
        P = self.predict_box(track, frame)
        D = np.asarray(box, dtype=np.float64)
        pr = track.edge_reliable
        beta = 0.65 if level >= prev_best else 0.35
        fused = np.empty(4)
        for e in range(4):
            if pr[e] and dr[e]:
                fused[e] = (1 - beta) * P[e] + beta * D[e]
            elif dr[e]:
                fused[e] = D[e]
            elif pr[e]:
                fused[e] = P[e]
            else:
                fused[e] = min(P[e], D[e]) if e < 2 else max(P[e], D[e])
        track.box = fused
        track.ref_frame = frame
        track.edge_reliable = pr | dr
        self._pred_cache.clear()

    def emit(self, frame):
        """-> list of (class_name, box_source_clipped, confidence)."""
        out = []
        for t in self.tracks:
            pred = self.predict_box(t, frame)
            clipped = clip_to_frame(pred)
            if clipped is None:
                continue
            full_area = max(0.0, (pred[2] - pred[0]) * (pred[3] - pred[1]))
            vis_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
            if (full_area > 0 and vis_area < 0.15 * full_area and
                    (pred[2] - pred[0] > 30 or pred[3] - pred[1] > 30)):
                continue
            total = t.class_scores.sum()
            if total <= 0:
                continue
            cls = int(np.argmax(t.class_scores))
            p = t.class_scores[cls] / total
            conf = (p * (1 - math.exp(-total / TAU))
                    * max(0.5, 1 - 0.005 * (frame - t.last_seen)))
            conf = min(0.99, max(0.01, conf))
            out.append((OBJECT_CLASSES[cls], clipped, conf))
            if os.environ.get('DRONE_SECOND_CLASS', '1') != '0':
                order = np.argsort(t.class_scores)[::-1]
                if len(order) > 1:
                    cls2 = int(order[1])
                    p2 = t.class_scores[cls2] / total
                    if p2 >= SECOND_P:
                        conf2 = min(0.99, max(0.01, conf * (p2 / p) * 0.8))
                        out.append((OBJECT_CLASSES[cls2], clipped, conf2))
        out.sort(key=lambda r: -r[2])
        return out[:500]
