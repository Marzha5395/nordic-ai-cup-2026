"""Pixel-space geometry helpers: source px <-> 960x540 view px."""
import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH

VIEW_W, VIEW_H = 960, 540


def region_scale(region):
    """Source px per view px: 4 at L0, 2 at L1, 1 at L2."""
    return (region[2] - region[0]) / float(VIEW_W)


def view_to_source_box(box_xyxy_view, region):
    s = region_scale(region)
    return np.array([
        region[0] + box_xyxy_view[0] * s,
        region[1] + box_xyxy_view[1] * s,
        region[0] + box_xyxy_view[2] * s,
        region[1] + box_xyxy_view[3] * s,
    ])


def source_to_view_box(box_xyxy_source, region):
    s = region_scale(region)
    return np.array([
        (box_xyxy_source[0] - region[0]) / s,
        (box_xyxy_source[1] - region[1]) / s,
        (box_xyxy_source[2] - region[0]) / s,
        (box_xyxy_source[3] - region[1]) / s,
    ])


def edge_reliability(box_source, region):
    """(left, top, right, bottom): an edge is reliable iff >= 3 source px from
    the corresponding view-region edge AND >= 3 px from the frame edge."""
    return np.array([
        box_source[0] - region[0] >= 3 and box_source[0] >= 3,
        box_source[1] - region[1] >= 3 and box_source[1] >= 3,
        region[2] - box_source[2] >= 3 and IMAGE_WIDTH - box_source[2] >= 3,
        region[3] - box_source[3] >= 3 and IMAGE_HEIGHT - box_source[3] >= 3,
    ], dtype=bool)


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-9))


def boxes_iou_matrix(A, B):
    """(N,4) x (M,4) -> (N,M) IoU matrix."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ix1 = np.maximum(A[:, 0, None], B[None, :, 0])
    iy1 = np.maximum(A[:, 1, None], B[None, :, 1])
    ix2 = np.minimum(A[:, 2, None], B[None, :, 2])
    iy2 = np.minimum(A[:, 3, None], B[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def clip_to_frame(box):
    """Clip a source-px box to the frame; None if w < 2 or h < 2 remains."""
    x1 = min(max(box[0], 0.0), float(IMAGE_WIDTH))
    y1 = min(max(box[1], 0.0), float(IMAGE_HEIGHT))
    x2 = min(max(box[2], 0.0), float(IMAGE_WIDTH))
    y2 = min(max(box[3], 0.0), float(IMAGE_HEIGHT))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return np.array([x1, y1, x2, y2])
