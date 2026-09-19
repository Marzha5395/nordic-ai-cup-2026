import cv2
import numpy as np


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def move_box(box, matrix):
    x1, y1, x2, y2 = box
    points = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], dtype=np.float32)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return np.concatenate((points.min(axis=0), points.max(axis=0)))


def estimate_motion(previous, current, previous_region, current_region):
    previous_region = np.asarray(previous_region, dtype=np.float32)
    current_region = np.asarray(current_region, dtype=np.float32)
    height, width = current.shape
    old_height, old_width = previous.shape
    old_scale = (previous_region[2:] - previous_region[:2]) / [old_width, old_height]
    new_scale = (current_region[2:] - current_region[:2]) / [width, height]
    scale = old_scale / new_scale
    shift = (previous_region[:2] - current_region[:2]) / new_scale
    alignment = np.array([[scale[0], 0, shift[0]], [0, scale[1], shift[1]]], dtype=np.float32)
    aligned = cv2.warpAffine(previous, alignment, (width, height))
    valid = cv2.warpAffine(np.full(previous.shape, 255, dtype=np.uint8), alignment, (width, height))
    valid = cv2.erode(valid, np.ones((25, 25), dtype=np.uint8))
    points = cv2.goodFeaturesToTrack(aligned, maxCorners=400, qualityLevel=0.015, minDistance=10, mask=valid, blockSize=7)
    if points is None or len(points) < 12:
        return IDENTITY.copy(), False
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(aligned, current, points, None, winSize=(21, 21), maxLevel=3)
    if next_points is None:
        return IDENTITY.copy(), False
    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(current, aligned, next_points, None, winSize=(21, 21), maxLevel=3)
    if back_points is None:
        return IDENTITY.copy(), False
    keep = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    keep &= np.linalg.norm(points[:, 0] - back_points[:, 0], axis=1) < 1.5
    keep &= (next_points[:, 0] >= 0).all(axis=1) & (next_points[:, 0] < [width, height]).all(axis=1)
    if keep.sum() < 12:
        return IDENTITY.copy(), False
    source = points[keep, 0] * new_scale + current_region[:2]
    target = next_points[keep, 0] * new_scale + current_region[:2]
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=float(max(new_scale) * 2), maxIters=1000, confidence=0.99)
    if matrix is None or inliers is None or inliers.sum() < 12 or inliers.mean() < 0.45 or not np.isfinite(matrix).all():
        return IDENTITY.copy(), False
    zoom = np.linalg.norm(matrix[:, 0])
    rotation = np.arctan2(matrix[1, 0], matrix[0, 0])
    if not 0.85 <= zoom <= 1.18 or abs(rotation) > np.deg2rad(15):
        return IDENTITY.copy(), False
    return matrix.astype(np.float32), True
