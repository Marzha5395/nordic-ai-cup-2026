"""Cut the annotated objects out of the reference frames, once, with SAM.

GrabCut finds nothing on several classes (the helicopter, the large tower and
the medium launcher are all green on green), and a fallback to the whole box
pastes a rectangle of Helsinki ground along with the object, so the detector
learns the rectangle. SAM 2.1, prompted with the annotated box on a 4x upscaled
crop, separates them. Its mask is clipped to the box, reduced to its largest
connected part, and instances whose coverage is far from their class's median
are dropped as bad cut-outs.

The result is cached in ``.training/sprites.pkl``. SAM is a training-time tool
only; nothing here runs at inference.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.training' / 'sprites.pkl'
WIDTH, HEIGHT = 3840, 2160


# Classes whose annotated box is much larger than the visible model: the box prompt pulls in
# terrain, so SAM gets the centre point alone. Chosen by inspecting the cut-outs, not by any score.
POINT_ONLY = {'medium_launcher'}


def _segment(model, image, box, point_only=False):
    x1, y1, x2, y2 = box
    pad = max(x2 - x1, y2 - y1)
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    crop = image[top:y2 + pad, left:x2 + pad]
    scale = 4
    large = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # A positive point at the box centre as well as the box: with the box alone SAM sometimes takes
    # the terrain around a small object (medium_launcher) or a thin line next to it (small_launcher).
    center = [((x1 + x2) / 2 - left) * scale, ((y1 + y2) / 2 - top) * scale]
    if point_only:
        result = model(large, points=[center], labels=[1], verbose=False)[0]
    else:
        result = model(large, bboxes=[[(x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale]],
                       points=[center], labels=[1], verbose=False)[0]
    mask = result.masks.data[0].cpu().numpy().astype(np.float32)
    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
    inside = np.zeros_like(mask)
    inside[y1 - top:y2 - top, x1 - left:x2 - left] = True
    mask &= inside
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    mask = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # The object sits in the middle of its box; a mask that misses the middle is something else.
    cx, cy = (x1 + x2) // 2 - left, (y1 + y2) // 2 - top
    reach = max(2, min(x2 - x1, y2 - y1) // 6)
    if not mask[cy - reach:cy + reach + 1, cx - reach:cx + reach + 1].any():
        return None
    return crop, mask, (x1 - left, y1 - top, x2 - left, y2 - top)


def build(scene=ROOT / 'src' / 'helsinki'):
    from ultralytics import SAM

    model = SAM(str(ROOT / '.training' / 'models' / 'sam2.1_b.pt'))
    found = [[] for _ in OBJECT_CLASSES]
    for annotation_path in sorted((scene / 'annotations').glob('*.json')):
        image = cv2.imread(str(scene / 'images' / (annotation_path.stem + '.png')))
        for item in json.loads(annotation_path.read_text())['annotations']:
            box = [int(round(value)) for value in item['bbox']]
            if box[0] <= 1 or box[1] <= 1 or box[2] >= WIDTH - 1 or box[3] >= HEIGHT - 1:
                continue
            segmented = _segment(model, image, box, item['object_id'] in POINT_ONLY)
            if segmented is None:
                continue
            crop, mask, local = segmented
            coverage = mask.sum() / ((box[2] - box[0]) * (box[3] - box[1]))
            found[OBJECT_CLASSES.index(item['object_id'])].append({
                'image': crop.copy(), 'alpha': mask.astype(np.float32), 'box': np.array(local, np.float32),
                'coverage': float(coverage), 'frame': annotation_path.stem,
            })
    sprites = []
    for label, items in enumerate(found):
        median = float(np.median([item['coverage'] for item in items]))
        kept = [item for item in items if abs(item['coverage'] - median) <= 0.35 * median]
        for item in kept:
            item['label'] = label
        sprites.append(kept)
        print(f'{OBJECT_CLASSES[label]:16s} kept {len(kept)}/{len(items)}  median coverage {median:.2f}', flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(sprites))
    return sprites


def load():
    """Per-class lists of dicts with ``image`` (BGR crop), ``alpha`` (0..1), ``box`` (annotated box in the crop)."""
    if not CACHE.is_file():
        return build()
    return pickle.loads(CACHE.read_bytes())


if __name__ == '__main__':
    build()


def _mask_box(alpha):
    ys, xs = np.nonzero(alpha > 0.5)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def render(sprite, angle, scale):
    """Rotate and scale a cut-out. Returns premultiplied BGR (float32), alpha, and the label box.

    The annotated boxes are not tight to the visible pixels (medium_launcher's is almost twice
    its visible size), so a label cannot come from the mask alone, and re-hulling a rotated
    axis-aligned box inflates it by up to 40%. The label is the rotated mask box, scaled per
    axis by the annotated/mask ratio of the original, with the ratios swapped in proportion
    to sin^2 of the rotation, and the annotation's centre offset rotated with the object.
    At zero rotation and unit scale this reproduces the annotated box exactly.
    """
    height, width = sprite['image'].shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(angle), float(scale))
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32) @ matrix[:, :2].T + matrix[:, 2]
    matrix[:, 2] -= np.floor(corners.min(axis=0))
    size = tuple(int(value) for value in np.ceil(corners.max(axis=0) - np.floor(corners.min(axis=0))) + 1)
    alpha = cv2.warpAffine(sprite['alpha'], matrix, size, flags=cv2.INTER_LINEAR)
    rgb = cv2.warpAffine(sprite['image'].astype(np.float32) * sprite['alpha'][..., None], matrix, size, flags=cv2.INTER_LINEAR)
    if (alpha > 0.5).sum() < 2:
        return None
    original, annotated = _mask_box(sprite['alpha']), sprite['box']
    ratio_x = (annotated[2] - annotated[0]) / (original[2] - original[0])
    ratio_y = (annotated[3] - annotated[1]) / (original[3] - original[1])
    offset = (annotated[:2] + annotated[2:]) / 2 - (original[:2] + original[2:]) / 2
    turned = _mask_box(alpha)
    cos2 = np.cos(np.deg2rad(angle)) ** 2
    factor = np.array([ratio_x * cos2 + ratio_y * (1 - cos2), ratio_y * cos2 + ratio_x * (1 - cos2)], np.float32)
    center = (turned[:2] + turned[2:]) / 2 + matrix[:, :2] @ offset
    half = (turned[2:] - turned[:2]) * factor / 2
    return rgb, alpha, np.concatenate([center - half, center + half]).astype(np.float32)
