"""Extract per-instance object masks from the supplied scene.

The reference scene gives one instance of each class, seen from a slightly
different angle in every frame it appears in. We need masks for two reasons:

* a rotated frame needs a tight box, and the axis-aligned hull of a rotated
  box is far too loose for small objects scored at IoU 0.50;
* copy-paste augmentation needs an alpha channel or it pastes a visible
  rectangle of foreign terrain.

GrabCut, seeded from the ground-truth box, does the work. The cleanup
afterwards matters more than the segmentation itself: keep the components that
touch the middle of the box, close the gaps that thin structures (rotor
blades, wings) leave behind, and never let the mask spill past the box.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCENE_ROOT = Path(__file__).resolve().parent.parent / 'src'


def _fill_holes(mask: np.ndarray, maximum_hole_fraction: float = 0.06) -> np.ndarray:
    """Fill small interior holes of a binary mask, and only small ones.

    A dark window on a tank is a hole worth filling. The triangle of grass
    between an aircraft's wing and its tail is not: filling it pastes a patch
    of the reference scene's terrain onto whatever background the object lands
    on, and it stretches the derived box out to cover that patch as well.
    """
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]

    limit = maximum_hole_fraction * float(mask.shape[0] * mask.shape[1])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    small = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] <= limit:
            small[labels == index] = 255
    return cv2.bitwise_or(mask, small)


def extract_mask(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = 14,
    iterations: int = 6,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """Segment one object out of a frame.

    Returns the padded BGR patch, its uint8 mask, and the ground-truth box in
    patch coordinates. ``None`` when nothing usable survives.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(width, x2 + pad), min(height, y2 + pad)
    patch = image[py1:py2, px1:px2].copy()
    if patch.size == 0:
        return None

    bx1, by1, bx2, by2 = x1 - px1, y1 - py1, x2 - px1, y2 - py1
    box_w, box_h = bx2 - bx1, by2 - by1
    if box_w < 3 or box_h < 3:
        return None

    state = np.full(patch.shape[:2], cv2.GC_BGD, np.uint8)
    # The margin around the box is background we are sure about only at its
    # outer edge; the ring just outside the box may still hold a shadow.
    state[max(0, by1 - 2):by2 + 2, max(0, bx1 - 2):bx2 + 2] = cv2.GC_PR_BGD
    state[by1:by2, bx1:bx2] = cv2.GC_PR_FGD
    # A small central seed: every one of these assets covers its own centre.
    sx = max(1, int(0.30 * box_w))
    sy = max(1, int(0.30 * box_h))
    state[by1 + sy:max(by1 + sy + 1, by2 - sy), bx1 + sx:max(bx1 + sx + 1, bx2 - sx)] = cv2.GC_FGD

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch, state, None, background_model, foreground_model,
                    iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    mask = np.where((state == cv2.GC_FGD) | (state == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Nothing outside the annotated box belongs to the object.
    outside = np.ones_like(mask)
    outside[by1:by2, bx1:bx2] = 0
    mask[outside.astype(bool)] = 0

    # Close the gaps thin structures leave, then fill what is enclosed.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _fill_holes(mask)

    # Drop specks that are not part of the object: keep components that reach
    # the middle third of the box, or that are large in their own right.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    centre = np.zeros_like(mask)
    cv2.rectangle(
        centre,
        (bx1 + box_w // 3, by1 + box_h // 3),
        (bx2 - box_w // 3, by2 - box_h // 3),
        255,
        -1,
    )
    largest = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    keep = np.zeros_like(mask)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        component = labels == index
        touches_centre = bool(np.any(component & (centre > 0)))
        if index == largest or touches_centre or area > 0.25 * stats[largest, cv2.CC_STAT_AREA]:
            keep[component] = 255
    mask = keep

    if mask.sum() < 255 * 0.05 * box_w * box_h:
        return None
    return patch, mask, (bx1, by1, bx2, by2)


def load_instances(scene: str = 'helsinki') -> List[Dict]:
    """Return every unclipped annotated instance of the scene, with a mask."""
    directory = SCENE_ROOT / scene
    instances: List[Dict] = []
    for annotation_path in sorted((directory / 'annotations').glob('frame_*.json')):
        payload = json.loads(annotation_path.read_text())
        frame = payload['frame']
        image_path = directory / 'images' / f'frame_{frame:06d}.png'
        image = None
        for annotation in payload['annotations']:
            x1, y1, x2, y2 = annotation['bbox']
            if x1 <= 0 or y1 <= 0 or x2 >= 3840 or y2 >= 2160:
                continue  # clipped: the box does not describe the whole object
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            if image is None:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    break
            result = extract_mask(image, (x1, y1, x2, y2))
            if result is None:
                continue
            patch, mask, box = result
            instances.append(
                {
                    'object_id': annotation['object_id'],
                    'frame': frame,
                    'patch': patch,
                    'mask': mask,
                    'box_in_patch': box,
                }
            )
    return instances
