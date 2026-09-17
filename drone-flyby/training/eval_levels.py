"""Score the detector alone, at each resolution level, on the real frames.

This answers the question the camera policy depends on: how much accuracy does
a level actually buy? It tiles every source frame with non-overlapping views of
the level's size, renders each one exactly as the evaluator does, runs the
detector, lifts the boxes back into source pixels and scores the result against
the real ground truth.

It is an upper bound rather than a score -- no real policy sees the whole frame
at level 1 or 2 in a single step -- but the gap between levels is real, and so
is the per-class breakdown.
"""

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402
from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE  # noqa: E402
from local_evaluator import score                                # noqa: E402
from utils import frame_numbers, load_frame                      # noqa: E402


def tile_centres(level: int):
    width, height = SOURCE_REGION_SIZES[level]
    columns = 3840 // width
    rows = 2160 // height
    return [
        (width // 2 + column * width, height // 2 + row * height)
        for row in range(rows)
        for column in range(columns)
    ]


def recolour(image, hue, saturation, value, gamma):
    """Push a frame's colour around without touching its geometry."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * value, 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(shifted, table)


def _iou(first, second) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_a + area_b - intersection)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument(
        '--shift', default=None,
        help='Recolour the frames before rendering, as "hue,saturation,value,'
             'gamma" -- e.g. "25,0.7,1.15,0.85". The evaluation flight is over '
             'different ground under different light, and there is no held-out '
             'terrain to measure that with, so this is the closest proxy: the '
             'same objects, the same geometry, a scene that does not look the '
             'same.',
    )
    arguments = parser.parse_args()

    shift = None
    if arguments.shift:
        values = [float(v) for v in arguments.shift.split(',')]
        assert len(values) == 4, 'expected hue,saturation,value,gamma'
        shift = values

    from solution.detector import Detector
    from solution import config

    detector = Detector(
        weights_path=arguments.weights or config.WEIGHTS_PATH,
        confidence=arguments.conf,
    )

    frames = frame_numbers(arguments.scene)
    for level in (int(value) for value in arguments.levels.split(',')):
        centres = tile_centres(level)
        predictions = {}
        for frame in frames:
            image = load_frame(frame, arguments.scene)
            if shift is not None:
                image = recolour(image, *shift)
            found = []
            for centre_x, centre_y in centres:
                width, height = SOURCE_REGION_SIZES[level]
                x1, y1 = centre_x - width // 2, centre_y - height // 2
                crop = image[y1:y1 + height, x1:x1 + width]
                if (crop.shape[1], crop.shape[0]) != TRANSMITTED_VIEW_SIZE:
                    crop = cv2.resize(crop, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
                found.extend(
                    detector.detect(crop, (x1, y1, x1 + width, y1 + height))
                )
            # Tiles abut rather than overlap, but an object on a seam is found
            # twice, so suppress within a class before scoring.
            found.sort(key=lambda detection: -detection.score)
            kept = []
            for detection in found:
                if any(
                    other.class_index == detection.class_index
                    and _iou(other.box, detection.box) > 0.45
                    for other in kept
                ):
                    continue
                kept.append(detection)
            predictions[frame] = [
                {
                    'object_id': OBJECT_CLASSES[detection.class_index],
                    'bbox': detection.box,
                    'confidence': detection.score,
                }
                for detection in kept
            ]
        mean_average_precision, by_class = score(arguments.scene, predictions)
        print()
        print(f'=== level {level} ===')
        for name, value in sorted(by_class.items(), key=lambda item: -item[1]):
            print(f'  {name:16s} {value:.3f}')
        print(f'  mAP@0.50 {mean_average_precision:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
