"""How often does the detector see something where there is nothing?

The validation attempt answered with 20 detections on its first frames and 80
on its last: false tracks accumulating. A track is only as good as the
detections that feed it, so this measures the detector's false alarm rate
directly, on terrain with every annotated object painted out.

The interesting number is not the rate on the reference terrain -- the model
trained on it -- but the rate on terrain it has never seen, which is what the
``--backgrounds`` directory is for.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'training'))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from dtos import OBJECT_CLASSES                                  # noqa: E402


def views_from(image: np.ndarray, level: int, rng, count: int):
    """Cut random level-sized regions and render them the way the evaluator does."""
    region = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}[level]
    height, width = image.shape[:2]
    if width < region[0] or height < region[1]:
        scale = max(region[0] / width, region[1] / height)
        image = cv2.resize(
            image, (int(width * scale) + 1, int(height * scale) + 1),
            interpolation=cv2.INTER_LINEAR,
        )
        height, width = image.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, width - region[0] + 1))
        y = int(rng.integers(0, height - region[1] + 1))
        crop = image[y:y + region[1], x:x + region[0]]
        if (crop.shape[1], crop.shape[0]) != (960, 540):
            crop = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
        yield crop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backgrounds', default=str(PROJECT / 'dataset3' / 'backgrounds'),
                        help='Directory of object-free images.')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--views', type=int, default=8, help='Random views per image.')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--shift', default=None, help='hue,saturation,value,gamma')
    arguments = parser.parse_args()

    from eval_levels import recolour
    from solution import config
    from solution.detector import Detector

    detector = Detector(
        weights_path=arguments.weights or config.WEIGHTS_PATH, confidence=arguments.conf
    )
    shift = [float(v) for v in arguments.shift.split(',')] if arguments.shift else None

    paths = sorted(
        p for p in Path(arguments.backgrounds).iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}
    )
    if not paths:
        print(f'no images in {arguments.backgrounds}', file=sys.stderr)
        return 1

    rng = np.random.default_rng(0)
    thresholds = (0.05, 0.10, 0.20, 0.30, 0.50)
    counts = {threshold: 0 for threshold in thresholds}
    by_class = Counter()
    views = 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if shift is not None:
            image = recolour(image, *shift)
        for view in views_from(image, arguments.level, rng, arguments.views):
            views += 1
            for detection in detector.detect(view, (0, 0, 3840, 2160)):
                for threshold in thresholds:
                    if detection.score >= threshold:
                        counts[threshold] += 1
                if detection.score >= 0.20:
                    by_class[OBJECT_CLASSES[detection.class_index]] += 1

    print(f'{views} empty views from {len(paths)} images, level {arguments.level}')
    for threshold in thresholds:
        print(f'  score >= {threshold:.2f}: {counts[threshold] / views:6.2f} false alarms per view')
    if by_class:
        print('  worst classes at 0.20:', ', '.join(
            f'{name} {count}' for name, count in by_class.most_common(6)
        ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
