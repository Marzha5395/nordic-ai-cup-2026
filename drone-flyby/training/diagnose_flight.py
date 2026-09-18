"""Where does a replayed scene lose its points?

mAP folds four different failures into one number. This replays a scene the
way ``offline_eval.py`` does and sorts every answered box into one of them:

* ``hit``        right class, IoU >= 0.5 with an unmatched object;
* ``loose``      right class, overlapping an object but under IoU 0.5 --
                 the tracker or the box size is off;
* ``wrong``      IoU >= 0.5 with an object of another class;
* ``phantom``    near no object at all -- a false alarm.

and counts the objects nobody answered for. Each is reported above a few
confidence levels, because a phantom at 0.05 costs far less than one at 0.6.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'training'))

from local_evaluator import score                                # noqa: E402
from offline_eval import replay                                   # noqa: E402
from utils import frame_numbers, load_annotations                 # noqa: E402


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify(predictions, truth, threshold):
    kinds = Counter()
    phantom_classes = Counter()
    matched = set()
    for p in sorted(predictions, key=lambda item: -item['confidence']):
        if p['confidence'] < threshold:
            continue
        best, best_index = 0.0, None
        for index, t in enumerate(truth):
            if t['object_id'] != p['object_id'] or index in matched:
                continue
            value = iou(p['bbox'], t['bbox'])
            if value > best:
                best, best_index = value, index
        if best >= 0.5:
            matched.add(best_index)
            kinds['hit'] += 1
        elif best > 0.05:
            kinds['loose'] += 1
        elif any(iou(p['bbox'], t['bbox']) >= 0.5 for t in truth):
            kinds['wrong'] += 1
        else:
            kinds['phantom'] += 1
            phantom_classes[p['object_id']] += 1
    kinds['missed'] = len(truth) - len(matched)
    return kinds, phantom_classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--weights', default=None,
                        help='Detector checkpoint to use instead of config.WEIGHTS_PATH.')
    arguments = parser.parse_args()
    if arguments.weights:
        from solution import config
        config.WEIGHTS_PATH = Path(arguments.weights)

    predictions, _, _, _ = replay(arguments.scene)
    mean_average_precision, by_class = score(arguments.scene, predictions)
    frames = frame_numbers(arguments.scene)
    truth_total = sum(len(load_annotations(f, arguments.scene)) for f in frames)
    print(f'{arguments.scene}: {len(frames)} frames, {truth_total} object-frames, '
          f'mAP@0.50 {mean_average_precision:.3f}')
    for threshold in (0.05, 0.2, 0.4, 0.6):
        totals, phantoms = Counter(), Counter()
        for frame in frames:
            kinds, phantom_classes = classify(
                predictions.get(frame, []), load_annotations(frame, arguments.scene), threshold
            )
            totals.update(kinds)
            phantoms.update(phantom_classes)
        print(f'  conf >= {threshold:.2f}: ' + '  '.join(
            f'{key} {totals[key] / len(frames):5.2f}'
            for key in ('hit', 'loose', 'wrong', 'phantom', 'missed')
        ) + '   (per frame)')
        if threshold == 0.2 and phantoms:
            print('      phantoms by class: ' + ', '.join(
                f'{name} {count}' for name, count in phantoms.most_common(6)))
    worst = sorted(by_class.items(), key=lambda item: item[1])[:5]
    print('  weakest classes: ' + ', '.join(f'{n} {v:.2f}' for n, v in worst))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
