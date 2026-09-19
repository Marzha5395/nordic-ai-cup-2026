import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from detector import overlap
from dtos import OBJECT_CLASSES
from utils import load_annotations, load_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=Path('.training/errors.jpg'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    scene = report['settings']['scene']
    statistics = {name: Counter() for name in OBJECT_CLASSES}
    examples = defaultdict(list)
    for frame_text, predictions in report['predictions'].items():
        frame = int(frame_text)
        for target in load_annotations(frame, scene):
            name = target['object_id']
            box = np.array(target['bbox'], dtype=np.float32)
            statistics[name]['targets'] += 1
            ious = overlap(box, [item['bbox'] for item in predictions])
            best = predictions[int(ious.argmax())] if len(ious) else None
            iou = float(ious.max()) if len(ious) else 0
            if iou >= 0.5:
                statistics[name]['localized'] += 1
                statistics[name]['correct' if best['object_id'] == name else 'wrong_class'] += 1
                if best['object_id'] != name:
                    statistics[name]['confused_with_' + best['object_id']] += 1
            if min(box[:2]) > 1 and box[2] < 3839 and box[3] < 2159:
                examples[name].append((frame, box, best, iou))
    failures = sorted(OBJECT_CLASSES, key=lambda name: report['ap_by_class'].get(name, 0))[:6]
    canvas = np.full((6 * 240, 3 * 320, 3), 25, dtype=np.uint8)
    for row, name in enumerate(failures):
        samples = examples[name]
        if not samples:
            continue
        for column, index in enumerate(np.linspace(0, len(samples) - 1, 3).astype(int)):
            frame, box, best, iou = samples[index]
            image = load_frame(frame, scene)
            padding = max(12, round(max(box[2:] - box[:2]) * 0.4))
            x1, y1 = np.maximum(box[:2].astype(int) - padding, 0)
            x2, y2 = np.minimum(box[2:].astype(int) + padding, [3840, 2160])
            crop = image[y1:y2, x1:x2].copy()
            scale = min(300 / crop.shape[1], 180 / crop.shape[0])
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            truth = ((box - [x1, y1, x1, y1]) * scale).astype(int)
            cv2.rectangle(crop, tuple(truth[:2]), tuple(truth[2:]), (0, 255, 0), 1)
            label = 'no overlapping detection'
            if best and iou > 0:
                predicted = ((np.array(best['bbox']) - [x1, y1, x1, y1]) * scale).astype(int)
                cv2.rectangle(crop, tuple(predicted[:2]), tuple(predicted[2:]), (0, 0, 255), 1)
                label = f'{best["object_id"]} {best["confidence"]:.2f} IoU {iou:.2f}'
            x, y = column * 320, row * 240
            canvas[y + 40:y + 40 + crop.shape[0], x + 10:x + 10 + crop.shape[1]] = crop
            cv2.putText(canvas, f'{name} f{frame}', (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(canvas, label, (x + 5, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(json.dumps(statistics, indent=2))


if __name__ == '__main__':
    main()
