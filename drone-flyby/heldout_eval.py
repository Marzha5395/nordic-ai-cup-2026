"""Measure a detector on ground it was never trained on.

The reference scene cannot measure generalization: the detector is trained on
it. This builds a fixed set of 3840x2160 source frames from the held-out
terrain places (``.training/terrain/test``, never used for training), renders
level 0/1/2 views exactly like the evaluator (crop, then INTER_AREA down to
960x540), runs the real ONNX inference path in ``detector.py``, and reports:

  * false alarms per view on clean held-out terrain, per level;
  * mAP@0.5 per level on held-out terrain with the objects pasted in at random
    positions and rotations, labelled by ``sprites.render`` (the object cut-outs
    are the reference scene's own, so this recall is optimistic; the false
    alarms are not).

The set is generated once with a fixed seed and cached, so every model is
measured on identical views.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import sprites
from dtos import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parent
SOURCE_WIDTH, SOURCE_HEIGHT = 3840, 2160
VIEW_SIZE = (960, 540)
# Ground sample distance of the reference scene: ~13.9 m flown per frame shows up as ~65 source px of drift.
SOURCE_GSD = 0.21


def source_frame(tile_path, rng):
    tile = cv2.imread(str(tile_path))
    gsd = json.loads((tile_path.parent.parent / 'gsd.json').read_text())[f'{tile_path.parent.name}/{tile_path.name}']
    scale = gsd / SOURCE_GSD
    tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pad_x, pad_y = max(0, SOURCE_WIDTH - tile.shape[1]), max(0, SOURCE_HEIGHT - tile.shape[0])
    if pad_x or pad_y:
        tile = cv2.copyMakeBorder(tile, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2, cv2.BORDER_REFLECT)
    y = int(rng.integers(tile.shape[0] - SOURCE_HEIGHT + 1))
    x = int(rng.integers(tile.shape[1] - SOURCE_WIDTH + 1))
    return tile[y:y + SOURCE_HEIGHT, x:x + SOURCE_WIDTH].copy()


def paste(frame, objects, rng, style='alpha'):
    placed = []
    for label in rng.permutation(len(OBJECT_CLASSES)):
        sprite = objects[label][int(rng.integers(len(objects[label])))]
        rendered = sprites.render(sprite, float(rng.uniform(0, 360)), float(rng.uniform(0.85, 1.15)))
        if rendered is None:
            continue
        rgb, mask, local = rendered
        height, width = mask.shape
        for _ in range(50):
            x = int(rng.integers(40, SOURCE_WIDTH - width - 40))
            y = int(rng.integers(40, SOURCE_HEIGHT - height - 40))
            box = local + [x, y, x, y]
            if all(min(box[2], other[2]) <= max(box[0], other[0]) - 30 or min(box[3], other[3]) <= max(box[1], other[1]) - 30 for other, _ in placed):
                break
        else:
            continue
        region = frame[y:y + height, x:x + width].astype(np.float32)
        gain = rng.uniform(0.85, 1.15)
        plain = (rgb * gain + region * (1 - mask[..., None])).clip(0, 255)
        if style == 'poisson':
            # As synthetic_flight.py --style poisson: half re-lit to the surroundings, half plain.
            source = np.where(mask[..., None] > 0.01, rgb / np.maximum(mask[..., None], 1e-3), region).clip(0, 255).astype(np.uint8)
            binary = cv2.dilate((mask > 0.5).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
            cloned = cv2.seamlessClone(source, frame, binary, (x + width // 2, y + height // 2), cv2.NORMAL_CLONE)[y:y + height, x:x + width]
            plain = 0.5 * plain + 0.5 * cloned.astype(np.float32)
        frame[y:y + height, x:x + width] = plain.astype(np.uint8)
        placed.append((box, int(label)))
    return placed


def view_regions(rng, placed):
    regions = [(0, (0, 0, SOURCE_WIDTH, SOURCE_HEIGHT))]
    for level, count in ((1, 4), (2, 8)):
        width, height = SOURCE_WIDTH >> level, SOURCE_HEIGHT >> level
        for index in range(count):
            if placed and index % 2 == 0:
                # Half the zoomed views are aimed near an object, so there is something to find.
                box = placed[int(rng.integers(len(placed)))][0]
                center = (box[:2] + box[2:]) / 2 + rng.uniform(-0.3, 0.3, 2) * [width, height]
            else:
                center = rng.uniform([0, 0], [SOURCE_WIDTH, SOURCE_HEIGHT])
            cx = int(np.clip(center[0], width / 2, SOURCE_WIDTH - width / 2))
            cy = int(np.clip(center[1], height / 2, SOURCE_HEIGHT - height / 2))
            regions.append((level, (cx - width // 2, cy - height // 2, cx - width // 2 + width, cy - height // 2 + height)))
    return regions


def build(output, seed, style='alpha', places=None):
    rng = np.random.default_rng(seed)
    objects = sprites.load()
    missing = [name for name, found in zip(OBJECT_CLASSES, objects) if not found]
    if missing:
        raise ValueError(f'No clean cut-out for {missing}')
    tiles = sorted((ROOT / '.training' / 'terrain' / 'test').glob('*.jpg'))
    if places:
        tiles = [tile for tile in tiles if tile.stem.rsplit('_', 1)[0] in places]
    views = []
    output.mkdir(parents=True, exist_ok=True)
    for index, tile in enumerate(tiles):
        for kind in ('clean', 'objects'):
            frame = source_frame(tile, rng)
            placed = paste(frame, objects, rng, style) if kind == 'objects' else []
            if style == 'poisson':
                # A different grade per frame: gamma and colour cast, as another renderer might produce.
                frame = (255 * (frame / 255.0) ** rng.uniform(0.8, 1.25) * rng.uniform(0.9, 1.1, 3)).clip(0, 255).astype(np.uint8)
            for view_index, (level, region) in enumerate(view_regions(rng, placed)):
                x1, y1, x2, y2 = region
                image = cv2.resize(frame[y1:y2, x1:x2], VIEW_SIZE, interpolation=cv2.INTER_AREA) if level else cv2.resize(frame, VIEW_SIZE, interpolation=cv2.INTER_AREA)
                name = f'{tile.stem}_{kind}_{view_index:02d}.png'
                cv2.imwrite(str(output / name), image)
                truth, ignore = [], []
                for box, label in placed:
                    clipped = np.array([max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2)])
                    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                        continue
                    visible = np.prod(clipped[2:] - clipped[:2]) / np.prod(box[2:] - box[:2])
                    (truth if visible >= 0.5 else ignore).append({'bbox': [float(v) for v in clipped], 'label': label})
                views.append({'file': name, 'kind': kind, 'level': level, 'region': list(region), 'truth': truth, 'ignore': ignore, 'place': tile.stem.rsplit('_', 1)[0]})
        print(f'  built {index + 1}/{len(tiles)}', flush=True)
    (output / 'views.json').write_text(json.dumps(views))


def overlap(box, boxes):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    inter = np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0).prod(axis=1)
    return inter / np.maximum(np.prod(box[2:] - box[:2]) + np.prod(boxes[:, 2:] - boxes[:, :2], axis=1) - inter, 1e-6)


def mean_ap(views, detections):
    from faster_coco_eval import COCO, COCOeval_faster

    images, annotations, predictions = [], [], []
    for image_id, (view, found) in enumerate(zip(views, detections), 1):
        images.append({'id': image_id, 'width': SOURCE_WIDTH, 'height': SOURCE_HEIGHT, 'file_name': view['file']})
        for item in view['truth']:
            x1, y1, x2, y2 = item['bbox']
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': item['label'] + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
        ignored = [item['bbox'] for item in view['ignore']]
        for box, label, score in found:
            if ignored and overlap(np.asarray(box), ignored).max() > 0.3:
                continue
            predictions.append({'image_id': image_id, 'category_id': label + 1, 'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], 'score': score})
    present = sorted({item['category_id'] for item in annotations})
    if not predictions:
        return 0.0, {}
    ground_truth = COCO({'images': images, 'annotations': annotations, 'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(OBJECT_CLASSES)]})
    evaluator = COCOeval_faster(ground_truth, ground_truth.loadRes(predictions), 'bbox')
    evaluator.params.imgIds = [image['id'] for image in images]
    evaluator.params.catIds = present
    evaluator.params.iouThrs = np.array([0.5])
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision']
    per_class = {}
    for index, category in enumerate(present):
        values = precision[0, :, index, 0, -1]
        values = values[values > -1]
        per_class[OBJECT_CLASSES[category - 1]] = float(values.mean()) if values.size else 0.0
    return float(np.mean(list(per_class.values()))), per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT / 'weights' / 'detector.onnx')
    parser.add_argument('--set', type=Path, default=ROOT / '.training' / 'heldout')
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--confidence', type=float, default=0.12, help='Detector score floor, as served.')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--style', choices=['alpha', 'poisson'], default='alpha', help='Compositing of the set when it is first built')
    parser.add_argument('--places', help='Comma-separated held-out places to build the set from (default: all)')
    parser.add_argument('--torch-width', type=int, help='Run a .pt model on the GPU at this native input width (no canonical rescaling)')
    parser.add_argument('--json', type=Path, help='Write the numbers here as well.')
    args = parser.parse_args()
    if not (args.set / 'views.json').is_file():
        print(f'Building the held-out set in {args.set}')
        build(args.set, args.seed, args.style, args.places.split(',') if args.places else None)
    views = json.loads((args.set / 'views.json').read_text())

    import os
    os.environ['DRONE_CONFIDENCE'] = str(args.confidence)
    from detector import Detector
    if args.torch_width:
        from gpu_detector import TorchDetector
        detector = TorchDetector(args.model, width=args.torch_width, threshold=args.confidence, canonical=False)
    else:
        detector = Detector(args.model)
    detections = []
    for view in views:
        image = cv2.imread(str(args.set / view['file']))
        request = SimpleNamespace(view=SimpleNamespace(source_region_xyxy=view['region']))
        detections.append([(item.box.tolist(), int(item.scores.argmax()), item.confidence) for item in detector.detect(image, request)])

    report = {'model': str(args.model), 'views': len(views)}
    print(f'{args.model}  ({len(views)} views over {len({v["place"] for v in views})} held-out places)')
    for level in (0, 1, 2):
        clean = [found for view, found in zip(views, detections) if view['kind'] == 'clean' and view['level'] == level]
        alarms = {threshold: float(np.mean([sum(score >= threshold for _, _, score in found) for found in clean])) for threshold in (0.25, 0.5)}
        chosen = [(view, found) for view, found in zip(views, detections) if view['kind'] == 'objects' and view['level'] == level]
        score, per_class = mean_ap([view for view, _ in chosen], [found for _, found in chosen])
        report[f'level{level}'] = {'false_alarms_per_clean_view@0.25': alarms[0.25], 'false_alarms_per_clean_view@0.5': alarms[0.5], 'mAP50_with_objects': score, 'per_class': per_class}
        print(f'  level {level}: false alarms/clean view  {alarms[0.25]:6.2f} (>=0.25)  {alarms[0.5]:6.2f} (>=0.5)   mAP50 with objects {score:.3f}')
        if args.per_class:
            print('           ' + '  '.join(f'{name} {value:.2f}' for name, value in per_class.items()))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
