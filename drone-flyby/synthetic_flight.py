"""Whole flights over held-out ground, in the reference scene's layout.

``local_evaluator.py --scene heldout_flight_N`` replays one through the real endpoint,
tracker, camera policy and scorer. The ground is a mosaic of one held-out place's
tiles (``.training/terrain/test``, never trained on) at the source scale; the
objects are the reference cut-outs, placed by ``sprites.render`` at random
positions and rotations. The camera drifts over the ground at the reference
scene's ~65 source px per frame, so objects enter at the top and leave at the
bottom, as they do in the reference flight.

The objects are the same cut-outs the detector is trained on, so recall here is
optimistic; false alarms and everything downstream of the detector are real.
These scenes are for measurement only and are never trained on.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sprites
from dtos import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 3840, 2160
SOURCE_GSD = 0.21
STEP = 65


def ground(place, height, rng):
    scales = json.loads((ROOT / '.training' / 'terrain' / 'gsd.json').read_text())
    tiles = []
    for path in sorted((ROOT / '.training' / 'terrain' / 'test').glob(f'{place}_*.jpg')):
        image = cv2.imread(str(path))
        factor = scales[f'test/{path.name}'] / SOURCE_GSD
        tiles.append(cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC))
    # A grid of fully filled cells, each a random crop of a random tile of the place (every scaled tile is >= 1170 px).
    cell = 1024
    canvas = np.zeros((height, WIDTH, 3), np.uint8)
    for top in range(0, height, cell):
        for left in range(0, WIDTH, cell):
            tile = np.rot90(tiles[int(rng.integers(len(tiles)))], int(rng.integers(4)))
            y = int(rng.integers(tile.shape[0] - cell + 1))
            x = int(rng.integers(tile.shape[1] - cell + 1))
            piece = tile[y:y + cell, x:x + cell]
            h, w = min(cell, height - top), min(cell, WIDTH - left)
            canvas[top:top + h, left:left + w] = piece[:h, :w]
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--places', default='ee_ida,nl_zeeland,us_maine,ee_voru')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--density', type=float, default=1.5, help='Instances per class per flight, on average.')
    parser.add_argument('--seed', type=int, default=515)
    parser.add_argument('--prefix', default='heldout_flight')
    parser.add_argument('--style', choices=['alpha', 'poisson'], default='alpha',
                        help='poisson: seamlessClone the objects (re-lit to their surroundings) and give each flight its own '
                             'colour cast and gamma, so the check does not share the training compositing')
    args = parser.parse_args()
    objects = sprites.load()
    for index, place in enumerate(args.places.split(',')):
        rng = np.random.default_rng(args.seed + index)
        height = HEIGHT + STEP * (args.frames - 1)
        canvas = ground(place, height, rng)
        placed = []
        for label in range(len(OBJECT_CLASSES)):
            for _ in range(max(1, int(rng.poisson(args.density)))):
                sprite = objects[label][int(rng.integers(len(objects[label])))]
                rendered = sprites.render(sprite, float(rng.uniform(0, 360)), float(rng.uniform(0.9, 1.1)))
                if rendered is None:
                    continue
                rgb, alpha, local = rendered
                h, w = alpha.shape
                for _ in range(100):
                    x, y = int(rng.integers(0, WIDTH - w)), int(rng.integers(0, height - h))
                    box = local + [x, y, x, y]
                    if all(min(box[2], o[2]) <= max(box[0], o[0]) - 40 or min(box[3], o[3]) <= max(box[1], o[1]) - 40 for o, _ in placed):
                        break
                else:
                    continue
                if args.style == 'poisson':
                    if x < 2 or y < 2 or x + w > WIDTH - 2 or y + h > height - 2:
                        continue
                    source = np.where(alpha[..., None] > 0.01, rgb / np.maximum(alpha[..., None], 1e-3), canvas[y:y + h, x:x + w]).clip(0, 255).astype(np.uint8)
                    mask = cv2.dilate((alpha > 0.5).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
                    # Half Poisson (re-lit to the surroundings), half plain alpha: full Poisson recolours far more than any renderer.
                    plain = (rgb + canvas[y:y + h, x:x + w].astype(np.float32) * (1 - alpha[..., None])).clip(0, 255)
                    cloned = cv2.seamlessClone(source, canvas, mask, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)[y:y + h, x:x + w].astype(np.float32)
                    canvas[y:y + h, x:x + w] = (0.5 * plain + 0.5 * cloned).astype(np.uint8)
                else:
                    region = canvas[y:y + h, x:x + w].astype(np.float32)
                    canvas[y:y + h, x:x + w] = (rgb * rng.uniform(0.9, 1.1) + region * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
                placed.append((box, label))
        if args.style == 'poisson':
            gamma = float(rng.uniform(0.8, 1.25))
            cast = rng.uniform(0.9, 1.1, 3)
            canvas = (255 * (canvas / 255.0) ** gamma * cast).clip(0, 255).astype(np.uint8)
        scene = ROOT / 'src' / f'{args.prefix}_{index}'
        (scene / 'images').mkdir(parents=True, exist_ok=True)
        (scene / 'annotations').mkdir(parents=True, exist_ok=True)
        totals = {}
        for frame in range(args.frames):
            top = height - HEIGHT - STEP * frame
            cv2.imwrite(str(scene / 'images' / f'frame_{frame:06d}.png'), canvas[top:top + HEIGHT], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            annotations = []
            for box, label in placed:
                x1, y1, x2, y2 = box - [0, top, 0, top]
                cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, WIDTH), min(y2, HEIGHT)
                if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                    continue
                annotations.append({'object_id': OBJECT_CLASSES[label], 'bbox': [int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))]})
            counts = {}
            for item in annotations:
                counts[item['object_id']] = counts.get(item['object_id'], 0) + 1
            (scene / 'annotations' / f'frame_{frame:06d}.json').write_text(json.dumps({'frame': frame, 'annotations': annotations, 'object_counts': counts}, indent=1))
        for _, label in placed:
            totals[OBJECT_CLASSES[label]] = totals.get(OBJECT_CLASSES[label], 0) + 1
        (scene / 'run_metadata.json').write_text(json.dumps({'capture': {'altitude_m': 600, 'num_frames': args.frames, 'synthetic_from': place}, 'total_objects': len(placed), 'object_totals': totals}, indent=1))
        print(f'{scene.name}: {place}, {len(placed)} objects, {args.frames} frames')


if __name__ == '__main__':
    main()
