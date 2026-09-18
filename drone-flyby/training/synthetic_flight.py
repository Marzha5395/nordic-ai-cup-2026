"""Build whole flights over terrain the detector has never seen.

The reference scene cannot say how the solution does anywhere else: the
detector was trained on those 25 frames, so replaying them measures memory.
The validation attempt made the point -- about 0.85 locally, 0.22 there.

This writes scenes in the same layout as ``src/helsinki`` (frames,
annotations, run metadata), so ``local_evaluator.py --scene`` and
``training/offline_eval.py --scene`` replay them unchanged. Each flight is
built from the held-out mosaics in ``external/test`` only:

* the ground is a canvas tiled from several held-out mosaics, with the
  object cut-outs pasted onto it at random positions and orientations;
* each frame looks at that ground through a camera that drifts along the
  flight line and slowly zooms, so the motion model faces an affine field
  and not a translation;
* annotations are the pasted boxes carried through the same transform and
  clipped to the frame, the way the supplied ground truth clips them.

What it cannot do is show objects the detector has never seen: the cut-outs
are the reference scene's own. So recall here is still optimistic, but every
false alarm is honest, and false alarms are what failed validation.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from compose import (                                             # noqa: E402
    _intersection_over_area, build_sprites, modelled_hull, paste_sprite,
)
from dtos import OBJECT_CLASSES                                   # noqa: E402

WIDTH, HEIGHT = 3840, 2160


def build_ground(rng, mosaics, width, height):
    """Tile randomly chosen mosaics into a canvas of at least the given size."""
    tile = cv2.imread(str(mosaics[0])).shape[0]
    columns = int(math.ceil(width / tile))
    rows = int(math.ceil(height / tile))
    canvas = np.zeros((rows * tile, columns * tile, 3), np.uint8)
    for row in range(rows):
        for column in range(columns):
            image = cv2.imread(str(mosaics[int(rng.integers(len(mosaics)))]))
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            if rng.random() < 0.5:
                image = cv2.flip(image, int(rng.integers(-1, 2)))
            canvas[row * tile:(row + 1) * tile, column * tile:(column + 1) * tile] = image
    return canvas[:height, :width].copy()


def place_objects(rng, canvas, sprites_by_class, count, margin):
    """Paste ``count`` objects, every class at least once, none overlapping."""
    names = sorted(sprites_by_class)
    wanted = list(names)
    while len(wanted) < count:
        wanted.append(names[int(rng.integers(len(names)))])
    rng.shuffle(wanted)
    height, width = canvas.shape[:2]
    placed = []
    for name in wanted:
        options = sprites_by_class[name]
        for _ in range(60):
            sprite = options[int(rng.integers(len(options)))]
            centre_x = float(rng.uniform(margin, width - margin))
            centre_y = float(rng.uniform(margin, height - margin))
            rotation = float(rng.uniform(0.0, 2.0 * math.pi))
            scale = float(rng.uniform(0.92, 1.08))
            hull_w, hull_h = modelled_hull(sprite.length, sprite.width, sprite.angle, rotation)
            tentative = (
                centre_x - scale * hull_w, centre_y - scale * hull_h,
                centre_x + scale * hull_w, centre_y + scale * hull_h,
            )
            if any(_intersection_over_area(tentative, box) > 0.0 for _, box in placed):
                continue
            box = paste_sprite(
                canvas, sprite, centre_x, centre_y, rotation, scale,
                bool(rng.random() < 0.5), rng,
            )
            if box is not None:
                placed.append((OBJECT_CLASSES.index(name), box))
                break
    return placed


def render_flight(rng, ground, objects, frames, drift, zoom_start, zoom_rate, output):
    """Write the frames and annotations of one flight.

    Frame pixel ``(x, y)`` looks at ground point
    ``((x - W/2) / s + cx, (y - H/2) / s + cy)``: a scale ``s`` about the frame
    centre and a centre ``(cx, cy)`` moving against the drift, which is what
    makes the ground move by ``drift`` per frame and spread out from the
    middle as ``s`` grows.
    """
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(parents=True, exist_ok=True)
    ground_height, ground_width = ground.shape[:2]
    # The centre moves against the drift; start it half the travel ahead of
    # the middle of the ground so the flight is centred on it.
    travel = sum(1.0 / (zoom_start * (1.0 + zoom_rate) ** k) for k in range(frames - 1))
    start_x = ground_width / 2.0 + drift[0] * travel / 2.0
    start_y = ground_height / 2.0 + drift[1] * travel / 2.0
    totals = {name: 0 for name in OBJECT_CLASSES}
    seen = set()
    for frame in range(frames):
        scale = zoom_start * (1.0 + zoom_rate) ** frame
        centre_x = start_x - sum(drift[0] / (zoom_start * (1.0 + zoom_rate) ** k)
                                 for k in range(frame))
        centre_y = start_y - sum(drift[1] / (zoom_start * (1.0 + zoom_rate) ** k)
                                 for k in range(frame))
        # Inverse map: frame pixel -> ground pixel.
        matrix = np.array([
            [1.0 / scale, 0.0, centre_x - (WIDTH / 2.0) / scale],
            [0.0, 1.0 / scale, centre_y - (HEIGHT / 2.0) / scale],
        ])
        image = cv2.warpAffine(
            ground, matrix, (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101,
        )
        cv2.imwrite(str(output / 'images' / f'frame_{frame:06d}.png'), image)

        annotations = []
        counts = {}
        for index, (class_index, box) in enumerate(objects):
            x1 = (box[0] - centre_x) * scale + WIDTH / 2.0
            y1 = (box[1] - centre_y) * scale + HEIGHT / 2.0
            x2 = (box[2] - centre_x) * scale + WIDTH / 2.0
            y2 = (box[3] - centre_y) * scale + HEIGHT / 2.0
            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(WIDTH), x2), min(float(HEIGHT), y2)
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            name = OBJECT_CLASSES[class_index]
            annotations.append({
                'object_id': name,
                'bbox': [int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))],
            })
            counts[name] = counts.get(name, 0) + 1
            if index not in seen:
                seen.add(index)
                totals[name] += 1
        with open(output / 'annotations' / f'frame_{frame:06d}.json', 'w') as handle:
            json.dump({'frame': frame, 'annotations': annotations, 'object_counts': counts},
                      handle, indent=1)
    metadata = {
        'capture': {'synthetic': True, 'num_frames': frames, 'drift_px': list(drift),
                    'zoom_start': zoom_start, 'zoom_rate': zoom_rate},
        'total_objects': len(seen),
        'object_totals': {name: count for name, count in totals.items() if count},
    }
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=1))
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--terrain', default=str(PROJECT / 'external' / 'test'))
    parser.add_argument('--flights', type=int, default=4)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--density', type=float, default=1.2,
                        help='Objects per million ground pixels.')
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--prefix', default='synth')
    arguments = parser.parse_args()

    from masks import load_instances

    sprites_by_class = build_sprites(load_instances('helsinki'))
    mosaics = sorted(Path(arguments.terrain).glob('*.jpg'))
    if not mosaics:
        print(f'no mosaics in {arguments.terrain}', file=sys.stderr)
        return 1

    for flight in range(arguments.flights):
        rng = np.random.default_rng(arguments.seed + flight)
        # Mostly the reference geometry -- ground moving down the frame at
        # about 65 px per frame -- with some variety in speed and direction.
        speed = float(rng.uniform(55.0, 75.0))
        heading = 0.0 if flight % 2 == 0 else float(rng.uniform(-0.35, 0.35))
        drift = (speed * math.sin(heading), speed * math.cos(heading))
        zoom_rate = float(rng.uniform(0.002, 0.006))
        zoom_start = 1.0 / (1.0 + zoom_rate) ** (arguments.frames / 2.0)

        span_x = WIDTH / zoom_start + abs(drift[0]) * arguments.frames / zoom_start + 64
        span_y = HEIGHT / zoom_start + abs(drift[1]) * arguments.frames / zoom_start + 64
        ground = build_ground(rng, mosaics, int(span_x), int(span_y))
        count = max(16, int(round(arguments.density * span_x * span_y / 1e6)))
        objects = place_objects(rng, ground, sprites_by_class, count, margin=40)

        output = PROJECT / 'src' / f'{arguments.prefix}_{flight}'
        total = render_flight(
            rng, ground, objects, arguments.frames, drift, zoom_start, zoom_rate, output
        )
        print(f'{output.name}: {arguments.frames} frames, {total} objects in view, '
              f'drift ({drift[0]:.1f}, {drift[1]:.1f}) px/frame, zoom {zoom_rate:.4f}/frame')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
