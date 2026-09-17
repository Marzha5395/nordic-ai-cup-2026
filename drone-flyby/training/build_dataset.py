"""Write a YOLO dataset of composed 960x540 views to disk.

Everything is generated from the reference scene, so the only knobs that
matter are how many samples and which seed. Run it again with a different seed
to refresh the set.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose import OBJECT_CLASSES, load_scene_assets, render_sample  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

_ASSETS = None


def _initialise(scene_directory: str, cache_directory: str) -> None:
    global _ASSETS
    cv2.setNumThreads(1)
    _ASSETS = load_scene_assets(Path(scene_directory), Path(cache_directory))


def _render(job):
    index, seed, image_directory, label_directory = job
    rng = np.random.default_rng(seed)
    image, labels = render_sample(rng, _ASSETS)
    name = f'{index:06d}'
    cv2.imwrite(str(Path(image_directory) / f'{name}.png'), image)
    with open(Path(label_directory) / f'{name}.txt', 'w') as handle:
        for class_index, cx, cy, w, h in labels:
            handle.write(f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
    return len(labels)


def build_split(name, count, seed_base, output, scene_directory, cache_directory, workers):
    image_directory = output / 'images' / name
    label_directory = output / 'labels' / name
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)
    jobs = [
        (index, seed_base + index, str(image_directory), str(label_directory))
        for index in range(count)
    ]
    total = 0
    if workers <= 1:
        _initialise(str(scene_directory), str(cache_directory))
        for job in jobs:
            total += _render(job)
    else:
        import multiprocessing as mp

        context = mp.get_context('spawn')
        with context.Pool(
            workers, initializer=_initialise,
            initargs=(str(scene_directory), str(cache_directory)),
        ) as pool:
            for done, boxes in enumerate(pool.imap_unordered(_render, jobs, chunksize=16), 1):
                total += boxes
                if done % 500 == 0:
                    print(f'  {name}: {done}/{count}', flush=True)
    print(f'{name}: {count} images, {total} boxes')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=str(PROJECT / 'src' / 'helsinki'))
    parser.add_argument('--output', default=str(PROJECT / 'dataset'))
    parser.add_argument('--cache', default=str(PROJECT / 'dataset' / 'backgrounds'))
    parser.add_argument('--train', type=int, default=16000)
    parser.add_argument('--val', type=int, default=800)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    scene_directory = Path(arguments.scene)
    cache_directory = Path(arguments.cache)
    # Build the inpainted backgrounds once, in the parent, so that the worker
    # processes all find them cached instead of racing to write them.
    _initialise(str(scene_directory), str(cache_directory))

    build_split('val', arguments.val, arguments.seed + 5_000_000, output,
                scene_directory, cache_directory, arguments.workers)
    build_split('train', arguments.train, arguments.seed, output,
                scene_directory, cache_directory, arguments.workers)

    names = '\n'.join(f'  {index}: {name}' for index, name in enumerate(OBJECT_CLASSES))
    (output / 'drone.yaml').write_text(
        f'path: {output.resolve()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'names:\n{names}\n'
    )
    print('wrote', output / 'drone.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
