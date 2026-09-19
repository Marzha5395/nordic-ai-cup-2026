import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA

from dtos import OBJECT_CLASSES
from terrain_data import TRAINING_SCENES, TrainingScenes


class Samples(torch.utils.data.IterableDataset):
    """Endless batches from TrainingScenes, one independent random stream per worker process."""

    def __init__(self, data, batch, seed):
        self.data, self.batch, self.seed = data, batch, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        self.data.reseed(self.seed * 1000 + (info.id if info else 0))
        while True:
            yield self.data.batch(self.batch)


ROOT = Path(__file__).resolve().parent


def checkpoint(model, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': deepcopy(model).cpu().half(), 'train_args': {'task': 'detect', 'imgsz': metadata['image_size']}, 'training_metadata': metadata}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'src')
    parser.add_argument('--pretrained', default='yolo11n.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'weights' / 'detector.pt')
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--initialize', type=Path)
    parser.add_argument('--photometric', action='store_true', help='Stronger global gamma, contrast and colour-cast augmentation')
    parser.add_argument('--appearance', action='store_true', help='Per-object colour restyling and size-weighted class sampling')
    parser.add_argument('--shadows', action='store_true', help='Cast shadows from one random sun direction per image')
    parser.add_argument('--p2', action='store_true', help="Add V2's stride-4 head (train_v2.create_model); use with --pretrained yolo11s-obb.pt")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.accumulate, args.size) <= 0 or args.size % 32:
        parser.error('steps, batch, and accumulate must be positive; size must be a positive multiple of 32')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    device = torch.device(args.device)
    source = YOLO(str(args.initialize or args.pretrained)).model
    if args.p2:
        from train_v2 import create_model

        model = create_model(source, True)
    else:
        model = DetectionModel(deepcopy(source.yaml), nc=len(OBJECT_CLASSES), verbose=False)
        model.load(source)
        model.names = dict(enumerate(OBJECT_CLASSES))
        model.args = get_cfg(overrides={'box': 7.5, 'cls': 0.5, 'dfl': 1.5})
    model = model.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if '.dfl.' in name:
            parameter.requires_grad_(False)
    del source
    groups = [[], []]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[int(parameter.ndim == 1 or name.endswith('.bias'))].append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': groups[0], 'weight_decay': 0.01},
        {'params': groups[1], 'weight_decay': 0.0},
    ], lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = ModelEMA(model)
    print('Loading training-only reference imagery and extracting randomized foregrounds.', flush=True)
    data = TrainingScenes(args.data, args.size, args.seed, photometric=args.photometric, appearance=args.appearance, shadows=args.shadows)
    fingerprint = hashlib.sha256()
    for path in sorted(path for scene in TRAINING_SCENES for path in (args.data / scene).glob('annotations/*.json')):
        fingerprint.update(path.read_bytes())
    metadata = {
        'seed': args.seed,
        'image_size': args.size,
        'p2': args.p2,
        'photometric': args.photometric,
        'appearance': args.appearance,
        'shadows': args.shadows,
        'class_weights': dict(zip(OBJECT_CLASSES, np.round(data.class_weights, 4).tolist())),
        'planned_steps': args.steps,
        'batch_size': args.batch,
        'gradient_accumulation': args.accumulate,
        'learning_rate': args.learning_rate,
        'antialiased_resampling': True,
        'initial_weights_sha256': hashlib.sha256(Path(args.initialize or args.pretrained).read_bytes()).hexdigest(),
        'training_code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('train_terrain.py', 'terrain_data.py', 'sprites.py')},
        'classes': list(OBJECT_CLASSES),
        'source_frames': len(data.frames),
        'annotation_sha256': fingerprint.hexdigest(),
        'sprites_per_class': {name: len(sprites) for name, sprites in zip(OBJECT_CLASSES, data.sprites)},
        'terrain_tiles': len(data.terrain),
        'selection': 'fixed training schedule, final exponential moving average; measured only by heldout_eval.py on held-out terrain places',
        'augmentation': '80% SAM cut-outs composited at source resolution on outside terrain (train places only) or object-free reference ground, '
                        'with unlabelled look-alike patches, reduced by the view level factor with INTER_AREA; 20% real reference crops; '
                        'quarter turns, flips, colour, blur, noise',
    }
    print(json.dumps(metadata), flush=True)
    start = time.monotonic()
    loss_sum = np.zeros(3)
    optimizer.zero_grad(set_to_none=True)
    loader = iter(torch.utils.data.DataLoader(Samples(data, args.batch, args.seed), batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None))
    for step in range(args.steps):
        batch = {key: value.to(device, non_blocking=True) for key, value in next(loader).items()}
        batch['img'] = batch['img'].float() / 255
        warmup = min(1.0, (step + 1) / 200)
        decay = 0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * warmup * decay
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            loss, items = model(batch)
            loss = loss.sum() / args.batch / args.accumulate
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at training step {step + 1}')
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulate == 0 or step + 1 == args.steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        loss_sum += items.detach().cpu().numpy()
        if (step + 1) % 50 == 0:
            print(f'step={step + 1}/{args.steps} training_loss={(loss_sum / 50).round(4).tolist()} elapsed_s={time.monotonic() - start:.1f}', flush=True)
            loss_sum[:] = 0
        if (step + 1) % 500 == 0 or step + 1 == args.steps:
            metadata['completed_steps'] = step + 1
            metadata['elapsed_seconds'] = round(time.monotonic() - start, 2)
            checkpoint(ema.ema, args.output, metadata)
            args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved training artifact: {args.output}', flush=True)


if __name__ == '__main__':
    main()
