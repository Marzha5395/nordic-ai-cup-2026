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

from data_v2 import DiverseScenes
from dtos import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parent


def create_model(source, p2):
    config = deepcopy(source.yaml)
    config['nc'] = len(OBJECT_CLASSES)
    config['head'][-1][2:] = ['Detect', ['nc']]
    if p2:
        config['head'] = config['head'][:6] + [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 2], 1, 'Concat', [1]],
            [-1, 2, 'C3k2', [256, False]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 16], 1, 'Concat', [1]],
            [-1, 2, 'C3k2', [256, False]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 13], 1, 'Concat', [1]],
            [-1, 2, 'C3k2', [512, False]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 2, 'C3k2', [1024, True]],
            [[19, 22, 25, 28], 1, 'Detect', ['nc']],
        ]
    model = DetectionModel(config, nc=len(OBJECT_CLASSES), verbose=False)
    original = source.float().state_dict()
    target = model.state_dict()
    transferred = {}
    for key, value in original.items():
        parts = key.split('.')
        layer = int(parts[1])
        destinations = [layer]
        if p2:
            destinations = {16: [16, 19, 22], 17: [20, 23], 18: [24], 19: [25], 20: [26], 21: [27], 22: [28], 23: [29]}.get(layer, [layer])
        for destination in destinations:
            mapped = parts.copy()
            mapped[1] = str(destination)
            branches = [int(parts[3])] if layer == 23 and len(parts) > 3 and parts[2] in ('cv2', 'cv3') else [None]
            if p2 and branches[0] is not None:
                branches = [0, 1] if branches[0] == 0 else [branches[0] + 1]
            for branch in branches:
                if branch is not None:
                    mapped[3] = str(branch)
                name = '.'.join(mapped)
                if name in target and target[name].shape == value.shape:
                    transferred[name] = value
    model.load_state_dict(transferred, strict=False)
    model.names = dict(enumerate(OBJECT_CLASSES))
    model.args = get_cfg(overrides={'box': 7.5, 'cls': 0.7, 'dfl': 1.5})
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('.dfl.' not in name)
    print(f'Transferred {len(transferred)}/{len(target)} tensors; strides={model.stride.tolist()}', flush=True)
    return model


def save(run, model, ema, optimizer, scaler, data, step, metadata):
    inference = {'model': deepcopy(ema.ema).cpu().half(), 'train_args': {'task': 'detect', 'imgsz': metadata['image_size']}, 'training_metadata': metadata}
    snapshot = run / f'step_{step:06d}.pt'
    temporary = snapshot.with_suffix('.tmp')
    torch.save(inference, temporary)
    temporary.replace(snapshot)
    training = {
        'config': model.yaml,
        'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
        'ema': {name: value.detach().cpu() for name, value in ema.ema.state_dict().items()},
        'ema_updates': ema.updates,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'step': step,
        'metadata': metadata,
        'data_rng': data.rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = run / 'resume.tmp'
    torch.save(training, temporary)
    temporary.replace(run / 'resume.pt')
    (run / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default='yolo11s-obb.pt')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--schedule-steps', type=int, default=24000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--learning-rate', type=float, default=0.0008)
    parser.add_argument('--canonical-scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=417)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-p2', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.schedule_steps, args.checkpoint_every, args.batch, args.accumulate, args.size) <= 0 or args.size % 32:
        parser.error('Positive schedule, batch, accumulation and a size divisible by 32 are required')
    if args.steps % args.accumulate or args.checkpoint_every % args.accumulate:
        parser.error('Steps and checkpoint interval must be divisible by accumulation for exact resume')
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / 'resume.pt').exists() and not args.resume:
        parser.error('This run already contains a checkpoint; use --resume or a new run directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    state = torch.load(args.run / 'resume.pt', map_location='cpu', weights_only=False) if args.resume else None
    if state:
        model = DetectionModel(state['config'], nc=len(OBJECT_CLASSES), verbose=False)
        model.load_state_dict(state['model'])
        model.names = dict(enumerate(OBJECT_CLASSES))
        model.args = get_cfg(overrides={'box': 7.5, 'cls': 0.7, 'dfl': 1.5})
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('.dfl.' not in name)
    else:
        source = YOLO(args.pretrained).model
        model = create_model(source, not args.no_p2)
        del source
    model = model.to(device).train()
    groups = [[], []]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[int(parameter.ndim == 1 or name.endswith('.bias'))].append(parameter)
    optimizer = torch.optim.AdamW([{'params': groups[0], 'weight_decay': 0.01}, {'params': groups[1], 'weight_decay': 0.0}], lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    ema = ModelEMA(model)
    data = DiverseScenes(ROOT / 'src', ROOT / '.training/sprites', ROOT / '.training/external', size=args.size, seed=args.seed, canonical_scale=args.canonical_scale)
    metadata = {
        'pretrained': args.pretrained,
        'pretrained_sha256': hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        'pretrained_reference': 'https://docs.ultralytics.com/models/yolo11/',
        'external_backgrounds': json.loads((ROOT / '.training/external/provenance.json').read_text()),
        'foreground_masks': json.loads((ROOT / '.training/sprites/provenance.json').read_text()),
        'training_frames': data.selected_frames,
        'training_backgrounds': len(data.external_paths),
        'classes': list(OBJECT_CLASSES),
        'sprites_per_class': dict(zip(OBJECT_CLASSES, map(len, data.sprites))),
        'image_size': args.size,
        'canonical_scale': args.canonical_scale,
        'p2': not args.no_p2,
        'seed': args.seed,
        'batch_size': args.batch,
        'gradient_accumulation': args.accumulate,
        'learning_rate': args.learning_rate,
        'schedule_steps': args.schedule_steps,
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('train_v2.py', 'data_v2.py', 'training_data.py')},
        'limitations': 'Held-out views are still the same object instances; synthetic background transfer is not independent real-world validation.',
    }
    start_step = 0
    if state:
        ema.ema.load_state_dict(state['ema'])
        ema.updates = state['ema_updates']
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        data.rng.bit_generator.state = state['data_rng']
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] and device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step = state['step']
        del state
    print(json.dumps(metadata), flush=True)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    losses = np.zeros(3)
    for step in range(start_step, args.steps):
        batch = {key: value.to(device) for key, value in data.batch(args.batch).items()}
        batch['img'] = batch['img'].float() / 255
        warmup = min(1.0, (step + 1) / 200)
        decay = 0.1 + 0.9 * (1 + math.cos(math.pi * min(step / args.schedule_steps, 1))) / 2
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * warmup * decay
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            loss, items = model(batch)
            loss = loss.sum() / args.batch / args.accumulate
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulate == 0 or step + 1 == args.steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        losses += items.detach().cpu().numpy()
        if (step + 1) % 100 == 0:
            event = {'step': step + 1, 'loss': (losses / 100).tolist(), 'seconds': time.monotonic() - start}
            print(json.dumps(event), flush=True)
            with (args.run / 'training.jsonl').open('a') as handle:
                handle.write(json.dumps(event) + '\n')
            losses[:] = 0
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            metadata['completed_steps'] = step + 1
            snapshot = save(args.run, model, ema, optimizer, scaler, data, step + 1, metadata)
            print(f'Saved {snapshot}; resumable optimizer and RNG state saved separately.', flush=True)


if __name__ == '__main__':
    main()
