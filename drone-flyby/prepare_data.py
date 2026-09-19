import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import requests

from dtos import OBJECT_CLASSES
from training_data import TrainingScenes


ROOT = Path(__file__).resolve().parent
BACKGROUND_CLASSES = {'agricultural', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'intersection', 'mediumresidential', 'river', 'sparseresidential', 'tenniscourt'}
DATA_URL = 'https://huggingface.co/datasets/yuanliuyang/ucmerced/resolve/main/UCMerced_LandUse.zip'
DATA_SHA256 = '06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb'


def download_backgrounds(directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'UCMerced_LandUse.zip'
    if not archive.is_file():
        temporary = archive.with_suffix('.part')
        with requests.get(DATA_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        temporary.replace(archive)
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    if digest != DATA_SHA256:
        raise ValueError('Public dataset checksum differs from the published LFS object')
    count = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            parts = Path(member.filename).parts
            if len(parts) < 2 or parts[-2] not in BACKGROUND_CLASSES or not parts[-1].endswith('.tif'):
                continue
            if member.file_size > 4 * 1024 * 1024:
                raise ValueError('Unexpectedly large dataset image')
            image = cv2.imdecode(np.frombuffer(source.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {member.filename}')
            stem = Path(parts[-1]).stem
            number = int(stem[-2:])
            split = 'holdout' if number % 5 == 0 else 'train'
            target = directory / split / (stem + '.png')
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), image)
            count += 1
    metadata = {'source': DATA_URL, 'sha256': digest, 'license': 'Public domain USGS imagery; mirror declares CC0-1.0', 'license_reference': 'https://huggingface.co/datasets/yuanliuyang/ucmerced/raw/main/README.md', 'classes': sorted(BACKGROUND_CLASSES), 'images': count, 'holdout_rule': 'image suffix modulo five equals zero; never used for training'}
    (directory / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def inspect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    examples = [[] for _ in OBJECT_CLASSES]
    for path in sorted((ROOT / 'src').glob('*/annotations/*.json')):
        payload = json.loads(path.read_text())
        image = cv2.imread(str(path.parent.parent / 'images' / (path.stem + '.png')))
        height, width = image.shape[:2]
        for annotation in payload['annotations']:
            box = np.array(annotation['bbox'], dtype=np.float32)
            if box[0] <= 1 or box[1] <= 1 or box[2] >= width - 1 or box[3] >= height - 1:
                continue
            label = OBJECT_CLASSES.index(annotation['object_id'])
            examples[label].append((image, box, payload['frame']))
    canvas = np.full((4 * 260, 4 * 320, 3), 35, dtype=np.uint8)
    masks = canvas.copy()
    for label, samples in enumerate(examples):
        image, box, frame = max(samples, key=lambda item: np.prod(item[1][2:] - item[1][:2]))
        sprite = TrainingScenes._extract(image, box, label)
        for target, patch in [(canvas, sprite.image), (masks, (sprite.image * sprite.alpha[..., None]).astype(np.uint8))]:
            scale = min(300 / patch.shape[1], 205 / patch.shape[0])
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            x, y = (label % 4) * 320, (label // 4) * 260
            target[y + 35:y + 35 + patch.shape[0], x + 10:x + 10 + patch.shape[1]] = patch
            cv2.putText(target, f'{OBJECT_CLASSES[label]} f{frame}', (x + 8, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(directory / 'objects.jpg'), canvas)
    cv2.imwrite(str(directory / 'masks.jpg'), masks)


def generate_masks(directory):
    import torch
    from ultralytics import SAM

    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = SAM('sam2.1_t.pt')
    counts = dict.fromkeys(OBJECT_CLASSES, 0)
    for path in sorted((ROOT / 'src').glob('*/annotations/*.json')):
        payload = json.loads(path.read_text())
        image = cv2.imread(str(path.parent.parent / 'images' / (path.stem + '.png')))
        height, width = image.shape[:2]
        for item in payload['annotations']:
            x1, y1, x2, y2 = item['bbox']
            if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
                continue
            label = OBJECT_CLASSES.index(item['object_id'])
            output = directory / f'{path.parent.parent.name}_{payload["frame"]:06d}_{label:02d}.npz'
            if output.is_file():
                counts[item['object_id']] += 1
                continue
            padding = max(6, round(min(x2 - x1, y2 - y1) * 0.25))
            left, top = max(0, x1 - padding), max(0, y1 - padding)
            right, bottom = min(width, x2 + padding), min(height, y2 + padding)
            crop = image[top:bottom, left:right].copy()
            box = np.array([x1 - left, y1 - top, x2 - left, y2 - top], dtype=np.float32)
            scale = min(512 / crop.shape[1], 512 / crop.shape[0])
            enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            result = model(enlarged, bboxes=(box * scale)[None].tolist(), device='cuda:0', verbose=False, imgsz=1024)[0]
            alpha = cv2.resize(result.masks.data[0].float().cpu().numpy(), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
            keep = np.zeros(crop.shape[:2], dtype=np.float32)
            bx1, by1, bx2, by2 = box.astype(int)
            keep[by1:by2, bx1:bx2] = 1
            alpha *= keep
            if alpha.sum() < 0.02 * (x2 - x1) * (y2 - y1):
                old = TrainingScenes._extract(image, np.array(item['bbox']), label)
                crop, box, alpha = old.image, old.box, old.alpha
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0.35)
            np.savez_compressed(output, image=crop, alpha=alpha, box=box, label=label, frame=payload['frame'])
            counts[item['object_id']] += 1
        print(f'Segmented frame {payload["frame"]}', flush=True)
    (directory / 'provenance.json').write_text(json.dumps({'model': 'sam2.1_t.pt', 'reference': 'https://docs.ultralytics.com/models/sam-2', 'counts': counts}, indent=2) + '\n')
    canvas = np.full((1040, 1280, 3), 35, dtype=np.uint8)
    for label, name in enumerate(OBJECT_CLASSES):
        paths = list(directory.glob(f'*_{label:02d}.npz'))
        samples = [np.load(path) for path in paths]
        sample = max(samples, key=lambda item: np.prod(item['box'][2:] - item['box'][:2]))
        patch = (sample['image'] * sample['alpha'][..., None]).astype(np.uint8)
        scale = min(300 / patch.shape[1], 205 / patch.shape[0])
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        x, y = (label % 4) * 320, (label // 4) * 260
        canvas[y + 35:y + 35 + patch.shape[0], x + 10:x + 10 + patch.shape[1]] = patch
        cv2.putText(canvas, name, (x + 8, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for sample in samples:
            sample.close()
    cv2.imwrite(str(directory / 'masks.jpg'), canvas)


def prepare_validation(directory):
    import shutil
    from data_v2 import DiverseScenes, HOLDOUT_FRAMES, build_transfer_scene

    if (directory / 'transfer/provenance.json').exists():
        raise ValueError('Validation scene already exists; use a different output directory rather than change a test set')
    data = DiverseScenes(ROOT / 'src', ROOT / '.training/sprites', ROOT / '.training/external', seed=829, split='holdout')
    build_transfer_scene(data, directory / 'transfer')
    for folder in ('images', 'annotations'):
        (directory / 'heldout' / folder).mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / 'src').glob('*/annotations/*.json')):
        if int(path.stem.split('_')[-1]) not in HOLDOUT_FRAMES:
            continue
        shutil.copy2(path, directory / 'heldout/annotations' / path.name)
        image = path.parent.parent / 'images' / (path.stem + '.png')
        shutil.copy2(image, directory / 'heldout/images' / image.name)
    canvas = np.full((1040, 1280, 3), 35, dtype=np.uint8)
    for label, samples in enumerate(data.sprites):
        sample = max(samples, key=lambda item: np.prod(item.box[2:] - item.box[:2]))
        patch = (sample.image * sample.alpha[..., None]).astype(np.uint8)
        scale = min(300 / patch.shape[1], 205 / patch.shape[0])
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        x, y = (label % 4) * 320, (label // 4) * 260
        canvas[y + 35:y + 35 + patch.shape[0], x + 10:x + 10 + patch.shape[1]] = patch
        cv2.putText(canvas, OBJECT_CLASSES[label], (x + 8, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(directory / 'refined_masks.jpg'), canvas)
    print(f'Prepared held-out views and a fixed synthetic transfer diagnostic under {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['backgrounds', 'inspect', 'masks', 'validation'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.action == 'backgrounds':
        download_backgrounds(args.output or ROOT / '.training' / 'external')
    elif args.action == 'masks':
        generate_masks(args.output or ROOT / '.training' / 'sprites')
    elif args.action == 'validation':
        prepare_validation(args.output or ROOT / '.training' / 'validation')
    else:
        inspect(args.output or ROOT / '.training' / 'inspection')


if __name__ == '__main__':
    main()
