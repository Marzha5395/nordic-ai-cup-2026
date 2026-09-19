import json
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES
from training_data import Sprite, TrainingScenes, clip_labels, transform_boxes, warp_resampled


HOLDOUT_FRAMES = frozenset({1, 8, 15, 19, 24})


def refine_alpha(image, alpha, box):
    x1, y1, x2, y2 = box.astype(int)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.grabCut(image, mask, (x1, y1, x2 - x1, y2 - y1), np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    coverage = foreground.sum() / max(1, (x2 - x1) * (y2 - y1))
    if 0.08 < coverage < 0.85:
        agreement = np.minimum(foreground, alpha).sum() / max(1, min(foreground.sum(), alpha.sum()))
        if agreement < 0.35 or alpha.sum() > 2.5 * foreground.sum():
            return cv2.GaussianBlur(foreground, (3, 3), 0.35)
    return alpha


class DiverseScenes(TrainingScenes):
    def __init__(self, directory, sprites, backgrounds, size=640, seed=2026, split='train', canonical_scale=0.5):
        all_frames = {int(path.stem.split('_')[-1]) for path in Path(directory).glob('*/annotations/*.json')}
        selected = all_frames & HOLDOUT_FRAMES if split == 'holdout' else all_frames - HOLDOUT_FRAMES
        super().__init__(directory, size=size, seed=seed, frames=selected)
        self.selected_frames = sorted(selected)
        self.canonical_scale = canonical_scale
        self.split = split
        self.sprites = [[] for _ in OBJECT_CLASSES]
        for path in sorted(Path(sprites).glob('*.npz')):
            with np.load(path) as data:
                if int(data['frame']) not in selected:
                    continue
                label = int(data['label'])
                image, box = data['image'].copy(), data['box'].copy()
                alpha = refine_alpha(image, data['alpha'].copy(), box)
                self.sprites[label].append(Sprite(image, alpha, label, box))
        missing = [name for name, samples in zip(OBJECT_CLASSES, self.sprites) if not samples]
        if missing:
            raise ValueError(f'Missing {split} foregrounds: {missing}')
        self.external_paths = sorted((Path(backgrounds) / split).glob('*.png'))
        self.external = [cv2.imread(str(path)) for path in self.external_paths]
        if not self.external or any(image is None for image in self.external):
            raise ValueError('Download and prepare the external background split first')
        self.class_frames = [[index for index, (_, _, labels) in enumerate(self.frames) if label in labels] for label in range(len(OBJECT_CLASSES))]

    def _background(self):
        if self.split == 'holdout' or self.rng.random() < 0.55:
            image = self.external[int(self.rng.integers(len(self.external)))]
            image = np.rot90(image, int(self.rng.integers(4)))
            return cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return super()._background()

    def _real(self):
        label = int(self.rng.integers(len(OBJECT_CLASSES)))
        indices = self.class_frames[label]
        image, boxes, classes = self.frames[int(self.rng.choice(indices))]
        focus = boxes[np.flatnonzero(classes == label)[0]]
        center = focus.reshape(2, 2).mean(axis=0)
        scale = self.canonical_scale * float(self.rng.uniform(0.85, 1.15))
        center += self.rng.uniform(-0.35, 0.35, 2) * self.size / scale
        angle = float(self.rng.uniform(-180, 180))
        matrix = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        matrix[:, 2] += self.size / 2 - center
        output = warp_resampled(image, matrix, (self.size, self.size), border_value=(114, 114, 114))
        boxes, classes = clip_labels(transform_boxes(boxes, matrix), classes, self.size, self.size)
        return output, boxes, classes

    def _synthetic(self):
        image = self._background()
        boxes, classes = [], []
        count = 0 if self.rng.random() < 0.08 else int(self.rng.integers(6, 21))
        order = self.rng.permutation(len(OBJECT_CLASSES)).tolist()
        for index in range(count):
            label = int(order[index % len(order)])
            samples = self.sprites[label]
            sprite = samples[int(self.rng.integers(len(samples)))]
            scale = self.canonical_scale * float(self.rng.uniform(0.85, 1.15))
            height, width = sprite.image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(self.rng.uniform(-180, 180)), scale)
            matrix[:, :2] *= self.rng.uniform(0.94, 1.06, (1, 2))
            bounds = transform_boxes(np.array([[0, 0, width, height]], dtype=np.float32), matrix)[0]
            matrix[:, 2] -= bounds[:2]
            out_width, out_height = np.ceil(bounds[2:] - bounds[:2]).astype(int)
            if min(out_width, out_height) < 3 or max(out_width, out_height) >= self.size:
                continue
            premultiplied = sprite.image.astype(np.float32) * sprite.alpha[..., None]
            rgb = warp_resampled(premultiplied, matrix, (out_width, out_height))
            alpha = warp_resampled(sprite.alpha, matrix, (out_width, out_height)).clip(0, 1)
            local_box = transform_boxes(sprite.box[None], matrix)[0]
            for _ in range(12):
                x = int(self.rng.integers(self.size - out_width + 1))
                y = int(self.rng.integers(self.size - out_height + 1))
                box = local_box + [x, y, x, y]
                if not boxes:
                    break
                existing = np.asarray(boxes)
                intersection = np.maximum(np.minimum(existing[:, 2:], box[2:]) - np.maximum(existing[:, :2], box[:2]), 0).prod(axis=1)
                if intersection.max() == 0:
                    break
            else:
                continue
            gain = self.rng.uniform(0.8, 1.2) * self.rng.uniform(0.92, 1.08, 3)
            rgb = (rgb * gain).clip(0, 255 * alpha[..., None])
            region = image[y:y + out_height, x:x + out_width]
            if self.rng.random() < 0.35:
                shadow_matrix = np.array([[1, 0, self.rng.uniform(-3, 3)], [0, 1, self.rng.uniform(0, 4)]], dtype=np.float32)
                shadow = cv2.warpAffine(alpha, shadow_matrix, (out_width, out_height))
                shadow = cv2.GaussianBlur(shadow, (5, 5), 1)
                region[:] = (region * (1 - shadow[..., None] * self.rng.uniform(0.12, 0.3))).astype(np.uint8)
            region[:] = (rgb + region * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
            boxes.append(box)
            classes.append(label)
        return image, *clip_labels(boxes, classes, self.size, self.size)

    def sample(self):
        image, boxes, classes = super().sample()
        level = int(self.rng.choice([0, 1, 2], p=[0.65, 0.25, 0.1]))
        ratio = (0.25 * 2 ** level) / self.canonical_scale
        if ratio < 1:
            canvas = image.transpose(1, 2, 0)
            pixels = max(1, round(self.size * ratio))
            canvas = cv2.resize(canvas, (pixels, pixels), interpolation=cv2.INTER_AREA)
            canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            image = np.ascontiguousarray(canvas.transpose(2, 0, 1))
        return image, boxes, classes


def build_transfer_scene(data, directory, count=64):
    directory = Path(directory)
    (directory / 'images').mkdir(parents=True, exist_ok=True)
    (directory / 'annotations').mkdir(parents=True, exist_ok=True)
    old_size = data.size
    data.size = 1920
    for frame in range(count):
        image, boxes, labels = data._synthetic()
        top = (data.size - 1080) // 2
        image = image[top:top + 1080]
        boxes -= [0, top, 0, top]
        boxes, labels = clip_labels(boxes, labels, 1920, 1080, minimum_visibility=0.1)
        image = cv2.resize(image, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        annotations = [{'object_id': OBJECT_CLASSES[int(label)], 'bbox': [int(round(value * 2)) for value in box]} for box, label in zip(boxes, labels)]
        cv2.imwrite(str(directory / 'images' / f'frame_{frame:06d}.png'), image)
        (directory / 'annotations' / f'frame_{frame:06d}.json').write_text(json.dumps({'frame': frame, 'annotations': annotations}) + '\n')
    data.size = old_size
    (directory / 'provenance.json').write_text(json.dumps({'description': 'Synthetic domain-transfer diagnostic, not an independent real-world benchmark', 'sprite_frames': data.selected_frames, 'background_split': data.split, 'frames': count}, indent=2) + '\n')
