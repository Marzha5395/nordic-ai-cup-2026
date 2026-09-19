import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES


@dataclass
class Sprite:
    image: np.ndarray
    alpha: np.ndarray
    label: int
    box: np.ndarray


def warp_resampled(image, matrix, size, border_value=0):
    scale = min(np.linalg.norm(matrix[:, 0]), np.linalg.norm(matrix[:, 1]))
    if scale < 0.95:
        height, width = image.shape[:2]
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        resize_scale = np.array([new_width / width, new_height / height])
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        matrix = matrix.copy()
        matrix[:, 2] += matrix[:, :2] @ (0.5 / resize_scale - 0.5)
        matrix[:, :2] /= resize_scale
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def transform_boxes(boxes, matrix):
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    points = corners @ matrix[:, :2].T + matrix[:, 2]
    return np.concatenate((points.min(axis=1), points.max(axis=1)), axis=1)


def clip_labels(boxes, classes, width, height, minimum_visibility=0.45):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int64)
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    size = clipped[:, 2:] - clipped[:, :2]
    keep = (size.min(axis=1) >= 2) & (size.prod(axis=1) >= minimum_visibility * area)
    return clipped[keep], classes[keep]


class TrainingScenes:
    def __init__(self, directory, size=640, seed=2026, frames=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.frames = []
        self.sprites = [[] for _ in OBJECT_CLASSES]
        self.backgrounds = []
        for annotation_path in sorted(Path(directory).glob('*/annotations/*.json')):
            payload = json.loads(annotation_path.read_text())
            if frames is not None and payload['frame'] not in frames:
                continue
            image_path = annotation_path.parent.parent / 'images' / (annotation_path.stem + '.png')
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f'Cannot read training image: {image_path}')
            boxes = np.asarray([item['bbox'] for item in payload['annotations']], dtype=np.float32).reshape(-1, 4)
            classes = np.asarray([OBJECT_CLASSES.index(item['object_id']) for item in payload['annotations']])
            self.frames.append((image, boxes, classes))
            background = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            mask = np.zeros(background.shape[:2], dtype=np.uint8)
            for box, label in zip(boxes, classes):
                x1, y1, x2, y2 = np.round(box * 0.25).astype(int)
                cv2.rectangle(mask, (max(0, x1 - 3), max(0, y1 - 3)), (x2 + 3, y2 + 3), 255, -1)
                sprite = self._extract(image, box, int(label))
                if sprite is not None:
                    self.sprites[label].append(sprite)
            self.backgrounds.append(cv2.inpaint(background, mask, 3, cv2.INPAINT_TELEA))
        missing = [name for name, samples in zip(OBJECT_CLASSES, self.sprites) if not samples]
        if not self.frames or missing:
            raise ValueError(f'Training needs complete, unclipped examples for all classes: {missing}')

    @staticmethod
    def _extract(image, box, label):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = np.round(box).astype(int)
        if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            return None
        padding = max(4, round(min(x2 - x1, y2 - y1) * 0.12))
        left, top = max(0, x1 - padding), max(0, y1 - padding)
        right, bottom = min(width, x2 + padding), min(height, y2 + padding)
        crop = image[top:bottom, left:right].copy()
        local_box = np.array([x1 - left, y1 - top, x2 - left, y2 - top], dtype=np.float32)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        rect = (x1 - left, y1 - top, x2 - x1, y2 - y1)
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
        if alpha.sum() < 0.06 * (x2 - x1) * (y2 - y1):
            alpha[y1 - top:y2 - top, x1 - left:x2 - left] = 1
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
        return Sprite(crop, alpha, label, local_box)

    def _real(self):
        image, boxes, classes = self.frames[int(self.rng.integers(len(self.frames)))]
        scale = float(self.rng.choice([0.25, 0.5, 1.0]) * self.rng.uniform(0.8, 1.25))
        if len(boxes) and self.rng.random() < 0.85:
            center = boxes[int(self.rng.integers(len(boxes)))].reshape(2, 2).mean(axis=0)
            center += self.rng.uniform(-0.35, 0.35, 2) * self.size / scale
        else:
            center = self.rng.uniform([0, 0], image.shape[1::-1])
        angle = float(self.rng.uniform(-180, 180))
        matrix = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        matrix[:, 2] += self.size / 2 - center
        output = warp_resampled(image, matrix, (self.size, self.size), border_value=(114, 114, 114))
        boxes = transform_boxes(boxes, matrix)
        boxes, classes = clip_labels(boxes, classes, self.size, self.size)
        return output, boxes, classes

    def _background(self):
        if self.rng.random() < 0.7:
            image, boxes, _ = self.frames[int(self.rng.integers(len(self.frames)))]
            scale = float(self.rng.choice([0.5, 1.0]) * self.rng.uniform(0.8, 1.3))
            extent = min(int(self.size / scale), min(image.shape[:2]))
            for _ in range(12):
                x = int(self.rng.integers(image.shape[1] - extent + 1))
                y = int(self.rng.integers(image.shape[0] - extent + 1))
                overlap = (boxes[:, 0] < x + extent) & (boxes[:, 2] > x) & (boxes[:, 1] < y + extent) & (boxes[:, 3] > y)
                if not overlap.any():
                    return cv2.resize(image[y:y + extent, x:x + extent], (self.size, self.size), interpolation=cv2.INTER_AREA)
        background = self.backgrounds[int(self.rng.integers(len(self.backgrounds)))]
        extent = int(self.rng.uniform(0.5, 1.0) * min(background.shape[:2]))
        x = int(self.rng.integers(background.shape[1] - extent + 1))
        y = int(self.rng.integers(background.shape[0] - extent + 1))
        return cv2.resize(background[y:y + extent, x:x + extent], (self.size, self.size))

    def _synthetic(self):
        image = self._background()
        image = np.rot90(image, int(self.rng.integers(4))).copy()
        boxes, classes = [], []
        count = 0 if self.rng.random() < 0.08 else int(self.rng.integers(4, 22))
        camera_scale = float(self.rng.choice([0.25, 0.5, 1.0], p=[0.4, 0.35, 0.25]))
        for _ in range(count):
            label = int(self.rng.integers(len(OBJECT_CLASSES)))
            samples = self.sprites[label]
            sprite = samples[int(self.rng.integers(len(samples)))]
            scale = camera_scale * float(self.rng.uniform(0.65, 1.5))
            height, width = sprite.image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(self.rng.uniform(-180, 180)), scale)
            bounds = transform_boxes(np.array([[0, 0, width, height]], dtype=np.float32), matrix)[0]
            matrix[:, 2] -= bounds[:2]
            out_width, out_height = np.ceil(bounds[2:] - bounds[:2]).astype(int)
            if min(out_width, out_height) < 3 or max(out_width, out_height) >= self.size:
                continue
            rgb = warp_resampled(sprite.image.astype(np.float32) * sprite.alpha[..., None], matrix, (out_width, out_height))
            alpha = warp_resampled(sprite.alpha, matrix, (out_width, out_height))
            local_box = transform_boxes(sprite.box[None], matrix)[0]
            for _ in range(10):
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
            gain = self.rng.uniform(0.65, 1.35, 3)
            rgb = (rgb * gain).clip(0, 255 * alpha[..., None])
            region = image[y:y + out_height, x:x + out_width]
            region[:] = (rgb + region * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
            boxes.append(box)
            classes.append(label)
        boxes, classes = clip_labels(boxes, classes, self.size, self.size)
        return image, boxes, classes

    def sample(self):
        image, boxes, classes = self._real() if self.rng.random() < 0.25 else self._synthetic()
        if self.rng.random() < 0.5:
            image = cv2.flip(image, 1)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        if self.rng.random() < 0.5:
            image = cv2.flip(image, 0)
            boxes[:, [1, 3]] = self.size - boxes[:, [3, 1]]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.rng.uniform(-9, 9)) % 180
        hsv[..., 1] *= self.rng.uniform(0.55, 1.4)
        hsv[..., 2] *= self.rng.uniform(0.65, 1.35)
        image = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.rng.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), float(self.rng.uniform(0.3, 0.9)))
        if self.rng.random() < 0.15:
            noise = self.rng.normal(0, self.rng.uniform(1, 5), image.shape)
            image = (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        xywh = np.concatenate(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), axis=1) / self.size
        return np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)), xywh.astype(np.float32), classes

    def batch(self, count):
        import torch

        samples = [self.sample() for _ in range(count)]
        return {
            'img': torch.from_numpy(np.stack([item[0] for item in samples])),
            'bboxes': torch.from_numpy(np.concatenate([item[1] for item in samples])),
            'cls': torch.from_numpy(np.concatenate([item[2] for item in samples])).float().view(-1, 1),
            'batch_idx': torch.from_numpy(np.concatenate([np.full(len(item[2]), index, dtype=np.int64) for index, item in enumerate(samples)])),
        }
