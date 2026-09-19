"""Training samples that look like evaluator views over ground the model has never seen.

The reference scene is 25 frames of one place, with one instance of each class at
one orientation. A detector trained on crops of it learns that place, and on new
ground reports rocks, sheds and shadows as objects. So most samples here are
composed from scratch:

* the ground is real aerial terrain from other places (``fetch_terrain.py``,
  ``train`` split only; the ``test`` places are held out for ``heldout_eval.py``
  and ``synthetic_flight.py``), or object-free parts of the reference frames;
* the objects are the SAM cut-outs from ``sprites.py``, pasted at any rotation and
  at their real size (the drone always flies at 600 m), with mild lighting
  changes; labels come from ``sprites.render``, which does not inflate rotated boxes;
* unlabelled terrain patches are pasted with the same soft object-shaped masks,
  so a pasted-looking edge is not a shortcut for "object";
* everything is composited at source resolution and then reduced by the view
  level's exact factor with INTER_AREA, which is how the evaluator renders views.

About a fifth of the samples are real reference crops with their real labels,
rotated only by multiples of 90 degrees (exact for boxes), to anchor the real
appearance and the annotation convention.

``training_data.py`` is the V1/V2 recipe and is left as it is; ``data_v2.py`` imports it.
"""

import json
from pathlib import Path

import cv2
import numpy as np

import sprites as sprite_cache
from dtos import OBJECT_CLASSES


ROOT = Path(__file__).resolve().parent
SOURCE_GSD = 0.21
LEVELS = (0, 1, 2)
LEVEL_WEIGHTS = (0.35, 0.4, 0.25)
# Only these scenes are trained on: the held-out flights written by synthetic_flight.py live under src/ too.
TRAINING_SCENES = ('helsinki',)


def clip_labels(boxes, classes, width, height, minimum_visibility=0.45):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int64)
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    size = clipped[:, 2:] - clipped[:, :2]
    keep = (size.min(axis=1) >= 1.5) & (size.prod(axis=1) >= minimum_visibility * np.maximum(area, 1e-6))
    return clipped[keep], classes[keep]


def rotate_boxes90(boxes, size, turns):
    """Rotate boxes with the image by ``turns`` quarter turns counter-clockwise (np.rot90)."""
    for _ in range(turns % 4):
        boxes = np.stack([boxes[:, 1], size - boxes[:, 2], boxes[:, 3], size - boxes[:, 0]], axis=1)
    return boxes


class TrainingScenes:
    def __init__(self, directory, size=640, seed=2026, terrain=ROOT / '.training' / 'terrain', scenes=TRAINING_SCENES, photometric=False, appearance=False, shadows=False):
        self.size = size
        self.shadows = shadows
        self.photometric = photometric
        self.appearance = appearance
        self.rng = np.random.default_rng(seed)
        self.frames = []
        for annotation_path in sorted(path for scene in scenes for path in (Path(directory) / scene).glob('annotations/*.json')):
            payload = json.loads(annotation_path.read_text())
            image = cv2.imread(str(annotation_path.parent.parent / 'images' / (annotation_path.stem + '.png')))
            if image is None:
                raise ValueError(f'Cannot read training image: {annotation_path}')
            boxes = np.asarray([item['bbox'] for item in payload['annotations']], dtype=np.float32).reshape(-1, 4)
            classes = np.asarray([OBJECT_CLASSES.index(item['object_id']) for item in payload['annotations']], dtype=np.int64)
            self.frames.append((image, boxes, classes))
        self.sprites = sprite_cache.load()
        missing = [name for name, found in zip(OBJECT_CLASSES, self.sprites) if not found]
        if not self.frames or missing:
            raise ValueError(f'Training needs reference frames and cut-outs of every class: {missing}')
        # Classes are pasted uniformly, or (with ``appearance``) small classes a little more often:
        # weight = max(1, (median class area / class area)^1/4), so no class drops below an equal share
        # before normalising. Fixed from object sizes, not from any score.
        self.class_weights = np.full(len(OBJECT_CLASSES), 1 / len(OBJECT_CLASSES))
        if appearance:
            areas = np.array([np.median([np.prod(item['box'][2:] - item['box'][:2]) for item in found]) for found in self.sprites])
            raw = np.maximum(1.0, (np.median(areas) / areas) ** 0.25)
            self.class_weights = raw / raw.sum()
        scales = json.loads((terrain / 'gsd.json').read_text())
        self.terrain = []
        for path in sorted((terrain / 'train').glob('*.jpg')):
            image = cv2.imread(str(path))
            if image is not None:
                self.terrain.append((image, scales[f'train/{path.name}']))
        if not self.terrain:
            raise ValueError(f'No training terrain in {terrain / "train"}; run fetch_terrain.py')

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ ground

    def _terrain(self, extent):
        """A square of outside terrain covering ``extent`` source pixels, at roughly the source scale."""
        image, gsd = self.terrain[int(self.rng.integers(len(self.terrain)))]
        # Ground scale jitter, and a cap at the tile size: a small tile is shown slightly zoomed out.
        wanted = extent * SOURCE_GSD / gsd * float(self.rng.uniform(0.8, 1.25))
        side = int(min(wanted, min(image.shape[:2])))
        x = int(self.rng.integers(image.shape[1] - side + 1))
        y = int(self.rng.integers(image.shape[0] - side + 1))
        patch = image[y:y + side, x:x + side]
        patch = np.rot90(patch, int(self.rng.integers(4)))
        return cv2.resize(patch, (extent, extent), interpolation=cv2.INTER_AREA if side > extent else cv2.INTER_LINEAR)

    def _reference_ground(self, extent):
        """An object-free square of the reference frames, or None if none was found."""
        image, boxes, _ = self.frames[int(self.rng.integers(len(self.frames)))]
        side = min(extent, min(image.shape[:2]))
        for _ in range(20):
            x = int(self.rng.integers(image.shape[1] - side + 1))
            y = int(self.rng.integers(image.shape[0] - side + 1))
            touching = (boxes[:, 0] < x + side + 8) & (boxes[:, 2] > x - 8) & (boxes[:, 1] < y + side + 8) & (boxes[:, 3] > y - 8)
            if not touching.any():
                patch = image[y:y + side, x:x + side]
                return cv2.resize(patch, (extent, extent), interpolation=cv2.INTER_LINEAR) if side != extent else patch.copy()
        return None

    # ------------------------------------------------------------------ samples

    def _real(self):
        """A crop of a reference frame with its real labels, at a random level."""
        image, boxes, classes = self.frames[int(self.rng.integers(len(self.frames)))]
        level = int(self.rng.choice(LEVELS, p=LEVEL_WEIGHTS))
        extent = min(self.size << level, min(image.shape[:2]))
        if len(boxes) and self.rng.random() < 0.85:
            center = boxes[int(self.rng.integers(len(boxes)))].reshape(2, 2).mean(axis=0) + self.rng.uniform(-0.4, 0.4, 2) * extent
        else:
            center = self.rng.uniform([0, 0], image.shape[1::-1])
        x = int(np.clip(center[0] - extent / 2, 0, image.shape[1] - extent))
        y = int(np.clip(center[1] - extent / 2, 0, image.shape[0] - extent))
        crop = image[y:y + extent, x:x + extent]
        factor = self.size / extent
        output = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA) if factor < 1 else crop.copy()
        boxes, classes = clip_labels((boxes - [x, y, x, y]) * factor, classes, self.size, self.size)
        return output, boxes, classes

    def _paste(self, canvas, rgb, alpha, x, y):
        height, width = alpha.shape
        region = canvas[y:y + height, x:x + width].astype(np.float32)
        canvas[y:y + height, x:x + width] = (rgb + region * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)

    def _restyle(self, rgb, alpha, ground):
        """Change a cut-out's colours the way another scene's light and grading might.

        Every class is one instance with one set of colours; without this the detector
        learns those exact colours (on re-lit held-out views the hangar fell to 0.2 AP).
        Works on the unpremultiplied colours and returns premultiplied ones.
        """
        weight = alpha[..., None]
        inside = alpha > 0.5
        colour = np.where(weight > 0.01, rgb / np.maximum(weight, 1e-3), 0).astype(np.float32)
        if not inside.any():
            return rgb
        if self.rng.random() < 0.5:
            colour = 255 * (colour.clip(0, 255) / 255) ** self.rng.uniform(0.6, 1.6)
        mean = colour[inside].mean(axis=0)
        if self.rng.random() < 0.5:
            colour = (colour - mean) * self.rng.uniform(0.6, 1.4) + mean
        if self.rng.random() < 0.5:
            grey = colour.mean(axis=2, keepdims=True)
            colour = grey + (colour - grey) * self.rng.uniform(0.5, 1.5)
        if self.rng.random() < 0.5:
            # Partial colour transfer towards the ground it stands on (ambient light).
            colour = colour + self.rng.uniform(0, 0.6) * (ground.reshape(-1, 3).mean(axis=0) - colour[inside].mean(axis=0))
        colour = colour * self.rng.uniform(0.7, 1.3) * self.rng.uniform(0.85, 1.15, 3)
        return (colour.clip(0, 255) * weight).astype(np.float32)

    def _shadow(self, canvas, alpha, x, y, direction, length, darkness):
        """Darken the ground under the object's silhouette shifted along the sun direction.

        The validation flight has low sun and every object casts a dark shadow; the reference
        cut-outs have none. ``length`` is relative to the object's size, ``darkness`` is the
        remaining brightness in full shadow. The object is pasted over it afterwards.
        """
        height, width = alpha.shape
        shift = direction * length * max(width, height)
        pad = int(np.ceil(np.abs(shift).max())) + 3
        mask = np.zeros((height + 2 * pad, width + 2 * pad), np.float32)
        mask[pad:pad + height, pad:pad + width] = alpha
        mask = cv2.warpAffine(mask, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), mask.shape[::-1])
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.6, 0.02 * max(width, height)))
        top, left = y - pad, x - pad
        y0, x0 = max(0, top), max(0, left)
        y1, x1 = min(canvas.shape[0], top + mask.shape[0]), min(canvas.shape[1], left + mask.shape[1])
        if y1 <= y0 or x1 <= x0:
            return
        part = mask[y0 - top:y1 - top, x0 - left:x1 - left, None]
        region = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = (region * (1 - (1 - darkness) * part)).clip(0, 255).astype(np.uint8)

    def _synthetic(self):
        level = int(self.rng.choice(LEVELS, p=LEVEL_WEIGHTS))
        extent = self.size << level
        canvas = None
        if self.rng.random() < 0.15:
            canvas = self._reference_ground(extent)
        if canvas is None:
            canvas = self._terrain(extent)
        boxes, classes = [], []

        # Unlabelled look-alikes first: terrain from elsewhere, cut to an object's shape.
        for _ in range(int(self.rng.integers(0, 7))):
            label = int(self.rng.integers(len(OBJECT_CLASSES)))
            source = self.sprites[label][int(self.rng.integers(len(self.sprites[label])))]
            rendered = sprite_cache.render(source, self.rng.uniform(0, 360), self.rng.uniform(0.7, 1.6))
            if rendered is None:
                continue
            _, alpha, _ = rendered
            height, width = alpha.shape
            if width >= extent or height >= extent:
                continue
            texture = self._terrain(max(width, height))[:height, :width].astype(np.float32)
            x, y = int(self.rng.integers(extent - width + 1)), int(self.rng.integers(extent - height + 1))
            self._paste(canvas, texture * alpha[..., None], alpha, x, y)

        count = 0 if self.rng.random() < 0.1 else int(self.rng.integers(1, 11))
        # One sun per image: every object's shadow falls the same way, as in a render.
        sun = None
        if self.shadows and self.rng.random() < 0.7:
            angle = self.rng.uniform(0, 2 * np.pi)
            sun = (np.array([np.cos(angle), np.sin(angle)]), self.rng.uniform(0.08, 0.4), self.rng.uniform(0.35, 0.75))
        for _ in range(count):
            label = int(self.rng.choice(len(OBJECT_CLASSES), p=self.class_weights))
            source = self.sprites[label][int(self.rng.integers(len(self.sprites[label])))]
            rendered = sprite_cache.render(source, self.rng.uniform(0, 360), self.rng.uniform(0.85, 1.18))
            if rendered is None:
                continue
            rgb, alpha, local = rendered
            height, width = alpha.shape
            if width >= extent or height >= extent:
                continue
            for _ in range(10):
                x, y = int(self.rng.integers(extent - width + 1)), int(self.rng.integers(extent - height + 1))
                box = local + [x, y, x, y]
                if not boxes or (np.minimum(np.asarray(boxes)[:, 2:], box[2:]) <= np.maximum(np.asarray(boxes)[:, :2], box[:2])).any(axis=1).all():
                    break
            else:
                continue
            if self.appearance:
                rgb = self._restyle(rgb, alpha, canvas[y:y + height, x:x + width])
            else:
                # Lighting differs between scenes: overall gain and a slight colour cast per object.
                rgb = rgb * float(self.rng.uniform(0.8, 1.2)) * self.rng.uniform(0.93, 1.07, 3).astype(np.float32)
            if sun is not None:
                self._shadow(canvas, alpha, x, y, *sun)
            self._paste(canvas, np.minimum(rgb, 255 * alpha[..., None]), alpha, x, y)
            boxes.append(box)
            classes.append(label)

        factor = 1 << level
        image = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA) if factor > 1 else canvas
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4) / factor
        boxes, classes = clip_labels(boxes, classes, self.size, self.size)
        return image, boxes, classes

    def sample(self):
        image, boxes, classes = self._real() if self.rng.random() < 0.2 else self._synthetic()
        turns = int(self.rng.integers(4))
        image = np.ascontiguousarray(np.rot90(image, turns))
        boxes = rotate_boxes90(boxes, self.size, turns)
        if self.rng.random() < 0.5:
            image = cv2.flip(image, 1)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.rng.uniform(-6, 6)) % 180
        hsv[..., 1] *= self.rng.uniform(0.7, 1.3)
        hsv[..., 2] *= self.rng.uniform(0.75, 1.25)
        image = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.photometric:
            # Another scene may be rendered with other light and grading: global gamma, contrast and colour cast.
            image = image.astype(np.float32)
            if self.rng.random() < 0.5:
                image = 255 * (image / 255) ** self.rng.uniform(0.7, 1.4)
            if self.rng.random() < 0.5:
                image = (image - image.mean()) * self.rng.uniform(0.75, 1.25) + image.mean()
            if self.rng.random() < 0.5:
                image = image * self.rng.uniform(0.88, 1.12, 3)
            image = image.clip(0, 255).astype(np.uint8)
        if self.rng.random() < 0.1:
            image = cv2.GaussianBlur(image, (3, 3), float(self.rng.uniform(0.3, 0.8)))
        if self.rng.random() < 0.1:
            noise = self.rng.normal(0, self.rng.uniform(1, 4), image.shape)
            image = (image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        xywh = np.concatenate(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), axis=1) / self.size
        return np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)), xywh.astype(np.float32), classes

    def batch(self, count):
        return collate([self.sample() for _ in range(count)])


def collate(samples):
    import torch

    return {
        'img': torch.from_numpy(np.stack([item[0] for item in samples])),
        'bboxes': torch.from_numpy(np.concatenate([item[1] for item in samples])),
        'cls': torch.from_numpy(np.concatenate([item[2] for item in samples])).float().view(-1, 1),
        'batch_idx': torch.from_numpy(np.concatenate([np.full(len(item[2]), index, dtype=np.int64) for index, item in enumerate(samples)])),
    }
