"""Synthesise 960x540 training views out of the reference scene.

The competition hands the detector a 960x540 image taken from a 3840x2160
frame at one of three zoom levels, so every training image is built the same
way: compose at source resolution, then downsample by the level's exact
integer factor with ``INTER_AREA``, which is what the evaluator does.

Two problems the reference scene creates, and how this module answers them:

* **One orientation per class.** The scene holds a single instance of each
  class, at a single yaw. A detector trained on that learns the yaw as well as
  the object. So objects are cut out with a mask, rotated, and pasted back,
  and the box that comes with them is derived from an oriented-rectangle model
  rather than the loose hull of a rotated box.
* **One terrain.** Twenty-five frames of one place near Helsinki is not a
  distribution of terrain, it is a single sample from one. A detector trained
  on it alone reported twenty objects on the first frames of the validation
  flight and eighty on the last -- false tracks piling up on ground it had
  never seen. So most backgrounds come from ``external/``: public-domain USGS
  aerial imagery over deserts, farmland, forest, mountains, coast and city,
  downloaded by ``training/fetch_backgrounds.py`` and split so that a third of
  the locations are never trained on. The reference frames stay in the mix,
  with the real objects painted out, because they are the only *rendered*
  terrain available and the flight being scored is rendered.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160
VIEW_WIDTH = 960
VIEW_HEIGHT = 540

SOURCE_REGION_SIZES = {
    0: (3840, 2160),
    1: (1920, 1080),
    2: (960, 540),
}

OBJECT_CLASSES = (
    'hangar', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower',
    'medium_launcher', 'medium_plane', 'mine_roller', 'small_launcher',
    'small_plane', 'small_tower', 'ta-ta', 'tank', 'condor', 'jammer',
    'spacecraft',
)
CLASS_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #

def _donor_offset(
    box: Tuple[int, int, int, int],
    boxes: Sequence[Tuple[int, int, int, int]],
    shape: Tuple[int, int],
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Find a nearby patch of empty ground the same size as ``box``."""
    height, width = shape
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    for _ in range(60):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        distance = float(rng.uniform(1.4, 4.0)) * max(box_w, box_h)
        offset_x = int(round(math.cos(angle) * distance))
        offset_y = int(round(math.sin(angle) * distance))
        donor = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        if donor[0] < 0 or donor[1] < 0 or donor[2] > width or donor[3] > height:
            continue
        if any(
            donor[0] < other[2] and other[0] < donor[2]
            and donor[1] < other[3] and other[1] < donor[3]
            for other in boxes
        ):
            continue
        return offset_x, offset_y
    return None


def build_backgrounds(scene_directory: Path, cache_directory: Path) -> List[Path]:
    """Paint every annotated object out of the scene's frames.

    A composed image must not contain an object the label file does not list,
    or the detector is trained to call that object background.

    Inpainting leaves a smooth starburst where a textured forest used to be,
    which is a conspicuous mark on roughly ten places in every frame. Copying a
    nearby patch of empty ground over the object instead keeps the terrain
    statistics intact, and inpainting is only the fallback for the boxes that
    have no clear ground near them.
    """
    cache_directory.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for annotation_path in sorted((scene_directory / 'annotations').glob('frame_*.json')):
        payload = json.loads(annotation_path.read_text())
        frame = payload['frame']
        destination = cache_directory / f'background_{frame:06d}.png'
        paths.append(destination)
        if destination.exists():
            continue
        image = cv2.imread(
            str(scene_directory / 'images' / f'frame_{frame:06d}.png'), cv2.IMREAD_COLOR
        )
        if image is None:
            raise FileNotFoundError(f'missing frame {frame} of {scene_directory}')
        height, width = image.shape[:2]
        rng = np.random.default_rng(frame)

        margin = 8
        boxes = []
        for annotation in payload['annotations']:
            x1, y1, x2, y2 = (int(round(c)) for c in annotation['bbox'])
            boxes.append(
                (
                    max(0, x1 - margin), max(0, y1 - margin),
                    min(width, x2 + margin), min(height, y2 + margin),
                )
            )

        painted = image.copy()
        leftover = np.zeros((height, width), np.uint8)
        for box in boxes:
            offset = _donor_offset(box, boxes, (height, width), rng)
            if offset is None:
                cv2.rectangle(leftover, (box[0], box[1]), (box[2], box[3]), 255, -1)
                continue
            x1, y1, x2, y2 = box
            donor = image[y1 + offset[1]:y2 + offset[1], x1 + offset[0]:x2 + offset[0]]
            blend = np.zeros((y2 - y1, x2 - x1), np.float32)
            cv2.ellipse(
                blend,
                ((x2 - x1) // 2, (y2 - y1) // 2),
                (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                0.0, 0.0, 360.0, 1.0, -1,
            )
            radius = max(3, (min(x2 - x1, y2 - y1) // 4) | 1)
            blend = cv2.GaussianBlur(blend, (radius, radius), 0)[..., None]
            target = painted[y1:y2, x1:x2].astype(np.float32)
            painted[y1:y2, x1:x2] = (
                target * (1.0 - blend) + donor.astype(np.float32) * blend
            ).astype(np.uint8)

        if leftover.any():
            painted = cv2.inpaint(painted, leftover, 7, cv2.INPAINT_TELEA)
        cv2.imwrite(str(destination), painted)
    return paths


# --------------------------------------------------------------------------- #
# Oriented-rectangle box model
# --------------------------------------------------------------------------- #

def oriented_extent(
    mask: np.ndarray,
    box_in_patch: Tuple[int, int, int, int],
) -> Tuple[float, float, float]:
    """Model an object as an oriented rectangle: return (length, width, angle).

    The supplied boxes are axis-aligned hulls of objects that are themselves
    rotated, so rotating a box and taking its hull again would inflate it
    badly. Instead recover the rectangle whose hull *is* the supplied box: take
    the orientation from the mask and solve

        W = a |cos a| + b |sin a|
        H = a |sin a| + b |cos a|

    for the side lengths. Near 45 degrees that system is ill-conditioned, so
    fall back to scaling the mask's own minimum-area rectangle to match.
    """
    box_x1, box_y1, box_x2, box_y2 = box_in_patch
    width = float(box_x2 - box_x1)
    height = float(box_y2 - box_y1)

    points = cv2.findNonZero(mask)
    if points is None or len(points) < 5:
        return width, height, 0.0
    (_, _), (rect_w, rect_h), angle_degrees = cv2.minAreaRect(points)
    angle = math.radians(angle_degrees)
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))

    determinant = cosine * cosine - sine * sine
    if abs(determinant) > 0.20:
        long_side = (width * cosine - height * sine) / determinant
        short_side = (height * cosine - width * sine) / determinant
        if long_side > 1.0 and short_side > 1.0:
            return long_side, short_side, angle

    if rect_w < 1.0 or rect_h < 1.0:
        return width, height, 0.0
    hull_w = rect_w * cosine + rect_h * sine
    hull_h = rect_w * sine + rect_h * cosine
    scale = math.sqrt(max(1e-6, (width * height) / max(1e-6, hull_w * hull_h)))
    return rect_w * scale, rect_h * scale, angle


def modelled_hull(
    length: float,
    width: float,
    angle: float,
    extra_rotation: float,
) -> Tuple[float, float]:
    """Axis-aligned size of the oriented rectangle after an extra rotation."""
    total = angle + extra_rotation
    cosine, sine = abs(math.cos(total)), abs(math.sin(total))
    return length * cosine + width * sine, length * sine + width * cosine


# --------------------------------------------------------------------------- #
# Instance sprites
# --------------------------------------------------------------------------- #

class Sprite:
    """One cut-out instance, ready to be rotated and pasted."""

    __slots__ = ('object_id', 'class_index', 'bgr', 'alpha', 'length', 'width', 'angle')

    def __init__(self, object_id: str, patch: np.ndarray, mask: np.ndarray,
                 box_in_patch: Tuple[int, int, int, int]):
        self.object_id = object_id
        self.class_index = CLASS_INDEX[object_id]
        length, width, angle = oriented_extent(mask, box_in_patch)
        self.length = length
        self.width = width
        self.angle = angle

        # Centre the sprite on the annotated box centre: that is the point the
        # object is rotated about, and the point the modelled hull is centred
        # on afterwards.
        box_x1, box_y1, box_x2, box_y2 = box_in_patch
        centre_x = 0.5 * (box_x1 + box_x2)
        centre_y = 0.5 * (box_y1 + box_y2)
        half = int(math.ceil(0.5 * math.hypot(patch.shape[1], patch.shape[0]))) + 2
        size = 2 * half
        canvas = np.zeros((size, size, 3), np.uint8)
        alpha = np.zeros((size, size), np.uint8)
        offset_x = int(round(half - centre_x))
        offset_y = int(round(half - centre_y))
        canvas[offset_y:offset_y + patch.shape[0], offset_x:offset_x + patch.shape[1]] = patch
        alpha[offset_y:offset_y + mask.shape[0], offset_x:offset_x + mask.shape[1]] = mask
        # Ramp the alpha inwards from the mask boundary. A hard cut-out edge
        # leaves a halo of foreign terrain around every pasted object, and a
        # detector will happily learn the halo instead of the object.
        distance = cv2.distanceTransform((alpha > 127).astype(np.uint8), cv2.DIST_L2, 3)
        ramp = 2.0 if distance.max() > 4.0 else 1.0
        alpha = np.clip(distance / ramp, 0.0, 1.0)
        alpha = (cv2.GaussianBlur(alpha, (3, 3), 0) * 255.0).astype(np.uint8)
        self.bgr = canvas
        self.alpha = alpha


def build_sprites(instances: Sequence[Dict]) -> Dict[str, List[Sprite]]:
    """Group sprites by class so a sampler can balance across classes."""
    by_class: Dict[str, List[Sprite]] = {}
    for instance in instances:
        sprite = Sprite(
            instance['object_id'],
            instance['patch'],
            instance['mask'],
            instance['box_in_patch'],
        )
        by_class.setdefault(instance['object_id'], []).append(sprite)
    return by_class


# --------------------------------------------------------------------------- #
# Pasting
# --------------------------------------------------------------------------- #

def paste_sprite(
    canvas: np.ndarray,
    sprite: Sprite,
    centre_x: float,
    centre_y: float,
    rotation: float,
    scale: float,
    flip: bool,
    rng: np.random.Generator,
) -> Optional[Tuple[float, float, float, float]]:
    """Paste one rotated sprite and return its source-pixel box, or None.

    ``rotation`` is in radians and ``centre_x``/``centre_y`` are in canvas
    pixels. The box returned is the union of the modelled hull and the pasted
    alpha's own extent, so it can never be smaller than the pixels on screen.
    """
    bgr, alpha = sprite.bgr, sprite.alpha
    if flip:
        bgr = cv2.flip(bgr, 1)
        alpha = cv2.flip(alpha, 1)
    size = bgr.shape[0]
    centre = (size / 2.0 - 0.5, size / 2.0 - 0.5)
    matrix = cv2.getRotationMatrix2D(centre, math.degrees(rotation), scale)
    rotated_bgr = cv2.warpAffine(
        bgr, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    rotated_alpha = cv2.warpAffine(
        alpha, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0
    )

    # Photometric jitter, applied to the sprite alone: the same asset should
    # not always arrive with the same exposure.
    gain = float(rng.uniform(0.80, 1.20))
    bias = float(rng.uniform(-18.0, 18.0))
    rotated_bgr = np.clip(rotated_bgr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    # Rotation is about the sprite's own centre, which is the annotated box
    # centre, so the flip mirrors the modelled angle rather than moving it.
    angle = -sprite.angle if flip else sprite.angle
    hull_w, hull_h = modelled_hull(sprite.length, sprite.width, angle, rotation)
    hull_w *= scale
    hull_h *= scale

    canvas_height, canvas_width = canvas.shape[:2]
    top_left_x = int(round(centre_x - size / 2.0))
    top_left_y = int(round(centre_y - size / 2.0))

    dst_x1 = max(0, top_left_x)
    dst_y1 = max(0, top_left_y)
    dst_x2 = min(canvas_width, top_left_x + size)
    dst_y2 = min(canvas_height, top_left_y + size)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return None

    src_x1 = dst_x1 - top_left_x
    src_y1 = dst_y1 - top_left_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    sprite_alpha = rotated_alpha[src_y1:src_y2, src_x1:src_x2]
    if not sprite_alpha.any():
        return None
    weights = (sprite_alpha.astype(np.float32) / 255.0)[..., None]
    region = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    sprite_region = rotated_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = (
        region * (1.0 - weights) + sprite_region * weights
    ).astype(np.uint8)

    model_box = (
        centre_x - hull_w / 2.0,
        centre_y - hull_h / 2.0,
        centre_x + hull_w / 2.0,
        centre_y + hull_h / 2.0,
    )
    # The alpha's own extent, in canvas coordinates, as a floor on the box.
    columns = np.nonzero(rotated_alpha.any(axis=0))[0]
    rows = np.nonzero(rotated_alpha.any(axis=1))[0]
    alpha_box = (
        top_left_x + float(columns[0]),
        top_left_y + float(rows[0]),
        top_left_x + float(columns[-1]) + 1.0,
        top_left_y + float(rows[-1]) + 1.0,
    )
    return (
        min(model_box[0], alpha_box[0]),
        min(model_box[1], alpha_box[1]),
        max(model_box[2], alpha_box[2]),
        max(model_box[3], alpha_box[3]),
    )


# --------------------------------------------------------------------------- #
# Whole samples
# --------------------------------------------------------------------------- #

def _intersection_over_area(box: Sequence[float], other: Sequence[float]) -> float:
    left = max(box[0], other[0])
    top = max(box[1], other[1])
    right = min(box[2], other[2])
    bottom = min(box[3], other[3])
    if right <= left or bottom <= top:
        return 0.0
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return (right - left) * (bottom - top) / area


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Push the composed view around in colour, exposure and sharpness.

    The evaluation flight is over different terrain under different light, and
    25 frames of one location is not enough variation on its own.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-12, 13))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * float(rng.uniform(0.55, 1.45)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * float(rng.uniform(0.70, 1.30)), 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gamma = float(rng.uniform(0.75, 1.35))
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    image = cv2.LUT(image, table)

    if rng.random() < 0.30:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(0.4, 1.0)))
    if rng.random() < 0.50:
        noise = rng.normal(0.0, float(rng.uniform(1.5, 6.0)), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


LEVEL_WEIGHTS = {0: 0.08, 1: 0.70, 2: 0.22}
# How often the background comes from outside the reference scene. High,
# because one terrain is what produced a detector that fired on everything the
# validation flight showed it -- but not all, because the reference frames are
# the only rendered terrain there is and the scored flight is rendered.
EXTERNAL_BACKGROUND_PROBABILITY = 0.62
# Every background is resampled by a random factor before cropping. Without it
# the outside mosaics, which are smaller than a level-1 region and have to be
# enlarged, would be systematically softer than the reference frames -- and
# softness would become a reliable cue for "no object here".
BACKGROUND_RESCALE = (0.75, 1.55)
# Every labelled object in a composed image arrived by being cut out of one
# place and blended into another. If that is the only kind of blended patch in
# the training set, "blended patch" is a perfect predictor of "object", and a
# detector is entitled to learn it -- then anything unusual on unfamiliar
# terrain looks like an object. So some patches are blended in that are just
# more terrain, carry no label, and have to be answered with silence.
DISTRACTOR_PROBABILITY = 0.65
DISTRACTOR_COUNT = (1, 6)
EMPTY_SAMPLE_PROBABILITY = 0.10
# How often a sample keeps the real frame, objects and all, instead of the
# inpainted one. Pasted objects carry the orientation variety, but they also
# carry paste artefacts, and a detector that only ever sees pasted objects can
# learn the artefact. These samples are the anchor to the real renderer.
REAL_FRAME_PROBABILITY = 0.28
# How often a composed background is lifted out of the frame at an angle. The
# reference scene covers only about one and a half frames' worth of unique
# ground, and a rotated crop is both new terrain and a new sun direction.
ROTATED_BACKGROUND_PROBABILITY = 0.55


class ExternalBackgrounds:
    """Aerial mosaics, read from disk on demand and kept in a small cache.

    There are a couple of hundred of them at 2304x2304, which is more than a
    worker process wants resident, and each sample needs exactly one.
    """

    def __init__(self, directories: Sequence[Path], cache_size: int = 10):
        self.paths: List[Path] = []
        for directory in directories:
            if directory is None:
                continue
            directory = Path(directory)
            if not directory.is_dir():
                continue
            self.paths.extend(
                sorted(
                    path for path in directory.iterdir()
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                )
            )
        self.cache_size = cache_size
        self._cache: Dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def sample(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        path = self.paths[int(rng.integers(len(self.paths)))]
        image = self._cache.get(path)
        if image is None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                return None
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = image
        return image


class SceneAssets:
    """Everything the sampler needs: reference scene plus outside terrain."""

    def __init__(self, originals, annotations, backgrounds, sprites_by_class,
                 external=None):
        self.originals = originals
        self.annotations = annotations
        self.backgrounds = backgrounds
        self.sprites_by_class = sprites_by_class
        self.external = external or ExternalBackgrounds([])


def load_scene_assets(
    scene_directory: Path,
    cache_directory: Path,
    external_directories: Sequence[Path] = (),
) -> SceneAssets:
    from masks import load_instances

    background_paths = build_backgrounds(scene_directory, cache_directory)
    backgrounds = []
    for path in background_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        backgrounds.append(image)

    originals, annotations = [], []
    for annotation_path in sorted((scene_directory / 'annotations').glob('frame_*.json')):
        payload = json.loads(annotation_path.read_text())
        frame = payload['frame']
        image = cv2.imread(
            str(scene_directory / 'images' / f'frame_{frame:06d}.png'), cv2.IMREAD_COLOR
        )
        originals.append(image)
        annotations.append(
            [
                (CLASS_INDEX[a['object_id']], [float(c) for c in a['bbox']])
                for a in payload['annotations']
                if a['object_id'] in CLASS_INDEX
            ]
        )

    instances = load_instances(scene_directory.name)
    return SceneAssets(
        originals, annotations, backgrounds, build_sprites(instances),
        ExternalBackgrounds(external_directories),
    )


def paste_distractor(
    canvas: np.ndarray,
    donor: np.ndarray,
    rng: np.random.Generator,
    occupied: Sequence[Tuple[float, float, float, float]],
) -> bool:
    """Blend a patch of terrain into the canvas, with no label attached.

    Built the same way a sprite is -- an irregular soft-edged mask, a random
    rotation, its own exposure -- so that nothing but the contents tells it
    apart from a real object.
    """
    height, width = donor.shape[:2]
    size = int(rng.integers(18, 190))
    canvas_height, canvas_width = canvas.shape[:2]
    if width <= size + 2 or height <= size + 2:
        return False
    if canvas_width <= size + 2 or canvas_height <= size + 2:
        return False

    left = int(rng.integers(0, canvas_width - size))
    top = int(rng.integers(0, canvas_height - size))
    box = (float(left), float(top), float(left + size), float(top + size))
    if any(_intersection_over_area(box, other) > 0.0
           or _intersection_over_area(other, box) > 0.0 for other in occupied):
        return False

    x = int(rng.integers(0, width - size))
    y = int(rng.integers(0, height - size))
    patch = donor[y:y + size, x:x + size].copy()

    # An irregular blob rather than a disc: a few overlapping ellipses.
    mask = np.zeros((size, size), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        centre = (int(rng.integers(size // 4, 3 * size // 4)),
                  int(rng.integers(size // 4, 3 * size // 4)))
        axes = (max(2, int(rng.integers(size // 6, size // 2))),
                max(2, int(rng.integers(size // 6, size // 2))))
        cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    if not mask.any():
        return False
    distance = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / 2.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]

    gain = float(rng.uniform(0.70, 1.35))
    bias = float(rng.uniform(-28.0, 28.0))
    patch = np.clip(patch.astype(np.float32) * gain + bias, 0, 255)

    right, bottom = left + size, top + size
    region = canvas[top:bottom, left:right].astype(np.float32)
    canvas[top:bottom, left:right] = (
        region * (1.0 - alpha) + patch * alpha
    ).astype(np.uint8)
    return True


def _scaled_crop(
    source: np.ndarray,
    rng: np.random.Generator,
    region_width: int,
    region_height: int,
    rotate: bool,
) -> np.ndarray:
    """Take a region out of a background, at a random zoom and angle.

    Cropping ``region / factor`` pixels and resizing to ``region`` is the same
    as rescaling the whole background and cropping, without allocating a
    resampled copy of a 4K frame for every sample. The factor is clamped so the
    crop always fits: the outside mosaics are 2304 square and a level-1 region
    is 1920x1080, so they are enlarged slightly and the reference frames can go
    either way -- which is the point. If every outside background were softer
    than every reference one, softness would be a reliable cue for "no object
    here", and the detector would learn it instead of the objects.
    """
    height, width = source.shape[:2]
    low, high = BACKGROUND_RESCALE
    lowest = max(region_width / width, region_height / height)
    if rotate:
        # A rotated crop reaches into the corners of its own bounding square.
        lowest = math.hypot(region_width, region_height) / min(width, height)
    factor = float(rng.uniform(max(low, lowest * 1.02), max(high, lowest * 1.25)))
    crop_width = min(width, max(8, int(round(region_width / factor))))
    crop_height = min(height, max(8, int(round(region_height / factor))))

    if rotate:
        margin = int(0.5 * math.hypot(crop_width, crop_height)) + 1
        centre_x = float(rng.uniform(min(margin, width * 0.5), max(margin, width - margin)))
        centre_y = float(rng.uniform(min(margin, height * 0.5), max(margin, height - margin)))
        matrix = cv2.getRotationMatrix2D(
            (centre_x, centre_y), float(rng.uniform(0.0, 360.0)), 1.0
        )
        matrix[0, 2] += crop_width / 2.0 - centre_x
        matrix[1, 2] += crop_height / 2.0 - centre_y
        crop = cv2.warpAffine(
            source, matrix, (crop_width, crop_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        origin_x = int(rng.integers(0, width - crop_width + 1))
        origin_y = int(rng.integers(0, height - crop_height + 1))
        crop = source[origin_y:origin_y + crop_height, origin_x:origin_x + crop_width]

    if (crop.shape[1], crop.shape[0]) != (region_width, region_height):
        interpolation = (
            cv2.INTER_AREA if crop.shape[1] > region_width else cv2.INTER_LINEAR
        )
        crop = cv2.resize(crop, (region_width, region_height), interpolation=interpolation)
    return np.ascontiguousarray(crop)


def render_sample(
    rng: np.random.Generator,
    assets: SceneAssets,
    level: Optional[int] = None,
    external_probability: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Build one 960x540 view and its YOLO labels.

    Returns the image and a list of ``(class_index, cx, cy, w, h)`` normalized
    to the transmitted view, which is what a YOLO label file wants.
    """
    if level is None:
        levels = list(LEVEL_WEIGHTS)
        level = int(rng.choice(levels, p=[LEVEL_WEIGHTS[value] for value in levels]))
    region_width, region_height = SOURCE_REGION_SIZES[level]

    frame_index = int(rng.integers(len(assets.backgrounds)))

    if external_probability is None:
        external_probability = EXTERNAL_BACKGROUND_PROBABILITY
    external = None
    if len(assets.external) and rng.random() < external_probability:
        external = assets.external.sample(rng)

    # A "real" sample keeps a reference frame exactly as it was rendered,
    # objects included, and is the only path that carries annotations.
    keep_real = external is None and rng.random() < REAL_FRAME_PROBABILITY
    origin_x = origin_y = 0
    if keep_real:
        source = assets.originals[frame_index]
        origin_x = int(rng.integers(0, SOURCE_WIDTH - region_width + 1))
        origin_y = int(rng.integers(0, SOURCE_HEIGHT - region_height + 1))
        canvas = source[
            origin_y:origin_y + region_height, origin_x:origin_x + region_width
        ].copy()
    else:
        source = external if external is not None else assets.backgrounds[frame_index]
        rotate = level > 0 and rng.random() < ROTATED_BACKGROUND_PROBABILITY
        canvas = _scaled_crop(source, rng, region_width, region_height, rotate)

    # Boxes are collected in canvas (source-pixel) coordinates and converted
    # once at the end.
    placed: List[Tuple[float, float, float, float]] = []
    entries: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if keep_real:
        for class_index, (x1, y1, x2, y2) in assets.annotations[frame_index]:
            box = (x1 - origin_x, y1 - origin_y, x2 - origin_x, y2 - origin_y)
            if box[2] <= 0 or box[3] <= 0 or box[0] >= region_width or box[1] >= region_height:
                continue
            placed.append(box)
            entries.append((class_index, box))

    if rng.random() < EMPTY_SAMPLE_PROBABILITY and not keep_real:
        count = 0                                  # pure background, no objects
    else:
        count = int(rng.integers(2, 11))
    class_names = list(assets.sprites_by_class)

    attempts = 0
    added = 0
    while added < count and attempts < count * 12:
        attempts += 1
        # Sample the class first so the rare instances are not starved by the
        # classes that happen to appear in every reference frame.
        name = class_names[int(rng.integers(len(class_names)))]
        options = assets.sprites_by_class[name]
        sprite = options[int(rng.integers(len(options)))]

        # Allow objects to hang off the edge: the ground truth keeps clipped
        # boxes, and the frame border is where every object first appears.
        centre_x = float(rng.uniform(-0.04, 1.04)) * region_width
        centre_y = float(rng.uniform(-0.04, 1.04)) * region_height
        rotation = float(rng.uniform(0.0, 2.0 * math.pi))
        scale = float(rng.uniform(0.88, 1.14))
        flip = bool(rng.random() < 0.5)

        hull_w, hull_h = modelled_hull(sprite.length, sprite.width, sprite.angle, rotation)
        tentative = (
            centre_x - scale * hull_w / 2.0, centre_y - scale * hull_h / 2.0,
            centre_x + scale * hull_w / 2.0, centre_y + scale * hull_h / 2.0,
        )
        if any(_intersection_over_area(tentative, other) > 0.05
               or _intersection_over_area(other, tentative) > 0.05 for other in placed):
            continue

        box = paste_sprite(canvas, sprite, centre_x, centre_y, rotation, scale, flip, rng)
        if box is None:
            continue
        placed.append(box)
        entries.append((sprite.class_index, box))
        added += 1

    # Unlabelled blended patches, placed clear of everything that is labelled.
    if rng.random() < DISTRACTOR_PROBABILITY:
        donor = assets.external.sample(rng)
        if donor is None:
            donor = assets.backgrounds[int(rng.integers(len(assets.backgrounds)))]
        wanted = int(rng.integers(*DISTRACTOR_COUNT))
        for _ in range(wanted * 3):
            if wanted <= 0:
                break
            if paste_distractor(canvas, donor, rng, placed):
                wanted -= 1

    scale_x = VIEW_WIDTH / float(region_width)
    scale_y = VIEW_HEIGHT / float(region_height)
    labels: List[Tuple[int, float, float, float, float]] = []
    for class_index, box in entries:
        clipped = (
            max(0.0, box[0]), max(0.0, box[1]),
            min(float(region_width), box[2]), min(float(region_height), box[3]),
        )
        full_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        visible = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if visible / full_area < 0.25:
            continue
        view_box = (
            clipped[0] * scale_x, clipped[1] * scale_y,
            clipped[2] * scale_x, clipped[3] * scale_y,
        )
        if view_box[2] - view_box[0] < 2.0 or view_box[3] - view_box[1] < 2.0:
            continue
        labels.append(
            (
                class_index,
                0.5 * (view_box[0] + view_box[2]) / VIEW_WIDTH,
                0.5 * (view_box[1] + view_box[3]) / VIEW_HEIGHT,
                (view_box[2] - view_box[0]) / VIEW_WIDTH,
                (view_box[3] - view_box[1]) / VIEW_HEIGHT,
            )
        )

    if (region_width, region_height) != (VIEW_WIDTH, VIEW_HEIGHT):
        canvas = cv2.resize(canvas, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
    return _photometric(canvas, rng), labels
