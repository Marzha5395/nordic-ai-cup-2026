from dataclasses import dataclass
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dtos import OBJECT_CLASSES


@dataclass
class Detection:
    box: np.ndarray
    scores: np.ndarray
    quality: float = 1.0

    @property
    def confidence(self):
        return float(self.scores.max())


def overlap(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    intersection = np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-6)


def suppress(detections, threshold=0.45, limit=250):
    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept = []
    while remaining and len(kept) < limit:
        best, remaining = remaining[0], remaining[1:]
        kept.append(best)
        if remaining:
            ious = overlap(best.box, [item.box for item in remaining])
            remaining = [item for item, iou in zip(remaining, ious) if iou < threshold]
    return kept


class Detector:
    def __init__(self, path=None):
        path = Path(path or os.environ.get('DRONE_MODEL_PATH', Path(__file__).resolve().parent / 'weights' / 'detector.onnx'))
        if not path.is_file():
            raise FileNotFoundError(f'Missing trained detector: {path}. Run train.py and export_model.py before serving.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get('DRONE_INFERENCE_THREADS', '4'))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(value, int) for value in shape[2:]):
            raise ValueError('Export a fixed-shape, batch-one RGB detector')
        self.height, self.width = shape[2:]
        metadata = self.session.get_modelmeta().custom_metadata_map
        if json.loads(metadata.get('drone_classes', 'null')) != list(OBJECT_CLASSES):
            raise ValueError('Model class order does not match the drone protocol')
        self.threshold = float(os.environ.get('DRONE_CONFIDENCE', '0.12'))
        if not 0 < self.threshold < 1:
            raise ValueError('DRONE_CONFIDENCE must be between zero and one')
        self.session.run(None, {self.input_name: np.zeros(shape, dtype=np.float32)})

    def detect(self, image, request):
        height, width = image.shape[:2]
        scale = min(self.width / width, self.height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left, top = (self.width - resized_width) // 2, (self.height - resized_height) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        tensor = np.ascontiguousarray(canvas[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        raw = self.session.run(None, {self.input_name: tensor})[0]
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + len(OBJECT_CLASSES):
            raise ValueError(f'Unexpected detector output shape: {raw.shape}')
        output = raw[0].T
        selected = output[np.isfinite(output).all(axis=1) & (output[:, 4:].max(axis=1) >= self.threshold)]
        if not len(selected):
            return []
        selected = selected[np.argsort(-selected[:, 4:].max(axis=1))[:1000]]
        boxes = np.concatenate((selected[:, :2] - selected[:, 2:4] / 2, selected[:, :2] + selected[:, 2:4] / 2), axis=1)
        boxes -= np.array([left, top, left, top])
        boxes /= np.array([resized_width / width, resized_height / height] * 2)
        region = np.asarray(request.view.source_region_xyxy, dtype=np.float32)
        source_scale = (region[2:] - region[:2]) / [width, height]
        detections = []
        for box, row in zip(boxes, selected):
            if min(box[2:] - box[:2]) < 1:
                continue
            quality = 0.6 if (box[:2] < 2).any() or (box[2:] > np.array([width, height]) - 2).any() else 1.0
            box[[0, 2]] = box[[0, 2]].clip(0, width)
            box[[1, 3]] = box[[1, 3]].clip(0, height)
            if min(box[2:] - box[:2]) <= 0:
                continue
            source_box = box * np.tile(source_scale, 2) + np.tile(region[:2], 2)
            detections.append(Detection(source_box.astype(np.float32), row[4:].clip(0, 1), quality))
        return suppress(detections)
