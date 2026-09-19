import os
from pathlib import Path

import cv2
import numpy as np
import torch

from detector import Detection, Detector, overlap, suppress
from dtos import OBJECT_CLASSES


class TorchDetector(Detector):
    def __init__(self, path, device='cuda:0', width=1536, threshold=0.05, canonical=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing local detector weights: {path}')
        if width <= 0 or width % 32 or not 0 < threshold < 1:
            raise ValueError('Width must be a positive multiple of 32 and threshold must be in (0, 1)')
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics import YOLO

        torch.set_num_threads(4)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled runtime, or DRONE_DEVICE=cpu for debugging only.')
        fast_half = self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device)[0] >= 7
        self.dtype = torch.float16 if fast_half else torch.float32
        self.full_width = width
        self.canonical = canonical
        self.width = width
        self.height = ((width * 9 // 16 + 31) // 32) * 32
        self.threshold = threshold
        self.model = YOLO(str(path)).model.fuse(verbose=False).eval().to(self.device, dtype=self.dtype)
        if [self.model.names[index] for index in range(len(self.model.names))] != list(OBJECT_CLASSES):
            raise ValueError('Detector classes disagree with the protocol')
        self.infer(np.zeros((1, 3, self.height, self.width), dtype=np.float32))

    def detect(self, image, request):
        if self.canonical:
            x1, y1, x2, y2 = request.view.source_region_xyxy
            scale = self.full_width / request.original_width
            self.width = max(32, int(np.ceil((x2 - x1) * scale / 32)) * 32)
            self.height = max(32, int(np.ceil((y2 - y1) * scale / 32)) * 32)
        return super().detect(image, request)

    def infer(self, tensor):
        with torch.inference_mode():
            output = self.model(torch.from_numpy(tensor).to(self.device, dtype=self.dtype))
            if isinstance(output, tuple):
                output = output[0]
            return output.float().cpu().numpy()


def fuse_views(views, threshold=0.55):
    groups = []
    ordered = sorted(((index, item) for index, view in enumerate(views) for item in view), key=lambda pair: pair[1].confidence, reverse=True)
    for index, item in ordered:
        best, best_iou = None, threshold
        for group in groups:
            if index in group['views']:
                continue
            iou = float(overlap(item.box, [group['box']])[0])
            if iou >= best_iou:
                best, best_iou = group, iou
        if best is None:
            groups.append({'views': {index}, 'items': [item], 'box': item.box.copy()})
        else:
            best['views'].add(index)
            best['items'].append(item)
            best['box'] = np.average([member.box for member in best['items']], axis=0, weights=[member.confidence for member in best['items']])
    return suppress([Detection(group['box'].astype(np.float32), sum(member.scores for member in group['items']) / len(views), min(member.quality for member in group['items'])) for group in groups])


class FlipDetector:
    def __init__(self, detector):
        self.detector = detector

    def detect(self, image, request):
        native = self.detector.detect(image, request)
        mirrored = self.detector.detect(cv2.flip(image, 1), request)
        axis_sum = request.view.source_region_xyxy[0] + request.view.source_region_xyxy[2]
        restored = []
        for item in mirrored:
            box = item.box.copy()
            box[0], box[2] = axis_sum - item.box[2], axis_sum - item.box[0]
            restored.append(Detection(box, item.scores.copy(), item.quality))
        return fuse_views([native, restored])
