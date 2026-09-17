"""The YOLO detector, wrapped so the rest of the code sees source pixels.

Detections are made on the 960x540 image that arrived with the request and
handed back in source-frame pixels, because that is the only coordinate system
the world model and the camera policy care about.

Every inference runs on one dedicated thread. uvicorn dispatches a synchronous
endpoint onto its own worker threads, and the first CUDA call on a new thread
costs about 1.6 seconds -- five emitted frames, gone, on the frame that matters
most. Owning the thread means it is warmed once, here, and never again.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class Detection(NamedTuple):
    class_index: int
    score: float
    box: Tuple[float, float, float, float]   # source pixels, [x1, y1, x2, y2]


class Detector:
    """A single loaded model, guarded by a lock so the server can be threaded."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
        confidence: Optional[float] = None,
        nms_iou: Optional[float] = None,
        max_detections: Optional[int] = None,
        half: Optional[bool] = None,
    ):
        weights_path = weights_path if weights_path is not None else config.WEIGHTS_PATH
        image_size = image_size if image_size is not None else config.DETECTOR_IMAGE_SIZE
        confidence = confidence if confidence is not None else config.DETECTOR_CONFIDENCE
        nms_iou = nms_iou if nms_iou is not None else config.DETECTOR_NMS_IOU
        max_detections = (
            max_detections if max_detections is not None
            else config.DETECTOR_MAX_DETECTIONS
        )
        half = half if half is not None else config.DETECTOR_HALF_PRECISION
        import torch
        from ultralytics import YOLO

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f'No detector weights at {weights_path}. Train one with '
                f'training/train.py, or point config.WEIGHTS_PATH somewhere else.'
            )

        if device is None:
            device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.half = bool(half) and device != 'cpu'
        self.image_size = image_size
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='drone-detector'
        )
        # Ultralytics renamed the half-precision flag; support both spellings
        # so a version bump cannot silently drop the model back to fp32.
        self._precision_kwargs = {'quantize': 16} if self.half else {}

        self.model = YOLO(str(weights_path))
        self.model.to('cuda:0' if device == '0' else device)
        self.class_names = [
            self.model.names[index] for index in sorted(self.model.names)
        ]
        self.warmup()

    def warmup(self, rounds: int = 3) -> None:
        """Run the first inferences now rather than during the first frame."""
        blank = np.zeros((540, 960, 3), np.uint8)
        if self.half:
            try:
                self._infer(blank)
            except (TypeError, ValueError, KeyError):
                logger.info('falling back to the legacy half-precision flag')
                self._precision_kwargs = {'half': True}
        for _ in range(rounds):
            self._infer(blank)

    def _infer(self, image: np.ndarray):
        """Run one inference on the detector's own thread."""
        return self._worker.submit(self._infer_here, image).result()

    def _infer_here(self, image: np.ndarray):
        return self.model.predict(
            image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.nms_iou,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
            augment=False,
            **self._precision_kwargs,
        )[0]

    def detect(
        self,
        image: np.ndarray,
        source_region: Sequence[int],
    ) -> List[Detection]:
        """Detect on a transmitted view and return source-pixel detections."""
        with self._lock:
            result = self._infer(image)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        region_x1, region_y1, region_x2, region_y2 = (float(c) for c in source_region)
        scale_x = (region_x2 - region_x1) / float(width)
        scale_y = (region_y2 - region_y1) / float(height)

        coordinates = boxes.xyxy.detach().float().cpu().numpy()
        scores = boxes.conf.detach().float().cpu().numpy()
        classes = boxes.cls.detach().int().cpu().numpy()

        detections: List[Detection] = []
        for (x1, y1, x2, y2), score, class_index in zip(coordinates, scores, classes):
            detections.append(
                Detection(
                    int(class_index),
                    float(score),
                    (
                        region_x1 + float(x1) * scale_x,
                        region_y1 + float(y1) * scale_y,
                        region_x1 + float(x2) * scale_x,
                        region_y1 + float(y2) * scale_y,
                    ),
                )
            )
        return detections
