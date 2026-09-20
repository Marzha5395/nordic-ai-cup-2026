"""YOLO detector wrapper: 960x540 view in, view-px detections out.

Env switches read across the policy package: DRONE_WEIGHTS, DRONE_DEVICE,
DRONE_CONF, DRONE_L0_UPSCALE, DRONE_TTA, DRONE_SWEEP_LEVEL, DRONE_L1_REFRESH,
DRONE_L2_DIPS,
DRONE_SECOND_CLASS, DRONE_SECOND_P, DRONE_CONF_TAU, DRONE_NEW_TRACK_CONF,
DRONE_MISS_DECAY_L2, DRONE_DROP_BELOW.
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

CONF = float(os.environ.get('DRONE_CONF', '0.10'))
TTA = os.environ.get('DRONE_TTA', '0') == '1'


class Detector:
    def __init__(self, weights=None, device=None):
        from ultralytics import YOLO
        self.weights = weights or os.environ.get(
            'DRONE_WEIGHTS', 'weights/policy_y11s_run2.pt')
        self.device = device or os.environ.get('DRONE_DEVICE') or (
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = str(self.device).startswith('cuda')
        self.model = YOLO(self.weights)
        logger.info('detector: weights=%s device=%s half=%s conf=%.2f',
                    self.weights, self.device, self.half, CONF)
        self.model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=960,
                           conf=CONF, verbose=False, half=self.half,
                           device=self.device)

    def detect(self, image_bgr, level):
        """-> list of (box_view_xyxy np.array(4), conf float, cls int)."""
        imgsz = 1920 if (level == 0 and
                         os.environ.get('DRONE_L0_UPSCALE') == '2') else 960
        result = self.model.predict(
            image_bgr, imgsz=imgsz, conf=CONF, iou=0.6, max_det=200,
            agnostic_nms=True, verbose=False, half=self.half,
            augment=TTA, device=self.device)[0]
        out = []
        boxes = result.boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            out.append((xyxy[i].astype(np.float64), float(confs[i]),
                        int(clss[i])))
        return out
