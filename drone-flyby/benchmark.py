import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from detector import Detector
from dtos import DroneFlybyPredictRequestDto
import local_evaluator as harness
from submission import FlybyPredictor
import utils
from utils import global_bbox_to_source, validate_response


ROOT = Path(__file__).resolve().parent


def run(weights, width=1920, device='cuda:0', policy='full', threshold=0.05, scene='helsinki', realtime=False, transform='native', stateless=False, canonical=True, tta='none'):
    cv2.setNumThreads(1)
    if Path(weights).suffix == '.onnx':
        detector = Detector(weights)
        detector.threshold = threshold
    else:
        from gpu_detector import TorchDetector

        detector = TorchDetector(weights, device=device, width=width, threshold=threshold, canonical=canonical)
    effective_width = getattr(detector, 'full_width', detector.width)
    if tta == 'flip':
        from gpu_detector import FlipDetector

        detector = FlipDetector(detector)
    engine = FlybyPredictor(detector, policy=policy)
    camera = harness.Camera()
    frames = harness.frame_numbers(scene)
    statistics = harness.Statistics(frames_total=len(frames))
    predictions = {}
    sequence = str(uuid.uuid4())
    index = 0
    started = time.monotonic()
    while index < len(frames):
        frame = frames[index]
        image = utils.load_frame(frame, scene)
        if transform == 'flip':
            image = cv2.flip(image, 1)
        elif transform == 'dim':
            image = (image.astype(np.float32) * 0.65).astype(np.uint8)
        elif transform == 'warm':
            image = (image.astype(np.float32) * [0.8, 1.0, 1.15]).clip(0, 255).astype(np.uint8)
        payload = harness.build_request(frame, index, camera, harness.render_view(image, camera), None)
        payload['sequence_id'] = sequence
        payload['request_id'] = f'{sequence}:{index}'
        request = DroneFlybyPredictRequestDto.model_validate(payload)
        if stateless:
            engine.sequences.clear()
        sent = time.monotonic()
        response = engine.predict(request)
        validate_response(response)
        duration = (time.monotonic() - sent) * 1000
        statistics.round_trip_ms.append(duration)
        statistics.frames_sent += 1
        statistics.responses_accepted += 1
        predictions[frame] = []
        for item in response.annotations:
            box = list(global_bbox_to_source(item.bbox, request.original_width, request.original_height))
            if transform == 'flip':
                box[0], box[2] = request.original_width - box[2], request.original_width - box[0]
            predictions[frame].append({'object_id': item.object_id, 'bbox': box, 'confidence': item.confidence})
        command = response.requested_view
        if command is not None:
            camera.apply(command.resolution_level, command.center_x, command.center_y)
            statistics.commands_applied += 1
        if realtime:
            next_index = max(index + 1, int((time.monotonic() - started) / harness.FRAME_INTERVAL_SECONDS))
            statistics.frames_skipped += max(0, min(next_index, len(frames)) - index - 1)
            index = next_index
        else:
            index += 1
    mean_ap, classes = harness.score(scene, predictions)
    return {'map50': mean_ap, 'ap_by_class': classes, 'statistics': asdict(statistics), 'predictions': predictions, 'settings': {'weights': str(weights), 'width': effective_width, 'device': device, 'policy': policy, 'threshold': threshold, 'scene': scene, 'realtime': realtime, 'transform': transform, 'stateless': stateless, 'canonical': canonical, 'tta': tta}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights' / 'flyby_v2.pt')
    parser.add_argument('--width', type=int)
    parser.add_argument('--device')
    parser.add_argument('--policy', choices=['full', 'adaptive', 'overview'])
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'src')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--stateless', action='store_true')
    parser.add_argument('--canonical', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tta', choices=['none', 'flip'])
    parser.add_argument('--transform', choices=['native', 'flip', 'dim', 'warm'], default='native')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config_path = args.weights.with_suffix('.runtime.json')
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    args.width = args.width if args.width is not None else config.get('input_width', 1920)
    args.device = args.device or config.get('device', 'cuda:0')
    args.policy = args.policy or config.get('camera_policy', 'full')
    args.threshold = args.threshold if args.threshold is not None else config.get('confidence', 0.05)
    args.tta = args.tta or config.get('tta', 'none')
    utils.DATA_DIRECTORY = args.data_root.resolve()
    result = run(args.weights, args.width, args.device, args.policy, args.threshold, args.scene, args.realtime, args.transform, args.stateless, args.canonical, args.tta)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    summary = {key: value for key, value in result.items() if key != 'predictions'}
    timings = summary['statistics'].pop('round_trip_ms')
    summary['statistics']['mean_ms'] = float(np.mean(timings))
    summary['statistics']['p95_ms'] = float(np.percentile(timings, 95))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
