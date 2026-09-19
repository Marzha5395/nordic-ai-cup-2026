"""Replay a scene in-process with camera commands that sometimes land one frame late.

On validation the service often applied a camera command one frame late: the round trip through the
tunnel is close to the 333 ms frame interval, so the next frame was rendered from the old view and the
command applied after it. ``local_evaluator.py`` applies every command at once, so it cannot show the
effect. Here each command is applied before the next frame with probability ``1 - late`` and after it
otherwise, through the evaluator's own ``Camera`` rules, with rejection feedback as the service sends it.
Scoring is the evaluator's. Measurement only.
"""

import argparse
import json
import uuid

import numpy as np

from dtos import DroneFlybyPredictRequestDto
import local_evaluator as harness
import utils
from utils import global_bbox_to_source, validate_response


def replay(engine, scene, late, seed):
    rng = np.random.default_rng(seed)
    camera = harness.Camera()
    sequence = str(uuid.uuid4())
    frames = harness.frame_numbers(scene)
    predictions, feedback, queued, rejected = {}, None, [], 0
    for index, frame in enumerate(frames):
        # Commands due before this frame is rendered, oldest first.
        for command in [item for due, item in queued if due <= index]:
            try:
                camera.apply(*command)
                feedback = None
            except harness.CameraRejection as error:
                rejected += 1
                feedback = {'frame': frames[index - 1], 'requested_view': dict(zip(('resolution_level', 'center_x', 'center_y'), command)), 'reason': str(error)}
        queued = [(due, item) for due, item in queued if due > index]
        payload = harness.build_request(frame, index, camera, harness.render_view(utils.load_frame(frame, scene), camera), feedback)
        payload['sequence_id'] = sequence
        payload['request_id'] = f'{sequence}:{index}'
        response = engine.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        validate_response(response)
        predictions[frame] = [{'object_id': item.object_id, 'bbox': list(global_bbox_to_source(item.bbox, 3840, 2160)), 'confidence': item.confidence} for item in response.annotations]
        command = response.requested_view
        if command is not None:
            queued.append((index + (2 if rng.random() < late else 1), (command.resolution_level, command.center_x, command.center_y)))
    mean_ap, _ = harness.score(scene, predictions)
    return mean_ap, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/flyby_v6.pt')
    parser.add_argument('--policy', default='adaptive')
    parser.add_argument('--late', type=float, default=0.44, help='Probability that a command lands one frame late (validation: 104/237)')
    parser.add_argument('--hold-pending', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scenes', default='heldout_flight_0,heldout_flight_1,heldout_flight_2,heldout_flight_3,heldout_flightB_0,heldout_flightB_1,heldout_flightB_2,heldout_flightB_3')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    from gpu_detector import TorchDetector
    from submission import FlybyPredictor

    detector = TorchDetector(args.weights, width=960, threshold=0.05, canonical=False)
    results = {}
    for scene in args.scenes.split(','):
        engine = FlybyPredictor(detector, policy=args.policy, hold_pending=args.hold_pending)
        results[scene] = replay(engine, scene, args.late, args.seed)
    scores = [value for value, _ in results.values()]
    print(json.dumps({'policy': args.policy, 'late': args.late, 'hold_pending': args.hold_pending, 'mean_map50': round(float(np.mean(scores)), 4),
                      'rejected_moves': sum(count for _, count in results.values()), 'per_scene': {k: round(v, 3) for k, (v, _) in results.items()}}))


if __name__ == '__main__':
    main()
