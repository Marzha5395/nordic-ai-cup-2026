"""Replay a scene through the solution in-process and score it.

``local_evaluator.py`` is the real harness and does this over HTTP. This one
imports the same camera, the same view rendering and the same scorer but calls
``predict`` directly, which removes the server and the network from the loop
and makes a full scene take seconds. Use it to tune; use ``local_evaluator.py``
to confirm.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from dtos import DroneFlybyPredictRequestDto                    # noqa: E402
from local_evaluator import (                                   # noqa: E402
    Camera, CameraRejection, build_request, render_view, score,
)
from utils import frame_numbers, global_bbox_to_source, load_frame  # noqa: E402


def replay(scene: str, verbose: bool = False, frame_step: int = 1):
    import example
    from solution import runtime

    runtime.reset()
    frames = frame_numbers(scene)[::frame_step]
    camera = Camera()
    feedback = None
    predictions = {}
    applied = refused = 0
    durations = []

    for frame_index, frame in enumerate(frames):
        image = load_frame(frame, scene)
        payload = build_request(
            frame, frame_index, camera, render_view(image, camera), feedback
        )
        request = DroneFlybyPredictRequestDto.model_validate(payload)

        started = time.perf_counter()
        response = example.predict(request)
        durations.append(1000.0 * (time.perf_counter() - started))

        predictions[frame] = [
            {
                'object_id': annotation.object_id,
                'bbox': global_bbox_to_source(
                    annotation.bbox, payload['original_width'], payload['original_height']
                ),
                'confidence': float(annotation.confidence),
            }
            for annotation in response.annotations
        ]
        if verbose:
            print(
                f'frame {frame:4d} L{camera.resolution_level} '
                f'({camera.center_x:4d},{camera.center_y:4d}) '
                f'-> {len(response.annotations):3d} annotations '
                f'{durations[-1]:6.1f} ms'
            )

        if response.requested_view is not None:
            requested = response.requested_view
            try:
                camera.apply(
                    requested.resolution_level, requested.center_x, requested.center_y
                )
                applied += 1
                feedback = None
            except CameraRejection as error:
                refused += 1
                feedback = {
                    'frame': frame,
                    'requested_view': {
                        'resolution_level': requested.resolution_level,
                        'center_x': requested.center_x,
                        'center_y': requested.center_y,
                    },
                    'reason': str(error),
                }
                print(f'frame {frame}: camera command refused: {error}')

    return predictions, applied, refused, durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--frame-step', type=int, default=1)
    arguments = parser.parse_args()

    predictions, applied, refused, durations = replay(
        arguments.scene, arguments.verbose, arguments.frame_step
    )
    mean_average_precision, by_class = score(arguments.scene, predictions)

    ordered = sorted(durations)
    print()
    print(f'camera moves applied {applied}, refused {refused}')
    print(
        f'per frame ms: mean {sum(ordered) / len(ordered):.1f} '
        f'median {ordered[len(ordered) // 2]:.1f} max {ordered[-1]:.1f}'
    )
    print()
    print('AP@0.50 by class')
    for name, value in sorted(by_class.items(), key=lambda item: -item[1]):
        print(f'  {name:16s} {value:.3f}')
    print()
    print(f'COCO mAP@0.50: {mean_average_precision:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
