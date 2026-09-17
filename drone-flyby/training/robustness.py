"""Checks for the things that lose a whole attempt rather than a few points.

The evaluation flight is 250 frames and one attempt. What matters here is not
accuracy but that nothing raises, nothing grows without bound, no camera
command is refused and no answer is late.
"""

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from dtos import DroneFlybyPredictRequestDto                     # noqa: E402
from local_evaluator import (                                    # noqa: E402
    Camera, CameraRejection, build_request, render_view,
)
from utils import frame_numbers, load_frame, validate_response    # noqa: E402


def _frames(scene: str):
    store = {frame: load_frame(frame, scene) for frame in frame_numbers(scene)}
    return store


def check_no_weights() -> None:
    """A missing or broken model must not stop the server answering."""
    import example
    from solution import config, runtime

    saved = config.WEIGHTS_PATH
    config.WEIGHTS_PATH = PROJECT / 'weights' / 'does-not-exist.pt'
    runtime.reset(drop_detector=True)
    try:
        camera = Camera()
        image = load_frame(0)
        payload = build_request(0, 0, camera, render_view(image, camera), None)
        response = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        validate_response(response)
        assert response.request_id == payload['request_id']
        assert response.frame == 0
        assert response.annotations == []
        assert response.requested_view is not None, 'camera should still be driven'
        print('no weights            ok (empty answer, camera still moving)')
    finally:
        config.WEIGHTS_PATH = saved
        runtime.reset(drop_detector=True)


def check_broken_image() -> None:
    """A view that will not decode must not stop the server answering."""
    import example
    from solution import runtime

    runtime.reset()
    camera = Camera()
    payload = build_request(0, 0, camera, render_view(load_frame(0), camera), None)
    payload['view']['image'] = 'not base64 at all'
    response = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    validate_response(response)
    assert response.frame == 0
    print('undecodable image     ok (valid empty answer)')


def check_gaps_and_restarts(scene: str = 'helsinki') -> None:
    """Skipped frames and a fresh sequence id must both be handled."""
    import example
    from solution import runtime

    runtime.reset()
    images = _frames(scene)
    frames = sorted(images)
    camera = Camera()

    # Every third frame only: the gaps are what a slow server sees.
    for frame_index, frame in enumerate(frames[::3]):
        payload = build_request(
            frame, frame_index * 3, camera, render_view(images[frame], camera), None
        )
        response = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
        validate_response(response)
        if response.requested_view is not None:
            requested = response.requested_view
            camera.apply(requested.resolution_level, requested.center_x, requested.center_y)
    tracks_first = len(runtime._states['local'].world.tracks)

    # A second attempt, same server, new identifier.
    camera = Camera()
    payload = build_request(0, 0, camera, render_view(images[frames[0]], camera), None)
    payload['sequence_id'] = 'another'
    payload['request_id'] = 'another:0'
    response = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    validate_response(response)
    assert 'another' in runtime._states
    assert len(runtime._states) == 1, 'the finished sequence should be dropped'
    print(f'gaps and restarts     ok ({tracks_first} tracks carried, world reset after)')


def check_rejected_command(scene: str = 'helsinki') -> None:
    """Feedback about a refused command must not derail the next frame."""
    import example
    from solution import runtime

    runtime.reset()
    camera = Camera()
    image = load_frame(0, scene)
    feedback = {
        'frame': 0,
        'requested_view': {'resolution_level': 2, 'center_x': 100, 'center_y': 100},
        'reason': 'cannot change directly from resolution level 0 to 2',
    }
    payload = build_request(1, 1, camera, render_view(image, camera), feedback)
    response = example.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    validate_response(response)
    print('rejected command      ok (answered, camera re-planned)')


def check_long_sequence(scene: str = 'helsinki', frames_wanted: int = 260) -> None:
    """Timing and memory over a sequence the length of a real attempt."""
    import example
    from solution import runtime

    runtime.reset()
    images = _frames(scene)
    available = sorted(images)
    camera = Camera()
    durations = []
    refused = 0

    for frame_index in range(frames_wanted):
        frame = available[frame_index % len(available)]
        payload = build_request(
            frame_index, frame_index, camera, render_view(images[frame], camera), None
        )
        request = DroneFlybyPredictRequestDto.model_validate(payload)
        started = time.perf_counter()
        response = example.predict(request)
        durations.append(1000.0 * (time.perf_counter() - started))
        validate_response(response)
        assert len(response.annotations) <= 500
        if response.requested_view is not None:
            requested = response.requested_view
            try:
                camera.apply(
                    requested.resolution_level, requested.center_x, requested.center_y
                )
            except CameraRejection as error:
                refused += 1
                print(f'  frame {frame_index}: refused: {error}')

    state = runtime._states[payload['sequence_id']]
    ordered = sorted(durations)
    early = sum(durations[:50]) / 50
    late = sum(durations[-50:]) / 50
    print(
        f'long sequence         ok ({frames_wanted} frames, {refused} refused, '
        f'first 50 mean {early:.0f} ms, last 50 mean {late:.0f} ms, '
        f'max {ordered[-1]:.0f} ms, {len(state.world.tracks)} tracks, '
        f'{len(state.flow.samples)} flow samples)'
    )
    assert late < 250.0, 'per-frame cost must not grow with the sequence'
    runtime.reset()


def main() -> int:
    check_no_weights()
    check_broken_image()
    check_rejected_command()
    check_gaps_and_restarts()
    check_long_sequence()
    print('\nall checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
