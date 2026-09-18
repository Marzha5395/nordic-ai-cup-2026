"""Try configuration variants against one or more scenes, in-process.

The reference scene is 25 frames with one instance of each class, and the
detector was trained on it, so a difference of a few hundredths there is noise
and a gain there may be memory. Tune on the held-out synthetic flights
(``training/synthetic_flight.py``), several at once with ``--scene
synth_0,synth_1,...``, and confirm on a second set that was not tuned on.
Look for settings that sit on a wide plateau, not for the single best number.
"""

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'training'))

import utils                                                     # noqa: E402


def cache_frames(scene: str) -> None:
    """Keep the 4K frames in memory; a sweep reloads them dozens of times."""
    original = utils.load_frame
    store = {}

    def cached(frame: int, scene_name: str = scene):
        key = (frame, scene_name)
        if key not in store:
            store[key] = original(frame, scene_name)
        return store[key]

    utils.load_frame = cached
    import local_evaluator
    local_evaluator.load_frame = cached


VARIANTS = {
    'baseline': {},
    'no-second-class': {'SECOND_CLASS_MINIMUM_SHARE': 1.01},
    'second-class-strong': {'SECOND_CLASS_CONFIDENCE_FACTOR': 0.5,
                            'SECOND_CLASS_MINIMUM_SHARE': 0.15},
    'staleness-flat': {'CONFIDENCE_STALENESS_DECAY': 0.0},
    'staleness-steep': {'CONFIDENCE_STALENESS_DECAY': 0.08},
    'keep-everything': {'MINIMUM_OUTPUT_CONFIDENCE': 0.001,
                        'MAXIMUM_MISSES': 8, 'UNCONFIRMED_MISSES': 4},
    'strict-tracks': {'MAXIMUM_MISSES': 2, 'UNCONFIRMED_MISSES': 1},
    'detector-conf-02': {'DETECTOR_CONFIDENCE': 0.02},
    'detector-conf-10': {'DETECTOR_CONFIDENCE': 0.10},
    'revisit-off': {'TRACK_REVISIT_WEIGHT': 0.0},
    'revisit-heavy': {'TRACK_REVISIT_WEIGHT': 200.0},
    'level-2': {'PREFERRED_LEVEL': 2},
    'coverage-cell-32': {'COVERAGE_CELL': 32},
    'age-cap-15': {'COVERAGE_MAXIMUM_AGE': 15.0},
    'size-follow': {'SIZE_SMOOTHING': 0.85},
    'bias-off': {'TRACK_MAXIMUM_BIAS': 0.0},
    'bias-wide': {'TRACK_MAXIMUM_BIAS': 15.0},
    'age-cap-8': {'COVERAGE_MAXIMUM_AGE': 8.0},
    'age-cap-12': {'COVERAGE_MAXIMUM_AGE': 12.0},
    'age-cap-20': {'COVERAGE_MAXIMUM_AGE': 20.0},
    'age-cap-30': {'COVERAGE_MAXIMUM_AGE': 30.0},
    'age20-revisit25': {'COVERAGE_MAXIMUM_AGE': 20.0, 'TRACK_REVISIT_WEIGHT': 25.0},
    'age20-revisit120': {'COVERAGE_MAXIMUM_AGE': 20.0, 'TRACK_REVISIT_WEIGHT': 120.0},
    'quality-070': {'LEVEL_DETECTION_QUALITY': {0: 0.35, 1: 0.70, 2: 1.0}},
    'quality-100': {'LEVEL_DETECTION_QUALITY': {0: 0.35, 1: 1.00, 2: 1.0}},
    'match-loose': {'MATCH_IOU': 0.05, 'MATCH_DISTANCE_RATIO': 1.1},
    'match-tight': {'MATCH_IOU': 0.30, 'MATCH_DISTANCE_RATIO': 0.5},
    'position-gain-100': {'TRACK_POSITION_GAIN': 1.0},
    'position-gain-060': {'TRACK_POSITION_GAIN': 0.6},
    'size-slow': {'SIZE_SMOOTHING': 0.25},
    'nms-same-030': {'OUTPUT_NMS_IOU_SAME_CLASS': 0.30},
    'nms-same-060': {'OUTPUT_NMS_IOU_SAME_CLASS': 0.60},
    'detector-conf-07': {'DETECTOR_CONFIDENCE': 0.07},
    'detector-conf-15': {'DETECTOR_CONFIDENCE': 0.15},
    'q1-070': {'LEVEL_DETECTION_QUALITY': {0: 0.35, 1: 0.70, 2: 1.0}},
    'q1-080': {'LEVEL_DETECTION_QUALITY': {0: 0.35, 1: 0.80, 2: 1.0}},
    'q1-095': {'LEVEL_DETECTION_QUALITY': {0: 0.35, 1: 0.95, 2: 1.0}},
    'revisit-0': {'TRACK_REVISIT_WEIGHT': 0.0},
    'revisit-60': {'TRACK_REVISIT_WEIGHT': 60.0},
    'unconf-weight-1': {'UNCONFIRMED_TRACK_WEIGHT': 1.0},
    'unconf-weight-6': {'UNCONFIRMED_TRACK_WEIGHT': 6.0},
    'step-128': {'CANDIDATE_STEP': 128},
    'cell-96': {'COVERAGE_CELL': 96},
    'age-20': {'COVERAGE_MAXIMUM_AGE': 20.0},
    'age-30': {'COVERAGE_MAXIMUM_AGE': 30.0},
    'age-15': {'COVERAGE_MAXIMUM_AGE': 15.0},
    'age-35': {'COVERAGE_MAXIMUM_AGE': 35.0},
    'age-45': {'COVERAGE_MAXIMUM_AGE': 45.0},
    'revisit-100': {'TRACK_REVISIT_WEIGHT': 100.0},
    'q1-060': {'LEVEL_DETECTION_QUALITY': {0: 0.30, 1: 0.60, 2: 1.0}},
    'q1-090': {'LEVEL_DETECTION_QUALITY': {0: 0.30, 1: 0.90, 2: 1.0}},
    'move-cost-1': {'MOVE_COST': 1.0},
    'move-cost-3': {'MOVE_COST': 3.0},
    'move-cost-8': {'MOVE_COST': 8.0},
    'move-cost-20': {'MOVE_COST': 20.0},
    'mc-0.3': {'MOVE_COST': 0.3},
    'mc-0.5': {'MOVE_COST': 0.5},
    'mc-0.7': {'MOVE_COST': 0.7},
    'mc-1.5': {'MOVE_COST': 1.5},
    'mc-2.0': {'MOVE_COST': 2.0},
    'mc-4.0': {'MOVE_COST': 4.0},
    'mc-6.0': {'MOVE_COST': 6.0},
    'minconf-005': {'MINIMUM_OUTPUT_CONFIDENCE': 0.05},
    'minconf-010': {'MINIMUM_OUTPUT_CONFIDENCE': 0.10},
    'minconf-020': {'MINIMUM_OUTPUT_CONFIDENCE': 0.20},
    'purity-off': {'SECOND_CLASS_MINIMUM_SHARE': 1.01, 'MINIMUM_OUTPUT_CONFIDENCE': 0.05},
    'support-off': {'SUPPORT_FLOOR': 1.0},
    'support-floor-060': {'SUPPORT_FLOOR': 0.60},
    'support-floor-015': {'SUPPORT_FLOOR': 0.15},
    'hit-rate-off': {'MINIMUM_HIT_RATE': 0.0},
    'hit-rate-050': {'MINIMUM_HIT_RATE': 0.50},
    'strength-best': {'STRENGTH_SAMPLES': 1},
    'strength-6': {'STRENGTH_SAMPLES': 6},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='helsinki',
                        help='Scene under src/, or several separated by commas.')
    parser.add_argument('--only', default=None, help='Comma-separated variant names.')
    parser.add_argument('--weights', default=None,
                        help='Detector checkpoint to score, instead of the '
                             'configured one.')
    parser.add_argument('--seeds', type=int, default=6,
                        help='Trajectories to average over. The greedy camera '
                             'policy is chaotic, so one run says very little.')
    arguments = parser.parse_args()

    scenes = arguments.scene.split(',')
    cache_frames(scenes[0])

    from solution import config
    if arguments.weights:
        config.WEIGHTS_PATH = Path(arguments.weights).resolve()
        from solution import runtime
        runtime.reset(drop_detector=True)
        print(f'detector: {config.WEIGHTS_PATH}')
    import offline_eval
    from local_evaluator import score

    wanted = arguments.only.split(',') if arguments.only else list(VARIANTS)
    results = []
    reload_model = False
    for name in wanted:
        overrides = VARIANTS[name]
        saved = {key: getattr(config, key) for key in overrides}
        for key, value in overrides.items():
            setattr(config, key, value)
        try:
            from solution import runtime
            if any(key.startswith('DETECTOR_') for key in overrides) or reload_model:
                runtime.reset(drop_detector=True)
                reload_model = True
            values, refused, durations = [], 0, []
            for scene in scenes:
                for seed in range(arguments.seeds):
                    config.COVERAGE_TIE_BREAK_SEED = seed
                    predictions, _, seed_refused, seed_durations = offline_eval.replay(scene)
                    values.append(score(scene, predictions)[0])
                    refused += seed_refused
                    durations.extend(seed_durations)
            value = sum(values) / len(values)
            spread = max(values) - min(values)
        finally:
            for key, previous in saved.items():
                setattr(config, key, previous)
            reload_model = any(key.startswith('DETECTOR_') for key in overrides)
        worst = max(durations)
        results.append((value, name, spread, worst))
        print(f'{name:22s} mAP@0.50 {value:.3f} (spread {spread:.3f})  '
              f'refused {refused}  max {worst:.0f} ms', flush=True)

    print()
    for value, name, spread, worst in sorted(results, reverse=True):
        print(f'  {value:.3f} +-{spread / 2:.3f}  {name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
