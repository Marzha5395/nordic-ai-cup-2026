"""The endpoint the evaluation service calls.

Run ``python api.py --experiment verifier-v1`` for the experimental crop
verifier, or ``--experiment v4s1`` for the matched detector-only control.
Without a preset, the baseline honours the existing DRONE_WEIGHTS setting.
``--report-conf 0.25`` filters final annotations without altering track updates.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException


def configure_experiment():
    import argparse

    root = Path(__file__).resolve().parent
    # Everything below v4s1 changes exactly one variable against it, so a
    # validation run measures that variable and nothing else.
    base = dict(DRONE_IMGSZ='960', DRONE_CONF='0.05', DRONE_ALTERNATES='2',
                DRONE_ALTERNATE_DAMPING='0.5', DRONE_ASSIGN='0', DRONE_MAX_MISSES='3',
                DRONE_MAX_AGE='45', DRONE_MIN_HITS='1', DRONE_REPORT_MAX='400',
                DRONE_DOMAIN='0', DRONE_CAMERA='loop', DRONE_DETECTOR='yolo',
                DRONE_AGNOSTIC_NMS='0', DRONE_SIZE_PRIOR='0',
                DRONE_SAME_CLASS_CONTAINMENT='0')
    presets = {
        'v4s1': {},
        'verifier-v1': {},
        # Detector boxes, DINOv2 reference-bank labels, background rejection
        # before tracking (EXPERIMENTS.md section 33). The detector's own class
        # head is discarded, so v4s1 is the matched control: same weights, same
        # tracking, same camera, only the naming and gating stage differs.
        # DOCUMENTED DEAD-END (section 34, 2026-09-19): offline on held-out
        # Helsinki it reached recall 0.762 / precision 0.691 and a local replay
        # scored 0.932, but back-to-back competition validation put it at
        # 0.0936 against 0.1563 for the control. A 259-crop single-scene bank
        # does not transfer to the validation terrain. Kept inert for the record.
        'dino-verify': {},
        # Measured by replaying the recorded flight: alternate damping, class
        # assignment, track age (15 or 90) and a 500 cap all leave real output
        # unchanged, so they are not offered. Only these move it.
        # Competition validation, v4s1 = 0.153: alt0 (104 boxes) = 0.129,
        # conf02 (260 boxes) = 0.1322. Both directions are worse, so output
        # volume is not the lever and this axis is exhausted.
        'alt0': {'DRONE_ALTERNATES': '0'},          # 104 boxes/frame
        'alt4': {'DRONE_ALTERNATES': '4'},          # 157, saturates by 4
        'conf02': {'DRONE_CONF': '0.02'},           # 260
        'conf25': {'DRONE_CONF': '0.25'},           # 28
        'cap60': {'DRONE_REPORT_MAX': '60'},        # 60
        'misses1': {'DRONE_MAX_MISSES': '1'},       # 69, drops confident ones too
        # Photometric rather than volume: matches each incoming view to the
        # training scene's saturation, brightness and sharpness.
        'domainfix': {'DRONE_DOMAIN': '1'},
        # Camera, not model. l0only scored 0.1157 against 0.153, so Level-1
        # resolution earns score on the real scene and the gradient runs
        # toward more of it, not less.
        'l0only': {'DRONE_CAMERA': 'l0'},
        'l1only': {'DRONE_CAMERA': 'l1'},
        'l1top': {'DRONE_CAMERA': 'l1top'},
        'l1heavy': {'DRONE_CAMERA': 'l1heavy'},
        # Learned linear camera scorer vs the fixed loop, same v4s1 detector.
        # DOCUMENTED DEAD-END (lab/exp15_camera_gate.py, 2026-09-18): differential
        # -evolution fit of camera_policy.LearnedCameraPolicy over the 13 features
        # does NOT beat the hand-tuned loop on held-out Helsinki frames: seed-median
        # gap -0.0023 (seeds swing -0.021..+0.028 = pure fit noise), 5-fold median
        # learned 0.9615 vs fixed 0.9604 (+0.001, under margin). The alternating
        # loop already captures the camera headroom. Kept inert for the record.
        'learned-cam': {'DRONE_CAMERA_CKPT': str(root / 'lab' / 'runs' / 'camera_gate_v1' / 'camera.json')},
        # Different detector, not a knob. Trained on the same 236 sprites over
        # 5000 foreign aerial backgrounds so background cannot predict class,
        # which is what broke every Helsinki-built representation so far.
        'v7n': {'DRONE_WEIGHTS': str(root / 'model' / 'v7n.pt')},
        # v7n then fine-tuned on native Helsinki renders mixed with foreign
        # ground. Fine-tuning on Helsinki alone re-learned the shortcut and
        # cost more than it recovered, so the two sources train together.
        'v8mix': {'DRONE_WEIGHTS': str(root / 'model' / 'v8mix.pt')},
        # v4s1's synthetic set plus 107 human-reviewed real validation-flyby
        # frames (70% temporal split). On the held-out 47 real frames it beat
        # v4s1 offline: mAP50 0.435 vs 0.345, mAP50-95 0.329 vs 0.157. Same
        # tracking/camera as v4s1; only the detector weights change.
        'v9real': {'DRONE_WEIGHTS': str(root / 'model' / 'v9real.pt')},
        # v9real's set plus the 47 previously held-out real frames folded into
        # training (all 154 real frames now train), so medium_launcher and
        # mine_roller enter training for the first time. Offline val is
        # memorization-inflated (rval is now in train); the API validation is
        # the real test. Same tracking/camera as v4s1; only the weights change.
        'v11all': {'DRONE_WEIGHTS': str(root / 'model' / 'v11all.pt')},
        # v11all's data with the 154 real frames oversampled 15x against the
        # 6000 synthetic ones, so real pixels dominate the gradient. Tests
        # whether the real-frame gain scales with weighting. v11all is the
        # matched control; only the weights change.
        'v12over': {'DRONE_WEIGHTS': str(root / 'model' / 'v12over.pt')},
        # Data-collection preset (NOT a scored experiment): best detector +
        # full-frame Level-2 raster camera. Launch with DRONE_RECORD_DIR set and
        # run one VALIDATION attempt to capture real L2 crops of every object;
        # whole-frame score is expected to be poor. Never evaluate.
        'record-l2': {'DRONE_WEIGHTS': str(root / 'model' / 'v11all.pt'),
                      'DRONE_CAMERA': 'l2raster'},
        # Second L2 capture. Same purpose as record-l2, better camera: the
        # raster saw 44.9% of the ground and four silent classes yielded no
        # labellable sighting. exp20 puts the top-band sweep at 87.2%.
        'record-l2top': {'DRONE_WEIGHTS': str(root / 'model' / 'v12over.pt'),
                         'DRONE_CAMERA': 'l2top'},
        # Trained only on exhaustively hand-labelled real frames plus the
        # synthetic set. dataset_v9real is deliberately absent: its labels come
        # from 24 approved proposals densified across the flight, so real
        # objects sit unboxed in those views and contradict these.
        'v17frames': {'DRONE_WEIGHTS': str(root / 'model' / 'v17frames.pt')},
        # Same weights, class-agnostic NMS. Section 21 rejected suppressing
        # extra boxes when the leading box was usually wrong; v17frames scores
        # 0.6527, so that premise needs retesting rather than inheriting.
        'v17nms': {'DRONE_WEIGHTS': str(root / 'model' / 'v17frames.pt'),
                   'DRONE_AGNOSTIC_NMS': '1'},
        # Same weights again, geometry priors instead: a box far from its
        # class's known ground size is damped, and a box swallowed by a peer of
        # comparable real size is dropped - two objects that size cannot
        # contain one another, so it is one object with two labels.
        'v17size': {'DRONE_WEIGHTS': str(root / 'model' / 'v17frames.pt'),
                    'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        'v17largest': {'DRONE_WEIGHTS': str(root / 'model' / 'v17frames.pt'),
                   'DRONE_SIZE_PRIOR': '1',
                   'DRONE_SAME_CLASS_CONTAINMENT': '1',
                   'DRONE_AGNOSTIC_NMS': '0'},
        'v19frames': {
            'DRONE_WEIGHTS': str(root / 'weights' / 'fv_v19frames.pt'),
            'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        'v19ens': {'DRONE_WEIGHTS': str(root / 'weights' / 'fv_v19frames.pt'),
                   'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0',
                   'DRONE_ENSEMBLE_WEIGHTS': str(root / 'weights' / 'policy_y11s_run2.pt')},
        'ours-only': {'DRONE_WEIGHTS': str(root / 'weights' / 'policy_y11s_run2.pt'),
                      'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        # v17's exact train_yolo recipe (fresh yolo11s, optimizer auto, 180-deg
        # rotation + flips, strong HSV) on the improved 70-frame relabelled set
        # at 15x, replacing the earlier hand-written fine-tune that lost.
        'v19recipe': {
            'DRONE_WEIGHTS': str(root / 'model' / 'v19recipe.pt'),
            'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        # v17 recipe on urban ceiling-class copy-paste composites: attacks the
        # zero-label classes on built-up terrain. Judge on generalisation, not
        # the same-flight wire.
        'v20urban': {
            'DRONE_WEIGHTS': str(root / 'model' / 'v20urban.pt'),
            'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        'v18composites': {
            'DRONE_WEIGHTS': str(root / 'lab' / 'runs' / 'v18composites' / 'weights' / 'best.pt'),
            'DRONE_SIZE_PRIOR': '1', 'DRONE_AGNOSTIC_NMS': '0'},
        'rfdetr-nano': {
            'DRONE_WEIGHTS': str(root / 'lab' / 'runs' / 'rfdetr_nano_v17' /
                                 'checkpoint_best_total.pth'),
            'DRONE_DETECTOR': 'rfdetr', 'DRONE_RFDETR_SIZE': 'nano',
            'DRONE_RFDETR_SHAPE': '960,544', 'DRONE_SIZE_PRIOR': '1',
            'DRONE_AGNOSTIC_NMS': '0'},
        'rfdetr-nano10': {
            'DRONE_WEIGHTS': str(root / 'lab' / 'runs' / 'rfdetr_nano_v17' /
                                 'checkpoint_best_total.pth'),
            'DRONE_DETECTOR': 'rfdetr', 'DRONE_RFDETR_SIZE': 'nano',
            'DRONE_RFDETR_SHAPE': '960,544', 'DRONE_SIZE_PRIOR': '1',
            'DRONE_AGNOSTIC_NMS': '0', 'DRONE_CONF': '0.10'},
        # Both, because they catch different duplicates: NMS needs high IoU,
        # while containment has low IoU when one box is much smaller.
        'v17both': {'DRONE_WEIGHTS': str(root / 'model' / 'v17frames.pt'),
                    'DRONE_AGNOSTIC_NMS': '1', 'DRONE_SIZE_PRIOR': '1'},
    }
    names = ('baseline',) + tuple(presets)
    selected = os.environ.get('DRONE_EXPERIMENT', 'v19frames')
    report_conf = float(os.environ.get('DRONE_REPORT_CONF', '0.0'))
    verifier_conf = os.environ.get('DRONE_VERIFIER_CONF') or None
    if verifier_conf is not None:
        verifier_conf = float(verifier_conf)
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--experiment', choices=names, default=selected)
        parser.add_argument('--report-conf', type=float, default=report_conf,
                            help='minimum final confidence after verification and tracking (default: 0)')
        parser.add_argument('--verifier-conf', type=float, default=verifier_conf,
                            help='override verifier rejection threshold before tracking (default: checkpoint value)')
        arguments = parser.parse_args()
        selected, report_conf = arguments.experiment, arguments.report_conf
        verifier_conf = arguments.verifier_conf
        if not 0.0 <= report_conf <= 1.0:
            parser.error('--report-conf must be finite and between 0 and 1')
        if verifier_conf is not None and (not 0.0 <= verifier_conf <= 1.0 or selected != 'verifier-v1'):
            parser.error('--verifier-conf requires verifier-v1 and a finite value between 0 and 1')
    if selected not in names:
        raise ValueError(f'Unknown DRONE_EXPERIMENT: {selected}')
    if not 0.0 <= report_conf <= 1.0:
        raise ValueError('DRONE_REPORT_CONF must be finite and between 0 and 1')
    if verifier_conf is not None and (not 0.0 <= verifier_conf <= 1.0 or selected != 'verifier-v1'):
        raise ValueError('DRONE_VERIFIER_CONF requires verifier-v1 and a finite value between 0 and 1')
    os.environ['DRONE_EXPERIMENT'] = selected
    os.environ['DRONE_REPORT_CONF'] = str(report_conf)
    if verifier_conf is not None:
        os.environ['DRONE_VERIFIER_CONF'] = str(verifier_conf)
    if selected in presets:
        os.environ.update(base, DRONE_WEIGHTS=str(root / 'weights' / 'fv_v19frames.pt'))
        os.environ.update(presets[selected])


configure_experiment()

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from fv.adapter import model_status, predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    if not model_status()['solver_loaded']:
        raise HTTPException(status_code=503, detail='Selected experiment failed to load; check /health')
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections (report_conf=%.2f)',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
        model_status()['report_conf'],
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/health')
def health():
    """Report loaded experiment identity, verifier state and runtime settings."""
    import fv.solution as solution
    loaded = model_status()
    ready = loaded['solver_loaded'] and loaded['verifier'].get('last_error') is None
    return {
        **loaded,
        'status': 'ready' if ready else 'DEGRADED - selected experiment unavailable or failed',
        'weights': solution.WEIGHTS,
        'weights_exist': Path(solution.WEIGHTS).is_file(),
        'device': solution.DEVICE,
        'imgsz': solution.INFER_IMGSZ,
        'confidence': solution.DETECT_CONF,
        'alternate_classes': solution.ALTERNATE_CLASSES,
        'max_age_unseen': solution.MAX_AGE,
        'min_hits_to_report': solution.MIN_HITS,
        'report_max': solution.REPORT_MAX,
        'domain_match': bool(solution.DOMAIN_MATCH),
        'camera_mode': os.environ.get('DRONE_CAMERA', 'loop'),
        'camera_loop': [list(entry) for entry in solution.CameraPolicy.LOOP],
        'recording_to': os.environ.get('DRONE_RECORD_DIR') or None,
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    from fv.adapter import SOLVER
    import fv.solution as solution

    if SOLVER is None:
        logger.error('=' * 70)
        logger.error('SELECTED EXPERIMENT DID NOT LOAD. /predict will return HTTP 503.')
        logger.error('Check that the weights exist and that torch can see the '
                     'GPU, then restart.')
        logger.error('=' * 70)
    else:
        logger.info('=' * 70)
        logger.info('experiment %s', model_status()['experiment'])
        logger.info('verifier   %s', model_status()['verifier'])
        logger.info('weights   %s', solution.WEIGHTS)
        logger.info('device    %s   imgsz %s   conf %s',
                    solution.DEVICE, solution.INFER_IMGSZ, solution.DETECT_CONF)
        logger.info('report confidence floor %s', model_status()['report_conf'])
        logger.info('alternates %s   camera loop %s positions',
                    solution.ALTERNATE_CLASSES, len(solution.CameraPolicy.LOOP))
        record = os.environ.get('DRONE_RECORD_DIR')
        logger.info('recording %s', record or 'off')
        logger.info('=' * 70)
        logger.info('Detector loaded and warmed up; serving on %s:%s', HOST, PORT)

    uvicorn.run(app, host=HOST, port=PORT)
