import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from dtos import IMAGE_WIDTH


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'weights' / 'flyby_v2.pt')
    parser.add_argument('--tta', choices=['none', 'flip'], default='none')
    args = parser.parse_args()
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
    training = checkpoint['training_metadata']
    measurements = []
    for path in args.reports:
        report = json.loads(path.read_text())
        record = {key: value for key, value in report.items() if key != 'predictions'}
        timings = record['statistics'].pop('round_trip_ms', [])
        if timings:
            record['statistics']['mean_ms'] = sum(timings) / len(timings)
        record['source_report'] = path.name
        measurements.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    shutil.copyfile(args.weights, temporary)
    temporary.replace(args.output)
    with args.output.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {
        'weights_sha256': digest,
        'training': training,
        'deployment': {'device': 'cuda:0', 'input_width': round(IMAGE_WIDTH * training['canonical_scale'] / 32) * 32, 'canonical_scale': training['canonical_scale'], 'confidence': 0.05, 'camera_policy': 'full', 'tta': args.tta, 'runtime_external_api_calls': False},
        'measurements': measurements,
        'interpretation': [
            'Helsinki is a reference/training scene, not the competition validation or evaluation sequence.',
            'Held-out frames contain the same object instances as training frames and are temporally correlated.',
            'Transfer images are synthetic composites of held-out reference views and held-out external backgrounds.',
            'Timing measurements are from the local MX570 laptop, not the separate validation GPU.',
            'No official remote validation or evaluation attempt was made.',
        ],
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    runtime = dict(manifest['deployment'], weights_sha256=digest)
    args.output.with_suffix('.runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    print(f'Packaged {args.output} (training step {training["completed_steps"]}, SHA256 {digest})')


if __name__ == '__main__':
    main()
