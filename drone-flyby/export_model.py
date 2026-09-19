import argparse
import json
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO

from dtos import OBJECT_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parent / 'weights' / 'detector.pt')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f'Missing trained weights: {args.weights}')
    torch.set_num_threads(4)
    model = YOLO(str(args.weights))
    if [model.names[index] for index in range(len(model.names))] != list(OBJECT_CLASSES):
        raise ValueError('Training classes do not match the protocol')
    path = model.export(format='onnx', imgsz=[544, 960], batch=1, dynamic=False, simplify=False, opset=17, device='cpu')
    artifact = onnx.load(path)
    properties = {item.key: item.value for item in artifact.metadata_props}
    properties['drone_classes'] = json.dumps(list(OBJECT_CLASSES))
    properties['training_only'] = 'No evaluation or scored checkpoint selection was performed.'
    onnx.helper.set_model_props(artifact, properties)
    onnx.checker.check_model(artifact)
    onnx.save(artifact, path)
    print(f'Exported {path}; graph checked without running any evaluation.')


if __name__ == '__main__':
    main()
