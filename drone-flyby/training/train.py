"""Train the detector on the composed dataset.

The defaults are the ones that matter for this use case:

* a P2/4 head, because the smallest classes are 10-20 pixels across in the
  transmitted view and count as much as the largest ones;
* no mosaic and only mild scale jitter, because the flight altitude is fixed
  and object scale is therefore known -- teaching the model to expect four
  times the scale range it will ever see only costs capacity;
* rotation left to the dataset builder, which rotates objects individually and
  keeps a tight box, rather than to the loader, which would rotate the whole
  image and inflate every box.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(PROJECT / 'dataset' / 'drone.yaml'))
    parser.add_argument('--model', default=str(HERE / 'cfg' / 'yolo11-p2.yaml'))
    parser.add_argument('--scale', default='s', choices=list('nsmlx'))
    parser.add_argument('--weights', default='yolo11s.pt',
                        help='COCO checkpoint to transfer the backbone from, or "" for none.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--name', default='p2s')
    parser.add_argument('--project', default=str(PROJECT / 'runs'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()

    from ultralytics import YOLO

    configuration = arguments.model
    if configuration.endswith('.yaml') and 'yolo11-p2' in configuration:
        # Ultralytics reads the scale out of the file name, so hand it a name
        # that carries the scale letter.
        configuration = str(Path(configuration).with_name(f'yolo11{arguments.scale}-p2.yaml'))
        if not Path(configuration).exists():
            Path(configuration).write_text(Path(arguments.model).read_text())

    model = YOLO(configuration)
    if arguments.weights:
        model = model.load(arguments.weights)

    model.train(
        data=arguments.data,
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        rect=True,
        device=arguments.device,
        workers=arguments.workers,
        project=arguments.project,
        name=arguments.name,
        exist_ok=True,
        resume=arguments.resume,
        cache=False,
        amp=True,
        optimizer='auto',
        cos_lr=True,
        patience=0,
        # The composer already randomises orientation, density, placement and
        # exposure, so the loader only needs colour and flips on top.
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        translate=0.12,
        scale=0.25,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        # Sixteen classes that differ in fine detail at fifteen pixels, and a
        # macro-averaged score that charges as much for confusing two of them
        # as for missing both, so the classification term is worth more here
        # than the default balance assumes.
        cls=0.8,
        plots=True,
        val=True,
    )
    print('best weights:', Path(arguments.project) / arguments.name / 'weights' / 'best.pt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
