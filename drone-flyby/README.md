# Drone flyby — how to run

Serves the detector and camera policy on port **9053**; submit `http://<your-host>:9053/predict`.

```sh
cd drone-flyby
source .venv/bin/activate
python3 api.py
```

The endpoint uses `weights/flyby_v6.pt` with the settings in `weights/flyby_v6.runtime.json`
(native 960 px input, confidence 0.05, `adaptive` camera policy, no late-command handling).
Wait for `Uvicorn running` before submitting, and run one worker only: the tracker keeps
per-sequence state.

## Requirements

- Python 3.12 with `requirements-gpu.txt` (torch 2.7.1 + torchvision 0.22.1 from the CUDA 12.8
  wheel index; earlier CUDA 12.4 builds have no kernels for RTX 50-series GPUs).
- A CUDA GPU with ~2 GB free.

```sh
python -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```

## Options

Set as environment variables before `python3 api.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DRONE_MODEL_PATH` | `weights/flyby_v6.pt` | Model to serve (`weights/flyby_v2.pt` and `flyby_v5.pt` are kept for comparison) |
| `DRONE_CAMERA_POLICY` | `adaptive` | `adaptive` zooms; `full` only ever requests the whole frame |
| `DRONE_LAG` | `none` | How to handle a camera command the service has not applied yet: `none`, `resend` or `ahead` |
| `DRONE_DEVICE` | `cuda:0` | `cpu` for debugging only; far too slow for an attempt |
| `DRONE_RECORD_DIR` | unset | Saves every request's view and answer for later inspection |

"Camera command rejected" warnings are expected and cost only that camera move.

## Checking it locally

```sh
python3 local_evaluator.py --realtime                       # replay the supplied scene, with the 3 fps clock
python3 local_evaluator.py --realtime --url http://<host>:9053/predict   # over the real network path
python3 -m unittest test_solution test_v2 test_terrain test_artifact
```

`--realtime` reports skipped frames: a frame that never arrives scores as a frame with no
detections, so 0 skipped is what a good run looks like.

## Rebuilding the model (optional)

```sh
python3 fetch_terrain.py                  # aerial terrain, split by place into train/test
python3 sprites.py                        # cut the reference objects out with SAM
python3 train_terrain.py --p2 --pretrained yolo11s-obb.pt --batch 12 --steps 10000 \
        --photometric --appearance --shadows --output weights/flyby_v6.pt
python3 heldout_eval.py                   # false alarms and mAP on held-out places
python3 synthetic_flight.py               # whole flights over held-out ground, scored with local_evaluator.py
```
