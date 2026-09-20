# Drone flyby — memory-policy solution (`api_policy.py`)

This is an alternative endpoint that lives next to the team's V2 `submission.py`
solution and shares nothing with it except `dtos.py`/`utils.py`. Start it with
`python api_policy.py` (same port 9053, same `/predict` route).

## What the endpoint does

`api_policy.py` → `policy_controller.predict()` → `policy/`:

1. **Detector** (`policy/detector.py`): YOLO11s trained on synthetic composites
   (Helsinki sprites cut out with SAM, pasted with random yaw/scale onto
   Inria aerial tiles and Helsinki frames, rendered at all three zoom levels).
   Labels follow the ground truth's *projected 3D box* convention, so the boxes
   are deliberately loose around the visible pixels.
2. **Flow model** (`policy/flow.py`): the ground moves between frames by a
   constant affine/homography of the source frame (the camera is pitched ~20°:
   ~51 px/frame at the top edge, ~80 at the bottom). A prior fitted to the
   Helsinki GT is refined online by phase correlation between consecutive views;
   the flow direction hypothesis is re-checked on the first L0→L1 transition.
3. **Object memory** (`policy/memory.py`): every detection is lifted to source
   coordinates and fused into tracks that are propagated with the flow, so the
   response always covers the *whole* frame, not just the current crop. Boxes
   fuse per edge (edges touching a view/frame border are treated as unknown),
   classes accumulate evidence weighted by zoom level, misses at L2 decay
   evidence, duplicates merge, and a second class is emitted at reduced
   confidence when the class is ambiguous.
4. **Camera scheduler** (`policy/scheduler.py`): six L1 views cover the whole
   frame at the start, then an L2 boustrophedon scans the strip where new
   ground enters, moving *with* the ground so each strip is seen once at native
   resolution (twice for the middle columns), with one L1 view at each
   turnaround for the larger objects. Every command is checked against the
   camera rules before it is sent, so nothing is ever refused.

## Run it on the GPU machine

```bash
pip install -r requirements-gpu.txt      # the team's pins (torch 2.7.1 cu128, ultralytics 8.3.203); scipy comes with ultralytics
python api_policy.py                     # serves :9053/predict with the memory policy
```

Check the server log on start-up: it must not contain `Detector init failed`.
The detector weights default to `weights/policy_y11s_run2.pt` (override with
`DRONE_WEIGHTS=...`); the device is CUDA when available (`DRONE_DEVICE=cpu`
forces CPU — that works but at ~200 ms/frame will drop frames at 3 fps).
Sanity check from a second terminal:

```bash
python local_evaluator.py --realtime      # Helsinki; also prints round-trip ms
```

Or containerised: `docker build -f Dockerfile.policy -t drone-flyby-policy . && docker run --gpus all -p 9053:9053 drone-flyby-policy`.

Tunables (environment variables, defaults chosen by the ablations below):
`DRONE_CONF` (0.10 detector threshold), `DRONE_L1_REFRESH` (1),
`DRONE_SWEEP_LEVEL` (2), `DRONE_SECOND_CLASS` (1) / `DRONE_SECOND_P` (0.25),
`DRONE_CONF_TAU` (1.0), `DRONE_NEW_TRACK_CONF` (no gating), `DRONE_TTA` (0),
`DRONE_L0_UPSCALE` (1).

## Reproducing the data and the model

```bash
python synth/extract_sprites.py                       # SAM sprites -> data/sprites (needs weights/sam2.1_b.pt)
python synth/make_dataset.py --canvases 3000 --workers 4 --seed 0 --out data/yolo
python synth/make_scene.py --name synth250 --frames 250 --seed 1   # evaluation flight under src/synth250
```

Training ran on a Colab T4 through `synth/colab_train.py` (yolo11s, imgsz 960,
24 epochs on data/yolo = run 1 (not committed); then 10 more epochs on
data/yolo + data/yolo2 (seed 1, 30 background tiles) = run 2 = `weights/policy_y11s_run2.pt`; synthetic-val mAP50 0.743 and 0.765). Local evaluation with per-object diagnostics:

```bash
python synth/eval_policy.py --scene synth250 --quiet          # uses weights/policy_y11s_run2.pt
synth/run_ablations.sh weights/policy_y11s_run2.pt synth250 synth250b
```

## Results (COCO mAP@0.5, `local_evaluator.py` scorer, 0 refused camera moves everywhere)

Scenes: `helsinki` = the 25 supplied frames (sprites and backgrounds seen in
training, so optimistic); `synth250`/`synth250b` = 250-frame synthetic flights
(111 / 84 objects) rendered with the fitted camera model on Inria tiles.

| detector | policy | helsinki | synth250 | synth250b |
|---|---|---|---|---|
| yolo11n, 6 ep (old labels) | L2 sweep | 0.593 | 0.448 | – |
| yolo11s run 1 (24 ep) | L2 sweep + L1 refresh | 0.654 | 0.716 | 0.686 |
| yolo11s run 1 (24 ep) | L1 sweep (shipped) | 0.847 | 0.735 | 0.699 |
| **yolo11s run 2 (+10 ep, 2 datasets) — shipped `weights/policy_y11s_run2.pt`** | L1 sweep (shipped) | **0.841** | **0.765** | **0.723** |

Policy ablations with run-1 weights (synth250 / synth250b): L2 sweep without
the L1 refresh 0.692 / 0.676; targeted L2 dips from the L1 sweep 0.686 / 0.645
(they mostly chased ghost tracks); test-time augmentation −0.01 and +60 %
latency; detector threshold 0.05 instead of 0.10 +0.003 but twice the false
positives; L0 upsampling, confidence temperature and second-class threshold all
within ±0.005.

Latency on the laptop's 2 GB MX570: ~45 ms per frame end to end (PNG decode,
detector, flow, memory), well inside the 333 ms frame interval; the 24 GB card
will be faster.

Known weak spots: `ta-ta` and `small_launcher` (17–30 px objects, ~12 px at the
L1 sweep scale) and objects that are already in the lower half of the frame at
frame 0 (they only get the L0 view and one L1 look). Per-object diagnostics for
any run: `synth/eval_policy.py --scene <name>` (drop `--quiet` for all rows).

## Files

`api_policy.py` (entry point), `policy_controller.py` (per-sequence controller),
`policy/` (detector, flow, memory, scheduler, geometry), `test_policy.py`
(13 unit tests: `python -m unittest -v test_policy`), `synth/` (sprite
extraction, dataset and scene generation, Colab training driver, evaluation and
ablation tooling, `helsinki_camera.json` = fitted camera/flow model),
`weights/policy_y11s_run2.pt` (19 MB detector), `Dockerfile.policy`.
