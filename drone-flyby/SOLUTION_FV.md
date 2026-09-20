# Drone flyby — ported "fierceviking" pipeline (`api_fv.py`, `fv/`)

A third, independent endpoint next to `submission.py` (V2) and `api_policy.py`
(memory policy). Start it with `python api_fv.py` (same port 9053, same
`/predict` route). Nothing in `fv/` is imported by the other two solutions.

## Provenance

Ported on 2026-09-20 from the public repository
`github.com/fierceviking/Nordic-AI-Cup-2026`, branch `mtp_drone`, commit
`cafe301` ("submission"). Files taken verbatim except for import paths and the
default weights path: their `solution.py` -> `fv/solution.py`, `example.py` ->
`fv/adapter.py`, `recorder.py` -> `fv/recorder.py`, `api.py` -> `api_fv.py`,
and `runs/detect/runs/v19frames/train/weights/best.pt` ->
`weights/fv_v19frames.pt` (YOLO11s, 19 MB). Their `utils.py`, `dtos.py` and
`local_evaluator.py` are byte-identical to the organisers' files we already use.

Their experiment log (3,100 lines, not copied) reports **official validation
0.7584** for preset `v17size` = `v17frames` weights + size prior. That model is
not public; `v19frames` (public) is `v17frames` fine-tuned at a tiny learning
rate on a larger hand-relabelled set of the same flight. Two caveats they state
themselves and that hold for us:

* The real training frames were recorded **from the validation flight** and
  labelled by hand, so their validation score is inflated relative to the
  evaluation flight (a different location).
* Their purely synthetic detectors scored only 0.10–0.15 on the validation
  flight; the jump to 0.65–0.76 came from those real, same-flight labels.

## What the pipeline does

1. **Detector** (`fv/solution.py: Detector`): YOLO11s on the received 960x540
   view, boxes lifted to source pixels; boxes touching a view border that is
   not a frame border are dropped. Optional size prior: a box far from its
   class's known ground size is damped/dropped, and a box swallowed by a peer of
   comparable real size is dropped (`DRONE_SIZE_PRIOR=1`).
2. **World model** (`WorldModel`): tracks are warped forward each frame by the
   constant inter-frame homography (translation corrected online by ORB
   matching in `MotionEstimator`), class votes are pooled, duplicates merged,
   confidence decays with staleness; the response always covers the whole
   frame. Up to two alternate classes per track are emitted at damped
   confidence.
3. **Camera** (`CameraPolicy`): fixed loop alternating the full Level-0 view
   with one Level-1 quadrant (`FULL, Q1, FULL, Q2, ...`); every command is
   legal by construction (one level step, clamped centre, hop shortened to the
   movement limit).
4. **Our addition** (`EnsembleDetector`): an optional second detector whose
   boxes are pooled with the first before tracking. `DRONE_ENSEMBLE_WEIGHTS`
   selects it; `DRONE_ENSEMBLE_SCALE` (1.0) and `DRONE_ENSEMBLE_CONF` (= primary
   confidence) tune it.

## Presets (`python api_fv.py --experiment NAME`, or `DRONE_EXPERIMENT=NAME`)

| preset | detector(s) | notes |
|---|---|---|
| `v19frames` (default) | `weights/fv_v19frames.pt` | their public model, size prior on |
| `ours-only` | `weights/policy_y11s_run2.pt` | our memory-policy detector inside their pipeline |
| `v19ens` | both | ensemble; ~2x detector time |

All other presets in `api_fv.py` are their experiment history and reference
weights we do not have; do not use them.

## Local measurements (official `local_evaluator.py` scorer, offline, 0 refused camera moves, 0 skipped frames)

Scenes: `helsinki` = the 25 supplied frames (both detectors were trained on
sprites cut from these, so optimistic for both); `synth250`/`synth250b` = our
250-frame synthetic flights (Helsinki sprites on Inria orthophotos, rendered
with the fitted camera model — in-domain for our detector, out-of-domain for
theirs). Laptop 2 GB MX570 round trips in ms.

| configuration | helsinki | synth250 | synth250b | round trip |
|---|---|---|---|---|
| `api_fv.py` `v19frames` (their detector + pipeline) | 0.965 | 0.104 | 0.118 | 43–54 |
| `api_fv.py` `ours-only` (our detector, their pipeline) | 0.966 | 0.633 | 0.640 | 48–69 |
| `api_fv.py` `v19ens` (both detectors, their pipeline) | 0.969 | 0.587 | 0.570 | 58–89 |
| `api_policy.py` (our detector, our pipeline; from SOLUTION.md) | 0.841 | 0.765 | 0.723 | ~45 |

`v19ens --realtime` on helsinki: 0.975, 0 skipped, mean 58 ms / max 62 ms.

Reading: their *pipeline* is worth +0.125 over ours on Helsinki with the same
detector; our *pipeline* is worth +0.08–0.13 on the synthetic flights with the
same detector. Their *detector* does not transfer to our synthetic flights at
all; whether ours transfers to a real unseen city is exactly what an official
validation attempt measures (our detector never saw the validation flight, so
its validation score is an honest estimate for the evaluation flight — theirs
is not).

## Run-book for the GPU machine

```bash
git clone git@github.com:Marzha5395/nordic-ai-cup-2026.git
cd nordic-ai-cup-2026/drone-flyby
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-fv.txt            # RTX 50-series: add --index-url https://download.pytorch.org/whl/cu128 for torch/torchvision
python -c "import torch; print(torch.cuda.is_available())"     # must print True

# self-check: both checkpoints load and warm up (needs src/helsinki, which is in git)
DRONE_EXPERIMENT=v19ens python -c "import api_fv, fv.adapter as a; assert a.SOLVER is not None; print('ok')"

# serve ONE worker. Recording every request is optional but cheap (~1.4 MB/frame):
DRONE_RECORD_DIR=recordings python api_fv.py --experiment ours-only
curl -s localhost:9053/health        # solver_loaded true, weights_exist true, status ready

# protocol + timing check from a second terminal (same machine):
python local_evaluator.py --scene helsinki --realtime     # expect ~0.97, 0 skipped, round trip << 333 ms
```

Then on https://cases.nordicaicup.com submit `http://<public-ip>:9053/predict`
(path included), run **Verify**, then **Validation** — as many times as
needed, changing only `--experiment` between runs (restart the server each
time; it keeps per-sequence state). Suggested order and how to read it:

1. `--experiment ours-only` — honest number (our detector never saw this flight).
2. `python api_policy.py` — same detector, our pipeline; also honest. Compares
   the two pipelines on a real unseen city.
3. `--experiment v19ens` — inflated (their detector trained on this flight),
   but shows what their detector adds on real imagery.

Pick for the one-shot **Evaluation**: the higher of (1) and (2) if either is
respectable (>= 0.5). If both are poor (< 0.3), our detector does not transfer
to real cities and `v19ens` is the better gamble: it keeps their real-imagery
detector and adds ours for recall. Never start Evaluation without a completed
Validation of the exact configuration being submitted.

While a run is live the server log prints one line per frame; a `Camera
command from frame N was ignored` warning means a refused move (none expected).
