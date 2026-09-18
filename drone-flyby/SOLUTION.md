# Drone flyby — solution notes

The protocol is asymmetric: each request carries a 960x540 crop of wherever the
camera is pointed, and each response has to describe the whole 3840x2160 source
frame. The camera can look at a quarter of the frame at a time. So the answer
is not "detect what is in the image" — it is **remember the whole frame, and
spend the camera on the parts you know least about.**

Three pieces, in the order they matter.

## 1. The ground flow is affine, not a translation

The obvious motion model — one translation per frame — is wrong by up to ten
pixels a frame. Phase correlating a grid of tiles between consecutive frames of
the reference scene gives a field that is linear in image position:

```
frame 12 -> 13, source pixels per frame

           x = 480      1440       2400      3360
    y= 360  (-9.9,58)  (-3.5,56)  (3.2,57)  (10.0,57)
    y=1080  (-9.9,65)  (-3.6,66)  (3.4,66)  (10.7,65)
    y=1800 (-11.8,72)  (-3.3,76)  (3.3,75)  (10.7,76)
```

Fitting `dx = a0 + a1 x + a2 y`, `dy = b0 + b1 x + b2 y` reproduces the
annotated object displacements to under a pixel. A single best translation is
off by up to ten pixels at the frame edges — and since a 32 px object needs its
centre within about 6 px to still clear IoU 0.50, and the camera returns to a
given patch of ground every four to six frames, that difference decides whether
a tracked object still counts.

Two timescales, because the field has two. The gradients are geometry and hold
steady, so they are fitted to the whole sequence. The offset moves — across the
reference scene the drift at the frame centre climbs from 62.8 to 69.7 px per
frame as the ground falls away beneath the drone — so it is re-measured from
the last few frames only. `solution/motion.py`.

One failure the reference scene never showed. Over crop rows or a road
running along the flight line, a tile has almost no texture in the direction
of travel. The Hanning window does not move with the ground, so such a tile
correlates best with itself at zero shift, with a confident response. On the
held-out flights, every flow sample under 5 px was one of these and no correct
sample was. They dragged the fitted y-gradient to the wrong sign for the
first forty frames, which put tracked boxes 30% of an object's height off. The
drone never hovers, so shifts under 8 px are now discarded (see
`MINIMUM_DRIFT_PER_FRAME`). That cut the flow error on those flights from 5.0
to 0.16 px per frame.

## 2. The world model is what answers the frame

`solution/tracking.py` keeps every object seen so far in source coordinates,
moves it with the flow field plus a small per-track bias for what the field
cannot know (how tall the object is, how high the ground under it sits), and
answers every frame from that — including objects the camera is not looking at.

Per track: accumulated class evidence rather than the last label, a noisy-or
over the sightings rather than the last score, and a confidence that decays
with how long ago the object was last seen. Duplicate tracks are merged, and
output is suppressed within a class, because the scorer counts a second box on
one object as a false positive.

Without this the recall ceiling is the fraction of the frame the camera covers,
which is a quarter.

## 3. Level 1, and a staleness map

The camera level is settled by arithmetic, not by search. New ground arrives at
the top of the frame at the drift rate: about 65 px per frame across 3840 px,
so roughly 250 000 px² per frame. What a level can cover is its view width
times how far its centre may move:

| Level | View of the source | Centre may move | Fresh ground per frame |
|---|---|---|---|
| 0 | 3840x2160, quarter scale | — | everything, resolving nothing |
| 1 | 1920x1080, half scale | 1102 px | 1102 x 1080 = 1.19 Mpx² |
| 2 | 960x540, full scale | 551 px | 551 x 540 = 0.30 Mpx² |

Level 1 covers new ground four times faster than it arrives, leaving room to
revisit and to turn around. Level 2 covers it 1.2 times faster, which does not
survive a single turn at the end of a sweep and would have to find every object
on its only look. Measured end to end on the reference scene, level 2 scores
0.44 against level 1's 0.80.

Given the level, the centre is chosen by a staleness map: a grid over the
source frame holding how long ago each patch was looked at, drifted with the
ground each frame so that the cells scrolling in at the edge are new ground.
The camera goes wherever the most staleness, plus the tracks most in need of a
second opinion, fits inside one view. `solution/planner.py`.

One detail that mattered more than any other parameter: **a small cost per
pixel the camera centre travels.** A pure greedy staleness maximiser jumps
across the frame whenever two candidates are nearly tied, which leaves no
overlap between consecutive views and makes the whole trajectory chaotic — the
same settings scored anywhere from 0.60 to 0.87 depending only on how ties
broke. With a cost of 0.85 per pixel the policy settles into a clean raster
sweep, the run becomes deterministic, and the score goes up.

## 4. The detector

A YOLO11-s with an added P2/4 head (`training/cfg/yolo11-p2.yaml`). The
smallest classes are 10-20 px across in the transmitted view and a
macro-averaged mAP charges as much for missing `ta-ta` as for missing `condor`,
so a stride-4 head is what makes them reachable at all.

Training data is a problem: the reference scene is 25 frames holding **one
instance of each class, at one orientation, over one terrain**. A detector
trained on that learns the yaw as much as the object. So `training/compose.py`
builds every training image from scratch:

* objects are cut out with a GrabCut mask, rotated freely, and pasted back.
  The mask fills only *small* holes: filling the triangle of grass between an
  aircraft's wing and its tail pastes a patch of Helsinki onto whatever
  background the object lands on, and stretches the derived box out to cover
  it;
* the box that comes with them is derived from an **oriented-rectangle model**
  rather than the axis-aligned hull of a rotated box — the supplied boxes are
  themselves hulls of rotated objects, so rotating and re-hulling would inflate
  every box by up to 40%;
* backgrounds are the reference frames with the real objects painted out by
  transplanting a nearby patch of empty ground, not by inpainting, which leaves
  a conspicuous starburst where a forest used to be;
* backgrounds are also lifted out at arbitrary angles, which is free terrain
  and a free sun direction, since there is no label on terrain;
* about a quarter of the samples keep the real frame and its real objects, as
  an anchor against learning the paste artefact instead of the object;
* composition happens at source resolution and is then downsampled by the
  level's exact integer factor with `INTER_AREA`, which is what the evaluator
  does, at a mix of levels 0, 1 and 2.
* most backgrounds (62%) come from the outside terrain in `external/train`,
  resampled by a random factor so that sharpness does not give away which
  source a background came from;
* unlabelled patches of terrain are blended in with the same soft masks as the
  objects. Otherwise "blended patch" would perfectly predict "object", and on
  new terrain anything unusual would look like one.

The current detector (v4) was fine-tuned from v3 on `dataset4` for 18 epochs.
Its class head was re-learned rather than carried over, because the checkpoint
is loaded into an 80-class model before the trainer re-heads it.

## 5. Overfitting, and how it is measured now

The first submission scored about 0.83 on the reference scene and **0.22 on
validation**. The server log showed why: about 20 answered boxes per frame at
the start of the validation flight and about 80 at the end. The detector had
learned Helsinki. On Helsinki terrain with the objects painted out it raised
0.01 false alarms per view; on real aerial terrain it had never seen, 4.4 per
view (score >= 0.2). The tracker kept every one of them.

The reference scene cannot measure this, because the detector is trained on
it. So there are now three held-out measurements, none of which touch the
training data:

* **Unseen terrain.** `training/fetch_backgrounds.py` downloads public-domain
  USGS aerial imagery over twenty kinds of ground (desert, farmland, forest,
  mountains, coast, city). The locations are split by place. `external/train`
  is composed into training images; `external/test` is never trained on.
* **A detector val split over that terrain.** `dataset4`'s val split pastes
  the objects only onto `external/test`, and it is what picks the best
  checkpoint.
* **Whole flights over it.** `training/synthetic_flight.py` builds 60-frame
  flights from `external/test`: a drifting, slowly zooming camera over tiled
  ground with objects pasted in. They are written in the reference scene's
  layout, so `local_evaluator.py --scene synth_0` replays them unchanged.
  `synth_*` is the tuning set. `synthB_*` (another seed, higher density) was
  used for nothing but the final check. `training/diagnose_flight.py` splits
  a flight's losses into hits, loose boxes, wrong classes, phantoms and misses.

The flights still use the reference scene's own object cut-outs, so recall on
them is optimistic. But every false alarm on them is real, and false alarms
are what failed validation.

| | v3 (validation 0.22) | v4 |
|---|---|---|
| false alarms / view, unseen terrain, level 1, score >= 0.2 | 4.41 | 0.12 |
| detector mAP50, objects on unseen terrain (`dataset4` val) | 0.797 | 0.969 |
| flights `synth_0..3`, end to end, before the flow fix | 0.487 | -- |
| flights `synth_0..3`, end to end | 0.578 | 0.810 |
| flights `synthB_0..3` (untouched), end to end | 0.693 | 0.858 |
| reference scene, end to end | 0.828 | 0.777 |

The reference scene falls a little. The detector alone does better on it
(0.965 against 0.947 at level 1); the end-to-end loss is `hangar` and
`medium_plane`, which enter at the top-left corner in the last five of its 25
frames, before the camera gets there. On a scene that short that is
trajectory luck, and the planner was not re-tuned to chase it.

Rules applied to every change: tune on `synth_*`, keep a change only if it
also holds on `synthB_*`, and ignore gains that show up only on the reference
scene. For example, retiring tracks after two misses instead of four scored
+0.011 on the tuning flights and -0.001 on the untouched ones, so it was not
adopted.

**Recording.** `solution/recorder.py` writes every request's image, request
and response to `recordings/<sequence_id>/` on a background thread (turn it
off with `DRONE_RECORD=0`). The rules allow keeping the validation sequence,
and it is the only rendered terrain besides Helsinki, so the next validation
attempt should be kept and looked at.

## 6. Latency

The budget is 3333 ms per request, but frames are emitted every 333 ms and only
the newest is sent, so anything over ~333 ms silently costs frames.

Steady state is about 45 ms per frame end to end. The one trap was the *first*
request: uvicorn dispatches a synchronous endpoint onto its own worker threads,
and the first CUDA call on a new thread costs 1.6 seconds — five emitted
frames, lost on the frame that matters most. Warming the model at import does
not help, because the warm-up runs on a different thread. The detector
therefore owns one dedicated thread and every inference goes through it. Cold
first request: 66 ms.

## Layout

| Path | What it is |
|---|---|
| `example.py` | The seam the template defines; hands off to `solution`. |
| `solution/detector.py` | YOLO wrapper, on its own thread, answering in source pixels. |
| `solution/motion.py` | Tiled phase correlation and the affine flow fit. |
| `solution/tracking.py` | The world model. |
| `solution/planner.py` | Staleness map and camera choice. |
| `solution/runtime.py` | Per-sequence state, warm-up, one request in, one response out. |
| `solution/config.py` | Every number worth arguing about. |
| `training/masks.py` | GrabCut instance masks. |
| `training/compose.py` | Sample synthesis. |
| `training/build_dataset.py` | Writes the YOLO dataset. |
| `training/train.py` | Trains the detector. |
| `training/offline_eval.py` | Replays a scene in-process and scores it. |
| `training/eval_levels.py` | Detector-only score per resolution level. |
| `training/sweep.py` | Configuration variants, averaged over trajectories. |
| `training/robustness.py` | The failures that cost an attempt rather than points. |
| `solution/recorder.py` | Keeps every attempt's views and answers under `recordings/`. |
| `training/fetch_backgrounds.py` | Downloads held-out and training terrain (USGS NAIP). |
| `training/synthetic_flight.py` | Whole flights over held-out terrain, in scene layout. |
| `training/diagnose_flight.py` | Where a replayed scene loses its points. |
| `training/false_positives.py` | Detector false alarms per view on object-free terrain. |

## Reproducing

```
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python training/fetch_backgrounds.py --pixels 2304          # external/train, external/test
python training/build_dataset.py --output dataset4 --cache dataset4/backgrounds \
    --train 30000 --val 1500 --seed 909090 --workers 20
python training/train.py --data dataset4/drone.yaml --scale s \
    --weights runs/p2s_v3/weights/best.pt --epochs 18 --batch 16 --workers 16 --name p2s_v4
cp runs/p2s_v4/weights/best.pt weights/detector.pt

python training/synthetic_flight.py                                   # synth_0..3
python training/synthetic_flight.py --seed 777 --prefix synthB --density 1.6
python training/sweep.py --scene synth_0,synth_1,synth_2,synth_3 --seeds 1 --only baseline

python api.py                       # in one terminal
python local_evaluator.py --realtime  # in another
```
