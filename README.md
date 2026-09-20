# Nordic AI Cup 2026 — Childbeating Catboost Connoisseurs

Solutions of the Nordic AI Cup team **Childbeating Catboost Connoisseurs** for the three use cases
of the 2026 competition: survival simulator, drone flyby and medical appointment.

Each solution is an HTTP endpoint that the competition's evaluator calls. All inference runs
locally: no cloud APIs are used in any request path, as the rules require.

## Hardware and environment

Everything was developed, trained and served on a single machine:

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 Laptop, 24 GB VRAM (compute capability 12.0, driver 610.88) |
| CPU | Intel Core Ultra 9 275HX, 24 threads |
| OS | Ubuntu on WSL2 (Linux 6.18, glibc 2.43) |
| Python | 3.12, one virtual environment per use case |

The GPU is the limiting resource: the medical appointment endpoint alone needs about 17 GB, so the
endpoints were run one at a time. On a laptop GPU, power and thermal limits matter — a throttled
GPU roughly halves inference speed, which costs score in the two time-limited use cases.

## The three solutions

| Use case | Port | What it runs | Result |
| --- | --- | --- | --- |
| [Survival simulator](survival-simulator/) | 9052 | Hand-written controller: per-agent memory and landmark localization, predator avoidance, foraging and reproduction policy | ~1480 mean score over 12 local seeds |
| [Drone flyby](drone-flyby/) | 9053 | YOLO11s detector with a stride-4 head, trained on the reference objects composited onto outside aerial terrain; tracker plus zooming camera policy | 0.412 validation |
| [Medical appointment](medical-appointment/) | 9054 | Whisper large-v3-turbo speech recognition, then a local Gemma 4 26B-A4B (llama.cpp) answering every question with a verbatim evidence quote mapped back to word timestamps | 0.762 validation |

Each use case folder has its own README with the exact commands to run it.

## Notes that apply to all three

- **One server worker.** All three endpoints keep per-run state and must not be run with multiple
  workers or processes.
- **Response time is part of the score.** Survival allows 1200 s of accumulated wait over 30000
  ticks (~40 ms per tick), drone flyby emits a frame every 333 ms and skips frames you are too slow
  for, and medical appointment allows 60 s per conversation.
- **Local scoring only.** The supplied `local_evaluator.py` / `validate.py` scripts replay the
  supplied data through the endpoints; they make no network submission.
- **Model weights and training data** (`weights/`, `models/`, `.runtime/`, `.training/`) are kept
  out of Git where they are large; each README says what to download or build.
