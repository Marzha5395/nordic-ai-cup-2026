# Medical appointment — how to run

Serves speech recognition and question answering on port **9054**; submit
`http://<your-host>:9054/predict`.

```sh
cd medical-appointment
source .venv/bin/activate
python3 api.py
```

Startup takes about 15 seconds: both models load and are warmed up before the first request.
Wait for `Application startup complete` before submitting, and run one worker only.

The endpoint runs Whisper large-v3-turbo for transcription and Gemma 4 26B-A4B through a local
llama.cpp server (started automatically on 127.0.0.1:9060) for the answers. It needs about 17 GB
of GPU memory, so nothing else should be using the GPU. A conversation takes 6–30 seconds of the
60-second budget, depending on GPU clocks.

## One-time setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 prepare_models.py --build-cuda
```

`prepare_models.py` downloads the pinned models into `models/` (Gemma 4 26B-A4B q4_0, 14.4 GB;
Whisper large-v3-turbo, 1.6 GB; Qwen3.5 as a fallback) and builds llama.cpp with CUDA into
`.runtime/`. That build needs `cmake`, `gh` and a CUDA toolkit; on this machine the toolkit came
from pip wheels (`nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cuda-cccl`,
`nvidia-cuda-crt`, `nvidia-nvvm`, all at matching versions), with the CUDA runtime libraries copied
next to the built `llama-server`. `LLAMA_SERVER=/path/to/llama-server` reuses an existing build.

## Options

Set as environment variables before `python3 api.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL` | Gemma 4 26B-A4B | `models/Qwen3.5-9B-Q4_K_M.gguf` switches language model |
| `LLM_REASONING_BUDGET` | 2048 on GPU | Tokens the model may think before answering; 0 disables |
| `MEDICAL_EVIDENCE_MODE` | `quote` | `range` makes the model return transcript line ranges instead of quotes |
| `MEDICAL_SHORT_CONTEXT` | 1.0 | Seconds below which a span gets the sentence it answers prepended; 0 disables |
| `ASR_DEVICE`, `ASR_COMPUTE_TYPE` | auto | Speech recognition device and precision |
| `MEDICAL_RECORD_DIR` | unset | Saves each request's audio, questions and answers |

## Checking it locally

```sh
python3 local_evaluator.py            # replay the 39 supplied conversations through the endpoint
python3 -m pytest -q                  # unit and HTTP contract tests
python3 benchmark.py --split dev --reuse-transcripts   # offline experiments, no endpoint needed
```
