# Medical appointment project

## Rules and architecture

- Never submit the official evaluation unless the user explicitly asks. `benchmark.py`, unit tests, and `local_evaluator.py` operate on the supplied training data only.
- Scoring is `0.4 * answer_accuracy + 0.6 * mean_positive_temporal_IoU`. Conversations do not individually have five positive questions; do not force a fixed yes/no count.
- The request budget is 60 seconds for the whole conversation. Models must be ready before the server accepts requests.
- Inference must be entirely local. Model downloads belong in `prepare_models.py`, never in `/predict`. No training CSV, training transcript cache, or filename-based answer lookup is used in the inference path.
- Active code: `api.py` -> `example.predict` -> `solver.MedicalSolver`, using `speech.py` and `llm.py`. `main.py` is the older experimental script and is not imported by the service.
- All input questions are answered together. The language model emits explicit booleans and verbatim evidence quotes; `solver.py` maps quotes back to word timestamps. Keep explicit booleans: inferring yes from the presence of a relevant quote also accepts evidence contradicting the question.
- Do not modify the scoring functions in `utils.py` or `local_evaluator.py`.

## Setup and verification

Use Python 3.12 (the system Python on the development laptop is 3.14). The local development environment is `.venv312`.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python local_evaluator.py --oracle
```

For a 16–24 GB NVIDIA GPU, the self-contained container downloads pinned weights during the build and compiles a pinned CUDA llama.cpp runtime:

```sh
docker build -t medical-appointment .
docker run --gpus all -p 9054:9054 medical-appointment
```

For native GPU execution, install CUDA 12.x, cuDNN 9, a C++ compiler, CMake, and GitHub CLI first:

```sh
python -m pip install -r requirements.txt
python prepare_models.py --build-cuda
python api.py
```

The default models are Whisper small.en and Qwen3.5-4B Q4_K_M, with explicit yes/no generation and question-aware evidence boundary refinement. This configuration was chosen from local development experiments. Both models use CUDA on a suitable GPU; Whisper uses int8 on CPU. Qwen3.5-9B is also supported with `prepare_models.py --size 9B` and `LLM_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf`, or `docker build --build-arg LLM_SIZE=9B`. The development laptop exposes only a 2 GB GPU, so automatic device selection uses CPU and leaves that GPU alone.

```sh
python prepare_models.py --size 4B --cpu-runtime
ASR_DEVICE=cpu REQUEST_BUDGET_SECONDS=600 python api.py
```

The CPU budget override is for debugging only, not competition deployment. `LLAMA_SERVER` selects an already-built llama-server executable. `LLM_MODEL` and `ASR_MODEL` select local model paths. `LLM_GPU_LAYERS=0` forces CPU LLM execution. `LLM_URL` can reuse an explicitly started loopback model server; use `LLM_API_KEY` if that server requires authentication. Only one automatic model server can use port 9060 at once.

Whisper large-v3-turbo is optional, via `prepare_models.py --asr dropbox-dash/faster-whisper-large-v3-turbo`, `ASR_MODEL=models/faster-whisper-large-v3-turbo`, and `ASR_BATCH_SIZE=8`. The default non-batched small.en path matches the local timing experiments.

## Local experiments

```sh
python benchmark.py --ids 4 5 6 --reuse-transcripts
python benchmark.py --split dev --reuse-transcripts
python benchmark.py --split holdout --reuse-transcripts
python local_evaluator.py --verbose
```

`benchmark.py` is an offline accuracy experiment with a relaxed LLM timeout, not a 60-second GPU benchmark. Its default development set is the first 29 supplied conversations; the last ten are the local holdout. ASR caches and diagnostic predictions are ignored by git. Do not reuse a transcript cache directory across different ASR configurations. Metadata records the prompt and environment for new benchmark runs.

Use the unchanged HTTP local evaluator on the actual GPU to verify latency as well as accuracy before considering an official attempt. A successful CPU accuracy test does not establish compliance with the GPU request deadline.
