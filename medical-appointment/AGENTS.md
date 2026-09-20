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

The default GPU models are Whisper large-v3-turbo (batch size 1) and Gemma 4 26B-A4B (Google's QAT q4_0 GGUF), with explicit yes/no generation and question-aware evidence boundary refinement. `LLM_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf` restores the previous language model. Compute type is `int8_float16` except on GPUs whose cuBLAS has no int8 GEMM for CTranslate2 (compute capability 12.x, RTX 50 series), where `speech.py` selects `float16`.

The GPU default is 2048 reasoning tokens before answering (`LLM_REASONING_BUDGET`, 0 disables it; 0 on CPU). If no answer has started 15 seconds before the request deadline, `llm.py` abandons the reasoning and re-asks without it, so a slow or power-throttled GPU still returns complete answers; llama.cpp cancels the abandoned generation. Span starts are moved 0.18 s later (`solver.SPAN_START_DELAY_SECONDS`), never past the first word's end: Whisper's word starts include part of the preceding pause, and spans covering exactly the annotated words started a median 0.18 s early on the first 29 conversations (0.16 s on the last 10). Both models use CUDA on a suitable GPU. This is the scored deployment configuration. The lightweight CPU fallback uses Whisper small.en and Qwen3.5-4B; it is not the configuration that achieved the final local score. The development laptop exposes only a 2 GB GPU, so automatic device selection uses CPU. Speech-only experiments explicitly used the small GPU once enough memory was free.

```sh
python prepare_models.py --size 4B --cpu-runtime
ASR_DEVICE=cpu REQUEST_BUDGET_SECONDS=600 python api.py
```

The CPU budget override is for debugging only, not competition deployment. `LLAMA_SERVER` selects an already-built llama-server executable. `LLM_MODEL` and `ASR_MODEL` select local model paths. `LLM_GPU_LAYERS=0` forces CPU LLM execution. `LLM_URL` can reuse an explicitly started loopback model server; use `LLM_API_KEY` if that server requires authentication. Only one automatic model server can use port 9060 at once.

Whisper small.en is available explicitly with `prepare_models.py --asr Systran/faster-whisper-small.en`, `ASR_MODEL=models/faster-whisper-small.en`, and `ASR_BATCH_SIZE=0`. Prefer large-v3-turbo for scoring: small.en sometimes drops punctuation across a long dialogue, causing evidence to be reduced to unhelpful keyword fragments.

## Local experiments

```sh
python benchmark.py --ids 4 5 6 --reuse-transcripts
python benchmark.py --split dev --reuse-transcripts
python benchmark.py --split holdout --reuse-transcripts
python local_evaluator.py --verbose
```

`benchmark.py` is an offline accuracy experiment with a relaxed LLM timeout, not a 60-second GPU benchmark. Its default development set is the first 29 supplied conversations; `--split holdout` selects the last ten. Those last ten were subsequently inspected during development, so they are no longer an untouched holdout. ASR caches and diagnostic predictions are ignored by git. The default cache directory is keyed by ASR settings. Do not reuse an explicit transcript cache directory across different ASR configurations. Metadata records the prompt and environment for new benchmark runs. `--predictions` re-scores saved model outputs without generating new ones and must not be used as a latency measurement.

Spans shorter than 1 second get the sentence they answer prepended (`solver.SHORT_SPAN_CONTEXT_SECONDS`, `MEDICAL_SHORT_CONTEXT` overrides, 0 disables): a bare "None." carries no evidence on its own. The supplied splits disagreed about it (+0.024 tIoU on the first 29 conversations, -0.016 on the last 10, where it touched two spans), so it was settled on validation: 0.7515 -> 0.7617 with everything else identical. `recorder.py` keeps each request's audio, questions, answers and spans under `MEDICAL_RECORD_DIR` when that is set; the README allows keeping the validation sequence.

Local scores with the current defaults, full supplied set through `api.py` and the unchanged evaluator: accuracy 0.990, mean tIoU 0.667, score 0.796 (Qwen3.5-9B without reasoning scored 0.766). Split measurements, first 29 / last 10 conversations: 0.812 / 0.792. Rejected after measuring on both splits: 1024-token reasoning budget, quote trimming to the question's clause, extending spans into adjacent sentences, self-contained-quote prompting, spans on questions answered no, keyword/TF-IDF relocation of spans, and torchaudio forced alignment of word timings (worse than Whisper's own). These remain local supplied-data results, not official validation.

Final configuration check on a different ten-conversation subset: sample IDs `10 20 39 42 43 47 48 50 52 54`, 100 questions, accuracy 0.990, mean positive tIoU 0.673, combined score approximately 0.800. Raw outputs are in `benchmark_results/turbo_clauses_9b.json`; final postprocessing was checked with `benchmark_results/final_100.json`. The transcript cache used was `transcripts/turbo_gpu`. These are local supplied-data results, not official validation or evaluation results. The 9B language model ran on CPU for this check. Large-v3-turbo speech recognition was measured on the 2 GB GPU at roughly 4–13 seconds per conversation.

`evidence_ranker.py` is an experimental ONNX passage-ranking baseline. It underperformed the LLM-based locator and is not imported by the service or included in the deployment container.

Verification completed: 28 unit/HTTP-contract tests pass; Python compilation and `git diff --check` pass. A live `/predict` smoke test used GPU transcription, CPU Qwen3.5-9B, offline model loading, and an unrelated audio filename; it returned the expected three booleans with valid spans and null negative evidence. The local CPU request budget was relaxed for that smoke test. The CUDA image tags were checked, but the full container build and full-GPU request latency were not tested on this laptop.

Use the unchanged HTTP local evaluator on the actual GPU to verify latency as well as accuracy before considering an official attempt. A successful CPU accuracy test does not establish compliance with the GPU request deadline.
