FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates
RUN git clone --depth 1 --branch b10917 https://github.com/ggml-org/llama.cpp.git /llama.cpp \
    && test "$(git -C /llama.cpp rev-parse HEAD)" = "8ea290247c87ced2ab245b056ffe96dbcf90d36c"
ARG CUDA_ARCHITECTURES="75;80;86;89;90"
RUN cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /llama.cpp/build --target llama-server -j 4

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 ca-certificates
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV LLAMA_SERVER=/opt/llama/llama-server
ENV LD_LIBRARY_PATH="/opt/llama:${LD_LIBRARY_PATH}"
ENV ASR_DEVICE=cuda
ENV PYTHONUNBUFFERED=1
COPY --from=builder /llama.cpp/build/bin/ /opt/llama/
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY prepare_models.py .
ARG LLM_SIZE=9B
RUN python prepare_models.py --size "${LLM_SIZE}"
ENV LLM_MODEL=/app/models/Qwen3.5-${LLM_SIZE}-Q4_K_M.gguf
ENV HF_HUB_OFFLINE=1

# Your model and the server. The sample data under data/ is not copied: only
# local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py solver.py speech.py llm.py recorder.py ./

EXPOSE 9054

CMD ["python", "api.py"]
