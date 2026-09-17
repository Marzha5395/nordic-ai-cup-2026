# The server, with CUDA. The detector runs in about 20 ms on a GPU and about a
# second on a CPU, and the budget for a whole request is 3333 ms with a frame
# emitted every 333 ms, so CPU-only inference works but loses most of the
# frames. Run it with `docker run --gpus all -p 9053:9053 drone-flyby`.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

# torch comes from the CUDA index; everything else from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt

# The server, the solution and the trained detector. The sample data under
# src/ is not copied: it is 400 MB and only local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py ./
COPY solution ./solution
COPY weights/detector.pt ./weights/detector.pt

EXPOSE 9053

CMD ["python", "api.py"]
