FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Your model and the server. The sample data under src/ is not copied: it is
# 400 MB and only local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py detector.py gpu_detector.py motion.py solution.py submission.py recorder.py ./
COPY weights/flyby_v6.pt weights/flyby_v6.runtime.json ./weights/

ENV YOLO_OFFLINE=true YOLO_AUTOINSTALL=false

EXPOSE 9053

CMD ["python", "api.py"]
