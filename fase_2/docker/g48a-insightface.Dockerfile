FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python-headless==4.11.0.86" \
    "onnxruntime==1.29.0" \
    "insightface==0.7.3"

WORKDIR /workspace
ENTRYPOINT ["python", "-u", "-m", "fase_2.scripts.g48a_run_all_frames"]
