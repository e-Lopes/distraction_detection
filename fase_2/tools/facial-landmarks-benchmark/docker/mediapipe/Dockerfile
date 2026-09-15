FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/mediapipe/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Modelo da Tasks API (FaceLandmarker, 478 pts incluindo íris). A API legada
# "Solutions" (mp.solutions.face_mesh) foi removida das builds recentes do
# pacote, por isso usamos a Tasks API, que requer este arquivo .task.
RUN mkdir -p /app/models && \
    wget -q -O /app/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

COPY src/ /app/src/

ENTRYPOINT ["python", "-m", "src.runners.run_mediapipe"]
