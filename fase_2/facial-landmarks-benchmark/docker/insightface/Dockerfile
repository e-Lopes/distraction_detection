# Base com CUDA 11.8 / cuDNN8 - compatível com o wheel padrão do onnxruntime-gpu==1.18.x
# (ver https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
#  para outras combinações de versão CUDA/cuDNN caso seu driver exija CUDA 12.x)
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY docker/insightface/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

# Cache de modelos do InsightFace (baixados automaticamente no 1º uso, em
# ~/.insightface/models) — persistido via volume nomeado no docker-compose.
ENV INSIGHTFACE_HOME=/root/.insightface

ENTRYPOINT ["python3", "-m", "src.runners.run_insightface"]
