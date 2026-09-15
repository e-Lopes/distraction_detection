# Build do OpenFace 2.2.0 (CE-CLM, 68 landmarks) a partir do código-fonte.
#
# ATENÇÃO: este é historicamente o build mais frágil dos três frameworks -
# o projeto está sem manutenção ativa e dependências do sistema mudam de
# versão para versão do Ubuntu. Se o build falhar, confira primeiro a wiki
# oficial: https://github.com/TadasBaltrusaitis/OpenFace/wiki
# e ajuste as versões de OpenCV/boost abaixo conforme a mensagem de erro.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ca-certificates \
        libopenblas-dev \
        liblapack-dev \
        libboost-all-dev \
        libopencv-dev \
        python3 \
        python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /openface

# Clona o repositório oficial (usar um commit/tag fixo ajuda a reprodutibilidade;
# ajuste a tag abaixo se quiser uma versão específica)
RUN git clone --depth 1 --branch OpenFace_2.2.0 https://github.com/TadasBaltrusaitis/OpenFace.git . \
    && bash download_models.sh

RUN mkdir -p build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE .. \
    && make -j"$(nproc)"

# Dependências Python para o wrapper de benchmark (orquestração + métricas)
WORKDIR /app
COPY docker/openface/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "-m", "src.runners.run_openface", "--binary", "/openface/build/bin/FeatureExtraction"]
