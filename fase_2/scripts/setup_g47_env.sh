#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_EXE:-/home/edu/anaconda3/bin/conda}"
SOURCE_ENV="yolo_env"
TARGET_ENV="g47_env"
ULTRALYTICS_VERSION="8.4.115"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
  echo "O ambiente ${TARGET_ENV} já existe; nada foi sobrescrito."
else
  "${CONDA_BIN}" create --name "${TARGET_ENV}" --clone "${SOURCE_ENV}" --yes
fi

TARGET_PYTHON="/home/edu/anaconda3/envs/${TARGET_ENV}/bin/python"
"${TARGET_PYTHON}" -m pip install "ultralytics==${ULTRALYTICS_VERSION}" psutil

G47_CONFIG_DIR="$(pwd)/fase_2/outputs/cache/G47/ultralytics_config"
mkdir -p "${G47_CONFIG_DIR}"
XDG_CONFIG_HOME="${G47_CONFIG_DIR}" "${TARGET_PYTHON}" -c '
import json
import torch
import mediapipe
import ultralytics
from ultralytics import YOLO

payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "mediapipe": mediapipe.__version__,
    "ultralytics": ultralytics.__version__,
}
if torch.cuda.is_available():
    payload["device"] = torch.cuda.get_device_name(0)
    payload["capability"] = torch.cuda.get_device_capability(0)
YOLO("yolo26n-pose.yaml")
print(json.dumps(payload, indent=2))
'
