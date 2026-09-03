#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-auto}"
CONDA_BASE="$(conda info --base)"
INSIGHTFACE_PYTHON="${CONDA_BASE}/envs/g48_insightface/bin/python"

if [[ ! -x "${INSIGHTFACE_PYTHON}" ]]; then
  echo "Python do ambiente g48_insightface não encontrado: ${INSIGHTFACE_PYTHON}" >&2
  exit 1
fi

echo "[pré-verificação] InsightFace/ONNX Runtime"
"${INSIGHTFACE_PYTHON}" - "${DEVICE}" <<'PY'
import sys

device = sys.argv[1]
try:
    import insightface  # noqa: F401
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "Ambiente g48_insightface incompleto: " + str(exc) + "\n"
        "Instale os pacotes usando explicitamente o Python indicado pelo script."
    ) from exc

providers = ort.get_available_providers()
print(f"ONNX Runtime providers: {providers}", flush=True)
if device == "cuda" and "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDA solicitada, mas CUDAExecutionProvider não está disponível.")
PY

echo "[1/2] MediaPipe Face Mesh (CPU)"
fase_2/.venv/bin/python -u -m fase_2.scripts.g48a_run_all_frames \
  --extractor mediapipe --progress-every 500

echo "[2/2] InsightFace (device=${DEVICE})"
"${INSIGHTFACE_PYTHON}" -u \
  -m fase_2.scripts.g48a_run_all_frames \
  --extractor insightface --device "${DEVICE}" --progress-every 500

echo "G48A concluída para MediaPipe e InsightFace."
