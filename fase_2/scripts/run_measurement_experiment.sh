#!/usr/bin/env bash
# Classification only. Raw re-extraction is a separate, explicitly authorized step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
measurement_python="${EXPERIMENT_PYTHON:-fase_2/.venv/bin/python}"
measurement_config="${1:-fase_2/configs/measurement_experiment.yaml}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
"$measurement_python" -c 'import torch; assert torch.cuda.is_available(), "PyTorch CUDA necessário para a LSTM"; print(torch.cuda.get_device_name(0))'
mkdir -p fase_2/outputs/measurement_v1/logs
measurement_log="fase_2/outputs/measurement_v1/logs/run_$(date +%Y%m%d_%H%M%S).log"
{
  "$measurement_python" -u -m fase_2 prepare --config "$measurement_config"
  "$measurement_python" -u -m fase_2 train --scope screening --family all --resume --config "$measurement_config"
  "$measurement_python" -u -m fase_2 report --config "$measurement_config"
} 2>&1 | tee "$measurement_log"
