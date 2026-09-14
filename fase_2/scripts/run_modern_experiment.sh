#!/usr/bin/env bash
# Run from repository root. Never extracts videos or launches legacy experiments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
experiment_python="${EXPERIMENT_PYTHON:-fase_2/.venv/bin/python}"
experiment_config="${1:-fase_2/configs/modern_experiment.yaml}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
"$experiment_python" -u -c 'import torch, aeon, mantis; assert torch.cuda.is_available(), "Instale PyTorch com CUDA no ambiente do experimento"; print("GPU:", torch.cuda.get_device_name(0), "VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 2), flush=True)'
mkdir -p fase_2/outputs/modern_v1/logs
experiment_log="fase_2/outputs/modern_v1/logs/run_$(date +%Y%m%d_%H%M%S).log"
{
  "$experiment_python" -u -m fase_2 prepare --config "$experiment_config"
  "$experiment_python" -u -m fase_2 train --scope screening --family all --resume --config "$experiment_config"
  "$experiment_python" -u -m fase_2 report --config "$experiment_config"
} 2>&1 | tee "$experiment_log"
