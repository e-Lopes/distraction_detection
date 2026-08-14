from __future__ import annotations

import pytest

from fase_2.src.training.g2_report import execution_table, validate_matrix


def _rows():
    rows = []
    for model in ("lstm", "tcn", "transformer"):
        for window in (30, 60, 150):
            for fold in (1, 2, 3, 4):
                run_id = f"{model}-{window}-{fold}"
                for subset in ("validation", "test"):
                    rows.append({
                        "run_id": run_id, "configuration_id": f"w{window}", "generation": "G2",
                        "model": model, "representation": "R0", "window_size_frames": window,
                        "fold": fold, "seed": 42, "parameter_count": 1, "model_size_bytes": 2,
                        "best_epoch": 3, "stopping_epoch": 7, "best_validation_macro_f1": 0.4,
                        "best_training_loss": 0.5, "best_validation_loss": 0.6,
                        "training_seconds": 1.0, "peak_gpu_memory_bytes": 10,
                        "resumed_checkpoint": False, "subset": subset, "num_windows": 10,
                        "accuracy": 0.7, "balanced_accuracy": 0.4, "macro_f1_all_classes": 0.4,
                    })
    return rows


def test_validate_and_collapse_official_matrix():
    rows = _rows()
    validate_matrix(rows)
    assert len(execution_table(rows)) == 36


def test_validate_rejects_incomplete_matrix():
    with pytest.raises(ValueError, match="Matriz G2 invalida"):
        validate_matrix(_rows()[:-2])
