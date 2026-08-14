import numpy as np
import pytest

from fase_2.src.training.g1_baselines import (
    aggregate_fold_statistics,
    aggregate_run_table,
    flatten_r0,
    planned_runs,
    predict_fixed_rule_window,
)
from fase_2.src.training.temporal_data import SequenceMetadata, SequenceSplit


RULES = {
    "ear_threshold": 0.25,
    "ear_consecutive_frames": 20,
    "mar_threshold": 0.55,
    "pitch_threshold": -15.0,
    "tie_priority": ("fatigue", "distraction", "alert"),
}


def _predict(values):
    return predict_fixed_rule_window(np.asarray(values, dtype=float), **RULES)


def test_fixed_rules_preserve_threshold_boundaries_and_priority():
    alert = np.tile([0.25, 0.55, -15.0, 0.0, 0.0], (30, 1))
    assert _predict(alert) == "alert"
    distraction = alert.copy()
    distraction[:, 2] = -15.01
    assert _predict(distraction) == "distraction"
    fatigue = alert.copy()
    fatigue[:, 1] = 0.56
    fatigue[:, 2] = -20.0
    assert _predict(fatigue) == "fatigue"


def test_ear_requires_twenty_consecutive_frames_and_window_majority():
    values = np.tile([0.3, 0.0, 0.0, 0.0, 0.0], (60, 1))
    values[:19, 0] = 0.2
    assert _predict(values) == "alert"
    values[:, 0] = 0.2
    assert _predict(values) == "fatigue"


def test_rule_tie_uses_frozen_priority():
    values = np.tile([0.3, 0.0, 0.0, 0.0, 0.0], (30, 1))
    values[:15, 1] = 0.6
    assert _predict(values) == "fatigue"


def test_flatten_r0_preserves_window_order():
    split = SequenceSplit(
        values=np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5),
        labels=np.asarray([0, 2]),
        metadata=(
            SequenceMetadata("video_01", 0, 2, "alert"),
            SequenceMetadata("video_01", 3, 5, "distraction"),
        ),
    )
    matrix, labels = flatten_r0(split)
    assert matrix.shape == (2, 15)
    assert matrix[0].tolist() == list(range(15))
    assert labels.tolist() == ["alert", "distraction"]


def test_flatten_rejects_non_r0_and_dry_run_matrix_is_unique():
    split = SequenceSplit(
        np.zeros((1, 2, 8), dtype=np.float32),
        np.asarray([0]),
        (SequenceMetadata("video_01", 0, 1, "alert"),),
    )
    with pytest.raises(ValueError, match="R0"):
        flatten_r0(split)
    config = {"generation": "G1", "configuration_id": "q", "seed": 42}
    matrix = planned_runs(
        config,
        windows=[30, 60, 150],
        folds=[1, 2, 3, 4],
        models=["fixed_rules", "svm", "random_forest", "xgboost"],
    )
    assert len(matrix) == 48
    assert len({row["run_id"] for row in matrix}) == 48


def test_g1_aggregation_has_one_execution_per_run_and_sample_std():
    base = {
        "generation": "G1",
        "model": "svm",
        "representation": "R0_flat",
        "window_size_frames": 60,
        "seed": 42,
        "balancing": "none",
        "train_seconds": 1.0,
        "resumed_checkpoint": False,
        "num_windows": 10,
        "accuracy": 0.5,
        "macro_f1_present_classes": 0.4,
    }
    rows = []
    for fold, macro in ((1, 0.3), (2, 0.5)):
        for subset in ("validation", "test"):
            rows.append(
                {
                    **base,
                    "run_id": f"run_{fold}",
                    "fold": fold,
                    "subset": subset,
                    "balanced_accuracy": macro,
                    "macro_f1_all_classes": macro,
                }
            )
    executions = aggregate_run_table(rows)
    assert len(executions) == 2
    assert executions[0]["validation_macro_f1_all_classes"] == 0.3
    statistics = aggregate_fold_statistics(rows)
    selected = next(
        row
        for row in statistics
        if row["subset"] == "validation" and row["metric"] == "macro_f1_all_classes"
    )
    assert selected["mean"] == pytest.approx(0.4)
    assert selected["standard_deviation"] == pytest.approx(0.141421356)
