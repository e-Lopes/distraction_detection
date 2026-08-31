import numpy as np
import pandas as pd
import pytest
from fase_2.src.features.pose_robustness import (
    apply_pose_correction,
    fit_pose_correction,
    rolling_perclos,
)


def _training_rows():
    pitch = np.linspace(-30, 30, 120)
    yaw = 15 * np.sin(np.linspace(0, 4 * np.pi, 120))
    ear = 0.30 + 0.002 * pitch - 0.001 * yaw
    return pd.DataFrame(
        {
            "video_id": ["train_video"] * 120,
            "ear": ear,
            "pitch": pitch,
            "yaw": yaw,
            "face_detected": [1] * 120,
            "behavior_label": ["alert"] * 120,
            "timestamp_seconds": np.arange(120, dtype=float),
        }
    )


def test_pose_correction_is_train_only_and_removes_pose_trend():
    training = _training_rows()
    model = fit_pose_correction(training)
    assert model.training_videos == ("train_video",)
    corrected = apply_pose_correction(training, model)
    before = abs(training["ear"].corr(training["pitch"]))
    after = abs(corrected["ear_pose_corrected"].corr(training["pitch"]))
    assert after < before * 0.75
    assert corrected["ear_pose_corrected"].median() == pytest.approx(1, abs=0.03)
    assert corrected["ear_expected_open"].between(model.expected_lower, model.expected_upper).all()


def test_perclos_bounds_and_coverage_gate():
    rows = apply_pose_correction(_training_rows(), fit_pose_correction(_training_rows()))
    rows.loc[:9, "eye_closed_calibrated"] = True
    rows.loc[30:90, "eye_closed_calibrated"] = pd.NA
    result = rolling_perclos(rows, window_seconds=20, minimum_coverage=0.5)
    available = result["perclos_20s"].dropna()
    assert available.between(0, 100).all()
    assert result.loc[50, "perclos_20s"] != result.loc[50, "perclos_20s"]


def test_pose_fit_rejects_missing_required_class_rows():
    rows = _training_rows().assign(behavior_label="fatigue")
    with pytest.raises(ValueError, match="insuficiente"):
        fit_pose_correction(rows)
