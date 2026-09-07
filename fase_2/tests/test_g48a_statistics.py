import numpy as np
import pandas as pd

from fase_2.src.evaluation.g48a_statistics import (
    calculate,
    concordance_correlation,
    longest_false_run,
    longest_false_run_masked,
    wilson_interval,
)


def _frame(detected, values):
    size = len(detected)
    return pd.DataFrame({
        "frame_id": np.arange(size), "face_detected": detected,
        "ear": values, "mar": values, "pitch": values, "yaw": values, "roll": values,
        "inference_time_ms": np.arange(size, dtype=float),
    }).set_index("frame_id", drop=False)


def test_helpers_cover_intervals_runs_and_concordance():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert longest_false_run(np.array([True, False, False, True, False])) == 2
    assert longest_false_run_masked(
        np.array([False, False, False, False]), np.array([True, False, False, True])
    ) == 1
    assert concordance_correlation(np.arange(4.0), np.arange(4.0)) == 1.0


def test_calculate_uses_paired_frames_and_candidate_minus_baseline():
    frames = {
        "mediapipe": _frame([True, True, False, False], [1.0, 2.0, np.nan, np.nan]),
        "openface": _frame([True, False, True, False], [2.0, np.nan, 4.0, np.nan]),
    }
    state = pd.Series(["valid", "valid", "operator_absent", "valid"], index=range(4))
    summary, availability, agreement = calculate(frames, fps=2.0, operational_state=state)
    assert set(summary["extractor"]) == {"mediapipe", "openface"}
    assert availability.iloc[0]["mediapipe_only"] == 1
    assert availability.iloc[0]["candidate_only"] == 1
    assert summary.query("extractor == 'openface'").iloc[0]["detected_operator_absent"] == 1
    ear = agreement.query("measure == 'ear'").iloc[0]
    assert ear["paired_valid"] == 1
    assert ear["bias_candidate_minus_mediapipe"] == 1.0
