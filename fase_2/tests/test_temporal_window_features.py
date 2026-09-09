import numpy as np
import pytest

from fase_2.src.features.temporal_window_features import (
    AVAILABLE_GROUPS,
    extract_temporal_window_features,
    temporal_feature_names,
)


def _row(ear, mar=0.1, pitch=0.0, yaw=0.0, roll=0.0, detected="1", timestamp=0.0):
    def encoded(value):
        return "" if value is None else str(value)

    return {
        "ear": encoded(ear),
        "mar": encoded(mar),
        "pitch": encoded(pitch),
        "yaw": encoded(yaw),
        "roll": encoded(roll),
        "face_detected": detected,
        "timestamp_seconds": str(timestamp),
        # Campo-alvo deliberadamente presente para provar que o extrator não depende dele.
        "frame_label": "fatigue",
    }


def _as_dict(rows, *, groups=AVAILABLE_GROUPS, **kwargs):
    values = extract_temporal_window_features(rows, groups=groups, **kwargs)
    return dict(zip(temporal_feature_names(groups), values, strict=True))


def test_distribution_and_deltas_have_stable_order_and_expected_values():
    rows = [_row(0.4, timestamp=0), _row(0.3, timestamp=0.1), _row(0.2, timestamp=0.2)]
    features = _as_dict(rows, groups=("signal_distribution", "signal_dynamics"))
    assert features["ear_mean"] == pytest.approx(0.3)
    assert features["ear_slope"] == pytest.approx(-0.1)
    assert features["ear_delta_mean"] == pytest.approx(-0.1)
    assert features["ear_mean_abs_delta"] == pytest.approx(0.1)
    assert features["ear_iqr"] == pytest.approx(0.1)
    assert features["ear_line_length"] == pytest.approx(0.2)
    assert features["ear_total_variation"] == pytest.approx(0.2)
    assert len(features) == 140


def test_behavioral_events_and_ratios_are_computed_without_using_the_target():
    rows = [
        _row(0.3, timestamp=0.0),
        _row(0.2, timestamp=0.1),
        _row(0.2, mar=0.7, timestamp=0.2),
        _row(0.3, mar=0.8, yaw=30, timestamp=0.3),
    ]
    features = _as_dict(rows, groups=("ocular", "oral", "head_pose"), fps=10)
    assert features["perclos"] == 0.5
    assert features["eye_closure_event_count"] == 1
    assert features["longest_eye_closure_fraction"] == 0.5
    assert features["mouth_open_ratio"] == 0.5
    assert features["pose_away_ratio"] == 0.25


def test_missing_frames_break_events_and_not_signal_dynamics():
    rows = [
        _row(0.2, timestamp=0.0),
        _row(None, mar=None, pitch=None, yaw=None, roll=None, detected="0", timestamp=0.1),
        _row(0.1, timestamp=0.2),
    ]
    features = _as_dict(rows)
    assert features["perclos"] == 1.0
    assert features["eye_closure_event_count"] == 2
    assert features["ear_delta_mean"] == 0.0
    assert features["face_detected_rate"] == pytest.approx(2 / 3)
    assert features["gap_count"] == 1
    assert features["longest_gap_fraction"] == pytest.approx(1 / 3)


def test_all_missing_window_is_finite_and_explicit():
    rows = [
        _row(None, mar=None, pitch=None, yaw=None, roll=None, detected="0", timestamp=index / 10)
        for index in range(3)
    ]
    values = extract_temporal_window_features(rows)
    features = dict(zip(temporal_feature_names(), values, strict=True))
    assert np.isfinite(values).all()
    assert features["face_detected_rate"] == 0
    assert features["missing_ratio"] == 1
    assert features["longest_gap_fraction"] == 1


def test_multivariate_and_interpolation_features_are_finite_for_constants():
    rows = [_row(0.3, mar=0.2, timestamp=0.0), _row(0.3, mar=0.2, timestamp=0.1)]
    rows[1]["was_interpolated"] = "1"
    features = _as_dict(rows, groups=("multivariate", "missingness"), fps=10)
    assert features["ear_mar_correlation"] == 0
    assert features["ear_mar_covariance"] == 0
    assert features["interpolated_ratio"] == 0.5


def test_configuration_validation_rejects_ambiguous_inputs():
    rows = [_row(0.3)]
    with pytest.raises(ValueError, match="duplicados"):
        extract_temporal_window_features(rows, groups=("ocular", "ocular"))
    with pytest.raises(ValueError, match="desconhecidos"):
        extract_temporal_window_features(rows, groups=("unknown",))
    with pytest.raises(ValueError, match="positivo"):
        extract_temporal_window_features(rows, thresholds={"ear_closed": 0})
