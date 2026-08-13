import pytest

from fase_2.src.data.splits import SplitBlock
from fase_2.src.preprocessing.strategies import (
    aggregate_preprocessed_window,
    fit_training_medians,
    preprocess_block,
    strategy_feature_names,
)


def _row(value: str, detected: str = "1") -> dict[str, str]:
    return {
        "ear": value,
        "mar": value,
        "pitch": value,
        "yaw": value,
        "roll": value,
        "face_detected": detected,
    }


def test_short_internal_gap_is_interpolated_and_original_validity_is_preserved():
    config = {
        "short_gap_max_frames": 2,
        "short_gap_method": "linear",
        "long_gap_fill": "training_median",
    }
    result = preprocess_block(
        [_row("1"), _row("", "0"), _row("", "0"), _row("4")],
        config,
        training_medians={name: 9.0 for name in ("ear", "mar", "pitch", "yaw", "roll")},
    )
    assert [float(row["ear"]) for row in result] == [1, 2, 3, 4]
    assert [row["face_detected"] for row in result] == ["1", "0", "0", "1"]
    assert [row["was_interpolated"] for row in result] == ["0", "1", "1", "0"]
    assert [row["missing_duration_so_far"] for row in result] == ["0", "1", "2", "0"]


def test_boundary_and_long_gaps_are_not_interpolated():
    config = {
        "short_gap_max_frames": 1,
        "short_gap_method": "linear",
        "long_gap_fill": "training_median",
    }
    medians = {name: 7.0 for name in ("ear", "mar", "pitch", "yaw", "roll")}
    result = preprocess_block(
        [_row("", "0"), _row("2"), _row("", "0"), _row("", "0"), _row("5")],
        config,
        training_medians=medians,
    )
    assert [float(row["ear"]) for row in result] == [7, 2, 7, 7, 5]
    assert not any(row["was_interpolated"] == "1" for row in result)


def test_training_medians_ignore_validation_and_test_values():
    series = {
        "video_01": [_row("1"), _row("3"), _row("100")],
        "video_02": [_row("200")],
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 1),
        SplitBlock(1, "validation", "video_01", 2, 2),
        SplitBlock(1, "test", "video_02", 0, 0),
    ]
    medians = fit_training_medians(series, blocks)
    assert set(medians.values()) == {2.0}


def test_strategy_features_include_flags_only_when_configured():
    base = {
        "name": "base",
        "add_flags": True,
        "flags": ["face_detected"],
        "window_missingness_features": ["missing_ratio", "gap_count", "longest_gap"],
    }
    flagged = {**base, "flags": ["face_detected", "was_interpolated", "missing_duration_so_far"]}
    assert len(strategy_feature_names(base)) == 14
    assert len(strategy_feature_names(flagged)) == 17
    rows = preprocess_block([_row("1"), _row("", "0")], {"long_gap_fill": "zero"})
    values = aggregate_preprocessed_window(rows, flagged)
    assert len(values) == 17
    assert values[strategy_feature_names(flagged).index("was_interpolated_rate")] == 0


def test_training_median_fill_requires_train_statistics():
    with pytest.raises(ValueError, match="training_medians"):
        preprocess_block(
            [_row("", "0")],
            {"long_gap_fill": "training_median"},
        )
