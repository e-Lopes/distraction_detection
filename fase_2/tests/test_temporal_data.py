import numpy as np

from fase_2.src.data.splits import SplitBlock
import pytest

from fase_2.src.training.temporal_data import (
    SequenceMetadata,
    SequenceSplit,
    build_sequence_fold,
    limit_sequence_split,
    representation_features,
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


def test_sequence_scaler_uses_training_frames_only():
    series = {
        "video_01": [_row("1"), _row("3"), _row("101"), _row("103")],
        "video_02": [_row("201"), _row("203")],
    }
    labels = {
        "video_01": ["alert", "alert", "fatigue", "fatigue"],
        "video_02": ["distraction", "distraction"],
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 1),
        SplitBlock(1, "validation", "video_01", 2, 3),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    preprocessing = {
        "short_gap_max_frames": 0,
        "long_gap_fill": "zero",
    }
    subsets, scaler, _ = build_sequence_fold(
        series,
        labels,
        blocks,
        preprocessing,
        fold=1,
        size_frames=2,
        stride_frames=2,
        minimum_proportion=0.6,
        representation="R0",
    )
    assert scaler.mean[0] == 2
    assert scaler.scale[0] == 1
    assert np.allclose(subsets["train"].values[0, :, 0], [-1, 1])
    assert subsets["validation"].values[0, 0, 0] == 99
    assert subsets["train"].metadata[0].video_id == "video_01"


def test_representations_have_the_dimensions_defined_by_the_integrated_plan():
    assert representation_features("R0") == ("ear", "mar", "pitch", "yaw", "roll")
    assert representation_features("r1") == representation_features("R0")
    assert len(representation_features("R2")) == 8
    with pytest.raises(ValueError, match="R0, R1 ou R2"):
        representation_features("R3")


def test_r0_builds_five_signals_and_rejects_interpolation():
    series = {
        "video_01": [_row("1"), _row("3"), _row("101"), _row("103")],
        "video_02": [_row("201"), _row("203")],
    }
    labels = {
        "video_01": ["alert", "alert", "fatigue", "fatigue"],
        "video_02": ["distraction", "distraction"],
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 1),
        SplitBlock(1, "validation", "video_01", 2, 3),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    subsets, _, _ = build_sequence_fold(
        series,
        labels,
        blocks,
        {"short_gap_max_frames": 0, "long_gap_fill": "zero"},
        fold=1,
        size_frames=2,
        stride_frames=2,
        minimum_proportion=0.6,
        representation="R0",
    )
    assert subsets["train"].values.shape == (1, 2, 5)
    with pytest.raises(ValueError, match="R0 exige"):
        build_sequence_fold(
            series,
            labels,
            blocks,
            {"short_gap_max_frames": 15, "long_gap_fill": "zero"},
            fold=1,
            size_frames=2,
            stride_frames=2,
            minimum_proportion=0.6,
            representation="R0",
        )


def test_smoke_limit_is_reproducible_and_preserves_observed_classes():
    labels = np.repeat(np.arange(3), 10)
    split = SequenceSplit(
        np.arange(30 * 2 * 5, dtype=np.float32).reshape(30, 2, 5),
        labels,
        tuple(SequenceMetadata("video_01", i * 2, i * 2 + 1, str(labels[i])) for i in range(30)),
    )
    first = limit_sequence_split(split, 9, seed=42)
    second = limit_sequence_split(split, 9, seed=42)
    assert len(first.labels) == 9
    assert set(first.labels) == {0, 1, 2}
    assert np.array_equal(first.values, second.values)


def test_r1_interpolation_does_not_use_a_neighbor_outside_the_window():
    series = {
        "video_01": [_row("1"), _row("", "0"), _row("3"), _row("4"), _row("5"), _row("6")],
        "video_02": [_row("7"), _row("8")],
    }
    labels = {
        "video_01": ["alert"] * 4 + ["fatigue"] * 2,
        "video_02": ["distraction"] * 2,
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 3),
        SplitBlock(1, "validation", "video_01", 4, 5),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    subsets, _, _ = build_sequence_fold(
        series,
        labels,
        blocks,
        {
            "short_gap_max_frames": 1,
            "short_gap_method": "linear",
            "long_gap_fill": "training_median",
        },
        fold=1,
        size_frames=2,
        stride_frames=2,
        minimum_proportion=0.6,
        representation="R1",
    )
    assert subsets["train"].metadata[0].missing_ratio == 0.5
    assert subsets["train"].metadata[0].interpolated_ratio == 0.0


def test_r2_keeps_binary_flags_unscaled_and_declares_feature_order():
    series = {
        "video_01": [_row("1"), _row("", "0"), _row("3"), _row("4"), _row("5"), _row("6")],
        "video_02": [_row("7"), _row("8")],
    }
    labels = {
        "video_01": ["alert"] * 4 + ["fatigue"] * 2,
        "video_02": ["distraction"] * 2,
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 3),
        SplitBlock(1, "validation", "video_01", 4, 5),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    preprocessing = {
        "short_gap_max_frames": 1,
        "short_gap_method": "linear",
        "long_gap_fill": "training_median",
        "flags": ["face_detected", "was_interpolated", "missing_duration_so_far"],
    }
    subsets, scaler, _ = build_sequence_fold(
        series,
        labels,
        blocks,
        preprocessing,
        fold=1,
        size_frames=2,
        stride_frames=2,
        minimum_proportion=0.6,
        representation="R2",
    )
    assert subsets["train"].values.shape[-1] == 8
    assert scaler.mean[5:7] == (0.0, 0.0)
    assert scaler.scale[5:7] == (1.0, 1.0)
    assert set(np.unique(subsets["train"].values[:, :, 5])).issubset({0.0, 1.0})
    assert set(np.unique(subsets["train"].values[:, :, 6])).issubset({0.0, 1.0})


def test_r2_rejects_missing_or_reordered_flags():
    series = {"video_01": [_row("1"), _row("2")], "video_02": [_row("3"), _row("4")]}
    labels = {"video_01": ["alert", "alert"], "video_02": ["fatigue", "fatigue"]}
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 0),
        SplitBlock(1, "validation", "video_01", 1, 1),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    with pytest.raises(ValueError, match="R2 exige flags na ordem"):
        build_sequence_fold(
            series,
            labels,
            blocks,
            {"short_gap_max_frames": 1, "long_gap_fill": "training_median", "flags": []},
            fold=1,
            size_frames=1,
            stride_frames=1,
            minimum_proportion=0.6,
            representation="R2",
        )
