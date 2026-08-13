from fase_2.src.data.splits import SplitBlock
from fase_2.src.data.windowing import WindowRecord
from fase_2.src.training.preprocessing_comparison import build_fold_features


def _row(value: str, detected: str = "1") -> dict[str, str]:
    return {
        "ear": value,
        "mar": value,
        "pitch": value,
        "yaw": value,
        "roll": value,
        "face_detected": detected,
    }


def test_fold_features_do_not_interpolate_across_block_boundary():
    series = {
        "video_01": [_row("1"), _row("", "0"), _row("3"), _row("4")],
        "video_02": [_row("5"), _row("6")],
    }
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 1),
        SplitBlock(1, "validation", "video_01", 2, 3),
        SplitBlock(1, "test", "video_02", 0, 1),
    ]
    windows = [
        WindowRecord("video_01", 0, 1, 2, "alert", 1.0, 2),
        WindowRecord("video_01", 2, 3, 2, "fatigue", 1.0, 2),
        WindowRecord("video_02", 0, 1, 2, "distraction", 1.0, 2),
    ]
    strategy = {
        "name": "test",
        "short_gap_max_frames": 1,
        "short_gap_method": "linear",
        "long_gap_fill": "training_median",
        "add_flags": True,
        "flags": ["face_detected", "was_interpolated"],
        "window_missingness_features": ["missing_ratio"],
    }
    subsets, medians = build_fold_features(series, windows, blocks, strategy, fold=1)
    names = [
        f"{metric}_{stat}"
        for metric in ("ear", "mar", "pitch", "yaw", "roll")
        for stat in ("mean", "std")
    ] + ["missing_ratio", "face_detected_rate", "was_interpolated_rate"]
    train_values = subsets["train"][0].values
    assert medians["ear"] == 1
    assert train_values[names.index("ear_mean")] == 1
    assert train_values[names.index("was_interpolated_rate")] == 0
