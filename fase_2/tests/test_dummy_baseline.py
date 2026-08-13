from fase_2.src.data.splits import SplitBlock
from fase_2.src.training.dummy_baseline import (
    FeatureWindow,
    aggregate_window,
    evaluate_dummy,
)


def _row(value: str, detected: str) -> dict[str, str]:
    return {
        "ear": value,
        "mar": value,
        "pitch": value,
        "yaw": value,
        "roll": value,
        "face_detected": detected,
    }


def test_aggregate_zero_fills_only_missing_metrics_and_keeps_detection_flag():
    values = aggregate_window([_row("0", "1"), _row("", "0")])
    assert values[0] == 0
    assert values[10] == 0.5
    assert values[11] == 0.5
    assert values[12] == 1
    assert values[13] == 1


def test_dummy_uses_train_majority_and_marks_absent_test_class():
    windows = [
        FeatureWindow("video_01", 0, 3, 4, "alert", (0.0,)),
        FeatureWindow("video_01", 4, 7, 4, "alert", (0.0,)),
        FeatureWindow("video_01", 8, 11, 4, "fatigue", (0.0,)),
        FeatureWindow("video_01", 20, 23, 4, "distraction", (0.0,)),
        FeatureWindow("video_01", 24, 27, 4, "alert", (0.0,)),
        FeatureWindow("video_02", 0, 3, 4, "alert", (0.0,)),
        FeatureWindow("video_02", 4, 7, 4, "distraction", (0.0,)),
    ]
    blocks = [
        SplitBlock(1, "train", "video_01", 0, 11),
        SplitBlock(1, "validation", "video_01", 20, 27),
        SplitBlock(1, "test", "video_02", 0, 7),
    ]
    summary, per_class, _ = evaluate_dummy(windows, blocks)
    assert {row["train_majority_class"] for row in summary} == {"alert"}
    test_summary = next(row for row in summary if row["subset"] == "test")
    assert '"fatigue"' in test_summary["absent_classes"]
    fatigue = next(
        row for row in per_class if row["subset"] == "test" and row["label"] == "fatigue"
    )
    assert fatigue["support"] == 0
    assert fatigue["f1"] == ""
