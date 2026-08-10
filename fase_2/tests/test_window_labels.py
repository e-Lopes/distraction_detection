import pytest
from fase_2.src.data.windowing import build_windows, label_window

CLASSES = {"alert", "fatigue", "distraction"}


def test_majority_label_above_threshold():
    label, proportion, count = label_window(
        ["fatigue"] * 7 + ["alert"] * 3,
        behavior_classes=CLASSES,
        minimum_proportion=0.60,
    )
    assert (label, proportion, count) == ("fatigue", 0.7, 10)


def test_exact_sixty_percent_is_accepted():
    label, proportion, _ = label_window(
        ["alert"] * 6 + ["distraction"] * 4,
        behavior_classes=CLASSES,
        minimum_proportion=0.60,
    )
    assert label == "alert"
    assert proportion == pytest.approx(0.60)


def test_insufficient_majority_and_tie_are_mixed():
    insufficient = label_window(
        ["alert"] * 5 + ["fatigue"] * 3 + ["distraction"] * 2,
        behavior_classes=CLASSES,
        minimum_proportion=0.60,
    )
    tie = label_window(
        ["alert"] * 5 + ["fatigue"] * 5,
        behavior_classes=CLASSES,
        minimum_proportion=0.60,
    )
    assert insufficient[0] == "mixed"
    assert tie[0] == "mixed"


def test_operational_and_missing_frames_are_not_behavior_labels():
    label, proportion, count = label_window(
        [None, "operator_absent", "alert", "alert"],
        behavior_classes=CLASSES,
        minimum_proportion=0.60,
    )
    assert (label, proportion, count) == ("alert", 1.0, 2)
    assert label_window([None, "operator_absent"], behavior_classes=CLASSES)[0] == "mixed"


def test_windows_respect_start_end_stride_and_video_boundary():
    windows = build_windows(
        {"video_01": ["alert"] * 8, "video_02": ["fatigue"] * 5},
        size_frames=4,
        stride_frames=3,
        behavior_classes=CLASSES,
    )
    assert [(w.video_id, w.start_frame, w.end_frame) for w in windows] == [
        ("video_01", 0, 3),
        ("video_01", 3, 6),
        ("video_02", 0, 3),
    ]


def test_video_shorter_than_window_produces_no_window():
    assert (
        build_windows(
            {"video_01": ["alert"] * 3},
            size_frames=4,
            stride_frames=1,
            behavior_classes=CLASSES,
        )
        == []
    )
