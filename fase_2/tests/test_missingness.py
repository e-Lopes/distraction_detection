import csv
from pathlib import Path

import pytest

from fase_2.src.preprocessing.missingness import (
    consecutive_false_lengths,
    diagnose_file,
    summarize_detection,
)


def test_gaps_include_both_edges():
    assert consecutive_false_lengths([False, False, True, False, True, False]) == [2, 1, 1]


def test_summary_distinguishes_short_and_long_gaps():
    summary = summarize_detection(
        "video_01", [False, True, False, False, True, False, False, False], short_gap_max_frames=2
    )
    assert summary.missing_frames == 6
    assert summary.gap_count == 3
    assert summary.short_gap_count == 2
    assert summary.long_gap_count == 1
    assert summary.longest_gap_frames == 3
    assert summary.median_gap_frames == 2


def _write_series(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "video_id", "frame_index", "timestamp_seconds", "ear", "mar", "pitch", "yaw",
        "roll", "face_detected",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnosis_preserves_valid_zeros(tmp_path: Path):
    path = tmp_path / "video.csv"
    _write_series(
        path,
        [
            {"video_id": "video_01", "frame_index": 0, "timestamp_seconds": 0, "ear": 0,
             "mar": 0, "pitch": 0, "yaw": 0, "roll": 0, "face_detected": 1},
            {"video_id": "video_01", "frame_index": 1, "timestamp_seconds": 0.1, "ear": "",
             "mar": "", "pitch": "", "yaw": "", "roll": "", "face_detected": 0},
        ],
    )
    summary = diagnose_file(path, short_gap_max_frames=15)
    assert summary.detected_frames == 1
    assert summary.missing_frames == 1


def test_diagnosis_rejects_metric_on_missing_frame(tmp_path: Path):
    path = tmp_path / "video.csv"
    _write_series(
        path,
        [{"video_id": "video_01", "frame_index": 0, "timestamp_seconds": 0, "ear": 0.2,
          "mar": "", "pitch": "", "yaw": "", "roll": "", "face_detected": 0}],
    )
    with pytest.raises(ValueError, match="face_detected=0"):
        diagnose_file(path, short_gap_max_frames=15)
