import csv
from pathlib import Path

import pytest

from fase_2.src.features.extract_facial_series import (
    SERIES_FIELDS,
    legacy_heuristic_state,
    load_roi,
    main,
    parse_video_specs,
    write_demo,
)


def test_parse_video_specs_uses_anonymous_defaults():
    assert parse_video_specs(None) == [
        ("video_01", "1.mp4"),
        ("video_02", "2.mp4"),
        ("video_03", "3.mp4"),
        ("video_04", "4.mp4"),
    ]


def test_parse_video_specs_rejects_paths_and_duplicates():
    with pytest.raises(ValueError):
        parse_video_specs(["video_01=subdir/1.mp4"])
    with pytest.raises(ValueError):
        parse_video_specs(["video_01=1.mp4", "video_01=2.mp4"])


def test_roi_validation_on_load(tmp_path: Path):
    config = tmp_path / "roi.json"
    config.write_text('{"roi_cadeira": [1, 2, 30, 40]}', encoding="utf-8")
    assert load_roi(config) == (1, 2, 30, 40)
    config.write_text('{"roi_cadeira": [1, 2, 0, 40]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_roi(config)


def test_legacy_state_is_explicitly_separate_from_target():
    assert legacy_heuristic_state(0.1, 0.1, 0.0, 20) == "fatigue"
    assert legacy_heuristic_state(0.3, 0.6, 0.0, 0) == "fatigue"
    assert legacy_heuristic_state(0.3, 0.1, -20.0, 0) == "distraction"
    assert legacy_heuristic_state(0.3, 0.1, 0.0, 0) == "alert"


def test_demo_requires_explicit_call_and_preserves_missingness(tmp_path: Path):
    write_demo(tmp_path, overwrite=False)
    output = tmp_path / "demo_facial_series.csv"
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == SERIES_FIELDS
    assert len(rows) == 5
    assert rows[2]["face_detected"] == "0"
    assert rows[2]["ear"] == ""
    assert rows[2]["operational_state"] == "face_missing"
    with pytest.raises(FileExistsError):
        write_demo(tmp_path, overwrite=False)


def test_real_execution_never_falls_back_to_demo():
    with pytest.raises(ValueError, match="--video-dir"):
        main([])
    with pytest.raises(ValueError, match="não pode ser combinado"):
        main(["--demo", "--video-dir", "videos"])
