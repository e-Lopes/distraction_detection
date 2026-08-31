import csv
from pathlib import Path

import numpy as np
import pytest
from fase_2.src.features.extract_facial_series import (
    SERIES_FIELDS,
    canonicalize_lateral_angle,
    compute_eye_measurement,
    legacy_heuristic_state,
    load_roi,
    main,
    parse_video_specs,
    write_demo,
)


class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _face_points(scale=1.0, angle=0.0, offset=(0.0, 0.0)):
    points = [_Point(0.5, 0.5) for _ in range(468)]
    eye = np.array([[0, 0], [1, 0.5], [2, 0.5], [3, 0], [2, -0.5], [1, -0.5]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for indices, center in (
        ([33, 160, 158, 133, 153, 144], (-2, 0)),
        ([362, 385, 387, 263, 373, 380], (2, 0)),
    ):
        transformed = (eye + center) @ rotation.T * scale + np.asarray(offset) + 50
        for index, value in zip(indices, transformed, strict=True):
            points[index] = _Point(value[0] / 100, value[1] / 100)
    return points


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
    with pytest.raises(ValueError, match="--workers"):
        main(["--workers", "0"])


def test_normalized_ear_is_invariant_to_similarity_transform():
    base, _ = compute_eye_measurement(_face_points(), [362, 385, 387, 263, 373, 380], 100, 100)
    transformed, _ = compute_eye_measurement(
        _face_points(scale=1.7, angle=0.6, offset=(8, -5)),
        [362, 385, 387, 263, 373, 380],
        100,
        100,
    )
    assert transformed == pytest.approx(base, rel=1e-6)


def test_lateral_angle_canonicalization_removes_180_degree_jump():
    assert canonicalize_lateral_angle(-175) == 5
    assert canonicalize_lateral_angle(155) == -25
    assert canonicalize_lateral_angle(-20) == -20
