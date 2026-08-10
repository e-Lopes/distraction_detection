from pathlib import Path

from fase_2.src.data.annotations import (
    AnnotationInterval,
    intervals_to_frames,
    load_annotations,
    parse_time,
    validate_annotations,
)


def interval(**overrides):
    values = {
        "video_id": "video_01",
        "start_second": 0,
        "end_second": 9,
        "behavior_label": "alert",
        "operational_state": "valid",
        "source_label": "Alerta",
        "annotation_version": "v1",
        "source_row": 2,
    }
    values.update(overrides)
    return AnnotationInterval(**values)


def validate(items, **overrides):
    arguments = {
        "expected_video_ids": {"video_01"},
        "behavior_classes": {"alert", "fatigue", "distraction"},
        "operational_states": {"valid", "operator_absent"},
    }
    arguments.update(overrides)
    return validate_annotations(items, **arguments)


def test_parse_time_supports_minutes_and_hours():
    assert parse_time("29:45") == 1_785
    assert parse_time("00:29:45") == 1_785


def test_parse_time_rejects_negative_or_invalid_seconds():
    for value in ("-1:10", "01:60", "invalid"):
        try:
            parse_time(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Tempo inválido aceito: {value}")


def test_invalid_class_inverted_interval_and_unknown_video():
    issues = validate(
        [
            interval(
                video_id="other",
                start_second=10,
                end_second=5,
                behavior_label="sleeping",
            )
        ]
    )
    assert {issue.code for issue in issues} >= {
        "unknown_video",
        "inverted_interval",
        "invalid_behavior_class",
        "missing_video_annotations",
    }


def test_overlap_gap_and_outside_duration_are_detected():
    issues = validate(
        [
            interval(start_second=0, end_second=10, source_row=2),
            interval(start_second=10, end_second=20, source_row=3),
            interval(start_second=25, end_second=35, source_row=4),
        ],
        video_duration_seconds={"video_01": 30.0},
    )
    codes = {issue.code for issue in issues}
    assert "overlap" in codes
    assert "unlabeled_gap" in codes
    assert "outside_video" in codes


def test_subsecond_annotation_overshoot_is_warning_and_clipped():
    issues = validate(
        [interval(start_second=0, end_second=10)],
        video_duration_seconds={"video_01": 9.5},
    )
    matching = [issue for issue in issues if issue.code == "end_clipped_to_video"]
    assert len(matching) == 1
    assert matching[0].severity == "warning"


def test_operator_absent_is_not_behavior_class():
    assert (
        validate(
            [
                interval(
                    behavior_label=None,
                    operational_state="operator_absent",
                    source_label="Ausente",
                )
            ]
        )
        == []
    )
    issues = validate([interval(operational_state="operator_absent")])
    assert any(issue.code == "behavior_during_operational_failure" for issue in issues)


def test_missing_required_csv_value_is_reported(tmp_path: Path):
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "video_id,start_time,end_time,behavior_label,operational_state,source_label,annotation_version\n"
        "video_01,00:00,00:10,alert,valid,,v1\n",
        encoding="utf-8",
    )
    intervals, issues = load_annotations(csv_path)
    assert intervals == []
    assert [issue.code for issue in issues] == ["missing_required_value"]


def test_inclusive_seconds_convert_to_frames_and_clip_at_video_end():
    rows = intervals_to_frames(
        [interval(start_second=1, end_second=2)],
        {"video_01": {"fps": 10.0, "num_frames": 25}},
    )
    assert rows[0]["start_frame"] == 10
    assert rows[0]["end_frame"] == 24
