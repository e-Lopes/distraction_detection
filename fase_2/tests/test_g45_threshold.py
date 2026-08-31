import csv
import json

import pytest

from fase_2.src.training.g45_threshold import (
    crossfit_thresholds,
    deduplicate_predictions,
    load_validation_predictions,
    predict_with_threshold,
    reconstruct_predictions,
    threshold_grid,
)


def _row(video, start, actual, probabilities, fold=1):
    return {
        "fold": fold,
        "video_id": video,
        "start_frame": start,
        "end_frame": start + 59,
        "actual": actual,
        "prob_alert": probabilities[0],
        "prob_fatigue": probabilities[1],
        "prob_distraction": probabilities[2],
    }


def test_threshold_grid_is_inclusive_and_exact():
    grid = threshold_grid(0.05, 0.50, 0.01)
    assert len(grid) == 46
    assert grid[0] == 0.05
    assert grid[-1] == 0.50


def test_deduplication_averages_and_renormalizes_repeated_windows():
    rows = [
        _row("video_01", 0, "fatigue", (0.2, 0.6, 0.2), fold=1),
        _row("video_01", 0, "fatigue", (0.1, 0.3, 0.1), fold=2),
    ]
    result = deduplicate_predictions(rows)
    assert len(result) == 1
    assert result[0]["source_count"] == 2
    assert json.loads(result[0]["source_folds"]) == [1, 2]
    probability_sum = sum(
        result[0][f"prob_{label}"] for label in ("alert", "fatigue", "distraction")
    )
    assert probability_sum == pytest.approx(1)
    assert result[0]["prob_fatigue"] == pytest.approx(0.6)


def test_crossfit_never_uses_target_video_for_calibration():
    rows = []
    for index, video in enumerate(("video_01", "video_02", "video_03", "video_04")):
        rows.extend(
            [
                _row(video, index * 100, "alert", (0.8, 0.1, 0.1)),
                _row(video, index * 100 + 60, "fatigue", (0.2, 0.6, 0.2)),
                _row(video, index * 100 + 120, "distraction", (0.1, 0.1, 0.8)),
            ]
        )
    decisions, curves = crossfit_thresholds(rows, threshold_grid(0.05, 0.50, 0.01))
    assert len(decisions) == 4
    assert len(curves) == 4 * 46
    for row in curves:
        assert row["target_video"] not in json.loads(row["calibration_videos"])


def test_prediction_rule_prioritizes_fatigue_threshold_then_binary_argmax():
    row = _row("video_01", 0, "fatigue", (0.6, 0.3, 0.1))
    assert predict_with_threshold(row, 0.30) == "fatigue"
    assert predict_with_threshold(row, 0.31) == "alert"
    row["prob_distraction"] = 0.7
    assert predict_with_threshold(row, 0.31) == "distraction"


def test_reconstruction_applies_threshold_of_each_video():
    rows = [
        _row("video_01", 0, "fatigue", (0.6, 0.3, 0.1)),
        _row("video_02", 0, "alert", (0.6, 0.3, 0.1)),
    ]
    decisions = [
        {"target_video": "video_01", "threshold": 0.25},
        {"target_video": "video_02", "threshold": 0.35},
    ]
    result = reconstruct_predictions(rows, decisions)
    assert [row["predicted_crossfit"] for row in result] == ["fatigue", "alert"]


def test_loader_rejects_test_artifact_before_reading(tmp_path):
    path = tmp_path / "run__fold_1__seed_42__test.csv"
    path.write_text("unused", encoding="utf-8")
    with pytest.raises(ValueError, match="somente predições de validação"):
        load_validation_predictions([path])


def test_loader_normalizes_amp_probabilities(tmp_path):
    path = tmp_path / "run__fold_1__seed_42__validation.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "video_id", "start_frame", "end_frame", "actual",
                "prob_alert", "prob_fatigue", "prob_distraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "video_01", "start_frame": 0, "end_frame": 59,
                "actual": "alert", "prob_alert": 0.5, "prob_fatigue": 0.25,
                "prob_distraction": 0.249,
            }
        )
    rows = load_validation_predictions([path])
    probability_sum = sum(
        rows[0][f"prob_{label}"] for label in ("alert", "fatigue", "distraction")
    )
    assert probability_sum == pytest.approx(1)
