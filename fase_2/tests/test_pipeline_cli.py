from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

from fase_2.__main__ import build_parser
from fase_2.src import pipeline


def _write_csv(path: Path, header: list[str], row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(row)


def _config(tmp_path: Path) -> Path:
    manifest = tmp_path / "videos.csv"
    annotations = tmp_path / "annotations.csv"
    splits = tmp_path / "splits.csv"
    extraction = tmp_path / "extraction.csv"
    counts = tmp_path / "window_counts.csv"
    _write_csv(manifest, ["video_id", "fps", "num_frames", "relative_path"],
               ["video_01", 10, 10, "missing.mp4"])
    _write_csv(annotations, ["video_id", "start_frame", "end_frame", "behavior_label"],
               ["video_01", 0, 9, "alert"])
    _write_csv(splits, ["fold", "subset", "video_id", "start_frame", "end_frame"],
               [1, "test", "video_01", 0, 9])
    _write_csv(extraction, ["video_id"], ["video_01"])
    with counts.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "subset", "video_id", "window_size_frames", "num_windows"])
        for fold in range(1, 5):
            for subset, amount in (("train", 100), ("validation", 20), ("test", 30)):
                writer.writerow([fold, subset, "video_01", 60, amount])
    config = {
        "data": {"manifest": str(manifest), "annotations": str(annotations),
                 "facial_series": str(tmp_path / "series"), "extraction_manifest": str(extraction),
                 "classes": ["alert", "fatigue", "distraction"], "videos": ["video_01"]},
        "preprocessing": {},
        "windowing": {"sizes_frames": [30, 60, 150], "stride_frames": 15},
        "features": {"temporal_behavior_v1": {"config": str(tmp_path / "features.yaml")}},
        "splits": {"manifest": str(splits), "folds": [1, 2, 3, 4]},
        "models": {"screening": {
            "feature": {"enabled": True, "representation": "temporal_behavior_v1", "windows": [60],
                        "balancing": "class_weights", "candidates": [
                            {"model": model, "parameters": {}} for model in
                            ("logistic_regression", "svm", "random_forest", "xgboost")]},
            "distance": {"enabled": True, "representation": "R0", "windows": [60], "candidates": [
                {"model": "knn_dtw", "parameters": {"k": 1, "sakoe_chiba_radius": 6},
                 "cost": {"max_pairs_without_confirmation": 1000, "cache_distances": True}}]},
            "shapelet": {"enabled": True, "representation": "R0", "windows": [60], "candidates": [
                {"model": "random_shapelet_ridge", "parameters": {}}]},
            "transform": {"enabled": True, "representation": "R0", "windows": [30, 60, 150],
                          "candidates": [{"model": "minirocket_ridge", "parameters": {}}]},
            "ensemble": {"enabled": False, "candidates": []},
        }},
        "training": {"screening": {"seed": 42}, "confirmation": {
            "seeds": [42, 123, 456, 789, 2026],
            "promotion_file": str(tmp_path / "promotion.yaml"), "candidates": [
            {"family": "classical", "model": "svm", "representation": "R0_flat",
             "window": 60, "balancing": "class_weights", "enabled": True, "deterministic": True},
            {"family": "temporal", "model": "lstm", "representation": "R0",
             "window": 60, "balancing": "class_weights", "enabled": True},
        ]}},
        "balancing": {}, "thresholds": {}, "event_evaluation": {}, "compute_metrics": {},
        "outputs": {"root": str(tmp_path / "output"), "registry": str(tmp_path / "registry.csv"),
                    "window_counts": str(counts)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_screening_plan_is_light_and_has_36_runs(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: None)
    torch_was_loaded = "torch" in sys.modules

    plan = pipeline.build_plan(config, "all", "screening", "all")

    assert len(plan) == 36
    assert sum(run.paradigm == "feature" for run in plan) == 16
    assert sum(run.paradigm == "distance" for run in plan) == 4
    assert sum(run.paradigm == "shapelet" for run in plan) == 4
    assert sum(run.paradigm == "transform" for run in plan) == 12
    assert sum(run.status == "dependency_missing" for run in plan) == 16
    assert sum(run.status == "blocked" for run in plan) == 4
    assert torch_was_loaded == ("torch" in sys.modules)


def test_registry_round_trip_preserves_allowed_status(tmp_path, monkeypatch):
    config_path = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_historical_compatible", lambda *args: None)
    plan = pipeline.sync_plan_registry(config_path, "classical", "screening", "feature")
    config = pipeline.load_config(config_path)

    pipeline.update_registry(config, plan[0], "completed", artifact="checkpoint.joblib",
                             duration_seconds="1.25")

    row = pipeline.load_registry(config)[plan[0].run_id]
    assert row["status"] == "completed"
    assert row["artifact"] == "checkpoint.joblib"
    assert row["scope"] == "screening"
    assert row["paradigm"] == "feature"


def test_confirmation_reuses_seed_42_and_blocks_new_seeds(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline, "_historical_compatible",
                        lambda model, balancing, fold, seed: f"old-{model}-{fold}" if seed == 42 else None)
    plan = pipeline.build_plan(config, "all", "confirmation", "all")

    assert len(plan) == 24
    assert sum(run.status == "reused" for run in plan) == 8
    assert sum(run.status == "blocked" for run in plan) == 16
    assert {run.source for run in plan} <= {"new", "reused"}


def test_public_parser_exposes_only_expected_commands():
    parser = build_parser()
    subcommands = next(action for action in parser._actions
                       if isinstance(action, __import__("argparse")._SubParsersAction))

    assert set(subcommands.choices) == {"interface", "status", "prepare", "train", "report", "all",
                                        "chain", "check-data", "extract"}
    args = parser.parse_args(["train", "--family", "temporal", "--plan"])
    assert args.family == "temporal" and args.plan is True
    args = parser.parse_args(["train", "--scope", "screening", "--paradigm", "shapelet", "--plan"])
    assert args.scope == "screening" and args.paradigm == "shapelet"


def test_interface_summary_is_read_only_and_reports_missing_inputs(tmp_path):
    from fase_2.src.terminal import overview

    config = _config(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    output = overview(config)
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    assert before == after
    assert "séries ausentes = video_01" in output
    assert "cache desatualizado" in output
    assert "minimum_iou" in output
    assert "screening:" in output and "confirmation:" in output


def test_interface_keeps_filters_and_survives_action_failure(tmp_path, monkeypatch):
    from fase_2.src import terminal

    config = _config(tmp_path)
    choices = iter(["8", "9", "deep", "3", "5", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(choices))
    calls = []

    def action(path, arguments):
        calls.append(arguments)
        return 1

    monkeypatch.setattr(terminal, "run_action", action)
    assert terminal.main(config) == 0
    assert calls == [
        ["train", "--plan", "--scope", "confirmation", "--paradigm", "deep"],
        ["train", "--scope", "confirmation", "--paradigm", "deep"],
    ]


def test_interface_eof_exits_without_running(tmp_path, monkeypatch):
    from fase_2.src import terminal

    def eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(terminal, "run_action", lambda *_: pytest.fail("unexpected action"))
    assert terminal.main(_config(tmp_path)) == 0


def test_all_stops_if_prepare_fails(monkeypatch):
    from fase_2 import __main__ as cli

    monkeypatch.setattr(cli, "prepare", lambda *a, **kw: 2)
    monkeypatch.setattr(cli, "train", lambda *a, **kw: pytest.fail("unexpected training"))
    monkeypatch.setattr(cli, "generate_report", lambda *a, **kw: pytest.fail("unexpected report"))
    assert cli.main(["all"]) == 2


def test_series_validation_rejects_bad_timestamps(tmp_path):
    series = tmp_path / "video_01.csv"
    header = ["video_id", "frame_index", "timestamp_seconds", "ear", "mar", "pitch", "yaw",
              "roll", "face_detected"]
    series.parent.mkdir(parents=True, exist_ok=True)
    with series.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(["video_01", 0, 0.5, "", "", "", "", "", 0])
        writer.writerow(["video_01", 1, 0.4, "", "", "", "", "", 0])

    with pytest.raises(ValueError, match="timestamp"):
        pipeline._validate_series(series, expected_rows=2, fps=10.0)
