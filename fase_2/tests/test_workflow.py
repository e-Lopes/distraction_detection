from dataclasses import replace
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pytest
import yaml

from test_pipeline_cli import _config, _write_csv
from fase_2.src import pipeline, workflow
from fase_2.src.data_audit import audit


def test_audit_rejects_missing_data_and_split_bounds(tmp_path):
    path = _config(tmp_path)
    config = pipeline.load_config(path)
    config["splits"]["purge_gap_frames"] = 150
    path.write_text(yaml.safe_dump(config))
    _write_csv(Path(config["splits"]["manifest"]),
               ["fold", "subset", "video_id", "start_frame", "end_frame"],
               [1, "test", "video_01", 0, 50])
    result = audit(path)
    assert result["status"] == "failed"
    assert any("fora dos limites" in e for e in result["errors"])
    assert any("Série facial" in e for e in result["errors"])
    assert json.loads((tmp_path / "output/data_audit.json").read_text())["status"] == "failed"


def test_failed_data_check_never_starts_training(tmp_path, monkeypatch):
    path = _config(tmp_path)
    monkeypatch.setattr(workflow, "extract", lambda *a, **kw: 0)
    monkeypatch.setattr(pipeline, "prepare", lambda *a: (_ for _ in ()).throw(ValueError("bad data")))
    monkeypatch.setattr(pipeline, "_train_screening", lambda *a, **kw: pytest.fail("training started"))
    assert workflow.chain(path) == 2
    assert json.loads((tmp_path / "output/chain_state.json").read_text())["status"] == "failed"


def test_chain_keeps_completed_step_when_next_fails_and_resumes(tmp_path, monkeypatch):
    from fase_2.src import reporting

    path = _config(tmp_path)
    config = pipeline.load_config(path)
    selected = pipeline.build_plan(path, "all", "screening", "feature")[:2]
    completed = set()
    calls = []
    monkeypatch.setattr(workflow, "extract", lambda *a, **kw: 0)
    monkeypatch.setattr(pipeline, "prepare", lambda *a: 0)
    monkeypatch.setattr(pipeline, "sync_plan_registry", lambda *a: [
        replace(r, status="completed" if r.run_id in completed else "pending") for r in selected])
    monkeypatch.setattr(reporting, "consolidate_metrics", lambda c: [])
    fail = [True]

    def train(config, runs, **kw):
        run = runs[0]
        calls.append(run.run_id)
        if run == selected[1] and fail[0]:
            raise RuntimeError("interrupted second run")
        artifact = tmp_path / f"{run.run_id}.joblib"
        artifact.write_text("saved")
        pipeline.update_registry(config, run, "completed", artifact=str(artifact))
        completed.add(run.run_id)

    monkeypatch.setattr(pipeline, "_train_screening", train)
    assert workflow.chain(path) == 2
    state = json.loads((tmp_path / "output/chain_state.json").read_text())
    assert state["completed"] == [selected[0].run_id]
    fail[0] = False
    assert workflow.chain(path) == 0
    assert calls == [selected[0].run_id, selected[1].run_id, selected[1].run_id]


def test_saved_classical_fit_is_reused_before_evaluation(tmp_path):
    from sklearn.linear_model import LogisticRegression

    path = _config(tmp_path)
    config = pipeline.load_config(path)
    config["outputs"]["checkpoints"] = str(tmp_path / "checkpoints")
    run = pipeline.build_plan(path, "all", "screening", "feature")[0]
    values, labels = [[0], [1], [2], [3]], [0, 0, 1, 1]
    fitted = pipeline._fit_or_resume(LogisticRegression(), values, labels, config, run)
    # Refit em dados vazios falharia: a retomada deve carregar o modelo salvo.
    resumed = pipeline._fit_or_resume(LogisticRegression(), [], [], config, run)
    assert list(fitted.predict(values)) == list(resumed.predict(values))


def test_missing_face_discards_stale_values():
    from fase_2.src.preprocessing.strategies import preprocess_block, METRICS

    rows = [{"face_detected": "0", **{name: "999" for name in METRICS}}]
    output = preprocess_block(rows, {"short_gap_max_frames": 0, "long_gap_fill": "zero"})
    assert all(float(output[0][name]) == 0 for name in METRICS)
    assert rows[0]["ear"] == "999"


def tiny_config(tmp_path):
    from fase_2.src.data.splits import generate_leave_one_video_out

    path = _config(tmp_path)
    config = pipeline.load_config(path)
    videos = ["video_01", "video_02"]
    config["data"]["videos"] = videos
    config["windowing"].update(sizes_frames=[5], stride_frames=3, minimum_target_proportion=0.6)
    config["splits"].update(folds=[1, 2], purge_gap_frames=5)
    config["models"]["screening"] = {"feature": {"enabled": True,
        "representation": "temporal_behavior_v1", "windows": [5], "balancing": "class_weights",
        "candidates": [{"model": "logistic_regression", "parameters": {"max_iter": 50}}]}}
    config["training"]["confirmation"]["candidates"] = []
    config["outputs"].update(checkpoints=str(tmp_path / "checkpoints"),
        predictions=str(tmp_path / "predictions"), metrics=str(tmp_path / "metrics.csv"))
    series_dir = Path(config["data"]["facial_series"])
    series_dir.mkdir()
    manifests, annotations = [], []
    for video in videos:
        manifests.append(dict(video_id=video, fps=10, num_frames=90, width=64, height=64,
                              duration_seconds=9, relative_path=f"{video}.mp4"))
        for start in range(0, 90, 9):
            annotations.append(dict(video_id=video, start_frame=start, end_frame=start+8,
                                    behavior_label=config["data"]["classes"][(start//9) % 3]))
        rows = [dict(video_id=video, frame_index=i, timestamp_seconds=i/10, face_detected=1,
                     **{name: str(0.1 + ((i//9) % 3)/10) for name in ("ear", "mar", "pitch", "yaw", "roll")})
                for i in range(90)]
        pipeline._write_csv_atomic(series_dir / f"{video}.csv", rows)
    pipeline._write_csv_atomic(Path(config["data"]["manifest"]), manifests)
    pipeline._write_csv_atomic(Path(config["data"]["annotations"]), annotations)
    pipeline._write_csv_atomic(Path(config["splits"]["manifest"]), [asdict(b) for b in
        generate_leave_one_video_out({v: 90 for v in videos}, validation_fraction=0.3, purge_gap_frames=5)])
    Path(config["features"]["temporal_behavior_v1"]["config"]).write_text(yaml.safe_dump(
        {"groups": ["signal_distribution"], "thresholds": {}}))
    path.write_text(yaml.safe_dump(config))
    return path


def test_real_small_chain_saves_models_predictions_and_resumes(tmp_path, monkeypatch):
    path = tiny_config(tmp_path)
    assert audit(path)["status"] == "passed"
    assert workflow.chain(path) == 0
    config = pipeline.load_config(path)
    runs = pipeline.build_plan(path, "all", "screening", "feature")
    assert len(runs) == 2 and all(r.status == "completed" for r in runs)
    for run in runs:
        assert Path(run.artifact).is_file()
        for subset in ("validation", "test"):
            predictions = pipeline._read_csv(Path(config["outputs"]["predictions"]) / f"{run.run_id}__{subset}.csv")
            assert predictions and "prob_fatigue" in predictions[0]
    assert (tmp_path / "output/window_counts.csv").is_file()
    assert (tmp_path / "output/missingness_by_class.csv").is_file()
    monkeypatch.setattr(pipeline, "_train_screening", lambda *a, **kw: pytest.fail("unnecessary retraining"))
    assert workflow.chain(path) == 0


def test_extraction_preserves_completed_video_and_rebuilds_manifest(tmp_path, monkeypatch):
    from fase_2.src.features import extract_facial_series as extractor

    path = tiny_config(tmp_path)
    config = pipeline.load_config(path)
    videos = tmp_path / "videos"
    videos.mkdir()
    for row in pipeline._read_csv(Path(config["data"]["manifest"])):
        (videos / Path(row["relative_path"]).name).write_bytes(b"source-test-only")
    calls = []

    @dataclass
    class Summary:
        video_id: str
        complete: bool = True

    def worker(job):
        calls.append(job["video_id"])
        if job["video_id"] == "video_02" and len(calls) == 2:
            raise RuntimeError("interrupted extraction")
        return Summary(job["video_id"])

    monkeypatch.setattr(extractor, "_extract_video_worker", worker)
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.extract(path, video_dir=videos)
    Path(config["data"]["extraction_manifest"]).unlink()
    assert workflow.extract(path, video_dir=videos) == 0
    assert calls == ["video_01", "video_02", "video_02"]
    assert len(pipeline._read_csv(Path(config["data"]["extraction_manifest"]))) == 2


def test_extraction_uses_configured_video_directory(tmp_path, monkeypatch):
    from fase_2.src.features import extract_facial_series as extractor

    path = tiny_config(tmp_path)
    config = pipeline.load_config(path)
    video_dir = tmp_path / "fixed-videos"
    video_dir.mkdir()
    config["data"]["video_dir"] = str(video_dir)
    for row in pipeline._read_csv(Path(config["data"]["manifest"])):
        (video_dir / Path(row["relative_path"]).name).write_bytes(b"source")
    path.write_text(yaml.safe_dump(config))
    seen = []

    @dataclass
    class Summary:
        video_id: str
        complete: bool = True

    def worker(job):
        seen.append(job["video_path"].parent)
        return Summary(job["video_id"])

    monkeypatch.setattr(extractor, "_extract_video_worker", worker)
    assert workflow.extract(path) == 0
    assert seen == [video_dir, video_dir]


def test_temporal_wrapper_executes_only_requested_fold_and_seed(tmp_path, monkeypatch):
    path = tiny_config(tmp_path)
    config = pipeline.load_config(path)
    canonical = pipeline.load_config(pipeline.DEFAULT_CONFIG)
    config["training"] = canonical["training"]
    config["training"].update(max_epochs=1, batch_size=16, amp=False)
    config["training"]["confirmation"].update(seeds=[42, 123],
        promotion_file=str(tmp_path / "promotion.yaml"), candidates=[
        {"family": "temporal", "model": "lstm", "representation": "R0", "window": 5,
         "balancing": "none", "enabled": True}])
    config["models"]["temporal"] = {"parameters": {"lstm":
        {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0, "pooling": "last"}}}
    config["preprocessing"] = {"historical_representations": {"R0":
        {"short_gap_max_frames": 0, "long_gap_fill": "zero"}}}
    config["outputs"].update(resolved=str(tmp_path / "resolved"), figures=str(tmp_path / "figures"))
    (tmp_path / "promotion.yaml").write_text("reviewed: true")
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(pipeline, "_historical_compatible", lambda *a: None)
    run = next(r for r in pipeline.build_plan(path, "all", "confirmation", "all")
               if r.fold == 2 and r.seed == 123)
    pipeline._train_temporal(path, config, [run], False)
    records = list((tmp_path / "output/runs").glob("*.json"))
    assert len(records) == 1
    result = json.loads(records[0].read_text())
    assert (result["fold"], result["seed"]) == (2, 123)
    saved = pipeline.load_registry(config)[run.run_id]
    assert saved["status"] == "completed"
    assert Path(saved["artifact"]).name == "best_macro_f1.pt"
    assert Path(saved["artifact"]).with_name("progress.json").is_file()


def test_confirmation_svm_saves_each_fold_and_preprocessing(tmp_path, monkeypatch):
    path = tiny_config(tmp_path)
    config = pipeline.load_config(path)
    config["models"]["classical"] = {"svm": {"C": 1.0, "kernel": "linear", "probability": True}}
    config["preprocessing"] = {"historical_representations": {"R0":
        {"short_gap_max_frames": 0, "long_gap_fill": "zero"}}}
    config["training"]["confirmation"]["candidates"] = [
        {"family": "classical", "model": "svm", "representation": "R0_flat", "window": 5,
         "balancing": "class_weights", "enabled": True, "deterministic": True}]
    Path(config["training"]["confirmation"]["promotion_file"]).write_text("reviewed: true")
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(pipeline, "_historical_compatible", lambda *a: None)
    assert workflow.chain(path, scope="confirmation") == 0
    runs = pipeline.build_plan(path, "all", "confirmation", "all")
    assert len(runs) == 2 and all(r.status == "completed" for r in runs)
    for run in runs:
        assert Path(run.artifact).with_name(f"{run.run_id}__preprocessing.json").is_file()


def test_deleted_predictions_make_run_pending_again(tmp_path):
    path = tiny_config(tmp_path)
    assert workflow.chain(path) == 0
    config = pipeline.load_config(path)
    first = pipeline.build_plan(path, "all", "screening", "feature")[0]
    (Path(config["outputs"]["predictions"]) / f"{first.run_id}__test.csv").unlink()
    assert pipeline.build_plan(path, "all", "screening", "feature")[0].status == "pending"
