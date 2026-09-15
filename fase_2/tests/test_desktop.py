from __future__ import annotations

import os
import sys
import time

import pytest

from fase_2.src.desktop import Job


@pytest.mark.parametrize("name", ["Qualidade das medidas · 14/09", "Famílias modernas · 14/09"])
def test_new_experiments_use_training_without_legacy_extraction(name):
    from fase_2.src.desktop import EXPERIMENTS, execution_command, simple_overview, protocol_path
    from fase_2.src.pipeline import load_config
    from fase_2.__main__ import build_parser

    path = EXPERIMENTS[name]
    command = execution_command(path, "chain", "Comparar modelos", "Redes neurais", "videos")
    args = build_parser().parse_args(command[4:])
    assert args.command == "train"
    assert args.scope == "screening"
    assert args.paradigm == "deep"
    assert args.config == str(path)
    assert "--video-dir" not in command
    assert "Treinamentos previstos:" in simple_overview(path)
    assert protocol_path(load_config(path)).is_file()


def test_measurement_extraction_options_and_validation():
    from fase_2.src.desktop import EXPERIMENTS, execution_command
    from fase_2.__main__ import build_parser

    path = EXPERIMENTS["Qualidade das medidas · 14/09"]
    def command(**kwargs):
        return execution_command(path, "measurement-extract", "screening", "all", **kwargs)

    args = build_parser().parse_args(command(video="video_02", start_frame="150",
        max_frames="30", output="sample folder")[4:])
    assert (args.video, args.start_frame, args.max_frames, args.output, args.full) == (
        "video_02", 150, 30, "sample folder", False)
    assert "--full" in command(video="video_02", full=True)
    for options in ({"video": "unknown"}, {"video": "video_01", "max_frames": "91"},
                    {"video": "video_01", "start_frame": "-1"},
                    {"video": "video_01", "full": True, "start_frame": "1"}):
        with pytest.raises(ValueError):
            command(**options)


def test_job_streams_output_and_failure_code():
    job = Job()
    job.start([sys.executable, "-u", "-c", "print('output'); raise SystemExit(3)"])
    job.reader.join(timeout=5)
    assert not job.running
    assert job.events.get(timeout=1) == ("log", "output\n")
    assert job.events.get(timeout=1) == ("exit", 3)


def test_job_rejects_concurrent_execution_and_stops():
    job = Job()
    job.start([sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"])
    try:
        assert job.events.get(timeout=5) == ("log", "ready\n")
        with pytest.raises(RuntimeError, match="andamento"):
            job.start([sys.executable, "-c", "pass"])
    finally:
        job.stop()
        job.reader.join(timeout=5)
        job.stop(force=True)
    assert not job.running
    assert job.events.get(timeout=1)[0] == "exit"


def test_interface_dispatches_to_desktop_and_summary_remains_textual(monkeypatch):
    from fase_2 import __main__ as cli
    from fase_2.src import desktop, terminal

    calls = []
    monkeypatch.setattr(desktop, "main", lambda config: calls.append("desktop") or 0)
    monkeypatch.setattr(terminal, "main", lambda config, **kw: calls.append(kw) or 0)
    assert cli.main(["interface"]) == 0
    assert cli.main(["interface", "--summary"]) == 0
    assert calls == ["desktop", {"summary": True}]


@pytest.mark.skipif(os.environ.get("RUN_DESKTOP_TESTS") != "1",
                    reason="Requer sessão gráfica e execução explícita")
def test_desktop_widgets_and_async_job(monkeypatch):
    import tkinter as tk
    from fase_2.src.desktop import Desktop
    from fase_2.src.pipeline import DEFAULT_CONFIG

    root = tk.Tk()
    root.withdraw()
    app = Desktop(root, DEFAULT_CONFIG)
    def wait_for(condition):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            root.update()
            if condition():
                return
            time.sleep(0.02)
        raise AssertionError('Interface não atualizou a tempo')
    try:
        wait_for(lambda: len(app.table.get_children()) == 36)
        assert not app.progress.winfo_manager()
        assert app.message.get() == ''
        assert len(app.tabs.tabs()) == 3
        assert len(app.table.get_children()) == 36
        app.scope.set("confirmation")
        app.paradigm.set("deep")
        app.refresh_plan()
        wait_for(lambda: len(app.table.get_children()) == 20)
        assert len(app.table.get_children()) == 20
        calls = []
        start = app.job.start

        def short_job(command):
            calls.append(command)
            start([sys.executable, "-u", "-c", "print('desktop smoke ok')"])

        monkeypatch.setattr(app.job, "start", short_job)
        app.execute("train")
        assert app.busy
        assert app.progress.winfo_manager() == 'pack'
        deadline = time.monotonic() + 5
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert not app.busy
        assert not app.progress.winfo_manager()
        assert calls[0][4:9] == ["train", "--scope", "confirmation", "--paradigm", "deep"]
        assert "desktop smoke ok" in app.texts["logs"].get("1.0", "end")
        from fase_2.src.desktop import EXPERIMENTS
        app.select_config(EXPERIMENTS['Qualidade das medidas · 14/09'])
        wait_for(lambda: len(app.table.get_children()) == 60)
        assert app.counter_vars['blocked'].get() == '60'
        assert app.counter_vars['completed'].get() == '0'
        app.open_options()
        assert app.options_window.winfo_exists()
        app.options_window.withdraw()
        app.select_config(EXPERIMENTS['Famílias modernas · 14/09'])
        wait_for(lambda: len(app.table.get_children()) == 32)
        assert app.options_window is None
    finally:
        app.job.stop(force=True)
        app.close()


def test_plan_counts_keep_reused_separate_from_completed():
    from types import SimpleNamespace
    from fase_2.src.desktop import plan_counts
    plan = [SimpleNamespace(status=s) for s in
            ['completed', 'reused', 'reused', 'pending', 'running', 'blocked', 'dependency_missing', 'failed']]
    assert plan_counts(plan) == dict(total=8, completed=1, reused=2, pending=1, running=1, blocked=3)


def test_result_rows_exclude_incompatible_and_incomplete_runs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fase_2.src import desktop
    (tmp_path / 'screening_metrics.csv').write_text(
        'run_id,model,macro_f1_all_classes\ngood,svm,0.4\nstale,svm,0.9\n', encoding='utf-8')
    monkeypatch.setattr(desktop, 'load_config', lambda _: {'outputs': {'root': str(tmp_path)}})
    plan = [SimpleNamespace(run_id='good', status='completed', artifact='good.joblib'),
            SimpleNamespace(run_id='stale', status='pending', artifact='stale.joblib')]
    assert [row['run_id'] for row in desktop.result_rows('config', plan)] == ['good']
