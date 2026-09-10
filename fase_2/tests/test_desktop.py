from __future__ import annotations

import os
import sys
import time

import pytest

from fase_2.src.desktop import Job


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
    app = Desktop(root, DEFAULT_CONFIG)
    try:
        root.update()
        assert len(app.tabs.tabs()) == 5
        assert len(app.table.get_children()) == 36
        app.scope.set("confirmation")
        app.paradigm.set("deep")
        app.refresh_plan()
        assert len(app.table.get_children()) == 20
        calls = []
        start = app.job.start

        def short_job(command):
            calls.append(command)
            start([sys.executable, "-u", "-c", "print('desktop smoke ok')"])

        monkeypatch.setattr(app.job, "start", short_job)
        app.execute("train")
        assert app.busy
        deadline = time.monotonic() + 5
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert not app.busy
        assert calls[0][4:9] == ["train", "--scope", "confirmation", "--paradigm", "deep"]
        assert "desktop smoke ok" in app.texts["logs"].get("1.0", "end")
    finally:
        app.job.stop(force=True)
        app.close()
