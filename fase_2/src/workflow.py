"""Extração por vídeo e fila de treinos com resultados persistidos por execução."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from . import pipeline


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def extract(config_path, *, video_dir=None):
    config = pipeline.load_config(config_path)
    configured_video_dir = video_dir or config["data"].get("video_dir")
    output = Path(config["data"]["facial_series"])
    manifest = Path(config["data"]["extraction_manifest"])
    prior = {r["video_id"]: r for r in pipeline._read_csv(manifest)}
    videos = pipeline._read_csv(Path(config["data"]["manifest"]))
    figure_dir = Path(config["outputs"].get(
        "figures", str(Path(config["outputs"]["root"]) / "figures"))) / "facial_indicators"
    from .features.extract_facial_series import load_roi
    roi_path = config["data"].get("roi_config")
    roi = load_roi(Path(roi_path)) if roi_path else None
    for index, video in enumerate(videos, 1):
        video_id = video["video_id"]
        source = Path(video["relative_path"])
        if configured_video_dir:
            source = Path(configured_video_dir) / source.name
        target = output / f"{video_id}.csv"
        progress_path = output / f"{video_id}.extraction.json"
        source_stat = source.stat() if source.is_file() else None
        signature = {"source": str(source.resolve()),
                     "size": source_stat.st_size if source_stat else None,
                     "mtime_ns": source_stat.st_mtime_ns if source_stat else None,
                     "extractor_version": 2, "roi": list(roi) if roi else None}
        if target.is_file() and progress_path.is_file():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if (saved.get("signature") == signature and saved.get("status") == "completed"
                    and saved.get("output_sha256") == pipeline._sha256(target) and saved.get("summary")):
                pipeline._validate_series(target, int(video["num_frames"]), float(video["fps"]))
                prior[video_id] = saved["summary"]
                pipeline._write_csv_atomic(manifest, prior.values())
                from .features.plot_facial_indicators import plot_video
                combined = figure_dir / f"{video_id}_indicators_over_time.png"
                if not combined.is_file():
                    print(f"Gerando gráficos de {video_id}.", flush=True)
                    plot_video(target, combined)
                print(f"Vídeo {index}/{len(videos)} já extraído: {video_id}.", flush=True)
                continue
        if not source.is_file():
            raise FileNotFoundError(f"Selecione a pasta dos vídeos. Arquivo ausente: {source}")
        from .features.extract_facial_series import _extract_video_worker
        print(f"Extraindo vídeo {index}/{len(videos)}: {source.name}. "
              "O resultado será salvo ao concluir este vídeo.", flush=True)
        save_state(progress_path, {"status": "running", "signature": signature})
        summary = _extract_video_worker({"video_id": video_id, "video_path": source,
            "output_path": target, "roi": roi, "overwrite": True,
            "hash_video": True, "progress_every": 1000})
        pipeline._validate_series(target, int(video["num_frames"]), float(video["fps"]))
        if not summary.complete:
            raise ValueError(f"Extração incompleta: {video_id}.")
        prior[video_id] = asdict(summary)
        pipeline._write_csv_atomic(manifest, prior.values())
        save_state(progress_path, {"status": "completed", "signature": signature,
                                   "output_sha256": pipeline._sha256(target), "summary": asdict(summary)})
        from .features.plot_facial_indicators import plot_video
        figures = plot_video(target, figure_dir / f"{video_id}_indicators_over_time.png")
        print(f"Vídeo {index}/{len(videos)} salvo em {target}.", flush=True)
        print(f"Visualizações salvas: {len(figures)} arquivos PNG e versões SVG em {figure_dir}.",
              flush=True)
    return 0


def chain(config_path, *, video_dir=None, scope="screening", paradigm="all"):
    config = pipeline.load_config(config_path)
    root = Path(config["outputs"]["root"])
    state_path = root / "chain_state.json"
    state = {"status": "running", "stage": "dados", "completed": [], "pending": [],
             "updated_at": pipeline._now()}
    save_state(state_path, state)
    try:
        if video_dir or any(not (Path(config["data"]["facial_series"]) / f"{v}.csv").is_file()
                            for v in config["data"]["videos"]):
            extract(config_path, video_dir=video_dir)
        pipeline.prepare(config_path)
        plan = pipeline.sync_plan_registry(config_path, "all", scope, paradigm)
        runnable = [r for r in plan if r.status == "pending"]
        state["pending"] = [r.run_id for r in plan if r.status in {"blocked", "dependency_missing"}]
        for index, run in enumerate(runnable, 1):
            required = ["numpy", "sklearn", "joblib"]
            if run.model == "xgboost":
                required.append("xgboost")
            if (run.family == "temporal" or run.representation == "R0_flat"
                    or run.paradigm in {"distance", "shapelet", "transform"}):
                required.append("torch")
            absent = [name for name in required if importlib.util.find_spec(name) is None]
            if absent:
                state["pending"].append(run.run_id)
                pipeline.update_registry(config, run, "dependency_missing", message=", ".join(absent))
                save_state(state_path, state)
                print(f"{run.model} aguardando instalação: {', '.join(absent)}.", flush=True)
                continue
            state.update(stage="treinamento", current=run.run_id, position=index, total=len(runnable))
            save_state(state_path, state)
            print(f"\nEtapa {index}/{len(runnable)}: {run.model}, divisão {run.fold}, "
                  f"repetição {run.seed}. Esta etapa aparece como em andamento e será marcada "
                  "como salva somente quando modelo, métricas e predições estiverem completos.", flush=True)
            if run.scope == "screening" or run.family == "classical":
                pipeline._train_screening(config, [run], force=False, allow_expensive=False)
            elif run.family == "temporal":
                pipeline._train_temporal(Path(config_path), config, [run], False)
            saved = pipeline.load_registry(config).get(run.run_id, {})
            if saved.get("status") != "completed" or not Path(saved.get("artifact", "")).is_file():
                raise RuntimeError("A etapa não salvou o modelo completo. A fila foi interrompida.")
            state["completed"].append(run.run_id)
            state["updated_at"] = pipeline._now()
            save_state(state_path, state)
            from .reporting import consolidate_metrics
            consolidate_metrics(config)
            print("Modelo e resultados salvos. Seguindo para a próxima etapa.", flush=True)
        state.update(status="paused" if state["pending"] else "completed", stage="finalizado")
        save_state(state_path, state)
        print("Sequência encerrada. Há etapas aguardando dependências ou revisão."
              if state["pending"] else "Todas as etapas desta seleção foram salvas.")
        return 2 if state["pending"] else 0
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     message=str(error), updated_at=pipeline._now())
        save_state(state_path, state)
        print(f"Sequência interrompida: {error}. Os resultados já salvos foram preservados.", flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 2
