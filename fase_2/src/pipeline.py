"""Orquestração pública e econômica das etapas prepare, train e report."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


DEFAULT_CONFIG = Path("fase_2/configs/final_experiment.yaml")
REGISTRY_FIELDS = (
    "run_id", "scope", "paradigm", "family", "phase", "model", "representation", "window_size_frames",
    "fold", "seed", "balancing", "fingerprint", "status", "source", "artifact",
    "duration_seconds", "updated_at", "message",
)
VALID_STATUSES = {"pending", "running", "completed", "failed", "reused", "skipped",
                  "blocked", "dependency_missing"}


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    scope: str
    paradigm: str
    family: str
    phase: str
    model: str
    representation: str
    window_size_frames: int
    fold: int
    seed: int
    balancing: str
    fingerprint: str
    status: str
    source: str = "final"
    artifact: str = ""
    reason: str = ""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "data", "preprocessing", "windowing", "features", "splits", "models",
        "training", "balancing", "thresholds", "event_evaluation", "compute_metrics",
        "outputs",
    }
    missing = sorted(required - set(data or {}))
    if missing:
        raise ValueError(f"Configuração principal sem seções: {', '.join(missing)}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_fingerprint(config_path: Path, config: Mapping[str, object]) -> str:
    """Hash somente da configuração e dos pequenos manifestos canônicos."""
    digest = hashlib.sha256(config_path.read_bytes())
    paths = (
        Path(config["data"]["manifest"]),
        Path(config["data"]["annotations"]),
        Path(config["splits"]["manifest"]),
        Path(config["data"]["extraction_manifest"]),
    )
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(_sha256(path).encode() if path.is_file() else b"missing")
    return digest.hexdigest()


def run_fingerprint(base: str, settings: Mapping[str, object]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{base}:{payload}".encode()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv_atomic(path: Path, rows: Iterable[Mapping[str, object]],
                      fields: Sequence[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    selected_fields = list(fields or (materialized[0].keys() if materialized else ()))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected_fields, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry_path(config: Mapping[str, object]) -> Path:
    return Path(config["outputs"]["registry"])


def load_registry(config: Mapping[str, object]) -> dict[str, dict[str, str]]:
    return {row["run_id"]: row for row in _read_csv(registry_path(config))}


def save_registry(config: Mapping[str, object], rows: Mapping[str, Mapping[str, object]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    _write_csv_atomic(registry_path(config), ordered, REGISTRY_FIELDS)


def update_registry(config: Mapping[str, object], run: PlannedRun, status: str,
                    *, artifact: str = "", duration_seconds: float | str = "",
                    message: str = "") -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Status inválido: {status}")
    rows = load_registry(config)
    row = asdict(run)
    row.pop("reason", None)
    row.update(status=status, artifact=artifact or run.artifact,
               duration_seconds=duration_seconds, updated_at=_now(), message=message)
    rows[run.run_id] = row
    save_registry(config, rows)


def _historical_compatible(model: str, balancing: str, fold: int, seed: int) -> str | None:
    if seed != 42:
        return None
    rows = _read_csv(Path("fase_2/outputs/metrics/G4/g4_runs.csv"))
    strategy = {"class_weights": "B", "weighted_sampling": "C"}.get(balancing)
    matches = [row for row in rows if row.get("model") == model
               and row.get("scenario") == strategy and int(row.get("fold", 0)) == fold
               and int(row.get("seed", 0)) == seed and row.get("subset") == "validation"]
    return matches[0].get("run_id", "historical_G4") if matches else None


def _legacy_build_plan(config_path: str | Path, family: str = "all") -> list[PlannedRun]:
    path = Path(config_path)
    config = load_config(path)
    base = pipeline_fingerprint(path, config)
    existing = load_registry(config)
    folds = [int(value) for value in config["splits"]["folds"]]
    runs: list[PlannedRun] = []

    if family in {"classical", "all"}:
        for size in config["windowing"]["sizes_frames"]:
            for fold in folds:
                settings = {"family": "classical", "phase": "screening",
                            "model": "logistic_regression", "representation": "R0_flat",
                            "window": int(size), "fold": fold, "seed": 42,
                            "balancing": "none"}
                fingerprint = run_fingerprint(base, settings)
                run_id = f"final__screening__logistic_regression__r0_flat__w{size}__fold_{fold}__seed_42"
                old = existing.get(run_id)
                status = "completed" if old and old.get("fingerprint") == fingerprint and old.get("status") in {"completed", "reused"} else "pending"
                runs.append(PlannedRun(run_id, "classical", "screening", "logistic_regression",
                                       "R0_flat", int(size), fold, 42, "none",
                                       fingerprint, status,
                                       artifact=old.get("artifact", "") if old else ""))

    candidates = config["training"]["confirmation"]["candidates"]
    seeds = [int(value) for value in config["training"]["confirmation"]["seeds"]]
    for candidate in candidates:
        if not candidate.get("enabled", False) or family not in {candidate["family"], "all"}:
            continue
        candidate_seeds = [42] if candidate.get("deterministic", False) else seeds
        for fold in folds:
            for seed in candidate_seeds:
                settings = {**candidate, "fold": fold, "seed": seed, "phase": "confirmation"}
                fingerprint = run_fingerprint(base, settings)
                representation = str(candidate["representation"])
                run_id = (f"final__confirmation__{candidate['model']}__{representation.lower()}__"
                          f"w{candidate['window']}__{candidate['balancing']}__fold_{fold}__seed_{seed}")
                historical = _historical_compatible(str(candidate["model"]),
                                                     str(candidate["balancing"]), fold, seed)
                old = existing.get(run_id)
                if historical:
                    status, source, artifact = "reused", "historical", historical
                elif old and old.get("fingerprint") == fingerprint and old.get("status") in {"completed", "reused"}:
                    status, source, artifact = old["status"], old.get("source", "final"), old.get("artifact", "")
                else:
                    status, source, artifact = "pending", "final", ""
                runs.append(PlannedRun(run_id, str(candidate["family"]), "confirmation",
                                       str(candidate["model"]), representation,
                                       int(candidate["window"]), fold, seed,
                                       str(candidate["balancing"]), fingerprint, status,
                                       source=source, artifact=artifact,
                                       reason="deterministic: uma execução" if candidate.get("deterministic") else ""))
    return runs


def _matches_filters(run_family: str, run_scope: str, run_paradigm: str, *,
                     family: str, scope: str, paradigm: str) -> bool:
    return (family in {"all", run_family} and scope in {"all", run_scope}
            and paradigm in {"all", run_paradigm})


def _dtw_pair_count(config: Mapping[str, object], fold: int, window: int) -> tuple[int, int, int]:
    rows = [row for row in _read_csv(Path(config["outputs"]["window_counts"]))
            if int(row["fold"]) == fold and int(row["window_size_frames"]) == window]
    counts = {subset: sum(int(row["num_windows"]) for row in rows if row["subset"] == subset)
              for subset in ("train", "validation", "test")}
    evaluations = counts["validation"] + counts["test"]
    return counts["train"], evaluations, counts["train"] * evaluations


def _new_run_status(old: Mapping[str, str] | None, fingerprint: str, *,
                    dependency_ok: bool, blocked_reason: str = "") -> tuple[str, str, str]:
    if old and old.get("fingerprint") == fingerprint and old.get("status") in {"completed", "reused"}:
        return old["status"], old.get("source", "reused"), old.get("artifact", "")
    if not dependency_ok:
        return "dependency_missing", "new", ""
    if blocked_reason:
        return "blocked", "new", ""
    source = "resumed" if old and old.get("artifact") else "new"
    return "pending", source, old.get("artifact", "") if old else ""


def build_plan(config_path: str | Path, family: str = "all", scope: str = "all",
               paradigm: str = "all") -> list[PlannedRun]:
    path = Path(config_path)
    config = load_config(path)
    base = pipeline_fingerprint(path, config)
    existing = load_registry(config)
    folds = [int(value) for value in config["splits"]["folds"]]
    runs: list[PlannedRun] = []
    seed = int(config["training"]["screening"]["seed"])
    optional_available = importlib.util.find_spec("sktime") is not None

    for run_paradigm, specification in config["models"]["screening"].items():
        if run_paradigm == "ensemble" or not specification.get("enabled", False):
            continue
        run_family = "classical"
        if not _matches_filters(run_family, "screening", run_paradigm, family=family,
                                scope=scope, paradigm=paradigm):
            continue
        representation = str(specification["representation"])
        balancing = str(specification.get("balancing", "none"))
        for candidate in specification["candidates"]:
            model = str(candidate["model"])
            dependency_ok = optional_available or run_paradigm not in {"shapelet", "transform"}
            for window in specification["windows"]:
                for fold in folds:
                    settings = {"scope": "screening", "paradigm": run_paradigm,
                                "family": run_family, "model": model,
                                "representation": representation, "window": int(window),
                                "fold": fold, "seed": seed, "balancing": balancing,
                                "parameters": candidate["parameters"]}
                    fingerprint = run_fingerprint(base, settings)
                    run_id = (f"final__screening__{run_paradigm}__{model}__"
                              f"{representation.lower()}__w{window}__fold_{fold}__seed_{seed}")
                    blocked_reason = ""
                    reason = ""
                    if run_paradigm == "distance":
                        train_count, eval_count, pairs = _dtw_pair_count(config, fold, int(window))
                        maximum = int(candidate["cost"]["max_pairs_without_confirmation"])
                        reason = (f"DTW train={train_count} eval={eval_count} pairs={pairs:,}; "
                                  f"limit={maximum:,}; cache={candidate['cost']['cache_distances']}")
                        if pairs > maximum:
                            blocked_reason = reason
                    status, source, artifact = _new_run_status(
                        existing.get(run_id), fingerprint, dependency_ok=dependency_ok,
                        blocked_reason=blocked_reason)
                    if status == "dependency_missing":
                        reason = 'sktime absent; python -m pip install -e "fase_2[tsc]"'
                    runs.append(PlannedRun(
                        run_id, "screening", run_paradigm, run_family, "screening", model,
                        representation, int(window), fold, seed, balancing, fingerprint, status,
                        source=source, artifact=artifact, reason=reason))

    promotion_file = Path(config["training"]["confirmation"]["promotion_file"])
    promoted = promotion_file.is_file()
    for candidate in config["training"]["confirmation"]["candidates"]:
        if not candidate.get("enabled", False):
            continue
        run_family = str(candidate["family"])
        run_paradigm = "deep" if run_family == "temporal" else "feature"
        if not _matches_filters(run_family, "confirmation", run_paradigm, family=family,
                                scope=scope, paradigm=paradigm):
            continue
        seeds = ([42] if candidate.get("deterministic") else
                 [int(value) for value in config["training"]["confirmation"]["seeds"]])
        for fold in folds:
            for candidate_seed in seeds:
                settings = {**candidate, "scope": "confirmation", "paradigm": run_paradigm,
                            "fold": fold, "seed": candidate_seed}
                fingerprint = run_fingerprint(base, settings)
                representation = str(candidate["representation"])
                run_id = (f"final__confirmation__{candidate['model']}__{representation.lower()}__"
                          f"w{candidate['window']}__{candidate['balancing']}__fold_{fold}__seed_{candidate_seed}")
                historical = _historical_compatible(str(candidate["model"]),
                                                     str(candidate["balancing"]), fold, candidate_seed)
                old = existing.get(run_id)
                if historical:
                    status, source, artifact = "reused", "reused", historical
                    reason = "historical compatible result"
                elif old and old.get("fingerprint") == fingerprint and old.get("status") == "completed":
                    status, source, artifact = "completed", old.get("source", "new"), old.get("artifact", "")
                    reason = ""
                elif not promoted:
                    status, source, artifact = "blocked", "new", ""
                    reason = f"confirmation blocked until screening promotion: {promotion_file}"
                else:
                    status, source, artifact, reason = "pending", "new", "", ""
                runs.append(PlannedRun(
                    run_id, "confirmation", run_paradigm, run_family, "confirmation",
                    str(candidate["model"]), representation, int(candidate["window"]), fold,
                    candidate_seed, str(candidate["balancing"]), fingerprint, status,
                    source=source, artifact=artifact, reason=reason))
    return runs


def sync_plan_registry(config_path: str | Path, family: str = "all", scope: str = "all",
                       paradigm: str = "all") -> list[PlannedRun]:
    config = load_config(config_path)
    registry = load_registry(config)
    plan = build_plan(config_path, family, scope, paradigm)
    active_ids = {run.run_id for run in build_plan(config_path, "all", "all", "all")}
    for run_id, row in registry.items():
        row["scope"] = row.get("scope") or row.get("phase") or "historical"
        if not row.get("paradigm"):
            row["paradigm"] = ("deep" if row.get("family") == "temporal" else
                               "rule" if row.get("model") in {"dummy", "fixed_rules"} else "feature")
        if row.get("source") not in {"historical", "new", "reused", "resumed"}:
            row["source"] = ("historical" if row.get("phase") == "historical" else
                             "reused" if row.get("status") == "reused" else "new")
        if (run_id.startswith("final__") and run_id not in active_ids
                and row.get("status") in {"pending", "failed"}):
            row.update(status="skipped", source="new", updated_at=_now(),
                       message="superseded by current canonical configuration")
    for run in plan:
        current = registry.get(run.run_id)
        if current is None or run.status == "reused" or current.get("status") not in {"completed"}:
            row = asdict(run)
            row.pop("reason", None)
            row.update(duration_seconds="", updated_at=_now(), message=run.reason)
            registry[run.run_id] = row
    save_registry(config, registry)
    return plan


def sync_historical_registry(config_path: str | Path) -> int:
    """Import existing scientific executions without reading heavy artifacts."""
    config = load_config(config_path)
    registry = load_registry(config)
    sources = (
        ("fase_2/outputs/metrics/G1/g1_execution_table.csv", "classical"),
        ("fase_2/outputs/metrics/G2/g2_execution_table.csv", "temporal"),
        ("fase_2/outputs/metrics/G3/g3_execution_table.csv", "temporal"),
        ("fase_2/outputs/metrics/G4/g4_runs.csv", "mixed"),
        ("fase_2/outputs/metrics/G45/focal_r0_w60__runs.csv", "temporal"),
        ("fase_2/outputs/metrics/G45/g45c_runs.csv", "mixed"),
    )
    imported = 0
    for source_name, default_family in sources:
        source = Path(source_name)
        seen: set[str] = set()
        for index, item in enumerate(_read_csv(source), 1):
            run_id = item.get("run_id") or f"historical__{source.stem}__{index}"
            if run_id in seen:
                continue
            seen.add(run_id)
            model = item.get("model", "unknown")
            family = default_family
            if family == "mixed":
                family = "temporal" if model in {"lstm", "tcn", "transformer"} else "classical"
            paradigm = ("rule" if model in {"dummy", "fixed_rules"} else
                        "deep" if family == "temporal" else "feature")
            row = {
                "run_id": run_id, "scope": "historical", "paradigm": paradigm,
                "family": family, "phase": "historical",
                "model": model, "representation": item.get("representation", item.get("ablation", "")),
                "window_size_frames": item.get("window_size_frames", ""),
                "fold": item.get("fold", ""), "seed": item.get("seed", "42"),
                "balancing": item.get("balancing", item.get("strategy", "")),
                "fingerprint": "historical_artifact", "status": "reused",
                "source": "historical", "artifact": source_name,
                "duration_seconds": item.get("training_seconds", item.get("train_seconds", "")),
                "updated_at": _now(), "message": "historical result imported; do not retrain",
            }
            if run_id not in registry:
                imported += 1
            registry[run_id] = {**registry.get(run_id, {}), **row}
    analyses = {
        "historical__dummy": "fase_2/outputs/metrics/dummy_baseline_summary.csv",
        "historical__missingness": "fase_2/outputs/metrics/facial_missingness.csv",
        "historical__threshold_crossfit": "fase_2/outputs/metrics/G45/g45a_summary.csv",
        "historical__pose_robustness": "fase_2/outputs/metrics/G46/g46_runs.csv",
        "historical__extractor_comparison": "fase_2/outputs/metrics/G48A/g48a_video_04_extractor_summary.csv",
        "historical__yolo_exploratory": "fase_2/outputs/metrics/G47/video_01__yolo26n-pose__cpu__summary.json",
    }
    for run_id, artifact in analyses.items():
        if Path(artifact).is_file() and run_id not in registry:
            registry[run_id] = {
                "run_id": run_id, "scope": "historical", "paradigm": "rule",
                "family": "analysis", "phase": "historical",
                "model": "", "representation": "", "window_size_frames": "", "fold": "",
                "seed": "", "balancing": "", "fingerprint": "historical_artifact",
                "status": "reused", "source": "historical", "artifact": artifact,
                "duration_seconds": "", "updated_at": _now(),
                "message": "historical analysis imported; do not retrain",
            }
            imported += 1
        elif run_id in registry:
            registry[run_id].update(scope="historical", paradigm="rule", source="historical")
    save_registry(config, registry)
    return imported


def format_plan(config_path: str | Path, family: str, scope: str = "all",
                paradigm: str = "all") -> str:
    path = Path(config_path)
    config = load_config(path)
    sync_historical_registry(path)
    plan = sync_plan_registry(path, family, scope, paradigm)
    pending = [run for run in plan if run.status == "pending"]
    reused = [run for run in plan if run.status in {"completed", "reused"}]
    prior_times = [float(row["duration_seconds"]) for row in load_registry(config).values()
                   if row.get("duration_seconds") not in {"", None}]
    estimate = (sum(prior_times) / len(prior_times) * len(pending)) if prior_times else None
    lines = [
        f"Plano: scope={scope} | paradigm={paradigm} | family={family}",
        f"Paradigmas: {', '.join(sorted({run.paradigm for run in plan}))}",
        f"Modelos: {', '.join(sorted({run.model for run in plan}))}",
        f"Representações: {', '.join(sorted({run.representation for run in plan}))}",
        f"Janelas: {sorted({run.window_size_frames for run in plan})}",
        f"Folds: {sorted({run.fold for run in plan})}",
        f"Seeds: {sorted({run.seed for run in plan})}",
        "Estados: " + " | ".join(
            f"{state}={sum(run.status == state for run in plan)}"
            for state in ("pending", "blocked", "dependency_missing", "reused", "completed")
            if any(run.status == state for run in plan)
        ),
        f"Runs: {len(plan)} | reutilizáveis/concluídos: {len(reused)} | pendentes: {len(pending)}",
        f"Estimativa: {estimate / 60:.1f} min baseada no registro" if estimate is not None else "Estimativa: indisponível até haver execuções finais comparáveis",
        f"Saída: {config['outputs']['root']}",
        f"Configuração resolvida: {path}",
        f"Fingerprint: {pipeline_fingerprint(path, config)}",
        "",
        "Runs:",
    ]
    lines.extend(
        f"  [{run.status.upper():9}] {run.run_id}" + (f" ({run.reason})" if run.reason else "")
        for run in plan
    )
    return "\n".join(lines)


def _validate_series(path: Path, expected_rows: int, fps: float) -> tuple[int, int]:
    required = {"video_id", "frame_index", "timestamp_seconds", "ear", "mar", "pitch",
                "yaw", "roll", "face_detected"}
    rows = detected = 0
    previous_timestamp = -1.0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: colunas ausentes {sorted(missing)}")
        for expected_index, row in enumerate(reader):
            if int(row["frame_index"]) != expected_index:
                raise ValueError(f"{path}: frame_index descontínuo em {expected_index}")
            timestamp = float(row["timestamp_seconds"])
            if not math.isfinite(timestamp) or timestamp < previous_timestamp:
                raise ValueError(f"{path}: timestamp inconsistent at frame {expected_index}")
            previous_timestamp = timestamp
            is_detected = row["face_detected"] == "1"
            values = [row[name] for name in ("ear", "mar", "pitch", "yaw", "roll")]
            if is_detected and (any(value == "" for value in values)
                                or not all(math.isfinite(float(value)) for value in values)):
                raise ValueError(f"{path}: valor facial inválido no frame {expected_index}")
            detected += int(is_detected)
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{path}: {rows} frames; esperado {expected_rows}")
    expected_duration = expected_rows / fps
    if abs(previous_timestamp - expected_duration) > max(2.0, expected_duration * 0.02):
        raise ValueError(f"{path}: final timestamp inconsistent with declared FPS")
    return rows, detected


def prepare(config_path: str | Path, *, force: bool = False) -> int:
    from .data.splits import SplitBlock, validate_split_blocks

    path = Path(config_path)
    config = load_config(path)
    output_root = Path(config["outputs"]["root"])
    state_path = output_root / "prepare_state.json"
    fingerprint = pipeline_fingerprint(path, config)
    if not force and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") == fingerprint and state.get("status") == "completed":
            print(f"[PREPARE REUSED] cache válido: {state_path}")
            return 0

    print("[PREPARE 1/7] Validando manifesto e metadados")
    videos = _read_csv(Path(config["data"]["manifest"]))
    expected_ids = set(config["data"]["videos"])
    if {row["video_id"] for row in videos} != expected_ids:
        raise ValueError("Manifesto não contém exatamente os quatro vídeos configurados")
    for row in videos:
        if (float(row["fps"]) <= 0 or int(row["num_frames"]) <= 0
                or int(row["width"]) <= 0 or int(row["height"]) <= 0
                or float(row["duration_seconds"]) <= 0):
            raise ValueError(f"Metadado inválido: {row['video_id']}")
    inconsistent_duration = [row["video_id"] for row in videos
                             if abs(float(row["duration_seconds"])
                                    - int(row["num_frames"]) / float(row["fps"])) > 0.1]
    if inconsistent_duration:
        raise ValueError(f"Duration and FPS disagree: {inconsistent_duration}")
    missing_videos = [row["relative_path"] for row in videos if not Path(row["relative_path"]).is_file()]

    print("[PREPARE 2/7] Validando intervalos anotados")
    intervals = _read_csv(Path(config["data"]["annotations"]))
    if not intervals or set(row["video_id"] for row in intervals) != expected_ids:
        raise ValueError("Intervalos anotados ausentes ou incompletos")
    allowed = set(config["data"]["classes"])
    if any(row["behavior_label"] and row["behavior_label"] not in allowed for row in intervals):
        raise ValueError("Classe comportamental desconhecida nos intervalos")

    print("[PREPARE 3/7] Validando e reutilizando séries faciais canônicas")
    series_dir = Path(config["data"]["facial_series"])
    series_summary = []
    for row in videos:
        series_path = series_dir / f"{row['video_id']}.csv"
        count, detected = _validate_series(series_path, int(row["num_frames"]), float(row["fps"]))
        series_summary.append({"video_id": row["video_id"], "frames": count,
                               "detected": detected, "missing_rate": 1 - detected / count})

    print("[PREPARE 4/7] Reutilizando diagnóstico de missingness")
    missingness_paths = [Path("fase_2/outputs/metrics/facial_missingness.csv"),
                         Path("fase_2/outputs/metrics/facial_missingness_by_class.csv"),
                         Path("fase_2/outputs/metrics/facial_missingness_by_window.csv"),
                         Path("fase_2/outputs/metrics/window_distribution.csv"),
                         Path("fase_2/outputs/metrics/split_window_distribution.csv"),
                         Path("fase_2/outputs/metrics/preprocessing_comparison_summary.csv"),
                         Path("fase_2/src/features/temporal_window_features.py"),
                         Path(config["features"]["temporal_behavior_v1"]["config"])]
    if not all(item.is_file() for item in missingness_paths):
        raise FileNotFoundError("Diagnóstico de missingness incompleto; gere-o pelo pipeline legado")

    print("[PREPARE 5/7] Validando folds e purge gap")
    blocks = [SplitBlock(fold=int(row["fold"]), subset=row["subset"], video_id=row["video_id"],
                         start_frame=int(row["start_frame"]), end_frame=int(row["end_frame"]))
              for row in _read_csv(Path(config["splits"]["manifest"]))]
    errors = validate_split_blocks(blocks, purge_gap_frames=int(config["splits"]["purge_gap_frames"]))
    if errors:
        raise ValueError("Splits inválidos: " + "; ".join(errors))

    print("[PREPARE 6/7] Auditando duração real das janelas por FPS")
    durations = []
    for row in videos:
        fps = float(row["fps"])
        for size in config["windowing"]["sizes_frames"]:
            durations.append({"video_id": row["video_id"], "fps": fps,
                              "window_size_frames": size, "duration_seconds": int(size) / fps})
    _write_csv_atomic(output_root / "window_durations.csv", durations)
    _write_csv_atomic(output_root / "series_validation.csv", series_summary)

    print("[PREPARE 7/7] Registrando fingerprint e cache")
    output_root.mkdir(parents=True, exist_ok=True)
    state = {"status": "completed", "fingerprint": fingerprint, "updated_at": _now(),
             "videos_available": len(videos) - len(missing_videos),
             "videos_missing_but_series_reused": missing_videos,
             "series": series_summary, "resampling": "not_required_for_historical_baseline",
             "note": "As durações variam por FPS; o baseline de 30/60/150 frames foi preservado."}
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, state_path)
    print(f"[DONE] prepare fingerprint={fingerprint[:12]} cache={state_path}")
    if missing_videos:
        print(f"[WARNING] {len(missing_videos)} vídeos brutos indisponíveis; séries derivadas válidas foram reutilizadas")
    return 0


def _train_logistic(config_path: Path, config: Mapping[str, object],
                    runs: Sequence[PlannedRun], force: bool) -> None:
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .data.splits import SplitBlock
    from .preprocessing.missingness import expand_behavior_labels
    from .training.classical_baselines import evaluate_predictions, feature_matrix, split_windows
    from .training.dummy_baseline import build_feature_windows, load_series

    pending = [run for run in runs if run.model == "logistic_regression"
               and (force or run.status == "pending")]
    if not pending:
        return
    series = load_series(Path(config["data"]["facial_series"]))
    intervals = _read_csv(Path(config["data"]["annotations"]))
    labels = expand_behavior_labels({key: len(value) for key, value in series.items()}, intervals)
    blocks = [SplitBlock(fold=int(row["fold"]), subset=row["subset"], video_id=row["video_id"],
                         start_frame=int(row["start_frame"]), end_frame=int(row["end_frame"]))
              for row in _read_csv(Path(config["splits"]["manifest"]))]
    by_size = {}
    parameters = dict(config["models"]["classical"]["logistic_regression"])
    output = Path(config["outputs"]["root"])
    metrics_path = output / "new_classical_metrics.csv"
    metric_rows = _read_csv(metrics_path)
    total = len(pending)
    for index, run in enumerate(pending, 1):
        print(f"[TRAIN {index}/{total}] family=classical model=logistic_regression "
              f"window={run.window_size_frames} fold={run.fold} seed={run.seed}")
        update_registry(config, run, "running")
        started = time.perf_counter()
        try:
            if run.window_size_frames not in by_size:
                by_size[run.window_size_frames] = build_feature_windows(
                    series, labels, size_frames=run.window_size_frames,
                    stride_frames=int(config["windowing"]["stride_frames"]),
                    minimum_proportion=float(config["windowing"]["minimum_target_proportion"]))
            subsets = split_windows(by_size[run.window_size_frames], blocks, run.fold)
            indices = tuple(range(len(subsets["train"][0].values)))
            x_train, y_train = feature_matrix(subsets["train"], indices)
            model = Pipeline([("scale", StandardScaler()),
                              ("model", LogisticRegression(random_state=run.seed, **parameters))])
            model.fit(x_train, y_train)
            checkpoint = Path(config["outputs"]["checkpoints"]) / f"{run.run_id}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, checkpoint)
            for subset_name in ("validation", "test"):
                values, expected = feature_matrix(subsets[subset_name], indices)
                predicted = model.predict(values)
                summary, _, _ = evaluate_predictions(
                    model_name="logistic_regression", ablation="historical_r3", fold=run.fold,
                    subset=subset_name, expected=expected, predicted=predicted,
                    train_seconds=time.perf_counter() - started, resumed=False)
                summary.update(run_id=run.run_id, seed=run.seed,
                               representation=run.representation,
                               window_size_frames=run.window_size_frames,
                               balancing=run.balancing, fingerprint=run.fingerprint)
                metric_rows = [row for row in metric_rows
                               if not (row.get("run_id") == run.run_id and row.get("subset") == subset_name)]
                metric_rows.append(summary)
                probabilities = model.predict_proba(values)
                prediction_rows = [
                    {"video_id": item.video_id, "start_frame": item.start_frame,
                     "end_frame": item.end_frame, "actual": actual, "predicted": prediction,
                     "prob_alert": probability[list(model.classes_).index("alert")],
                     "prob_fatigue": probability[list(model.classes_).index("fatigue")],
                     "prob_distraction": probability[list(model.classes_).index("distraction")]}
                    for item, actual, prediction, probability in zip(
                        subsets[subset_name], expected, predicted, probabilities, strict=True)]
                _write_csv_atomic(Path(config["outputs"]["predictions"])
                                  / f"{run.run_id}__{subset_name}.csv", prediction_rows)
            _write_csv_atomic(metrics_path, metric_rows)
            elapsed = time.perf_counter() - started
            update_registry(config, run, "completed", artifact=str(checkpoint),
                            duration_seconds=f"{elapsed:.3f}")
            print(f"[DONE] run_id={run.run_id} time={elapsed:.1f}s checkpoint={checkpoint}")
        except Exception as exc:
            update_registry(config, run, "failed", duration_seconds=f"{time.perf_counter()-started:.3f}",
                            message=str(exc))
            raise


def _train_temporal(config_path: Path, config: Mapping[str, object],
                    runs: Sequence[PlannedRun], force: bool) -> None:
    from .training.temporal_multiseed import main as temporal_main

    pending = [run for run in runs if run.family == "temporal" and (force or run.status == "pending")]
    if not pending:
        return
    groups: dict[tuple[str, int, str], list[PlannedRun]] = {}
    for run in pending:
        groups.setdefault((run.model, run.window_size_frames, run.balancing), []).append(run)
    for group_index, ((model, window, balancing), selected) in enumerate(groups.items(), 1):
        seeds = sorted({run.seed for run in selected})
        runtime = {
            "experiment_name": "final_confirmation",
            "generation": "final",
            "configuration_id": f"confirmation_{model}_{balancing}_w{window}",
            "representation": selected[0].representation,
            "window_size_frames": window,
            "preprocessing_config": "fase_2/configs/preprocessing/baseline_zero_fill.yaml",
            "models": [model], "seeds": config["training"]["confirmation"]["seeds"],
            "evaluation_subsets": ["validation", "test"],
            "training": {
                "max_epochs": config["training"]["max_epochs"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "weight_decay": config["training"]["weight_decay"],
                "monitor": "val_macro_f1", "mode": "max",
                "patience": config["training"]["early_stopping"]["patience"],
                "minimum_delta": config["training"]["early_stopping"]["minimum_delta"],
                "checkpoint_every_epochs": config["training"]["checkpoint_every_epochs"],
                "gradient_clip_norm": config["training"]["gradient_clip_norm"],
                "amp": config["training"]["amp"], "deterministic": True,
                "num_workers": config["training"]["num_workers"], "balancing": balancing,
            },
            "parameters": {model: config["models"]["temporal"]["parameters"][model]},
        }
        resolved = Path(config["outputs"]["resolved"]) / f"temporal_{model}_{balancing}_w{window}.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
        for run in selected:
            update_registry(config, run, "running")
        started = time.perf_counter()
        arguments = ["--experiment-config", str(resolved), "--checkpoint-dir", config["outputs"]["checkpoints"],
                     "--run-dir", str(Path(config["outputs"]["root"]) / "runs"),
                     "--prediction-dir", config["outputs"]["predictions"],
                     "--output-dir", str(Path(config["outputs"]["root"]) / "temporal_metrics"),
                     "--figure-dir", str(Path(config["outputs"]["figures"]) / f"{model}_{balancing}_w{window}"),
                     "--max-runs", str(len(selected))]
        for seed in seeds:
            arguments.extend(["--seed", str(seed)])
        if force:
            arguments.append("--no-resume")
        try:
            print(f"[TRAIN GROUP {group_index}/{len(groups)}] model={model} seeds={seeds}")
            temporal_main(arguments)
            elapsed = time.perf_counter() - started
            per_run = elapsed / len(selected)
            for run in selected:
                update_registry(config, run, "completed", artifact=str(resolved),
                                duration_seconds=f"{per_run:.3f}")
        except Exception as exc:
            for run in selected:
                update_registry(config, run, "failed", message=str(exc))
            raise


def _screening_candidate(config: Mapping[str, object], run: PlannedRun) -> Mapping[str, object]:
    section = config["models"]["screening"][run.paradigm]
    return next(item for item in section["candidates"] if item["model"] == run.model)


def _train_screening(config: Mapping[str, object], runs: Sequence[PlannedRun], *,
                     force: bool, allow_expensive: bool) -> None:
    """Executa a matriz pre-registrada; imports pesados ocorrem somente apos o plano."""
    import joblib
    import numpy as np

    from .data.splits import SplitBlock
    from .preprocessing.missingness import expand_behavior_labels
    from .training.classical_baselines import evaluate_predictions, feature_matrix, split_windows
    from .training.dummy_baseline import CLASSES, load_series
    from .training.tsc_adapters import (
        DependentDTW1NN, MiniRocketRidgeAdapter, ShapeletRidgeAdapter,
        build_temporal_feature_windows, enforce_dtw_limit, estimate_dtw_cost,
        feature_classifier, write_shapelet_metadata,
    )

    executable = [run for run in runs if force or run.status == "pending"
                  or (allow_expensive and run.status == "blocked" and run.paradigm == "distance")]
    unavailable = [run for run in runs if run.status in {"blocked", "dependency_missing"}
                   and run not in executable]
    for run in unavailable:
        print(f"[TRAIN][screening][{run.status}] paradigm={run.paradigm} model={run.model} "
              f"window={run.window_size_frames} fold={run.fold} reason={run.reason}")
    if not executable:
        return

    series = load_series(Path(config["data"]["facial_series"]))
    intervals = _read_csv(Path(config["data"]["annotations"]))
    labels = expand_behavior_labels({key: len(value) for key, value in series.items()}, intervals)
    blocks = [SplitBlock(fold=int(row["fold"]), subset=row["subset"], video_id=row["video_id"],
                         start_frame=int(row["start_frame"]), end_frame=int(row["end_frame"]))
              for row in _read_csv(Path(config["splits"]["manifest"]))]
    fps_by_video = {row["video_id"]: float(row["fps"])
                    for row in _read_csv(Path(config["data"]["manifest"]))}
    feature_config = yaml.safe_load(Path(config["features"]["temporal_behavior_v1"]["config"])
                                    .read_text(encoding="utf-8"))
    feature_windows = None
    sequence_cache = {}
    metric_path = Path(config["outputs"]["root"]) / "screening_metrics.csv"
    metric_rows = _read_csv(metric_path)

    for index, run in enumerate(executable, 1):
        print(f"[TRAIN][screening][{index}/{len(executable)}] paradigm={run.paradigm} "
              f"model={run.model} window={run.window_size_frames} fold={run.fold} seed={run.seed}")
        update_registry(config, run, "running")
        started = time.perf_counter()
        try:
            candidate = _screening_candidate(config, run)
            if run.paradigm == "feature":
                if feature_windows is None:
                    feature_windows = build_temporal_feature_windows(
                        series, labels, size_frames=run.window_size_frames,
                        stride_frames=int(config["windowing"]["stride_frames"]),
                        minimum_proportion=float(config["windowing"]["minimum_target_proportion"]),
                        groups=feature_config["groups"], thresholds=feature_config["thresholds"],
                        fps_by_video=fps_by_video)
                subsets = split_windows(feature_windows, blocks, run.fold)
                indices = range(len(subsets["train"][0].values))
                x_train, y_train = feature_matrix(subsets["train"], indices)
                model = feature_classifier(run.model, candidate["parameters"], seed=run.seed,
                                           balancing=run.balancing)
                model.fit(x_train, y_train)
                evaluation = {name: feature_matrix(subsets[name], indices)
                              for name in ("validation", "test")}
                metadata = {name: subsets[name] for name in ("validation", "test")}
            else:
                key = (run.fold, run.window_size_frames)
                if key not in sequence_cache:
                    from .training.temporal_data import build_sequence_fold
                    sequence_cache[key] = build_sequence_fold(
                        series, labels, blocks, config["preprocessing"]["historical_representations"]["R0"],
                        fold=run.fold, size_frames=run.window_size_frames,
                        stride_frames=int(config["windowing"]["stride_frames"]),
                        minimum_proportion=float(config["windowing"]["minimum_target_proportion"]),
                        representation="R0")[0]
                subsets = sequence_cache[key]
                x_train = subsets["train"].values
                y_train = np.asarray([CLASSES[int(value)] for value in subsets["train"].labels])
                if run.paradigm == "distance":
                    parameters = candidate["parameters"]
                    cost = estimate_dtw_cost(len(x_train),
                                             len(subsets["validation"].values) + len(subsets["test"].values),
                                             run.window_size_frames, x_train.shape[2])
                    enforce_dtw_limit(cost, candidate["cost"]["max_pairs_without_confirmation"],
                                      allow_expensive=allow_expensive)
                    print(f"[DTW] fold={run.fold} train_windows={len(x_train)} "
                          f"evaluation_windows={cost.evaluation_windows} estimated_pairs={cost.pairs:,}")
                    model = DependentDTW1NN(radius=int(parameters["sakoe_chiba_radius"]))
                elif run.paradigm == "shapelet":
                    print(f"[SHAPELET] fold={run.fold} candidates={candidate['parameters']['candidate_budget']}")
                    model = ShapeletRidgeAdapter(candidate["parameters"], seed=run.seed)
                else:
                    print(f"[MINIROCKET] window={run.window_size_frames} fold={run.fold} "
                          f"kernels={candidate['parameters']['num_kernels']}")
                    model = MiniRocketRidgeAdapter(candidate["parameters"], seed=run.seed)
                model.fit(x_train, y_train)
                evaluation = {name: (subsets[name].values,
                                     np.asarray([CLASSES[int(value)] for value in subsets[name].labels]))
                              for name in ("validation", "test")}
                metadata = {name: subsets[name].metadata for name in ("validation", "test")}

            checkpoint = Path(config["outputs"]["checkpoints"]) / f"{run.run_id}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, checkpoint)
            if run.paradigm == "shapelet":
                write_shapelet_metadata(checkpoint.with_suffix(".shapelets.json"), model)
            for subset_name, (values, expected) in evaluation.items():
                if run.paradigm == "distance":
                    model.cache_path = (Path(config["outputs"]["root"]) / "distance_cache"
                                        / f"{run.run_id}__{subset_name}.npy")
                predicted = model.predict(values)
                summary, _, _ = evaluate_predictions(
                    model_name=run.model, ablation=run.representation, fold=run.fold,
                    subset=subset_name, expected=expected, predicted=predicted,
                    train_seconds=time.perf_counter() - started, resumed=run.source == "resumed")
                summary.update(run_id=run.run_id, scope=run.scope, paradigm=run.paradigm,
                               family=run.family, seed=run.seed, representation=run.representation,
                               window_size_frames=run.window_size_frames, balancing=run.balancing,
                               fingerprint=run.fingerprint, source=run.source,
                               pr_auc_status="pending_probability_support",
                               event_metrics_status="pending_confirmation")
                if run.paradigm == "shapelet" and subset_name == "test":
                    presence_path = checkpoint.with_suffix(".external_shapelets.json")
                    presence_path.write_text(json.dumps(
                        model.closest_external_examples(metadata[subset_name]), indent=2),
                        encoding="utf-8")
                metric_rows = [row for row in metric_rows
                               if not (row.get("run_id") == run.run_id and row.get("subset") == subset_name)]
                metric_rows.append(summary)
                prediction_rows = [{"video_id": item.video_id, "start_frame": item.start_frame,
                                    "end_frame": item.end_frame, "actual": actual,
                                    "predicted": prediction}
                                   for item, actual, prediction in zip(
                                       metadata[subset_name], expected, predicted, strict=True)]
                _write_csv_atomic(Path(config["outputs"]["predictions"])
                                  / f"{run.run_id}__{subset_name}.csv", prediction_rows)
            _write_csv_atomic(metric_path, metric_rows)
            elapsed = time.perf_counter() - started
            update_registry(config, run, "completed", artifact=str(checkpoint),
                            duration_seconds=f"{elapsed:.3f}")
            print(f"[DONE] run_id={run.run_id} time={elapsed:.1f}s checkpoint={checkpoint}")
        except Exception as exc:
            update_registry(config, run, "failed", duration_seconds=f"{time.perf_counter()-started:.3f}",
                            message=str(exc))
            raise


def train(config_path: str | Path, family: str, *, scope: str = "screening",
          paradigm: str = "all", force: bool = False, allow_expensive: bool = False) -> int:
    path = Path(config_path)
    config = load_config(path)
    state = Path(config["outputs"]["root"]) / "prepare_state.json"
    if not state.is_file():
        raise RuntimeError("Execute `python -m fase_2 prepare` antes do treinamento")
    prepared = json.loads(state.read_text(encoding="utf-8"))
    if prepared.get("fingerprint") != pipeline_fingerprint(path, config):
        raise RuntimeError("Cache prepare desatualizado; execute `python -m fase_2 prepare` novamente")
    plan = sync_plan_registry(path, family, scope, paradigm)
    if scope in {"screening", "all"}:
        _train_screening(config, [run for run in plan if run.scope == "screening"],
                         force=force, allow_expensive=allow_expensive)
    confirmation = [run for run in plan if run.scope == "confirmation"]
    blocked = [run for run in confirmation if run.status == "blocked"]
    if blocked:
        print(f"[TRAIN][confirmation][blocked] {len(blocked)} runs aguardam promocao do screening")
    runnable_confirmation = [run for run in confirmation if run.status in {"pending", "running"}]
    if runnable_confirmation and family in {"temporal", "all"}:
        _train_temporal(path, config, runnable_confirmation, force)
    print("[DONE] train; resultados históricos compatíveis foram reutilizados")
    return 0


def status(config_path: str | Path = DEFAULT_CONFIG) -> str:
    path = Path(config_path)
    config = load_config(path)
    imported = sync_historical_registry(path)
    plan = sync_plan_registry(path, "all")
    temporal_pending = any(run.status in {"pending", "blocked"} and run.family == "temporal"
                           for run in plan)
    components = [
        ("dados e anotações", Path(config["data"]["annotations"]).is_file(), "concluído"),
        ("séries faciais", all((Path(config["data"]["facial_series"]) / f"{video}.csv").is_file()
                              for video in config["data"]["videos"]), "concluído"),
        ("missingness", Path("fase_2/outputs/metrics/facial_missingness.csv").is_file(), "concluído"),
        ("janelas e splits", Path(config["splits"]["manifest"]).is_file(), "concluído"),
        ("features temporais", Path("fase_2/src/features/temporal_window_features.py").is_file(), "implementado; treino pendente"),
        ("modelos clássicos", Path("fase_2/outputs/metrics/G1/g1_runs.csv").is_file(), "histórico concluído"),
        ("modelos temporais", Path("fase_2/outputs/metrics/G2/g2_runs.csv").is_file(), "seed 42 concluída"),
        ("avaliação por episódio", Path(config["outputs"]["event_metrics"]).is_file(), "parcial"),
        ("estabilidade multi-seed", not temporal_pending,
         "concluido" if not temporal_pending else "pendente"),
        ("relatório consolidado", Path(config["outputs"]["report"]).is_file(), "disponível"),
    ]
    lines = ["Estado do pipeline", "", "Componente | Disponível | Estado", "---|---|---"]
    lines.extend(f"{name} | {'sim' if available else 'não'} | {state}" for name, available, state in components)
    counts = {value: sum(run.status == value for run in plan) for value in VALID_STATUSES}
    lines.extend(["", f"Plano final: {len(plan)} runs | " + " | ".join(
        f"{key}={value}" for key, value in counts.items() if value),
        f"Registro: {config['outputs']['registry']}",
        f"Historicos novos importados nesta chamada: {imported}",
        f"Relatório: {config['outputs']['report']}"])
    return "\n".join(lines)
