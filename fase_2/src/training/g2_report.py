"""Consolida a matriz oficial G2 em evidencias versionaveis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from ..data.config import load_yaml, repository_path
from .classical_baselines import save_figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = ("alert", "fatigue", "distraction")
COLORS = {"lstm": "#377eb8", "tcn": "#e41a1c", "transformer": "#4daf4a"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_matrix(rows: Sequence[Mapping[str, object]]) -> None:
    runs = {str(row["run_id"]) for row in rows}
    combinations = {
        (
            str(row["model"]),
            int(row["window_size_frames"]),
            int(row["fold"]),
            int(row["seed"]),
        )
        for row in rows
    }
    subsets_by_run = {
        run_id: {str(row["subset"]) for row in rows if row["run_id"] == run_id}
        for run_id in runs
    }
    expected = {
        (model, window, fold, 42)
        for model in ("lstm", "tcn", "transformer")
        for window in (30, 60, 150)
        for fold in (1, 2, 3, 4)
    }
    if combinations != expected or len(runs) != 36:
        missing = sorted(expected - combinations)
        extra = sorted(combinations - expected)
        raise ValueError(f"Matriz G2 invalida; ausentes={missing}, extras={extra}")
    invalid = sorted(run_id for run_id, subsets in subsets_by_run.items() if subsets != {"validation", "test"})
    if invalid:
        raise ValueError(f"Runs sem validation/test completos: {invalid}")


def execution_table(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for run_id in sorted({str(row["run_id"]) for row in rows}):
        selected = [row for row in rows if row["run_id"] == run_id]
        base = selected[0]
        output: dict[str, object] = {
            key: base[key]
            for key in (
                "run_id", "configuration_id", "generation", "model", "representation",
                "window_size_frames", "fold", "seed", "parameter_count", "model_size_bytes",
                "best_epoch", "stopping_epoch", "best_validation_macro_f1",
                "best_training_loss", "best_validation_loss", "training_seconds",
                "peak_gpu_memory_bytes", "resumed_checkpoint",
            )
        }
        for row in selected:
            subset = str(row["subset"])
            for metric in ("num_windows", "accuracy", "balanced_accuracy", "macro_f1_all_classes"):
                output[f"{subset}_{metric}"] = row[metric]
        result.append(output)
    return result


def fold_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_all_classes")
    keys = sorted({(str(r["model"]), int(r["window_size_frames"]), str(r["subset"])) for r in rows})
    for model, window, subset in keys:
        selected = [r for r in rows if r["model"] == model and int(r["window_size_frames"]) == window and r["subset"] == subset]
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in selected])
            result.append({
                "model": model, "window_size_frames": window, "subset": subset,
                "metric": metric, "n_folds": len(values), "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)), "minimum": float(values.min()),
                "maximum": float(values.max()),
            })
    return result


def per_class_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    keys = sorted({(str(r["model"]), int(r["window_size_frames"]), str(r["subset"]), str(r["label"])) for r in rows})
    for model, window, subset, label in keys:
        selected = [r for r in rows if r["model"] == model and int(r["window_size_frames"]) == window and r["subset"] == subset and r["label"] == label]
        for metric in ("precision", "recall", "f1"):
            values = np.asarray([float(r[metric]) for r in selected if r[metric] != ""])
            result.append({
                "model": model, "window_size_frames": window, "subset": subset,
                "label": label, "metric": metric, "n_folds_with_support": len(values),
                "mean": float(values.mean()) if len(values) else "",
                "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else "",
                "minimum": float(values.min()) if len(values) else "",
                "maximum": float(values.max()) if len(values) else "",
            })
    return result


def prediction_distribution(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    keys = sorted({(str(r["run_id"]), str(r["subset"]), str(r["predicted"])) for r in rows})
    for run_id, subset, predicted in keys:
        selected = [r for r in rows if r["run_id"] == run_id and r["subset"] == subset]
        matching = [r for r in selected if r["predicted"] == predicted]
        count = sum(int(r["count"]) for r in matching)
        total = sum(int(r["count"]) for r in selected)
        base = selected[0]
        result.append({
            "run_id": run_id, "model": base["model"], "window_size_frames": base["window_size_frames"],
            "fold": base["fold"], "seed": base["seed"], "subset": subset,
            "predicted_class": predicted, "count": count, "proportion": count / total,
        })
    return result


def artifact_manifest(
    run_rows: Sequence[Mapping[str, object]],
    model_root: Path,
    log_root: Path,
    prediction_root: Path,
    root: Path,
) -> list[dict[str, object]]:
    result = []
    for row in run_rows:
        run_id = str(row["run_id"])
        paths = {
            "best_checkpoint": model_root / run_id / "best_macro_f1.pt",
            "last_checkpoint": model_root / run_id / "last.pt",
            "run_log": log_root / f"{run_id}.json",
            "validation_predictions": prediction_root / f"{run_id}__validation.csv",
            "test_predictions": prediction_root / f"{run_id}__test.csv",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Artefato ausente: {path}")
            result.append({
                "run_id": run_id, "artifact_type": kind, "relative_path": _relative(path, root),
                "size_bytes": path.stat().st_size, "sha256": _sha256(path),
            })
    return result


def _plot_metric(
    stats: Sequence[Mapping[str, object]],
    metric: str,
    output: Path,
    *,
    label: str | None = None,
) -> None:
    selected = [
        r
        for r in stats
        if r["subset"] == "validation"
        and r["metric"] == metric
        and (label is None or r.get("label") == label)
    ]
    windows = (30, 60, 150)
    models = ("lstm", "tcn", "transformer")
    x = np.arange(len(windows), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, model in zip((-0.22, 0.0, 0.22), models, strict=True):
        current = [next(r for r in selected if r["model"] == model and int(r["window_size_frames"]) == window) for window in windows]
        ax.bar(x + offset, [float(r["mean"]) for r in current], 0.2, yerr=[float(r["standard_deviation"]) for r in current], capsize=3, label=model.upper(), color=COLORS[model])
    ax.set_xticks(x, [str(window) for window in windows])
    ax.set_xlabel("Janela (frames)")
    ax.set_ylabel("Macro F1" if metric == "macro_f1_all_classes" else "F1 de Fatigue")
    ax.set_title("G2: media e desvio-padrao entre folds de validacao")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def _plot_epochs(run_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    models = ("lstm", "tcn", "transformer")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        selected = [r for r in run_rows if r["model"] == model]
        for window, marker in zip((30, 60, 150), ("o", "s", "^"), strict=True):
            current = [r for r in selected if int(r["window_size_frames"]) == window]
            ax.scatter([int(r["best_epoch"]) for r in current], [int(r["stopping_epoch"]) for r in current], label=f"w={window}", marker=marker, s=45)
        ax.set_title(model.upper())
        ax.set_xlabel("Melhor epoca")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Epoca de parada")
    axes[-1].legend()
    save_figure(fig, output)


def write_report(path: Path, fold_stats: Sequence[Mapping[str, object]], class_stats: Sequence[Mapping[str, object]]) -> None:
    macro = sorted(
        (r for r in fold_stats if r["subset"] == "validation" and r["metric"] == "macro_f1_all_classes"),
        key=lambda r: float(r["mean"]), reverse=True,
    )
    fatigue = {(r["model"], int(r["window_size_frames"])): r for r in class_stats if r["subset"] == "validation" and r["label"] == "fatigue" and r["metric"] == "f1"}
    lines = [
        "# G2 — modelos temporais em R0", "", "Matriz oficial: 36 execucoes (3 modelos x 3 janelas x 4 folds), seed 42, distribuicao original e sem balanceamento.", "",
        "| Modelo | Janela | Macro F1 de validacao (media +/- DP) | F1 Fatigue (media +/- DP) |", "|---|---:|---:|---:|",
    ]
    for row in macro:
        key = (row["model"], int(row["window_size_frames"]))
        fat = fatigue[key]
        lines.append(f"| {str(row['model']).upper()} | {row['window_size_frames']} | {float(row['mean']):.4f} +/- {float(row['standard_deviation']):.4f} | {float(fat['mean']):.4f} +/- {float(fat['standard_deviation']):.4f} |")
    lines += [
        "", "Os desvios acima medem variacao entre folds/sessoes, nao estabilidade entre seeds. A G2 de qualificacao nao confirma H3 nem autoriza selecao final: todas as configuracoes ainda foram avaliadas com uma unica seed.", "",
        "Matrizes de confusao, metricas por classe, historicos por epoca, distribuicoes de previsao, configuracoes resolvidas e hashes dos checkpoints acompanham este relatorio.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=3, required=True)
    parser.add_argument("--metrics-dir", default="fase_2/outputs/metrics/G2")
    parser.add_argument("--model-dir", default="fase_2/outputs/models/G2")
    parser.add_argument("--log-dir", default="fase_2/outputs/logs/G2")
    parser.add_argument("--prediction-dir", default="fase_2/outputs/predictions/G2")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/G2/combined")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    metrics_dir = repository_path(args.metrics_dir)
    model_dir = repository_path(args.model_dir)
    log_dir = repository_path(args.log_dir)
    prediction_dir = repository_path(args.prediction_dir)
    figure_dir = repository_path(args.figure_dir)
    configs = [load_yaml(repository_path(path)) for path in args.configs]
    prefixes = [str(config["configuration_id"]) for config in configs]

    summary = [row for prefix in prefixes for row in _read_csv(metrics_dir / f"{prefix}__runs.csv")]
    history = [row for prefix in prefixes for row in _read_csv(metrics_dir / f"{prefix}__history.csv")]
    per_class = [row for prefix in prefixes for row in _read_csv(metrics_dir / f"{prefix}__per_class.csv")]
    confusion = [row for prefix in prefixes for row in _read_csv(metrics_dir / f"{prefix}__confusion.csv")]
    validate_matrix(summary)

    runs = execution_table(summary)
    folds = fold_statistics(summary)
    classes = per_class_statistics(per_class)
    distribution = prediction_distribution(confusion)
    manifest = artifact_manifest(runs, model_dir, log_dir, prediction_dir, root)
    fingerprints = {
        str(row["run_id"]): json.loads((log_dir / f"{row['run_id']}.json").read_text(encoding="utf-8"))["fingerprint"]
        for row in runs
    }
    resolved = {
        "generation": "G2", "qualification_seed": 42, "official_run_count": 36,
        "balancing": "none", "run_fingerprints": fingerprints, "configs": [
            {"path": _relative(repository_path(path), root), "resolved": config}
            for path, config in zip(args.configs, configs, strict=True)
        ],
    }

    outputs = {
        "g2_runs.csv": summary, "g2_execution_table.csv": runs,
        "g2_history.csv": history, "g2_per_class.csv": per_class,
        "g2_per_class_statistics.csv": classes, "g2_confusion.csv": confusion,
        "g2_fold_statistics.csv": folds, "g2_prediction_distribution.csv": distribution,
        "g2_artifact_manifest.csv": manifest,
    }
    for name, rows in outputs.items():
        _write_csv(metrics_dir / name, rows)
    (metrics_dir / "g2_resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(metrics_dir / "g2_report.md", folds, classes)
    _plot_metric(folds, "macro_f1_all_classes", figure_dir / "macro_f1_by_model_window.svg")
    _plot_metric(
        classes,
        "f1",
        figure_dir / "fatigue_f1_by_model_window.svg",
        label="fatigue",
    )
    _plot_epochs(runs, figure_dir / "best_vs_stopping_epoch.svg")
    print(f"G2 consolidada: {len(runs)} runs oficiais em {metrics_dir}")


if __name__ == "__main__":
    main()
