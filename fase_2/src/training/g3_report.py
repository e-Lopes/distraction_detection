"""Consolida G3 com R0 reutilizado, deltas pareados e analise de Fatigue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from ..data.config import load_yaml, repository_path
from .classical_baselines import save_figure
from .dummy_baseline import CLASSES, load_series
from .g2_report import execution_table

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPRESENTATION_COLORS = {"R0": "#377eb8", "R1": "#e41a1c", "R2": "#4daf4a"}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _selected(row: Mapping[str, object]) -> bool:
    model = str(row["model"])
    window = int(row["window_size_frames"])
    return (model in {"lstm", "tcn"} and window == 60) or (
        model == "transformer" and window == 150
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def prediction_distribution(confusion: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(r["run_id"]), str(r["subset"]), str(r["predicted"])) for r in confusion})
    for run_id, subset, predicted in keys:
        selected = [r for r in confusion if r["run_id"] == run_id and r["subset"] == subset]
        count = sum(int(r["count"]) for r in selected if r["predicted"] == predicted)
        total = sum(int(r["count"]) for r in selected)
        base = selected[0]
        output.append(
            {
                "run_id": run_id,
                "model": base["model"],
                "representation": base["representation"],
                "window_size_frames": base["window_size_frames"],
                "fold": base["fold"],
                "seed": base["seed"],
                "subset": subset,
                "predicted_class": predicted,
                "count": count,
                "proportion": count / total,
            }
        )
    return output


def paired_deltas(
    summaries: Sequence[Mapping[str, object]],
    per_class: Sequence[Mapping[str, object]],
    distributions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    summary_lookup = {
        (str(r["model"]), int(r["fold"]), str(r["subset"]), str(r["representation"])): r
        for r in summaries
    }
    class_lookup = {
        (
            str(r["model"]), int(r["fold"]), str(r["subset"]),
            str(r["representation"]), str(r["label"]),
        ): r
        for r in per_class
    }
    prediction_lookup = {
        (
            str(r["model"]), int(r["fold"]), str(r["subset"]),
            str(r["representation"]), str(r["predicted_class"]),
        ): r
        for r in distributions
    }
    output = []
    comparisons = (("R1", "R0"), ("R2", "R0"), ("R2", "R1"))
    for model in ("lstm", "tcn", "transformer"):
        for fold in (1, 2, 3, 4):
            for subset in ("validation", "test"):
                for left, right in comparisons:
                    base_key = (model, fold, subset)
                    left_summary = summary_lookup[base_key + (left,)]
                    right_summary = summary_lookup[base_key + (right,)]
                    values = {
                        "macro_f1_all_classes": float(left_summary["macro_f1_all_classes"])
                        - float(right_summary["macro_f1_all_classes"]),
                        "balanced_accuracy": float(left_summary["balanced_accuracy"])
                        - float(right_summary["balanced_accuracy"]),
                        "fatigue_f1": float(class_lookup[base_key + (left, "fatigue")]["f1"] or 0)
                        - float(class_lookup[base_key + (right, "fatigue")]["f1"] or 0),
                        "fatigue_recall": float(class_lookup[base_key + (left, "fatigue")]["recall"] or 0)
                        - float(class_lookup[base_key + (right, "fatigue")]["recall"] or 0),
                        "distraction_f1": float(class_lookup[base_key + (left, "distraction")]["f1"] or 0)
                        - float(class_lookup[base_key + (right, "distraction")]["f1"] or 0),
                        "fatigue_predictions": int(prediction_lookup[base_key + (left, "fatigue")]["count"])
                        - int(prediction_lookup[base_key + (right, "fatigue")]["count"]),
                    }
                    for metric, delta in values.items():
                        output.append(
                            {
                                "model": model,
                                "window_size_frames": int(left_summary["window_size_frames"]),
                                "fold": fold,
                                "subset": subset,
                                "comparison": f"{left}-{right}",
                                "metric": metric,
                                "delta": delta,
                            }
                        )
    return output


def delta_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(r["model"]), str(r["subset"]), str(r["comparison"]), str(r["metric"])) for r in rows})
    for model, subset, comparison, metric in keys:
        selected = [float(r["delta"]) for r in rows if (r["model"], r["subset"], r["comparison"], r["metric"]) == (model, subset, comparison, metric)]
        mean, std = _mean_std(selected)
        output.append(
            {
                "model": model,
                "subset": subset,
                "comparison": comparison,
                "metric": metric,
                "n_folds": len(selected),
                "mean_delta": mean,
                "standard_deviation": std,
                "minimum": min(selected),
                "maximum": max(selected),
            }
        )
    return output


def _prediction_rows(
    executions: Sequence[Mapping[str, object]],
    *,
    root: Path,
    series: Mapping[str, Sequence[dict[str, str]]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    output: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in executions:
        run_id = str(run["run_id"])
        generation = "G2" if run["representation"] == "R0" else "G3"
        for subset in ("validation", "test"):
            path = root / f"fase_2/outputs/predictions/{generation}/{run_id}__{subset}.csv"
            rows: list[dict[str, object]] = [dict(row) for row in _read_csv(path)]
            for row in rows:
                if "missing_ratio" not in row:
                    window = series[str(row["video_id"])][
                        int(row["start_frame"]) : int(row["end_frame"]) + 1
                    ]
                    row["missing_ratio"] = sum(item["face_detected"] != "1" for item in window) / len(window)
                    row["interpolated_ratio"] = 0.0
                row.update(
                    {
                        "run_id": run_id,
                        "model": run["model"],
                        "representation": run["representation"],
                        "window_size_frames": run["window_size_frames"],
                        "fold": run["fold"],
                        "seed": run["seed"],
                        "subset": subset,
                    }
                )
            output[(run_id, subset)] = rows
    return output


def fatigue_analysis(
    executions: Sequence[Mapping[str, object]],
    predictions: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    class_distribution: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    train_counts = {
        (int(r["fold"]), int(r["window_size_frames"])): int(r["num_windows"])
        for r in class_distribution
        if r["subset"] == "train" and r["label"] == "fatigue"
    }
    output = []
    for run in executions:
        for subset in ("validation", "test"):
            rows = list(predictions[(str(run["run_id"]), subset)])
            actual_fatigue = [r for r in rows if r["actual"] == "fatigue"]
            predicted_fatigue = [r for r in rows if r["predicted"] == "fatigue"]
            false_positive = [r for r in rows if r["predicted"] == "fatigue" and r["actual"] != "fatigue"]
            false_negative = [r for r in rows if r["actual"] == "fatigue" and r["predicted"] != "fatigue"]
            probabilities = np.asarray([float(r["prob_fatigue"]) for r in rows])
            actual_probabilities = np.asarray([float(r["prob_fatigue"]) for r in actual_fatigue])
            output.append(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "representation": run["representation"],
                    "window_size_frames": run["window_size_frames"],
                    "fold": run["fold"],
                    "seed": run["seed"],
                    "subset": subset,
                    "train_actual_fatigue_windows": train_counts[(int(run["fold"]), int(run["window_size_frames"]))],
                    "actual_fatigue_windows": len(actual_fatigue),
                    "predicted_fatigue_windows": len(predicted_fatigue),
                    "fatigue_true_positives": len(actual_fatigue) - len(false_negative),
                    "fatigue_false_positives": len(false_positive),
                    "fatigue_false_negatives": len(false_negative),
                    "mean_probability_fatigue_all": float(probabilities.mean()),
                    "p95_probability_fatigue_all": float(np.quantile(probabilities, 0.95)),
                    "max_probability_fatigue_all": float(probabilities.max()),
                    "mean_probability_fatigue_actual": float(actual_probabilities.mean()) if len(actual_probabilities) else "",
                    "max_probability_fatigue_actual": float(actual_probabilities.max()) if len(actual_probabilities) else "",
                    "mean_missing_ratio_fatigue_actual": float(np.mean([float(r["missing_ratio"]) for r in actual_fatigue])) if actual_fatigue else "",
                    "mean_missing_ratio_fatigue_false_negative": float(np.mean([float(r["missing_ratio"]) for r in false_negative])) if false_negative else "",
                }
            )
    return output


def missingness_error(
    executions: Sequence[Mapping[str, object]],
    predictions: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    edges = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0000001)
    output = []
    for run in executions:
        rows = predictions[(str(run["run_id"]), "validation")]
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            selected = [r for r in rows if lower <= float(r["missing_ratio"]) < upper]
            if not selected:
                continue
            output.append(
                {
                    "model": run["model"],
                    "representation": run["representation"],
                    "window_size_frames": run["window_size_frames"],
                    "fold": run["fold"],
                    "missing_ratio_lower": lower,
                    "missing_ratio_upper": min(upper, 1.0),
                    "num_windows": len(selected),
                    "error_rate": sum(r["actual"] != r["predicted"] for r in selected) / len(selected),
                    "mean_probability_fatigue": float(np.mean([float(r["prob_fatigue"]) for r in selected])),
                    "fatigue_windows": sum(r["actual"] == "fatigue" for r in selected),
                }
            )
    return output


def artifact_manifest(executions: Sequence[Mapping[str, object]], root: Path) -> list[dict[str, object]]:
    output = []
    for run in executions:
        run_id = str(run["run_id"])
        generation = "G2" if run["representation"] == "R0" else "G3"
        paths = {
            "best_checkpoint": root / f"fase_2/outputs/models/{generation}/{run_id}/best_macro_f1.pt",
            "last_checkpoint": root / f"fase_2/outputs/models/{generation}/{run_id}/last.pt",
            "run_log": root / f"fase_2/outputs/logs/{generation}/{run_id}.json",
            "validation_predictions": root / f"fase_2/outputs/predictions/{generation}/{run_id}__validation.csv",
            "test_predictions": root / f"fase_2/outputs/predictions/{generation}/{run_id}__test.csv",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            output.append(
                {
                    "run_id": run_id,
                    "source_generation": generation,
                    "artifact_type": kind,
                    "relative_path": _relative(path, root),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return output


def select_representation(fold_stats: Sequence[Mapping[str, object]]) -> tuple[str, list[dict[str, object]]]:
    candidates = []
    for representation in ("R0", "R1", "R2"):
        selected = [r for r in fold_stats if r["subset"] == "validation" and r["metric"] == "macro_f1_all_classes" and r.get("representation") == representation]
        model_means = [float(r["mean"]) for r in selected]
        candidates.append(
            {
                "representation": representation,
                "mean_macro_f1_across_models": float(np.mean(model_means)),
                "mean_fold_standard_deviation": float(np.mean([float(r["standard_deviation"]) for r in selected])),
            }
        )
    candidates.sort(key=lambda row: (-float(row["mean_macro_f1_across_models"]), float(row["mean_fold_standard_deviation"]), str(row["representation"])))
    return str(candidates[0]["representation"]), candidates


def representation_fold_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_all_classes")
    keys = sorted({(str(r["model"]), str(r["representation"]), int(r["window_size_frames"]), str(r["subset"])) for r in rows})
    for model, representation, window, subset in keys:
        selected = [r for r in rows if (r["model"], r["representation"], int(r["window_size_frames"]), r["subset"]) == (model, representation, window, subset)]
        for metric in metrics:
            values = [float(r[metric]) for r in selected]
            mean, std = _mean_std(values)
            output.append({"model": model, "representation": representation, "window_size_frames": window, "subset": subset, "metric": metric, "n_folds": len(values), "mean": mean, "standard_deviation": std, "minimum": min(values), "maximum": max(values)})
    return output


def representation_class_statistics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(r["model"]), str(r["representation"]), int(r["window_size_frames"]), str(r["subset"]), str(r["label"])) for r in rows})
    for model, representation, window, subset, label in keys:
        selected = [r for r in rows if (r["model"], r["representation"], int(r["window_size_frames"]), r["subset"], r["label"]) == (model, representation, window, subset, label)]
        for metric in ("precision", "recall", "f1"):
            values = [float(r[metric]) for r in selected if r[metric] != ""]
            mean, std = _mean_std(values) if values else (float("nan"), float("nan"))
            output.append({"model":model,"representation":representation,"window_size_frames":window,"subset":subset,"label":label,"metric":metric,"n_folds_with_support":len(values),"mean":mean if values else "","standard_deviation":std if values else "","minimum":min(values) if values else "","maximum":max(values) if values else ""})
    return output


def _plot_macro(stats: Sequence[Mapping[str, object]], path: Path) -> None:
    models = ("lstm", "tcn", "transformer")
    reps = ("R0", "R1", "R2")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, rep in zip((-0.22, 0, 0.22), reps, strict=True):
        current = [next(r for r in stats if r["model"] == model and r["representation"] == rep and r["subset"] == "validation" and r["metric"] == "macro_f1_all_classes") for model in models]
        ax.bar(x + offset, [float(r["mean"]) for r in current], 0.2, yerr=[float(r["standard_deviation"]) for r in current], capsize=3, label=rep, color=REPRESENTATION_COLORS[rep])
    ax.set_xticks(x, [m.upper() for m in models]); ax.set_ylabel("Macro F1"); ax.set_title("G3: representacoes na validacao (media +/- DP entre folds)"); ax.legend(); ax.grid(axis="y", alpha=.25)
    save_figure(fig, path)


def _plot_fold(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ("lstm", "tcn", "transformer"), strict=True):
        for rep in ("R0", "R1", "R2"):
            current = sorted([r for r in rows if r["model"] == model and r["representation"] == rep and r["subset"] == "validation"], key=lambda r: int(r["fold"]))
            ax.plot([int(r["fold"]) for r in current], [float(r["macro_f1_all_classes"]) for r in current], marker="o", label=rep, color=REPRESENTATION_COLORS[rep])
        ax.set_title(model.upper()); ax.set_xlabel("Fold"); ax.set_xticks((1,2,3,4)); ax.grid(alpha=.25)
    axes[0].set_ylabel("Macro F1"); axes[-1].legend(); save_figure(fig, path)


def _plot_class_f1(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ("lstm", "tcn", "transformer"), strict=True):
        x = np.arange(3)
        for offset, rep in zip((-0.22, 0, 0.22), ("R0", "R1", "R2"), strict=True):
            values = []
            for label in CLASSES:
                current = [float(r["f1"] or 0) for r in rows if r["model"] == model and r["representation"] == rep and r["subset"] == "validation" and r["label"] == label]
                values.append(float(np.mean(current)))
            ax.bar(x + offset, values, .2, label=rep, color=REPRESENTATION_COLORS[rep])
        ax.set_xticks(x, CLASSES, rotation=25, ha="right"); ax.set_title(model.upper()); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("F1 medio entre folds"); axes[-1].legend(); save_figure(fig, path)


def _plot_prediction_distribution(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ("lstm", "tcn", "transformer"), strict=True):
        x = np.arange(3)
        for offset, rep in zip((-0.22, 0, .22), ("R0", "R1", "R2"), strict=True):
            values = []
            for label in CLASSES:
                current = [float(r["proportion"]) for r in rows if r["model"] == model and r["representation"] == rep and r["subset"] == "validation" and r["predicted_class"] == label]
                values.append(float(np.mean(current)))
            ax.bar(x + offset, values, .2, label=rep, color=REPRESENTATION_COLORS[rep])
        ax.set_xticks(x, CLASSES, rotation=25, ha="right"); ax.set_title(model.upper()); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Proporcao media predita"); axes[-1].legend(); save_figure(fig, path)


def _plot_fold_variability(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    labels=[]; values=[]
    for model in ("lstm", "tcn", "transformer"):
        for rep in ("R0", "R1", "R2"):
            labels.append(f"{model.upper()}\n{rep}")
            values.append([float(r["macro_f1_all_classes"]) for r in rows if r["model"] == model and r["representation"] == rep and r["subset"] == "validation"])
    fig,ax=plt.subplots(figsize=(11,5)); ax.boxplot(values,tick_labels=labels); ax.set_ylabel("Macro F1"); ax.set_title("G3: variabilidade entre os quatro folds"); ax.grid(axis="y",alpha=.25); save_figure(fig,path)


def _plot_deltas(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    selected = [r for r in rows if r["subset"] == "validation" and r["metric"] == "macro_f1_all_classes"]
    labels = [f"{m.upper()}\n{c}" for m in ("lstm", "tcn", "transformer") for c in ("R1-R0", "R2-R0", "R2-R1")]
    values = [next(float(r["mean_delta"]) for r in selected if r["model"] == m and r["comparison"] == c) for m in ("lstm", "tcn", "transformer") for c in ("R1-R0", "R2-R0", "R2-R1")]
    fig, ax = plt.subplots(figsize=(11, 5)); ax.bar(np.arange(len(values)), values, color=["#e41a1c" if v < 0 else "#4daf4a" for v in values]); ax.axhline(0, color="black", linewidth=1); ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right"); ax.set_ylabel("Delta de Macro F1"); ax.set_title("G3: deltas pareados medios entre folds"); ax.grid(axis="y", alpha=.25); save_figure(fig, path)


def _plot_fatigue_prob(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    selected = [r for r in rows if r["subset"] == "validation"]
    labels, values = [], []
    for model in ("lstm", "tcn", "transformer"):
        for rep in ("R0", "R1", "R2"):
            current = [float(r["mean_probability_fatigue_actual"]) for r in selected if r["model"] == model and r["representation"] == rep and r["mean_probability_fatigue_actual"] != ""]
            labels.append(f"{model.upper()}\n{rep}"); values.append(current)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.boxplot(values, tick_labels=labels); ax.axhline(1/3, linestyle="--", color="gray", label="1/3 (referencia)"); ax.set_ylabel("Probabilidade de Fatigue nas janelas reais"); ax.set_title("G3: proximidade da decisao para Fatigue"); ax.legend(); ax.grid(axis="y", alpha=.25); save_figure(fig, path)


def _plot_missingness(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for rep in ("R0", "R1", "R2"):
        grouped = defaultdict(list)
        for row in rows:
            if row["representation"] == rep:
                grouped[(float(row["missing_ratio_lower"]), float(row["missing_ratio_upper"]))].append(float(row["error_rate"]))
        x = [(a+b)/2 for a,b in sorted(grouped)]; y = [float(np.mean(grouped[key])) for key in sorted(grouped)]
        ax.plot(x, y, marker="o", label=rep, color=REPRESENTATION_COLORS[rep])
    ax.set_xlabel("Missing ratio (centro do intervalo)"); ax.set_ylabel("Taxa de erro"); ax.set_title("G3: missingness e erro de validacao"); ax.legend(); ax.grid(alpha=.25); save_figure(fig, path)


def _plot_curves(history: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, ("lstm", "tcn", "transformer"), strict=True):
        for rep in ("R0", "R1", "R2"):
            selected = [r for r in history if r["model"] == model and r["representation"] == rep]
            epochs = sorted({int(r["epoch"]) for r in selected})
            means = [float(np.mean([float(r["validation_macro_f1"]) for r in selected if int(r["epoch"]) == epoch])) for epoch in epochs]
            ax.plot(epochs, means, label=rep, color=REPRESENTATION_COLORS[rep])
        ax.set_title(model.upper()); ax.set_xlabel("Epoca"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Macro F1 de validacao"); axes[-1].legend(); save_figure(fig, path)


def _plot_confusions(confusion: Sequence[Mapping[str, object]], figure_dir: Path) -> None:
    for model in ("lstm", "tcn", "transformer"):
        for rep in ("R0", "R1", "R2"):
            matrix = np.zeros((3,3), dtype=int)
            for row in confusion:
                if row["model"] == model and row["representation"] == rep and row["subset"] == "validation":
                    matrix[CLASSES.index(str(row["actual"])), CLASSES.index(str(row["predicted"]))] += int(row["count"])
            fig, ax = plt.subplots(figsize=(5,4)); image=ax.imshow(matrix,cmap="Blues");
            for i in range(3):
                for j in range(3): ax.text(j,i,str(matrix[i,j]),ha="center",va="center")
            ax.set_xticks(range(3),CLASSES,rotation=30,ha="right"); ax.set_yticks(range(3),CLASSES); ax.set_xlabel("Predita"); ax.set_ylabel("Real"); ax.set_title(f"{model.upper()} / {rep} — validacao"); fig.colorbar(image,ax=ax); save_figure(fig, figure_dir / f"confusion_{model}_{rep.lower()}.svg")


def write_report(
    path: Path,
    stats: Sequence[Mapping[str, object]],
    fatigue: Sequence[Mapping[str, object]],
    delta_stats: Sequence[Mapping[str, object]],
    selection: str,
    ranking: Sequence[Mapping[str, object]],
    executions: Sequence[Mapping[str, object]],
) -> None:
    macro = [r for r in stats if r["subset"] == "validation" and r["metric"] == "macro_f1_all_classes"]
    lines = [
        "# G3 — missingness e representacao temporal", "",
        "## 1. Objetivo e hipotese", "",
        "Avaliar se interpolacao de gaps curtos (R1) e flags explicitas de missingness (R2) melhoram R0, mantendo modelos, folds, seed, distribuicao, orcamento e early stopping constantes.", "",
        "## 2. Relacao com G1 e G2", "",
        "G1 manteve SVM/60 como melhor qualificacao (0.4191). G3 reutiliza sem retreino os 12 runs R0 finalistas da G2 e adiciona 24 runs R1/R2.", "",
        "## 3. Configuracoes congeladas", "",
        "TCN/60, LSTM/60 e Transformer/150; seed 42; quatro folds; batch 1024; ate 150 epocas; patience 15; minimum delta 0.002; sem balanceamento. R1/R2 usam interpolacao linear offline de gaps internos de ate 15 frames e mediana do treino para gaps restantes.", "",
        "## 4–7. Resultados reutilizados, novos, por fold e agregados", "",
        "| Modelo | Representacao | Janela | Macro F1 validacao (media +/- DP) |", "|---|---|---:|---:|",
    ]
    for row in sorted(macro, key=lambda r: (str(r["model"]), str(r["representation"]))):
        lines.append(f"| {str(row['model']).upper()} | {row['representation']} | {row['window_size_frames']} | {float(row['mean']):.4f} +/- {float(row['standard_deviation']):.4f} |")
    lines += ["", "Os resultados completos por fold estao em `g3_runs.csv`; os desvios representam folds com seed 42, nao variabilidade entre seeds.", "", "## 8. Deltas pareados", "", "| Modelo | Comparacao | Delta medio de Macro F1 |", "|---|---|---:|"]
    for row in delta_stats:
        if row["subset"] == "validation" and row["metric"] == "macro_f1_all_classes":
            lines.append(f"| {str(row['model']).upper()} | {row['comparison']} | {float(row['mean_delta']):+.4f} |")
    val_fatigue = [r for r in fatigue if r["subset"] == "validation"]
    lines += ["", "## 9–10. Metricas por classe e Fatigue", "", "Nenhuma representacao produziu deteccao util de Fatigue: F1 e recall permaneceram zero. As probabilidades, falsos positivos, falsos negativos e contagens reais estao em `g3_fatigue_analysis.csv`.", "", "| Modelo | Representacao | Predicoes Fatigue (4 folds) | Prob. media nas janelas reais |", "|---|---|---:|---:|"]
    for model in ("lstm", "tcn", "transformer"):
        for rep in ("R0", "R1", "R2"):
            current=[r for r in val_fatigue if r["model"]==model and r["representation"]==rep]; probs=[float(r["mean_probability_fatigue_actual"]) for r in current if r["mean_probability_fatigue_actual"]!=""]
            lines.append(f"| {model.upper()} | {rep} | {sum(int(r['predicted_fatigue_windows']) for r in current)} | {float(np.mean(probs)):.4f} |")
    lstm_r2 = [r for r in val_fatigue if r["model"] == "lstm" and r["representation"] == "R2"]
    transformer_r2 = [r for r in val_fatigue if r["model"] == "transformer" and r["representation"] == "R2"]
    maximum_fatigue_probability = max(float(r["max_probability_fatigue_actual"]) for r in val_fatigue if r["max_probability_fatigue_actual"] != "")
    lines += [
        "",
        f"Havia {sum(int(r['actual_fatigue_windows']) for r in lstm_r2)} janelas reais de Fatigue na validacao de w60 e {sum(int(r['actual_fatigue_windows']) for r in transformer_r2)} em w150. Todas foram falsos negativos e nao houve falsos positivos. A maior probabilidade de Fatigue observada em uma janela real foi {maximum_fatigue_probability:.4f}, ainda abaixo da referencia de 1/3 e sem alteracao de threshold.",
        "",
        "As janelas de Fatigue dos blocos de validacao tinham missing ratio baixo (em geral zero ou proximo de zero). Assim, as flags de R2 nao forneceram um sinal forte para recuperar Fatigue nesses blocos; seus eventuais ganhos podem estar ligados sobretudo a outras classes e a correlacoes de missingness.",
    ]
    total_seconds=sum(float(r["training_seconds"]) for r in executions if r["representation"]!="R0")
    resumed=sum(str(r["resumed_checkpoint"]).lower()=="true" for r in executions if r["representation"]!="R0")
    lines += [
        "", "## 11–14. Confusao, previsoes, probabilidades e missingness", "",
        "As matrizes, distribuicoes, probabilidades de Fatigue e relacao entre missing rate e erro foram preservadas em CSV e SVG. Como missingness e classe sao associados, qualquer ganho de R2 deve ser interpretado como correlacao potencial, nao como evidencia causal de comportamento facial.", "",
        "## 15. Tempo e recursos", "", f"Os 24 novos runs consumiram {total_seconds:.1f} segundos de treino registrados e usaram CUDA/AMP na RTX 2060 SUPER. Tempos de inferencia, throughput, tamanho e pico de memoria constam em `g3_execution_table.csv`.", "",
        "## 16. Falhas e retomadas", "", f"Todos os 24 novos runs terminaram; {resumed} foram marcados como retomados. Os dois smokes permanecem separados e nenhuma falha foi incorporada.", "",
        "## 17. Limitacoes", "", "A G3 usa uma unica seed. R1/R2 com interpolacao linear sao offline. O kernel eficiente de atencao CUDA nao e deterministico bit a bit. Ha apenas quatro sessoes e poucos exemplos de Fatigue.", "",
        "## 18. Decisao para G4", "", f"A representacao recomendada e **{selection}**, selecionada somente pela media de Macro F1 de validacao e favorecida tambem pela menor complexidade. Ranking agregado: " + ", ".join(f"{r['representation']}={float(r['mean_macro_f1_across_models']):.4f}" for r in ranking) + ".", "",
        "R1 piorou de forma consistente. R2 recuperou parte da perda e melhorou ligeiramente o Transformer, mas nao superou R0 no conjunto dos modelos nem resolveu Fatigue. G4 ainda nao foi iniciada. H3 continua em aberto; estes resultados de uma seed nao sao finais.",
    ]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--metrics-dir",default="fase_2/outputs/metrics/G3"); parser.add_argument("--figure-dir",default="fase_2/outputs/figures/G3/combined"); args=parser.parse_args(argv)
    root=Path.cwd().resolve(); metrics=repository_path(args.metrics_dir); figures=repository_path(args.figure_dir)
    g2=repository_path("fase_2/outputs/metrics/G2")
    g3_prefixes=("qualification_r1_w60_b1024","qualification_r2_w60_b1024","qualification_r1_w150_b1024","qualification_r2_w150_b1024")
    summary=[r for r in _read_csv(g2/"g2_runs.csv") if _selected(r)] + [r for p in g3_prefixes for r in _read_csv(metrics/f"{p}__runs.csv")]
    history=[r for r in _read_csv(g2/"g2_history.csv") if _selected(r)] + [r for p in g3_prefixes for r in _read_csv(metrics/f"{p}__history.csv")]
    classes=[r for r in _read_csv(g2/"g2_per_class.csv") if _selected(r)] + [r for p in g3_prefixes for r in _read_csv(metrics/f"{p}__per_class.csv")]
    confusion=[r for r in _read_csv(g2/"g2_confusion.csv") if _selected(r)] + [r for p in g3_prefixes for r in _read_csv(metrics/f"{p}__confusion.csv")]
    runs=execution_table(summary)
    if len(runs)!=36 or sum(r["representation"]=="R0" for r in runs)!=12:
        raise ValueError("Matriz G3 consolidada deve conter 36 runs, sendo 12 R0")
    stats=representation_fold_statistics(summary); class_stats=representation_class_statistics(classes); distribution=prediction_distribution(confusion)
    predictions=_prediction_rows(runs,root=root,series=load_series(root/"fase_2/data/interim/legacy_extraction"))
    fatigue=fatigue_analysis(runs,predictions,_read_csv(root/"fase_2/outputs/metrics/split_window_distribution.csv")); missing=missingness_error(runs,predictions)
    deltas=paired_deltas(summary,classes,distribution); delta_stats=delta_statistics(deltas); artifacts=artifact_manifest(runs,root); selection,ranking=select_representation(stats)
    for name,rows in {"g3_execution_table.csv":runs,"g3_runs.csv":summary,"g3_history.csv":history,"g3_per_class.csv":classes,"g3_per_class_statistics.csv":class_stats,"g3_confusion.csv":confusion,"g3_fold_statistics.csv":stats,"g3_prediction_distribution.csv":distribution,"g3_paired_deltas.csv":deltas,"g3_paired_delta_statistics.csv":delta_stats,"g3_fatigue_analysis.csv":fatigue,"g3_missingness_error.csv":missing,"g3_artifact_manifest.csv":artifacts,"g3_representation_ranking.csv":ranking}.items(): _write_csv(metrics/name,rows)
    registry=load_yaml(root/"fase_2/configs/experiment/g3_representation.yaml"); fingerprints={str(r["run_id"]):json.loads((root/f"fase_2/outputs/logs/{'G2' if r['representation']=='R0' else 'G3'}/{r['run_id']}.json").read_text(encoding="utf-8"))["fingerprint"] for r in runs}
    (metrics/"g3_resolved_config.json").write_text(json.dumps({"registry":registry,"feature_order":{"R0":["ear","mar","pitch","yaw","roll"],"R1":["ear","mar","pitch","yaw","roll"],"R2":["ear","mar","pitch","yaw","roll","face_detected","was_interpolated","missing_duration_so_far"]},"run_fingerprints":fingerprints,"selected_for_g4":selection},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    write_report(metrics/"g3_report.md",stats,fatigue,delta_stats,selection,ranking,runs)
    figures.mkdir(parents=True,exist_ok=True); _plot_macro(stats,figures/"macro_f1_representations.svg"); _plot_fold(summary,figures/"macro_f1_by_fold.svg"); _plot_class_f1(classes,figures/"f1_by_class.svg"); _plot_prediction_distribution(distribution,figures/"prediction_distribution.svg"); _plot_fold_variability(summary,figures/"fold_variability.svg"); _plot_deltas(delta_stats,figures/"paired_deltas.svg"); _plot_fatigue_prob(fatigue,figures/"fatigue_probabilities.svg"); _plot_missingness(missing,figures/"missingness_vs_error.svg"); _plot_curves(history,figures/"training_curves.svg"); _plot_confusions(confusion,figures)
    print(f"G3 consolidada: {len(runs)} runs comparados; representacao para G4={selection}"); return 0


if __name__=="__main__": raise SystemExit(main())
