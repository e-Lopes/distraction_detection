"""Treina LSTM e TCN em múltiplas seeds e analisa estabilidade entre execuções."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..models.temporal import trainable_parameters
from ..preprocessing.missingness import expand_behavior_labels
from .classical_baselines import (
    evaluate_predictions,
    experiment_fingerprint,
    save_figure,
)
from .dummy_baseline import CLASSES, load_series
from .stability import descriptive_statistics
from .temporal_data import build_sequence_fold, limit_sequence_split, representation_features
from .temporal_engine import class_weights, predict_split, sample_weights, save_run_result, train_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_COLORS = {"lstm": "#377eb8", "tcn": "#e41a1c", "transformer": "#4daf4a"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def device_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return metadata


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([CLASSES[int(value)] for value in values])


def aggregate_by_seed(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["model"]), str(row["subset"]), int(row["seed"]))].append(row)
    result = []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_all_classes")
    for (model, subset, seed), rows in sorted(grouped.items()):
        result.append(
            {
                "model": model,
                "subset": subset,
                "seed": seed,
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in metrics
                },
                "best_epoch": float(np.mean([float(row["best_epoch"]) for row in rows])),
                "stopping_epoch": float(
                    np.mean([float(row["stopping_epoch"]) for row in rows])
                ),
                "best_validation_macro_f1": float(
                    np.mean([float(row["best_validation_macro_f1"]) for row in rows])
                ),
                "training_seconds": float(
                    np.sum([float(row["training_seconds"]) for row in rows])
                ),
            }
        )
    return result


def aggregate_statistics(seed_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1_all_classes",
        "best_epoch",
        "stopping_epoch",
        "best_validation_macro_f1",
        "training_seconds",
    )
    for model in sorted({str(row["model"]) for row in seed_rows}):
        for subset in sorted({str(row["subset"]) for row in seed_rows}):
            selected = [
                row
                for row in seed_rows
                if row["model"] == model and row["subset"] == subset
            ]
            for metric in metrics:
                statistics = descriptive_statistics(
                    [float(row[metric]) for row in selected]
                )
                output.append(
                    {"model": model, "subset": subset, "metric": metric, **statistics}
                )
    return output


def fold_seed_statistics(
    summary_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    metrics = (
        "macro_f1_all_classes",
        "best_epoch",
        "stopping_epoch",
        "best_validation_macro_f1",
    )
    keys = sorted(
        {
            (str(row["model"]), str(row["subset"]), int(row["fold"]))
            for row in summary_rows
        }
    )
    for model, subset, fold in keys:
        selected = [
            row
            for row in summary_rows
            if row["model"] == model
            and row["subset"] == subset
            and int(row["fold"]) == fold
        ]
        for metric in metrics:
            output.append(
                {
                    "model": model,
                    "subset": subset,
                    "fold": fold,
                    "metric": metric,
                    **descriptive_statistics([float(row[metric]) for row in selected]),
                }
            )
    return output


def plot_metric_variability(seed_rows: list[dict[str, object]], output: Path) -> None:
    available = {str(row["subset"]) for row in seed_rows}
    subset = "test" if "test" in available else "validation"
    selected = [row for row in seed_rows if row["subset"] == subset]
    models = sorted({str(row["model"]) for row in selected})
    figure, axis = plt.subplots(figsize=(8, 5))
    values = [
        [
            float(row["macro_f1_all_classes"])
            for row in selected
            if row["model"] == model
        ]
        for model in models
    ]
    axis.boxplot(values, tick_labels=[model.upper() for model in models], showmeans=True)
    for index, model_values in enumerate(values, start=1):
        axis.scatter(
            np.full(len(model_values), index),
            model_values,
            color=MODEL_COLORS.get(models[index - 1], "#777777"),
            zorder=3,
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel(f"Macro F1 {subset} — média dos quatro folds")
    axis.set_title("Variabilidade entre seeds independentes")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def _curves_by_seed(
    histories: Sequence[dict[str, object]], model: str, metric: str
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    selected = [row for row in histories if row["model"] == model]
    result = {}
    for seed in sorted({int(row["seed"]) for row in selected}):
        seed_rows = [row for row in selected if int(row["seed"]) == seed]
        epochs = sorted({int(row["epoch"]) for row in seed_rows})
        means = [
            np.mean(
                [
                    float(row[metric])
                    for row in seed_rows
                    if int(row["epoch"]) == epoch
                ]
            )
            for epoch in epochs
        ]
        result[seed] = (np.asarray(epochs), np.asarray(means))
    return result


def plot_validation_curves(histories: list[dict[str, object]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in histories})
    figure, raw_axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, model in zip(axes, models):
        for seed, (epochs, values) in _curves_by_seed(
            histories, model, "validation_macro_f1"
        ).items():
            axis.plot(epochs, values, marker=".", label=str(seed), alpha=0.8)
        axis.set_title(model.upper())
        axis.set_xlabel("Época")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro F1 de validação — média dos folds ativos")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(title="Seed")
    figure.suptitle("Estabilidade das curvas de validação")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_loss_curves(histories: list[dict[str, object]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in histories})
    figure, raw_axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, model in zip(axes, models):
        for seed, (epochs, values) in _curves_by_seed(
            histories, model, "validation_loss"
        ).items():
            axis.plot(epochs, values, marker=".", label=str(seed), alpha=0.8)
        axis.set_title(model.upper())
        axis.set_xlabel("Época")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Loss de validação — média dos folds ativos")
    axes[-1].legend(title="Seed")
    figure.suptitle("Variabilidade da loss entre seeds")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_early_stopping(summary: list[dict[str, object]], output: Path) -> None:
    validation = [row for row in summary if row["subset"] == "validation"]
    models = sorted({str(row["model"]) for row in validation})
    figure, raw_axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, model in zip(axes, models):
        selected = [row for row in validation if row["model"] == model]
        for fold in sorted({int(row["fold"]) for row in selected}):
            rows = [row for row in selected if int(row["fold"]) == fold]
            axis.scatter(
                [int(row["best_epoch"]) for row in rows],
                [int(row["stopping_epoch"]) for row in rows],
                label=f"fold {fold}",
                alpha=0.8,
            )
        limit = max(int(row["stopping_epoch"]) for row in selected) + 1
        axis.plot((0, limit), (0, limit), linestyle="--", color="gray", alpha=0.5)
        axis.set_title(model.upper())
        axis.set_xlabel("Melhor época")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Época de parada")
    axes[-1].legend()
    figure.suptitle("Comportamento do early stopping entre seeds e folds")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_fold_seed_heatmap(summary: list[dict[str, object]], output: Path) -> None:
    available = {str(row["subset"]) for row in summary}
    subset = "test" if "test" in available else "validation"
    selected = [row for row in summary if row["subset"] == subset]
    models = sorted({str(row["model"]) for row in selected})
    seeds = sorted({int(row["seed"]) for row in selected})
    folds = sorted({int(row["fold"]) for row in selected})
    figure, raw_axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, model in zip(axes, models):
        matrix = np.asarray(
            [
                [
                    next(
                        float(row["macro_f1_all_classes"])
                        for row in selected
                        if row["model"] == model
                        and int(row["fold"]) == fold
                        and int(row["seed"]) == seed
                    )
                    for seed in seeds
                ]
                for fold in folds
            ]
        )
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        for row_index in range(len(folds)):
            for column_index in range(len(seeds)):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < 0.45 else "black",
                )
        axis.set_xticks(range(len(seeds)), seeds)
        axis.set_yticks(range(len(folds)), folds)
        axis.set_xlabel("Seed")
        axis.set_title(model.upper())
    axes[0].set_ylabel(f"Fold {subset}")
    figure.colorbar(image, ax=axes, label=f"Macro F1 {subset}")
    figure.suptitle("Variação conjunta entre fold e seed")
    figure.subplots_adjust(top=0.83, right=0.88, wspace=0.15)
    save_figure(figure, output)
    plt.close(figure)


def write_report(
    path: Path,
    statistics: Sequence[dict[str, object]],
    seeds: Sequence[int],
    folds: Sequence[int],
) -> None:
    lines = [
        "# Modelos temporais — estabilidade entre seeds",
        "",
        f"Seeds independentes: `{list(seeds)}`. Cada seed foi executada nos folds `{list(folds)}`, "
        "com early stopping independente. A unidade da estatística abaixo é a média dos folds "
        f"de cada seed (`n={len(seeds)}`), não as janelas individuais.",
        "",
        "| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        if row["metric"] not in {
            "macro_f1_all_classes",
            "balanced_accuracy",
            "best_epoch",
            "stopping_epoch",
        }:
            continue
        lines.append(
            f"| {row['model']} | {row['subset']} | {row['metric']} | "
            f"{float(row['mean']):.4f} ± {float(row['standard_deviation']):.4f} | "
            f"{float(row['median']):.4f} | {float(row['minimum']):.4f} | "
            f"{float(row['maximum']):.4f} | "
            f"[{float(row['ci95_lower']):.4f}, {float(row['ci95_upper']):.4f}] |"
        )
    lines.extend(
        [
            "",
            "O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. "
            "Nenhuma configuração ou conclusão usa apenas a melhor seed observada.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="fase_2/configs/experiment/temporal_multiseed.yaml",
    )
    parser.add_argument("--data-config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument("--checkpoint-dir", default="fase_2/outputs/models/temporal_multiseed")
    parser.add_argument("--run-dir", default="fase_2/outputs/logs/temporal_multiseed")
    parser.add_argument("--prediction-dir", default="fase_2/outputs/predictions/temporal_multiseed")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/temporal_multiseed")
    parser.add_argument("--augmentation-audit-dir", default="fase_2/outputs/metrics/G4/augmentation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fold", type=int, action="append", help="Filtra folds; repetivel")
    parser.add_argument("--seed", type=int, action="append", help="Filtra seeds; repetivel")
    parser.add_argument("--model", action="append", help="Filtra modelos; repetivel")
    parser.add_argument("--dry-run", action="store_true", help="Lista a matriz sem treinar")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=40,
        help="Recusa matrizes maiores sem aumento explicito deste limite",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_path = repository_path(args.experiment_config)
    data_path = repository_path(args.data_config)
    interval_path = repository_path(args.frame_intervals)
    split_path = repository_path(args.splits)
    input_dir = repository_path(args.input_dir)
    experiment = load_yaml(experiment_path)
    if bool(experiment.get("training", {}).get("deterministic", False)):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data_config = load_yaml(data_path)
    preprocessing_path = repository_path(experiment["preprocessing_config"])
    preprocessing = load_yaml(preprocessing_path)
    generation = str(experiment.get("generation", "unversioned"))
    configuration_id = str(experiment.get("configuration_id", experiment["experiment_name"]))
    representation = str(experiment.get("representation", "R2")).upper()
    features = representation_features(representation)
    configured_order = tuple(str(value) for value in experiment.get("feature_order", features))
    if configured_order != features:
        raise ValueError(
            f"feature_order divergente para {representation}: {configured_order} != {features}"
        )
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponível")
    device = torch.device(device_name)
    metadata = device_metadata(device)
    metadata["git_commit"] = git_commit()
    print(json.dumps(metadata, indent=2), flush=True)

    blocks = [
        SplitBlock(
            fold=int(row["fold"]),
            subset=row["subset"],
            video_id=row["video_id"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
        )
        for row in _read_csv(split_path)
    ]
    folds = sorted({block.fold for block in blocks})
    if args.fold:
        folds = [fold for fold in folds if fold in set(args.fold)]
    seeds = [int(seed) for seed in experiment["seeds"]]
    if args.seed:
        seeds = [seed for seed in seeds if seed in set(args.seed)]
    models = [str(model) for model in experiment["models"]]
    if args.model:
        models = [model for model in models if model in set(args.model)]
    if not folds or not seeds or not models:
        raise ValueError("Filtros produziram uma matriz de execucao vazia")
    evaluation_subsets = tuple(
        str(subset) for subset in experiment.get("evaluation_subsets", ("validation", "test"))
    )
    if "validation" not in evaluation_subsets or not set(evaluation_subsets).issubset(
        {"validation", "test"}
    ):
        raise ValueError(
            "evaluation_subsets deve conter validation e aceita somente validation/test"
        )
    size = int(experiment["window_size_frames"])
    planned_runs = [
        {
            "run_id": (
                f"{generation}__{configuration_id}__{model}__{representation.lower()}__"
                f"w{size}__fold_{fold}__seed_{seed}"
            ),
            "generation": generation,
            "configuration_id": configuration_id,
            "model": model,
            "representation": representation,
            "window_size_frames": size,
            "fold": fold,
            "seed": seed,
        }
        for fold in folds
        for model in models
        for seed in seeds
    ]
    print(json.dumps({"planned_run_count": len(planned_runs), "runs": planned_runs}, indent=2))
    if args.dry_run:
        return 0
    if len(planned_runs) > args.max_runs:
        raise ValueError(
            f"Matriz possui {len(planned_runs)} runs e excede --max-runs={args.max_runs}; "
            "revise a matriz ou aumente o limite conscientemente"
        )

    series = load_series(input_dir)
    labels = expand_behavior_labels(
        {video_id: len(rows) for video_id, rows in series.items()}, _read_csv(interval_path)
    )
    fingerprint = experiment_fingerprint(
        [
            experiment_path,
            data_path,
            preprocessing_path,
            interval_path,
            split_path,
            input_dir / "extraction_manifest.csv",
        ],
        {
            "mode": "temporal_multiseed",
            "generation": generation,
            "configuration_id": configuration_id,
            "representation": representation,
            "features": features,
        },
    )
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    checkpoint_root = repository_path(args.checkpoint_dir)
    run_root = repository_path(args.run_dir)
    prediction_root = repository_path(args.prediction_dir)
    for fold in folds:
        augmentation_records: list[dict[str, object]] = []
        balancing = str(experiment["training"].get("balancing", "none"))
        splits, scaler, medians = build_sequence_fold(
            series,
            labels,
            blocks,
            preprocessing,
            fold=fold,
            size_frames=size,
            stride_frames=int(data_config["windowing"]["stride_frames"]),
            minimum_proportion=float(
                data_config["windowing"]["minimum_target_proportion"]
            ),
            representation=representation,
            training_augmentation=(experiment.get("augmentation") if balancing == "augmentation" else None),
            augmentation_seed=int(experiment.get("augmentation_seed", 42)),
            augmentation_audit=augmentation_records,
        )
        if augmentation_records:
            _write_csv(
                repository_path(args.augmentation_audit_dir)
                / f"{configuration_id}__w{size}__fold_{fold}.csv",
                augmentation_records,
            )
        max_samples = experiment.get("max_samples_per_subset")
        if max_samples is not None:
            splits = {
                subset: limit_sequence_split(split, int(max_samples), seed=fold)
                for subset, split in splits.items()
            }
        for model_name in models:
            for seed in seeds:
                run_name = (
                    f"{generation}__{configuration_id}__{model_name}__"
                    f"{representation.lower()}__w{size}__fold_{fold}__seed_{seed}"
                )
                result_path = run_root / f"{run_name}.json"
                completed = None
                if not args.no_resume and result_path.exists():
                    candidate = json.loads(result_path.read_text(encoding="utf-8"))
                    if candidate.get("fingerprint") == fingerprint:
                        completed = candidate
                if completed is None:
                    result = train_model(
                        model_name=str(model_name),
                        model_parameters=experiment["parameters"][model_name],
                        splits=splits,
                        training=experiment["training"],
                        seed=seed,
                        fold=fold,
                        device=device,
                        checkpoint_dir=checkpoint_root / run_name,
                        fingerprint=fingerprint,
                        resume=not args.no_resume,
                        feature_names=features,
                    )
                    run_history = [
                        {
                            "run_id": run_name,
                            "generation": generation,
                            "configuration_id": configuration_id,
                            "representation": representation,
                            "window_size_frames": size,
                            "model": model_name,
                            "fold": fold,
                            "seed": seed,
                            **row,
                        }
                        for row in result.history
                    ]
                    run_summary = []
                    run_classes = []
                    run_confusion = []
                    run_predictions: dict[str, list[dict[str, object]]] = {}
                    best_history = result.history[result.best_epoch - 1]
                    for subset in evaluation_subsets:
                        inference_started = time.perf_counter()
                        expected_ids, predicted_ids, probabilities = predict_split(
                            result.model,
                            splits[subset],
                            batch_size=int(experiment["training"]["batch_size"]),
                            device=device,
                            amp=bool(experiment["training"]["amp"]),
                        )
                        inference_seconds = time.perf_counter() - inference_started
                        expected = _decode(expected_ids)
                        predicted = _decode(predicted_ids)
                        summary, per_class, confusion = evaluate_predictions(
                            model_name=str(model_name),
                            ablation=f"temporal_{representation.lower()}",
                            fold=fold,
                            subset=subset,
                            expected=expected,
                            predicted=predicted,
                            train_seconds=result.training_seconds,
                            resumed=result.resumed,
                        )
                        context = {
                            "run_id": run_name,
                            "generation": generation,
                            "configuration_id": configuration_id,
                            "representation": representation,
                            "window_size_frames": size,
                            "seed": seed,
                            "best_epoch": result.best_epoch,
                            "stopping_epoch": result.stopping_epoch,
                            "best_validation_macro_f1": result.best_validation_macro_f1,
                            "best_training_loss": best_history["training_loss"],
                            "best_validation_loss": best_history["validation_loss"],
                            "training_seconds": result.training_seconds,
                            "parameter_count": trainable_parameters(result.model),
                            "model_size_bytes": (
                                checkpoint_root / run_name / "best_macro_f1.pt"
                            ).stat().st_size,
                            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                            "inference_seconds": inference_seconds,
                            "inference_windows_per_second": (
                                len(expected_ids) / inference_seconds
                                if inference_seconds > 0
                                else float("nan")
                            ),
                        }
                        summary.update(context)
                        row_context = {
                            "run_id": run_name,
                            "generation": generation,
                            "configuration_id": configuration_id,
                            "representation": representation,
                            "window_size_frames": size,
                            "seed": seed,
                        }
                        for row in per_class:
                            row.update(row_context)
                        for row in confusion:
                            row.update(row_context)
                        run_summary.append(summary)
                        run_classes.extend(per_class)
                        run_confusion.extend(confusion)
                        run_predictions[subset] = [
                            {
                                "video_id": item.video_id,
                                "start_frame": item.start_frame,
                                "end_frame": item.end_frame,
                                "actual": actual,
                                "predicted": prediction,
                                "missing_ratio": item.missing_ratio,
                                "interpolated_ratio": item.interpolated_ratio,
                                "prob_alert": probability[0],
                                "prob_fatigue": probability[1],
                                "prob_distraction": probability[2],
                            }
                            for item, actual, prediction, probability in zip(
                                splits[subset].metadata, expected, predicted, probabilities
                            )
                        ]
                    completed = {
                        "fingerprint": fingerprint,
                        "completed": True,
                        "model": model_name,
                        "fold": fold,
                        "seed": seed,
                        "run_id": run_name,
                        "generation": generation,
                        "configuration_id": configuration_id,
                        "representation": representation,
                        "feature_names": list(features),
                        "window_size_frames": size,
                        "evaluation_subsets": list(evaluation_subsets),
                        "scaler": {"mean": scaler.mean, "scale": scaler.scale},
                        "training_medians": medians,
                        "balancing": balancing,
                        "loss": str(experiment["training"].get("loss", "cross_entropy")),
                        "train_class_counts": {
                            CLASSES[index]: int(np.sum(splits["train"].labels == index))
                            for index in range(len(CLASSES))
                        },
                        "class_weights": (
                            class_weights(splits["train"].labels).tolist()
                            if balancing == "class_weights" else None
                        ),
                        "focal_alpha": (
                            class_weights(splits["train"].labels).tolist()
                            if str(experiment["training"].get("loss", "cross_entropy"))
                            == "focal"
                            else None
                        ),
                        "sampling": (
                            {
                                "replacement": True,
                                "num_samples_per_epoch": len(splits["train"].labels),
                                "seed": seed,
                                "sample_weight_min": float(sample_weights(splits["train"].labels).min()),
                                "sample_weight_max": float(sample_weights(splits["train"].labels).max()),
                            } if balancing == "weighted_sampling" else None
                        ),
                        "augmentation": experiment.get("augmentation") if balancing == "augmentation" else None,
                        "augmentation_record_count": len(augmentation_records),
                        "history": run_history,
                        "summary": run_summary,
                        "per_class": run_classes,
                        "confusion": run_confusion,
                        "metadata": metadata,
                    }
                    save_run_result(result_path, completed)
                    for subset, rows in run_predictions.items():
                        _write_csv(prediction_root / f"{run_name}__{subset}.csv", rows)
                history_rows.extend(completed["history"])
                summary_rows.extend(completed["summary"])
                class_rows.extend(completed["per_class"])
                confusion_rows.extend(completed["confusion"])

    seed_rows = aggregate_by_seed(summary_rows)
    statistics = aggregate_statistics(seed_rows)
    per_fold_statistics = fold_seed_statistics(summary_rows)
    output_dir = repository_path(args.output_dir)
    prefix = configuration_id
    _write_csv(output_dir / f"{prefix}__runs.csv", summary_rows)
    _write_csv(output_dir / f"{prefix}__history.csv", history_rows)
    _write_csv(output_dir / f"{prefix}__per_class.csv", class_rows)
    _write_csv(output_dir / f"{prefix}__confusion.csv", confusion_rows)
    _write_csv(output_dir / f"{prefix}__seed_means.csv", seed_rows)
    _write_csv(output_dir / f"{prefix}__statistics.csv", statistics)
    _write_csv(output_dir / f"{prefix}__fold_statistics.csv", per_fold_statistics)
    write_report(output_dir / f"{prefix}.md", statistics, seeds, folds)
    figure_dir = repository_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_variability(seed_rows, figure_dir / "metric_variability.png")
    plot_validation_curves(history_rows, figure_dir / "validation_curves.png")
    plot_loss_curves(history_rows, figure_dir / "validation_loss_curves.png")
    plot_early_stopping(summary_rows, figure_dir / "early_stopping_stability.png")
    plot_fold_seed_heatmap(summary_rows, figure_dir / "fold_seed_heatmap.png")
    print(f"Experimento temporal multi-seed concluído: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
