"""G1: regras fixas e classificadores classicos sobre janelas R0 achatadas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from .classical_baselines import (
    build_model,
    evaluate_predictions,
    experiment_fingerprint,
    fit_model,
    load_checkpoint,
    predict_model,
    save_checkpoint,
    save_figure,
)
from .dummy_baseline import CLASSES, load_series
from .temporal_data import SequenceScaler, SequenceSplit, build_sequence_fold, limit_sequence_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "fixed_rules": "#984ea3",
    "svm": "#377eb8",
    "random_forest": "#4daf4a",
    "xgboost": "#e41a1c",
}


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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Artefato fora do repositorio: {path}") from error


def aggregate_run_table(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_run: dict[str, dict[str, object]] = {}
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1_present_classes",
        "macro_f1_all_classes",
    )
    for row in rows:
        run_id = str(row["run_id"])
        current = by_run.setdefault(
            run_id,
            {
                key: row[key]
                for key in (
                    "run_id",
                    "generation",
                    "model",
                    "representation",
                    "window_size_frames",
                    "fold",
                    "seed",
                    "balancing",
                    "train_seconds",
                    "resumed_checkpoint",
                )
            },
        )
        subset = str(row["subset"])
        current[f"{subset}_num_windows"] = row["num_windows"]
        for metric in metric_names:
            current[f"{subset}_{metric}"] = row[metric]
    return [by_run[key] for key in sorted(by_run)]


def aggregate_fold_statistics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_all_classes")
    keys = sorted(
        {
            (str(row["model"]), int(row["window_size_frames"]), str(row["subset"]))
            for row in rows
        }
    )
    for model, window, subset in keys:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and int(row["window_size_frames"]) == window
            and row["subset"] == subset
        ]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            output.append(
                {
                    "model": model,
                    "window_size_frames": window,
                    "subset": subset,
                    "metric": metric,
                    "n_folds": len(values),
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return output


def aggregate_per_class_statistics(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = sorted(
        {
            (
                str(row["model"]),
                int(row["window_size_frames"]),
                str(row["subset"]),
                str(row["label"]),
            )
            for row in rows
        }
    )
    for model, window, subset, label in keys:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and int(row["window_size_frames"]) == window
            and row["subset"] == subset
            and row["label"] == label
        ]
        for metric in ("precision", "recall", "f1"):
            values = np.asarray(
                [float(row[metric]) for row in selected if row[metric] != ""], dtype=float
            )
            output.append(
                {
                    "model": model,
                    "window_size_frames": window,
                    "subset": subset,
                    "label": label,
                    "metric": metric,
                    "n_folds_with_support": len(values),
                    "mean": float(values.mean()) if len(values) else "",
                    "standard_deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else ""
                    ),
                }
            )
    return output


def prediction_distribution(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], int] = {}
    totals: dict[tuple[object, ...], int] = {}
    for row in rows:
        key = (
            row["model"],
            int(row["window_size_frames"]),
            int(row["fold"]),
            row["subset"],
            row["predicted"],
        )
        total_key = key[:-1]
        grouped[key] = grouped.get(key, 0) + int(row["count"])
        totals[total_key] = totals.get(total_key, 0) + int(row["count"])
    return [
        {
            "model": key[0],
            "window_size_frames": key[1],
            "fold": key[2],
            "subset": key[3],
            "predicted_class": key[4],
            "count": count,
            "proportion": count / totals[key[:-1]],
        }
        for key, count in sorted(grouped.items())
    ]


def artifact_manifest(
    run_rows: Sequence[Mapping[str, object]],
    checkpoint_root: Path,
    *,
    experiment_path: Path,
    fingerprint: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in run_rows:
        if row["model"] == "fixed_rules":
            output.append(
                {
                    "run_id": row["run_id"],
                    "artifact_type": "resolved_fixed_rules_config",
                    "relative_path": _relative(experiment_path),
                    "size_bytes": experiment_path.stat().st_size,
                    "sha256": _sha256(experiment_path),
                    "fingerprint": fingerprint,
                }
            )
            continue
        checkpoint = checkpoint_root / f"{row['run_id']}.joblib"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint G1 ausente: {checkpoint}")
        output.append(
            {
                "run_id": row["run_id"],
                "artifact_type": "joblib_checkpoint",
                "relative_path": _relative(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
                "fingerprint": fingerprint,
            }
        )
    return output


def predict_fixed_rule_window(
    values: np.ndarray,
    *,
    ear_threshold: float,
    ear_consecutive_frames: int,
    mar_threshold: float,
    pitch_threshold: float,
    tie_priority: Sequence[str],
) -> str:
    """Aplica as regras por frame e usa maioria; o streak reinicia em cada janela."""

    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Regras G1 exigem uma janela R0 N x 5")
    if not np.isfinite(values).all():
        raise ValueError("Regras G1 receberam NaN ou Inf")
    if set(tie_priority) != set(CLASSES):
        raise ValueError("tie_priority deve conter exatamente as tres classes")
    states: list[str] = []
    ear_streak = 0
    for ear, mar, pitch, _yaw, _roll in values:
        ear_streak = ear_streak + 1 if ear < ear_threshold else 0
        if ear_streak >= ear_consecutive_frames or mar > mar_threshold:
            state = "fatigue"
        elif pitch < pitch_threshold:
            state = "distraction"
        else:
            state = "alert"
        states.append(state)
    counts = Counter(states)
    maximum = max(counts.values())
    return next(label for label in tie_priority if counts[label] == maximum)


def predict_fixed_rules(
    split: SequenceSplit,
    scaler: SequenceScaler,
    rule_config: Mapping[str, object],
) -> np.ndarray:
    mean = np.asarray(scaler.mean, dtype=float)
    scale = np.asarray(scaler.scale, dtype=float)
    raw = split.values.astype(float) * scale + mean
    return np.asarray(
        [
            predict_fixed_rule_window(
                window,
                ear_threshold=float(rule_config["ear_threshold"]),
                ear_consecutive_frames=int(rule_config["ear_consecutive_frames"]),
                mar_threshold=float(rule_config["mar_threshold"]),
                pitch_threshold=float(rule_config["pitch_threshold"]),
                tie_priority=tuple(str(value) for value in rule_config["tie_priority"]),
            )
            for window in raw
        ]
    )


def flatten_r0(split: SequenceSplit) -> tuple[np.ndarray, np.ndarray]:
    if split.values.ndim != 3 or split.values.shape[-1] != 5:
        raise ValueError("B2a exige sequencias R0 N x 5")
    matrix = split.values.reshape(len(split.labels), -1)
    if not np.isfinite(matrix).all():
        raise ValueError("R0 achatado contem NaN ou Inf")
    labels = np.asarray([CLASSES[int(label)] for label in split.labels])
    return matrix, labels


def make_run_id(
    generation: str,
    configuration: str,
    model: str,
    window: int,
    fold: int,
    seed: int,
    max_samples_per_subset: int | None = None,
) -> str:
    run_id = (
        f"{generation}__{configuration}__{model}__r0_flat__"
        f"w{window}__fold_{fold}__seed_{seed}"
    )
    return (
        f"{run_id}__sample_{max_samples_per_subset}"
        if max_samples_per_subset is not None
        else run_id
    )


def planned_runs(
    config: Mapping[str, object],
    *,
    windows: Sequence[int],
    folds: Sequence[int],
    models: Sequence[str],
    max_samples_per_subset: int | None = None,
) -> list[dict[str, object]]:
    generation = str(config["generation"])
    configuration = str(config["configuration_id"])
    seed = int(config["seed"])
    return [
        {
            "run_id": make_run_id(
                generation,
                configuration,
                model,
                window,
                fold,
                seed,
                max_samples_per_subset,
            ),
            "generation": generation,
            "configuration_id": configuration,
            "model": model,
            "representation": "R0_flat",
            "window_size_frames": window,
            "fold": fold,
            "seed": seed,
            "max_samples_per_subset": max_samples_per_subset,
        }
        for window in windows
        for fold in folds
        for model in models
    ]


def plot_macro_f1(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    test = [row for row in rows if row["subset"] == "test"]
    models = sorted({str(row["model"]) for row in test})
    windows = sorted({int(row["window_size_frames"]) for row in test})
    figure, axis = plt.subplots(figsize=(10, 5))
    for model in models:
        means = [
            np.mean(
                [
                    float(row["macro_f1_all_classes"])
                    for row in test
                    if row["model"] == model and int(row["window_size_frames"]) == window
                ]
            )
            for window in windows
        ]
        axis.plot(windows, means, marker="o", label=model, color=COLORS[model])
    axis.set_xticks(windows)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Janela (frames)")
    axis.set_ylabel("Macro F1 medio nos folds externos")
    axis.set_title("G1: regras fixas e classicos R0 achatado")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_fold_variability(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    test = [row for row in rows if row["subset"] == "test"]
    models = sorted({str(row["model"]) for row in test})
    values = [
        [float(row["macro_f1_all_classes"]) for row in test if row["model"] == model]
        for model in models
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(values, tick_labels=models, showmeans=True)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro F1 por fold/janela")
    axis.set_title("G1: variabilidade entre sessoes e janelas")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_validation_test(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in rows})
    windows = sorted({int(row["window_size_frames"]) for row in rows})
    figure, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5), sharey=True)
    for axis, window in zip(np.atleast_1d(axes), windows):
        x_values = np.arange(len(models))
        for offset, subset in ((-0.18, "validation"), (0.18, "test")):
            values = [
                np.mean(
                    [
                        float(row["macro_f1_all_classes"])
                        for row in rows
                        if row["model"] == model
                        and int(row["window_size_frames"]) == window
                        and row["subset"] == subset
                    ]
                )
                for model in models
            ]
            axis.bar(x_values + offset, values, 0.36, label=subset)
        axis.set_xticks(x_values, models, rotation=25, ha="right")
        axis.set_title(f"Janela {window}")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Macro F1 medio")
    np.atleast_1d(axes)[-1].legend()
    figure.suptitle("G1: validacao interna versus teste externo descritivo")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_per_class_f1(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    test = [row for row in rows if row["subset"] == "test" and row["f1"] != ""]
    models = sorted({str(row["model"]) for row in test})
    windows = sorted({int(row["window_size_frames"]) for row in test})
    figure, axes = plt.subplots(1, len(windows), figsize=(6 * len(windows), 5), sharey=True)
    x_values = np.arange(len(CLASSES))
    width = 0.8 / len(models)
    for axis, window in zip(np.atleast_1d(axes), windows):
        for index, model in enumerate(models):
            values = [
                np.mean(
                    [
                        float(row["f1"])
                        for row in test
                        if row["model"] == model
                        and int(row["window_size_frames"]) == window
                        and row["label"] == label
                    ]
                )
                if any(
                    row["model"] == model
                    and int(row["window_size_frames"]) == window
                    and row["label"] == label
                    for row in test
                )
                else 0.0
                for label in CLASSES
            ]
            axis.bar(
                x_values + (index - (len(models) - 1) / 2) * width,
                values,
                width,
                label=model,
                color=COLORS[model],
            )
        axis.set_xticks(x_values, CLASSES)
        axis.set_title(f"Janela {window}")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("F1 medio nos folds com suporte")
    np.atleast_1d(axes)[-1].legend()
    figure.suptitle("G1: desempenho por classe")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_confusion(
    rows: Sequence[Mapping[str, object]], model: str, window: int, output: Path
) -> None:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and int(row["window_size_frames"]) == window
        and row["subset"] == "test"
    ]
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for row in selected:
        matrix[CLASSES.index(str(row["actual"])), CLASSES.index(str(row["predicted"]))] += int(
            row["count"]
        )
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row_index in range(len(CLASSES)):
        for column_index in range(len(CLASSES)):
            axis.text(
                column_index,
                row_index,
                f"{normalized[row_index, column_index]:.2f}\n(n={int(matrix[row_index, column_index])})",
                ha="center",
                va="center",
                color="white" if normalized[row_index, column_index] > 0.55 else "black",
            )
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_title(f"G1 {model}, janela {window}")
    figure.colorbar(image, ax=axis, label="Proporcao por classe real")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def write_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# G1 - regras fixas e classicos sobre R0 achatado",
        "",
        "Esta geracao usa distribuicao original (`balancing: none`). Os parametros e thresholds "
        "sao congelados antes da avaliacao; o teste externo nao seleciona modelos ou janelas.",
        "",
        "| Modelo | Janela | Subset | Macro F1 medio | Balanced accuracy media |",
        "|---|---:|---|---:|---:|",
    ]
    for model in sorted({str(row["model"]) for row in rows}):
        for window in sorted({int(row["window_size_frames"]) for row in rows}):
            for subset in ("validation", "test"):
                selected = [
                    row
                    for row in rows
                    if row["model"] == model
                    and int(row["window_size_frames"]) == window
                    and row["subset"] == subset
                ]
                if selected:
                    lines.append(
                        f"| {model} | {window} | {subset} | "
                        f"{np.mean([float(row['macro_f1_all_classes']) for row in selected]):.4f} | "
                        f"{np.mean([float(row['balanced_accuracy']) for row in selected]):.4f} |"
                    )
    lines.extend(
        [
            "",
            "Resultados de teste sao descritivos. Qualquer escolha para G2 deve usar apenas "
            "treino e validacao.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config", default="fase_2/configs/experiment/g1_baselines.yaml"
    )
    parser.add_argument("--data-config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument("--checkpoint-dir", default="fase_2/outputs/models/G1")
    parser.add_argument("--prediction-dir", default="fase_2/outputs/predictions/G1")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics/G1")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/G1")
    parser.add_argument("--window", type=int, action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--max-samples-per-subset", type=int)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=48)
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
    data_config = load_yaml(data_path)
    preprocessing_path = repository_path(experiment["preprocessing_config"])
    preprocessing = load_yaml(preprocessing_path)
    blocks = [
        SplitBlock(
            int(row["fold"]),
            row["subset"],
            row["video_id"],
            int(row["start_frame"]),
            int(row["end_frame"]),
        )
        for row in _read_csv(split_path)
    ]
    windows = [int(value) for value in experiment["window_sizes_frames"]]
    folds = sorted({block.fold for block in blocks})
    models = [str(value) for value in experiment["models"]]
    if args.window:
        windows = [value for value in windows if value in set(args.window)]
    if args.fold:
        folds = [value for value in folds if value in set(args.fold)]
    if args.model:
        models = [value for value in models if value in set(args.model)]
    allowed = {"fixed_rules", "svm", "random_forest", "xgboost"}
    if not windows or not folds or not models or not set(models) <= allowed:
        raise ValueError("Matriz G1 vazia ou com modelo desconhecido")
    matrix = planned_runs(
        experiment,
        windows=windows,
        folds=folds,
        models=models,
        max_samples_per_subset=args.max_samples_per_subset,
    )
    print(json.dumps({"planned_run_count": len(matrix), "runs": matrix}, indent=2))
    if args.dry_run:
        return 0
    if len(matrix) > args.max_runs:
        raise ValueError(
            f"Matriz G1 possui {len(matrix)} runs e excede --max-runs={args.max_runs}"
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
            "generation": "G1",
            "representation": "R0_flat",
            "max_samples_per_subset": args.max_samples_per_subset,
            "xgb_device": args.xgb_device or experiment["xgb_device"],
        },
    )
    seed = int(experiment["seed"])
    xgb_device = args.xgb_device or str(experiment["xgb_device"])
    checkpoint_root = repository_path(args.checkpoint_dir)
    prediction_root = repository_path(args.prediction_dir)
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for window in windows:
        for fold in folds:
            splits, scaler, _ = build_sequence_fold(
                series,
                labels,
                blocks,
                preprocessing,
                fold=fold,
                size_frames=window,
                stride_frames=int(data_config["windowing"]["stride_frames"]),
                minimum_proportion=float(
                    data_config["windowing"]["minimum_target_proportion"]
                ),
                representation="R0",
            )
            if args.max_samples_per_subset is not None:
                splits = {
                    subset: limit_sequence_split(
                        split, args.max_samples_per_subset, seed=seed + fold
                    )
                    for subset, split in splits.items()
                }
            for model_name in models:
                run_id = make_run_id(
                    str(experiment["generation"]),
                    str(experiment["configuration_id"]),
                    model_name,
                    window,
                    fold,
                    seed,
                    args.max_samples_per_subset,
                )
                train_seconds = 0.0
                resumed = False
                model = None
                if model_name != "fixed_rules":
                    checkpoint = checkpoint_root / f"{run_id}.joblib"
                    loaded = None if args.no_resume else load_checkpoint(checkpoint, fingerprint)
                    if loaded:
                        model, metadata = loaded
                        train_seconds = float(metadata.get("train_seconds", 0.0))
                        resumed = True
                    else:
                        x_train, y_train = flatten_r0(splits["train"])
                        x_validation, y_validation = flatten_r0(splits["validation"])
                        model = build_model(
                            model_name,
                            seed=seed,
                            xgb_device=xgb_device,
                            parameters=dict(experiment["parameters"][model_name]),
                        )
                        started = time.perf_counter()
                        model = fit_model(
                            model_name,
                            model,
                            x_train,
                            y_train,
                            x_validation,
                            y_validation,
                            balancing=str(experiment["balancing"]),
                        )
                        train_seconds = time.perf_counter() - started
                        save_checkpoint(
                            model,
                            checkpoint,
                            {
                                "fingerprint": fingerprint,
                                "run_id": run_id,
                                "train_seconds": train_seconds,
                                "balancing": experiment["balancing"],
                            },
                        )
                for subset in ("validation", "test"):
                    _, expected = flatten_r0(splits[subset])
                    predicted = (
                        predict_fixed_rules(splits[subset], scaler, experiment["fixed_rules"])
                        if model_name == "fixed_rules"
                        else predict_model(model_name, model, flatten_r0(splits[subset])[0])
                    )
                    summary, per_class, confusion = evaluate_predictions(
                        model_name=model_name,
                        ablation="r0_flat",
                        fold=fold,
                        subset=subset,
                        expected=expected,
                        predicted=predicted,
                        train_seconds=train_seconds,
                        resumed=resumed,
                    )
                    context = {
                        "run_id": run_id,
                        "generation": "G1",
                        "representation": "R0_flat",
                        "window_size_frames": window,
                        "seed": seed,
                        "balancing": experiment["balancing"],
                    }
                    summary.update(context)
                    for row in per_class:
                        row.update(context)
                    for row in confusion:
                        row.update(context)
                    summary_rows.append(summary)
                    class_rows.extend(per_class)
                    confusion_rows.extend(confusion)
                    _write_csv(
                        prediction_root / f"{run_id}__{subset}.csv",
                        [
                            {
                                "video_id": item.video_id,
                                "start_frame": item.start_frame,
                                "end_frame": item.end_frame,
                                "actual": actual,
                                "predicted": prediction,
                            }
                            for item, actual, prediction in zip(
                                splits[subset].metadata, expected, predicted
                            )
                        ],
                    )
                print(f"G1 concluido: {run_id}", flush=True)

    output_dir = repository_path(args.output_dir)
    run_table = aggregate_run_table(summary_rows)
    _write_csv(output_dir / "g1_runs.csv", summary_rows)
    _write_csv(output_dir / "g1_execution_table.csv", run_table)
    _write_csv(output_dir / "g1_per_class.csv", class_rows)
    _write_csv(
        output_dir / "g1_per_class_statistics.csv",
        aggregate_per_class_statistics(class_rows),
    )
    _write_csv(output_dir / "g1_confusion.csv", confusion_rows)
    _write_csv(output_dir / "g1_fold_statistics.csv", aggregate_fold_statistics(summary_rows))
    _write_csv(
        output_dir / "g1_prediction_distribution.csv",
        prediction_distribution(confusion_rows),
    )
    _write_csv(
        output_dir / "g1_artifact_manifest.csv",
        artifact_manifest(
            run_table,
            checkpoint_root,
            experiment_path=experiment_path,
            fingerprint=fingerprint,
        ),
    )
    (output_dir / "g1_resolved_config.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "windowing": data_config["windowing"],
                "fingerprint": fingerprint,
                "inputs": {
                    "experiment_config": _relative(experiment_path),
                    "data_config": _relative(data_path),
                    "frame_intervals": _relative(interval_path),
                    "splits": _relative(split_path),
                    "extraction_manifest": _relative(input_dir / "extraction_manifest.csv"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "g1_report.md", summary_rows)
    figure_dir = repository_path(args.figure_dir)
    plot_macro_f1(summary_rows, figure_dir / "macro_f1_by_window.png")
    plot_fold_variability(summary_rows, figure_dir / "fold_window_variability.png")
    plot_validation_test(summary_rows, figure_dir / "validation_vs_test.png")
    plot_per_class_f1(class_rows, figure_dir / "per_class_f1.png")
    for model in models:
        for window in windows:
            plot_confusion(
                confusion_rows,
                model,
                window,
                figure_dir / f"confusion_{model}_w{window}.png",
            )
    print(f"G1 finalizada: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
