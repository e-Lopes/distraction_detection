"""Treina SVM, Random Forest e XGBoost com checkpoints e gráficos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock, window_subset
from ..preprocessing.missingness import expand_behavior_labels
from .dummy_baseline import (
    CLASSES,
    FEATURE_NAMES,
    FeatureWindow,
    build_feature_windows,
    load_series,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ABLATIONS = {
    "facial_only": tuple(range(10)),
    "facial_plus_missingness": tuple(range(14)),
    "missingness_only": tuple(range(10, 14)),
}
MODEL_NAMES = ("svm", "random_forest", "xgboost")
COLORS = {"svm": "#377eb8", "random_forest": "#4daf4a", "xgboost": "#e41a1c"}


def save_figure(figure: plt.Figure, output: Path) -> None:
    """Salva uma versão raster para inspeção e uma vetorial para publicação."""

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    svg_output = output.with_suffix(".svg")
    figure.savefig(svg_output, bbox_inches="tight")
    svg = re.sub(r"[ \t]+(?=\r?$)", "", svg_output.read_text(encoding="utf-8"), flags=re.M)
    svg_output.write_text(svg, encoding="utf-8")


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


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_fingerprint(paths: Sequence[Path], settings: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(str(path).encode())
        digest.update(_file_digest(path).encode())
    return digest.hexdigest()


def split_windows(
    windows: Sequence[FeatureWindow], blocks: Sequence[SplitBlock], fold: int
) -> dict[str, list[FeatureWindow]]:
    subsets: dict[str, list[FeatureWindow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for window in windows:
        subset = window_subset(
            video_id=window.video_id,
            start_frame=window.start_frame,
            end_frame=window.end_frame,
            blocks=blocks,
            fold=fold,
        )
        if subset:
            subsets[subset].append(window)
    if any(not values for values in subsets.values()):
        raise ValueError(f"Fold {fold} contém subconjunto vazio")
    return subsets


def feature_matrix(
    windows: Sequence[FeatureWindow], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([[window.values[index] for index in indices] for window in windows])
    labels = np.asarray([window.label for window in windows])
    if not np.isfinite(matrix).all():
        raise ValueError("Features contêm NaN ou infinito")
    return matrix, labels


def balanced_sample_weights(labels: Sequence[str]) -> np.ndarray:
    counts = Counter(labels)
    total = len(labels)
    return np.asarray([total / (len(counts) * counts[label]) for label in labels])


def build_model(
    name: str,
    *,
    seed: int,
    xgb_device: str,
    parameters: dict[str, object],
) -> object:
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(**parameters)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(**parameters, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(**parameters, device=xgb_device, random_state=seed)
    raise ValueError(f"Modelo desconhecido: {name}")


def _encode(labels: Sequence[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(CLASSES)}
    return np.asarray([mapping[label] for label in labels])


def _decode(labels: Sequence[int]) -> np.ndarray:
    return np.asarray([CLASSES[int(label)] for label in labels])


def fit_model(
    model_name: str,
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> object:
    if model_name == "xgboost":
        model.fit(
            x_train,
            _encode(y_train),
            sample_weight=balanced_sample_weights(y_train),
            eval_set=[(x_validation, _encode(y_validation))],
            verbose=False,
        )
    else:
        model.fit(x_train, y_train)
    return model


def predict_model(model_name: str, model: object, values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mismatched devices.*")
        predictions = model.predict(values)
    return _decode(predictions) if model_name == "xgboost" else predictions


def save_checkpoint(model: object, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)
    metadata_path = path.with_suffix(".json")
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata_temp.replace(metadata_path)


def load_checkpoint(path: Path, fingerprint: str) -> tuple[object, dict[str, object]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        return None
    return joblib.load(path), metadata


def evaluate_predictions(
    *,
    model_name: str,
    ablation: str,
    fold: int,
    subset: str,
    expected: np.ndarray,
    predicted: np.ndarray,
    train_seconds: float,
    resumed: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    present = [label for label in CLASSES if label in set(expected)]
    balanced_accuracy = float(
        np.mean([np.mean(predicted[expected == label] == label) for label in present])
    )
    summary = {
        "model": model_name,
        "ablation": ablation,
        "fold": fold,
        "subset": subset,
        "num_windows": len(expected),
        "accuracy": accuracy_score(expected, predicted),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1_present_classes": f1_score(
            expected, predicted, labels=present, average="macro", zero_division=0
        ),
        "macro_f1_all_classes": f1_score(
            expected, predicted, labels=CLASSES, average="macro", zero_division=0
        ),
        "present_classes": json.dumps(present),
        "absent_classes": json.dumps([label for label in CLASSES if label not in present]),
        "train_seconds": train_seconds,
        "resumed_checkpoint": resumed,
    }
    precision, recall, f1_values, support = precision_recall_fscore_support(
        expected, predicted, labels=CLASSES, zero_division=0
    )
    per_class = []
    for index, label in enumerate(CLASSES):
        has_support = bool(support[index])
        per_class.append(
            {
                "model": model_name,
                "ablation": ablation,
                "fold": fold,
                "subset": subset,
                "label": label,
                "precision": precision[index] if has_support else "",
                "recall": recall[index] if has_support else "",
                "f1": f1_values[index] if has_support else "",
                "support": int(support[index]),
            }
        )
    counts = Counter(zip(expected, predicted))
    confusion = [
        {
            "model": model_name,
            "ablation": ablation,
            "fold": fold,
            "subset": subset,
            "actual": actual,
            "predicted": prediction,
            "count": counts[(actual, prediction)],
        }
        for actual in CLASSES
        for prediction in CLASSES
    ]
    return summary, per_class, confusion


def _mean_metric(
    rows: Sequence[dict[str, object]], model: str, ablation: str, metric: str
) -> float:
    selected = [
        float(row[metric])
        for row in rows
        if row["model"] == model and row["ablation"] == ablation and row["subset"] == "test"
    ]
    return float(np.mean(selected))


def plot_metric_comparison(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_values = np.arange(len(ABLATIONS))
    width = 0.24
    for model_index, model in enumerate(MODEL_NAMES):
        offset = (model_index - 1) * width
        for axis, metric, title in zip(
            axes,
            ("macro_f1_all_classes", "balanced_accuracy"),
            ("Macro F1", "Balanced accuracy"),
        ):
            values = [_mean_metric(rows, model, ablation, metric) for ablation in ABLATIONS]
            axis.bar(x_values + offset, values, width, label=model, color=COLORS[model])
            axis.set_title(title)
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x_values, [name.replace("_", "\n") for name in ABLATIONS])
        axis.set_ylabel("Média nos quatro folds de teste")
    axes[0].legend()
    figure.suptitle("Baselines clássicos — janela de 60 frames")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_fold_macro_f1(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for axis, ablation in zip(axes, ABLATIONS):
        for model in MODEL_NAMES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model
                    and row["ablation"] == ablation
                    and row["subset"] == "test"
                ),
                key=lambda row: int(row["fold"]),
            )
            axis.plot(
                [int(row["fold"]) for row in selected],
                [float(row["macro_f1_all_classes"]) for row in selected],
                marker="o",
                label=model,
                color=COLORS[model],
            )
        axis.set_title(ablation.replace("_", " "))
        axis.set_xlabel("Fold")
        axis.set_xticks((1, 2, 3, 4))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro F1 (3 classes)")
    axes[0].set_ylim(0, 1)
    axes[-1].legend()
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_per_class_f1(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    x_values = np.arange(len(CLASSES))
    width = 0.24
    for axis, ablation in zip(axes, ABLATIONS):
        for model_index, model in enumerate(MODEL_NAMES):
            values = []
            for label in CLASSES:
                selected = [
                    float(row["f1"])
                    for row in rows
                    if row["model"] == model
                    and row["ablation"] == ablation
                    and row["subset"] == "test"
                    and row["label"] == label
                    and row["f1"] != ""
                ]
                values.append(float(np.mean(selected)) if selected else 0.0)
            axis.bar(
                x_values + (model_index - 1) * width,
                values,
                width,
                label=model,
                color=COLORS[model],
            )
        axis.set_title(ablation.replace("_", " "))
        axis.set_xticks(x_values, CLASSES)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("F1 médio nos folds com suporte")
    axes[0].set_ylim(0, 1)
    axes[-1].legend()
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_confusion(rows: list[dict[str, object]], model: str, output: Path) -> None:
    selected = [
        row
        for row in rows
        if row["model"] == model
        and row["ablation"] == "facial_plus_missingness"
        and row["subset"] == "test"
    ]
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for row in selected:
        matrix[CLASSES.index(str(row["actual"])), CLASSES.index(str(row["predicted"]))] += int(
            row["count"]
        )
    denominators = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, denominators, out=np.zeros_like(matrix), where=denominators > 0)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row_index in range(len(CLASSES)):
        for column_index in range(len(CLASSES)):
            cell_text = (
                f"{normalized[row_index, column_index]:.2f}\n"
                f"(n={int(matrix[row_index, column_index])})"
            )
            axis.text(
                column_index,
                row_index,
                cell_text,
                ha="center",
                va="center",
                color="white" if normalized[row_index, column_index] > 0.55 else "black",
            )
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_title(f"{model} — matriz agregada nos testes")
    figure.colorbar(image, ax=axis, label="Proporção por classe real")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_feature_importance(
    values: Sequence[np.ndarray], model: str, output: Path
) -> None:
    importance = np.mean(np.asarray(values), axis=0)
    order = np.argsort(importance)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(
        np.asarray(FEATURE_NAMES)[order],
        importance[order],
        color=COLORS[model],
    )
    axis.set_xlabel("Importância média nos quatro folds")
    axis.set_title(f"{model} — indicadores + missingness")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_training_time(rows: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x_values = np.arange(len(ABLATIONS))
    width = 0.24
    for model_index, model in enumerate(MODEL_NAMES):
        values = []
        for ablation in ABLATIONS:
            selected = [
                float(row["train_seconds"])
                for row in rows
                if row["model"] == model
                and row["ablation"] == ablation
                and row["subset"] == "test"
            ]
            values.append(float(np.mean(selected)))
        axis.bar(
            x_values + (model_index - 1) * width,
            values,
            width,
            label=model,
            color=COLORS[model],
        )
    axis.set_xticks(x_values, [name.replace("_", "\n") for name in ABLATIONS])
    axis.set_ylabel("Segundos por fold")
    axis.set_title("Tempo médio de treinamento")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Baselines clássicos — janela de 60 frames",
        "",
        "Resultados médios nos quatro vídeos de teste. Macro F1 usa as três classes; "
        "classes ausentes continuam sem suporte nos CSVs por classe.",
        "",
        "| Modelo | Ablação | Macro F1 | Balanced accuracy |",
        "|---|---|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for ablation in ABLATIONS:
            lines.append(
                f"| {model} | {ablation} | "
                f"{_mean_metric(rows, model, ablation, 'macro_f1_all_classes'):.4f} | "
                f"{_mean_metric(rows, model, ablation, 'balanced_accuracy'):.4f} |"
            )
    lines.extend(
        [
            "",
            "Checkpoints e previsões individuais permanecem fora do Git em "
            "`outputs/models/classical_60/` e `outputs/predictions/classical_60/`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="fase_2/configs/experiment/classical_60.yaml",
    )
    parser.add_argument("--config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/classical_60")
    parser.add_argument("--checkpoint-dir", default="fase_2/outputs/models/classical_60")
    parser.add_argument("--prediction-dir", default="fase_2/outputs/predictions/classical_60")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_path = repository_path(args.experiment_config)
    experiment = load_yaml(experiment_path)
    window_size = args.window_size or int(experiment["window_size_frames"])
    seed = args.seed if args.seed is not None else int(experiment["seed"])
    xgb_device = args.xgb_device or str(experiment["xgb_device"])
    if tuple(experiment["models"]) != MODEL_NAMES:
        raise ValueError(f"Modelos esperados: {MODEL_NAMES}")
    if tuple(experiment["ablations"]) != tuple(ABLATIONS):
        raise ValueError(f"Ablações esperadas: {tuple(ABLATIONS)}")
    model_parameters = experiment["parameters"]
    config_path = repository_path(args.config)
    interval_path = repository_path(args.frame_intervals)
    split_path = repository_path(args.splits)
    input_dir = repository_path(args.input_dir)
    config = load_yaml(config_path)
    windowing = config["windowing"]
    series = load_series(input_dir)
    labels = expand_behavior_labels(
        {video_id: len(rows) for video_id, rows in series.items()}, _read_csv(interval_path)
    )
    windows = build_feature_windows(
        series,
        labels,
        size_frames=window_size,
        stride_frames=int(windowing["stride_frames"]),
        minimum_proportion=float(windowing["minimum_target_proportion"]),
    )
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
    manifest_path = input_dir / "extraction_manifest.csv"
    fingerprint = experiment_fingerprint(
        [experiment_path, config_path, interval_path, split_path, manifest_path],
        {
            "window_size": window_size,
            "seed": seed,
            "xgb_device": xgb_device,
            "model_parameters": model_parameters,
            "feature_names": FEATURE_NAMES,
            "ablations": ABLATIONS,
            "models": MODEL_NAMES,
        },
    )
    checkpoint_dir = repository_path(args.checkpoint_dir)
    prediction_dir = repository_path(args.prediction_dir)
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    importance_values: dict[str, list[np.ndarray]] = {
        "random_forest": [],
        "xgboost": [],
    }
    for ablation, indices in ABLATIONS.items():
        feature_names = [FEATURE_NAMES[index] for index in indices]
        for fold in sorted({block.fold for block in blocks}):
            subsets = split_windows(windows, blocks, fold)
            x_train, y_train = feature_matrix(subsets["train"], indices)
            x_validation, y_validation = feature_matrix(subsets["validation"], indices)
            for model_name in MODEL_NAMES:
                checkpoint = checkpoint_dir / f"{model_name}__{ablation}__fold_{fold}.joblib"
                loaded = None if args.no_resume else load_checkpoint(checkpoint, fingerprint)
                model = loaded[0] if loaded else None
                resumed = loaded is not None
                train_seconds = 0.0
                if model is None:
                    print(f"Treinando {model_name} | {ablation} | fold {fold}", flush=True)
                    model = build_model(
                        model_name,
                        seed=seed,
                        xgb_device=xgb_device,
                        parameters=model_parameters[model_name],
                    )
                    started = time.perf_counter()
                    try:
                        model = fit_model(
                            model_name,
                            model,
                            x_train,
                            y_train,
                            x_validation,
                            y_validation,
                        )
                    except Exception:
                        if model_name != "xgboost" or xgb_device != "cuda":
                            raise
                        print("XGBoost CUDA falhou; repetindo em CPU", flush=True)
                        model = build_model(
                            model_name,
                            seed=seed,
                            xgb_device="cpu",
                            parameters=model_parameters[model_name],
                        )
                        model = fit_model(
                            model_name,
                            model,
                            x_train,
                            y_train,
                            x_validation,
                            y_validation,
                        )
                    train_seconds = time.perf_counter() - started
                    save_checkpoint(
                        model,
                        checkpoint,
                        {
                            "fingerprint": fingerprint,
                            "model": model_name,
                            "ablation": ablation,
                            "fold": fold,
                            "feature_names": feature_names,
                            "train_seconds": train_seconds,
                        },
                    )
                else:
                    train_seconds = float(loaded[1].get("train_seconds", 0.0))
                    print(f"Checkpoint retomado: {checkpoint.name}", flush=True)
                if (
                    ablation == "facial_plus_missingness"
                    and model_name in importance_values
                ):
                    importance_values[model_name].append(
                        np.asarray(model.feature_importances_, dtype=float)
                    )
                for subset in ("validation", "test"):
                    x_values, expected = feature_matrix(subsets[subset], indices)
                    predicted = predict_model(model_name, model, x_values)
                    summary, per_class, confusion = evaluate_predictions(
                        model_name=model_name,
                        ablation=ablation,
                        fold=fold,
                        subset=subset,
                        expected=expected,
                        predicted=predicted,
                        train_seconds=train_seconds,
                        resumed=resumed,
                    )
                    summary_rows.append(summary)
                    class_rows.extend(per_class)
                    confusion_rows.extend(confusion)
                    prediction_rows = [
                        {
                            "video_id": window.video_id,
                            "start_frame": window.start_frame,
                            "end_frame": window.end_frame,
                            "actual": actual,
                            "predicted": prediction,
                        }
                        for window, actual, prediction in zip(
                            subsets[subset], expected, predicted
                        )
                    ]
                    _write_csv(
                        prediction_dir
                        / f"{model_name}__{ablation}__fold_{fold}__{subset}.csv",
                        prediction_rows,
                    )
    output_dir = repository_path(args.output_dir)
    _write_csv(output_dir / "classical_baselines_summary.csv", summary_rows)
    _write_csv(output_dir / "classical_baselines_per_class.csv", class_rows)
    _write_csv(output_dir / "classical_baselines_confusion.csv", confusion_rows)
    write_report(output_dir / "classical_baselines.md", summary_rows)
    figure_dir = repository_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_comparison(summary_rows, figure_dir / "metrics_comparison.png")
    plot_fold_macro_f1(summary_rows, figure_dir / "macro_f1_by_fold.png")
    plot_per_class_f1(class_rows, figure_dir / "per_class_f1.png")
    plot_training_time(summary_rows, figure_dir / "training_time.png")
    for model_name in MODEL_NAMES:
        plot_confusion(
            confusion_rows,
            model_name,
            figure_dir / f"confusion_{model_name}.png",
        )
    for model_name, values in importance_values.items():
        plot_feature_importance(
            values,
            model_name,
            figure_dir / f"feature_importance_{model_name}.png",
        )
    print(f"Experimento concluído: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
