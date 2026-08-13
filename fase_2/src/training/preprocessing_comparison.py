"""Compara zero-fill, interpolação curta e flags nos folds temporais congelados."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..data.windowing import WindowRecord, build_windows
from ..preprocessing.missingness import expand_behavior_labels
from ..preprocessing.strategies import (
    aggregate_preprocessed_window,
    fit_training_medians,
    preprocess_block,
    strategy_feature_names,
)
from .classical_baselines import (
    MODEL_NAMES,
    build_model,
    evaluate_predictions,
    experiment_fingerprint,
    feature_matrix,
    fit_model,
    load_checkpoint,
    predict_model,
    save_checkpoint,
    save_figure,
)
from .dummy_baseline import CLASSES, FeatureWindow, load_series

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_COLORS = {
    "zero_fill_baseline": "#377eb8",
    "short_gap_interpolation": "#4daf4a",
    "short_gap_interpolation_with_flags": "#984ea3",
}


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


def build_fold_features(
    series: Mapping[str, Sequence[dict[str, str]]],
    windows: Sequence[WindowRecord],
    blocks: Sequence[SplitBlock],
    strategy: Mapping[str, object],
    *,
    fold: int,
) -> tuple[dict[str, list[FeatureWindow]], dict[str, float]]:
    """Pré-processa cada bloco isoladamente e agrega somente janelas contidas nele."""

    fold_blocks = [block for block in blocks if block.fold == fold]
    use_medians = str(strategy.get("long_gap_fill", "zero")) == "training_median"
    medians = fit_training_medians(series, fold_blocks) if use_medians else {}
    prepared: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    by_video: dict[str, list[SplitBlock]] = {}
    for block in fold_blocks:
        original = series[block.video_id][block.start_frame : block.end_frame + 1]
        prepared[(block.video_id, block.start_frame, block.end_frame)] = preprocess_block(
            original,
            strategy,
            training_medians=medians if use_medians else None,
        )
        by_video.setdefault(block.video_id, []).append(block)

    subsets: dict[str, list[FeatureWindow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for window in windows:
        if window.label == "mixed":
            continue
        matches = [
            block
            for block in by_video.get(window.video_id, [])
            if window.start_frame >= block.start_frame and window.end_frame <= block.end_frame
        ]
        if len(matches) > 1:
            raise ValueError("Janela atribuída a múltiplos blocos")
        if not matches:
            continue
        block = matches[0]
        block_rows = prepared[(block.video_id, block.start_frame, block.end_frame)]
        relative_start = window.start_frame - block.start_frame
        rows = block_rows[relative_start : relative_start + window.size_frames]
        subsets[block.subset].append(
            FeatureWindow(
                video_id=window.video_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                size_frames=window.size_frames,
                label=window.label,
                values=aggregate_preprocessed_window(rows, strategy),
            )
        )
    if any(not values for values in subsets.values()):
        raise ValueError(f"Fold {fold} contém subconjunto vazio")
    return subsets, medians


def _mean_test_metric(
    rows: Sequence[dict[str, object]],
    model: str,
    strategy: str,
    window_size: int,
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
        if row["model"] == model
        and row["preprocessing"] == strategy
        and int(row["window_size_frames"]) == window_size
        and row["subset"] == "test"
    ]
    return float(np.mean(values))


def plot_metric_by_window(
    rows: list[dict[str, object]],
    strategies: Sequence[str],
    window_sizes: Sequence[int],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(17, 5), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES):
        for strategy in strategies:
            values = [
                _mean_test_metric(
                    rows, model, strategy, size, "macro_f1_all_classes"
                )
                for size in window_sizes
            ]
            axis.plot(
                window_sizes,
                values,
                marker="o",
                label=strategy.replace("_", " "),
                color=STRATEGY_COLORS[strategy],
            )
        axis.set_title(model)
        axis.set_xlabel("Tamanho da janela (frames)")
        axis.set_xticks(window_sizes)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro F1 médio nos folds externos")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(fontsize=8)
    figure.suptitle("Pré-processamento × contexto temporal")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_fatigue_recall(
    rows: list[dict[str, object]],
    strategies: Sequence[str],
    window_sizes: Sequence[int],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(17, 5), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES):
        for strategy in strategies:
            values = []
            for size in window_sizes:
                selected = [
                    float(row["recall"])
                    for row in rows
                    if row["model"] == model
                    and row["preprocessing"] == strategy
                    and int(row["window_size_frames"]) == size
                    and row["subset"] == "test"
                    and row["label"] == "fatigue"
                    and row["recall"] != ""
                ]
                values.append(float(np.mean(selected)) if selected else np.nan)
            axis.plot(
                window_sizes,
                values,
                marker="o",
                label=strategy.replace("_", " "),
                color=STRATEGY_COLORS[strategy],
            )
        axis.set_title(model)
        axis.set_xlabel("Tamanho da janela (frames)")
        axis.set_xticks(window_sizes)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Recall de fadiga nos folds com suporte")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(fontsize=8)
    figure.suptitle("Sensibilidade à fadiga por pré-processamento")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_validation_gap(
    rows: list[dict[str, object]], strategies: Sequence[str], output: Path
) -> None:
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(17, 5), sharey=True)
    x_values = np.arange(len(strategies))
    for axis, model in zip(axes, MODEL_NAMES):
        selected = [row for row in rows if row["model"] == model]
        for index, subset in enumerate(("validation", "test")):
            values = [
                np.mean(
                    [
                        float(row["macro_f1_all_classes"])
                        for row in selected
                        if row["preprocessing"] == strategy and row["subset"] == subset
                    ]
                )
                for strategy in strategies
            ]
            axis.bar(x_values + (index - 0.5) * 0.35, values, 0.35, label=subset)
        axis.set_title(model)
        axis.set_xticks(x_values, [name.replace("_", "\n") for name in strategies])
        axis.tick_params(axis="x", labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro F1 médio — todas as janelas")
    axes[0].set_ylim(0, 1)
    axes[-1].legend()
    figure.suptitle("Diferença entre validação e avaliação externa")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_fold_stability(
    rows: list[dict[str, object]], strategies: Sequence[str], output: Path
) -> None:
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(17, 5), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES):
        for strategy in strategies:
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["preprocessing"] == strategy
                and int(row["window_size_frames"]) == 60
                and row["subset"] == "test"
            ]
            selected.sort(key=lambda row: int(row["fold"]))
            axis.plot(
                [int(row["fold"]) for row in selected],
                [float(row["macro_f1_all_classes"]) for row in selected],
                marker="o",
                label=strategy.replace("_", " "),
                color=STRATEGY_COLORS[strategy],
            )
        axis.set_title(model)
        axis.set_xlabel("Fold externo")
        axis.set_xticks((1, 2, 3, 4))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro F1 — janela de 60 frames")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(fontsize=8)
    figure.suptitle("Estabilidade entre vídeos de teste")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def write_report(
    path: Path,
    summary: list[dict[str, object]],
    per_class: list[dict[str, object]],
    strategies: Sequence[str],
    window_sizes: Sequence[int],
) -> None:
    lines = [
        "# Comparação de pré-processamento",
        "",
        "Medianas são ajustadas somente no treino de cada fold. A interpolação ocorre dentro "
        "de cada bloco e nunca atravessa treino, validação, teste ou purge gaps.",
        "",
        "| Modelo | Estratégia | Janela | Macro F1 | Balanced accuracy | Recall fadiga* |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for strategy in strategies:
            for size in window_sizes:
                fatigue = [
                    float(row["recall"])
                    for row in per_class
                    if row["model"] == model
                    and row["preprocessing"] == strategy
                    and int(row["window_size_frames"]) == size
                    and row["subset"] == "test"
                    and row["label"] == "fatigue"
                    and row["recall"] != ""
                ]
                macro_f1 = _mean_test_metric(
                    summary, model, strategy, size, "macro_f1_all_classes"
                )
                balanced = _mean_test_metric(
                    summary, model, strategy, size, "balanced_accuracy"
                )
                lines.append(
                    f"| {model} | {strategy} | {size} | "
                    f"{macro_f1:.4f} | {balanced:.4f} | {np.mean(fatigue):.4f} |"
                )
    lines.extend(
        [
            "",
            "Observação: recall de fadiga usa apenas folds externos com suporte.",
            "",
            "Estes resultados comparam representações usando hiperparâmetros fixos. Eles não "
            "selecionam a configuração final pelo conjunto de teste.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="fase_2/configs/experiment/preprocessing_comparison.yaml",
    )
    parser.add_argument("--data-config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument(
        "--checkpoint-dir", default="fase_2/outputs/models/preprocessing_comparison"
    )
    parser.add_argument(
        "--prediction-dir", default="fase_2/outputs/predictions/preprocessing_comparison"
    )
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics")
    parser.add_argument(
        "--figure-dir", default="fase_2/outputs/figures/preprocessing_comparison"
    )
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"))
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
    strategy_paths = [repository_path(path) for path in experiment["preprocessing_configs"]]
    strategies = [load_yaml(path) for path in strategy_paths]
    strategy_names = [str(strategy["name"]) for strategy in strategies]
    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError("Nomes de estratégias duplicados")
    if tuple(experiment["models"]) != MODEL_NAMES:
        raise ValueError(f"Modelos esperados: {MODEL_NAMES}")
    window_sizes = [int(size) for size in experiment["window_sizes_frames"]]
    seed = int(experiment["seed"])
    xgb_device = args.xgb_device or str(experiment["xgb_device"])
    parameters = experiment["parameters"]

    series = load_series(input_dir)
    labels = expand_behavior_labels(
        {video_id: len(rows) for video_id, rows in series.items()}, _read_csv(interval_path)
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
    fingerprint = experiment_fingerprint(
        [
            experiment_path,
            data_path,
            interval_path,
            split_path,
            input_dir / "extraction_manifest.csv",
            *strategy_paths,
        ],
        {"mode": "preprocessing_comparison", "xgb_device": xgb_device},
    )
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    checkpoint_dir = repository_path(args.checkpoint_dir)
    prediction_dir = repository_path(args.prediction_dir)
    for size in window_sizes:
        windows = build_windows(
            labels,
            size_frames=size,
            stride_frames=int(data_config["windowing"]["stride_frames"]),
            behavior_classes=set(CLASSES),
            minimum_proportion=float(
                data_config["windowing"]["minimum_target_proportion"]
            ),
        )
        for strategy in strategies:
            strategy_name = str(strategy["name"])
            names = strategy_feature_names(strategy)
            indices = tuple(range(len(names)))
            for fold in sorted({block.fold for block in blocks}):
                subsets, medians = build_fold_features(
                    series, windows, blocks, strategy, fold=fold
                )
                x_train, y_train = feature_matrix(subsets["train"], indices)
                x_validation, y_validation = feature_matrix(subsets["validation"], indices)
                for model_name in MODEL_NAMES:
                    checkpoint = (
                        checkpoint_dir
                        / strategy_name
                        / f"window_{size}"
                        / f"{model_name}__fold_{fold}.joblib"
                    )
                    loaded = None if args.no_resume else load_checkpoint(checkpoint, fingerprint)
                    model = loaded[0] if loaded else None
                    resumed = loaded is not None
                    train_seconds = 0.0
                    if model is None:
                        print(
                            f"Treinando {model_name} | {strategy_name} | "
                            f"janela {size} | fold {fold}",
                            flush=True,
                        )
                        model = build_model(
                            model_name,
                            seed=seed,
                            xgb_device=xgb_device,
                            parameters=parameters[model_name],
                        )
                        started = time.perf_counter()
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
                                "preprocessing": strategy_name,
                                "window_size_frames": size,
                                "fold": fold,
                                "feature_names": names,
                                "training_medians": medians,
                                "train_seconds": train_seconds,
                            },
                        )
                    else:
                        train_seconds = float(loaded[1].get("train_seconds", 0.0))
                    for subset in ("validation", "test"):
                        samples = subsets[subset]
                        x_values, expected = feature_matrix(samples, indices)
                        predicted = predict_model(model_name, model, x_values)
                        summary, per_class, confusion = evaluate_predictions(
                            model_name=model_name,
                            ablation="all_features",
                            fold=fold,
                            subset=subset,
                            expected=expected,
                            predicted=predicted,
                            train_seconds=train_seconds,
                            resumed=resumed,
                        )
                        context = {
                            "preprocessing": strategy_name,
                            "window_size_frames": size,
                            "num_features": len(names),
                        }
                        summary.update(context)
                        for row in per_class:
                            row.update(context)
                        for row in confusion:
                            row.update(context)
                        summary_rows.append(summary)
                        class_rows.extend(per_class)
                        confusion_rows.extend(confusion)
                        predictions = [
                            {
                                "video_id": sample.video_id,
                                "start_frame": sample.start_frame,
                                "end_frame": sample.end_frame,
                                "actual": actual,
                                "predicted": prediction,
                            }
                            for sample, actual, prediction in zip(
                                samples, expected, predicted
                            )
                        ]
                        _write_csv(
                            prediction_dir
                            / strategy_name
                            / f"window_{size}__{model_name}__fold_{fold}__{subset}.csv",
                            predictions,
                        )

    output_dir = repository_path(args.output_dir)
    _write_csv(output_dir / "preprocessing_comparison_summary.csv", summary_rows)
    _write_csv(output_dir / "preprocessing_comparison_per_class.csv", class_rows)
    _write_csv(output_dir / "preprocessing_comparison_confusion.csv", confusion_rows)
    write_report(
        output_dir / "preprocessing_comparison.md",
        summary_rows,
        class_rows,
        strategy_names,
        window_sizes,
    )
    figure_dir = repository_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_by_window(
        summary_rows, strategy_names, window_sizes, figure_dir / "macro_f1_by_window.png"
    )
    plot_fatigue_recall(
        class_rows, strategy_names, window_sizes, figure_dir / "fatigue_recall.png"
    )
    plot_validation_gap(
        summary_rows, strategy_names, figure_dir / "validation_test_gap.png"
    )
    plot_fold_stability(
        summary_rows, strategy_names, figure_dir / "fold_stability_window_60.png"
    )
    print(f"Comparação concluída: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
