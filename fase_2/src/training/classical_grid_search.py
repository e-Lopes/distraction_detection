"""Grid search clássico com seleção exclusiva na validação e checkpoints leves."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from sklearn.metrics import f1_score

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from .classical_baselines import (
    ABLATIONS,
    MODEL_NAMES,
    build_model,
    evaluate_predictions,
    experiment_fingerprint,
    feature_matrix,
    fit_model,
    predict_model,
    plot_confusion,
    plot_fold_macro_f1,
    plot_metric_comparison,
    plot_per_class_f1,
    save_figure,
    save_checkpoint,
    split_windows,
)
from .dummy_baseline import FEATURE_NAMES, build_feature_windows, load_series

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parameter_grid(
    values: dict[str, list[object]] | list[dict[str, list[object]]],
) -> list[dict[str, object]]:
    if isinstance(values, list):
        return [candidate for subgrid in values for candidate in parameter_grid(subgrid)]
    keys = list(values)
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*(values[key] for key in keys))
    ]


def candidate_id(parameters: dict[str, object]) -> str:
    payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validation_metrics(expected: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    present = sorted(set(expected))
    return (
        f1_score(expected, predicted, labels=present, average="macro", zero_division=0),
        float(np.mean([np.mean(predicted[expected == label] == label) for label in present])),
    )


def save_candidate_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_candidate_result(path: Path, fingerprint: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result if result.get("fingerprint") == fingerprint else None


def plot_grid_scores(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row_index, model in enumerate(MODEL_NAMES):
        for column_index, ablation in enumerate(ABLATIONS):
            axis = axes[row_index, column_index]
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == model and row["ablation"] == ablation
                ),
                key=lambda row: float(row["validation_macro_f1"]),
                reverse=True,
            )
            values = [float(row["validation_macro_f1"]) for row in selected]
            axis.plot(range(1, len(values) + 1), values, marker=".", linewidth=1)
            axis.set_title(f"{model}\n{ablation.replace('_', ' ')}")
            axis.set_xlabel("Candidatos ordenados")
            axis.set_ylabel("Macro F1 validação")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
    figure.suptitle("Distribuição dos resultados do grid search")
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_best_by_fold(rows: list[dict[str, object]], output: Path) -> None:
    best = [row for row in rows if row["selected_best"]]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for axis, ablation in zip(axes, ABLATIONS):
        for model in MODEL_NAMES:
            selected = sorted(
                (
                    row
                    for row in best
                    if row["model"] == model and row["ablation"] == ablation
                ),
                key=lambda row: int(row["fold"]),
            )
            axis.plot(
                [int(row["fold"]) for row in selected],
                [float(row["validation_macro_f1"]) for row in selected],
                marker="o",
                label=model,
            )
        axis.set_title(ablation.replace("_", " "))
        axis.set_xticks((1, 2, 3, 4))
        axis.set_xlabel("Fold")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Melhor Macro F1 na validação")
    axes[0].set_ylim(0, 1)
    axes[-1].legend()
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def plot_model_boxplot(rows: list[dict[str, object]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    values = [
        [float(row["validation_macro_f1"]) for row in rows if row["model"] == model]
        for model in MODEL_NAMES
    ]
    axis.boxplot(values, tick_labels=MODEL_NAMES, showmeans=True)
    axis.set_ylabel("Macro F1 na validação")
    axis.set_title("Variação entre hiperparâmetros, ablações e folds")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output)
    plt.close(figure)


def write_grid_report(
    path: Path,
    selected: list[dict[str, object]],
    tuned_summary: list[dict[str, object]],
) -> None:
    """Resume desempenho externo e escolhas feitas apenas na validação."""

    lines = [
        "# Grid search dos modelos clássicos — janela de 60 frames",
        "",
        "Cada fold externo seleciona hiperparâmetros exclusivamente no respectivo bloco de "
        "validação. O vídeo de teste do fold não participa da seleção.",
        "",
        "## Desempenho externo médio",
        "",
        "| Modelo | Ablação | Macro F1 (3 classes) | Balanced accuracy |",
        "|---|---|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for ablation in ABLATIONS:
            rows = [
                row
                for row in tuned_summary
                if row["model"] == model
                and row["ablation"] == ablation
                and row["subset"] == "test"
            ]
            lines.append(
                f"| {model} | {ablation} | "
                f"{np.mean([float(row['macro_f1_all_classes']) for row in rows]):.4f} | "
                f"{np.mean([float(row['balanced_accuracy']) for row in rows]):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Vencedores por fold de validação",
            "",
            "| Modelo | Ablação | Fold | Macro F1 validação | Hiperparâmetros |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in sorted(
        selected,
        key=lambda item: (str(item["model"]), str(item["ablation"]), int(item["fold"])),
    ):
        lines.append(
            f"| {row['model']} | {row['ablation']} | {row['fold']} | "
            f"{float(row['validation_macro_f1']):.4f} | `{row['parameters']}` |"
        )
    lines.extend(
        [
            "",
            "A variação de parâmetros entre folds é parte da avaliação aninhada do "
            "procedimento. Nenhuma configuração final única deve ser escolhida olhando o "
            "desempenho de teste.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config", default="fase_2/configs/experiment/classical_60.yaml"
    )
    parser.add_argument("--data-config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument("--checkpoint-dir", default="fase_2/outputs/cache/grid_search_60")
    parser.add_argument("--model-dir", default="fase_2/outputs/models/grid_search_60")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/grid_search_60")
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_path = repository_path(args.experiment_config)
    data_config_path = repository_path(args.data_config)
    interval_path = repository_path(args.frame_intervals)
    split_path = repository_path(args.splits)
    input_dir = repository_path(args.input_dir)
    experiment = load_yaml(experiment_path)
    data_config = load_yaml(data_config_path)
    seed = int(experiment["seed"])
    xgb_device = args.xgb_device or str(experiment["xgb_device"])
    series = load_series(input_dir)
    labels = expand_behavior_labels(
        {video_id: len(rows) for video_id, rows in series.items()}, _read_csv(interval_path)
    )
    windows = build_feature_windows(
        series,
        labels,
        size_frames=int(experiment["window_size_frames"]),
        stride_frames=int(data_config["windowing"]["stride_frames"]),
        minimum_proportion=float(data_config["windowing"]["minimum_target_proportion"]),
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
            data_config_path,
            interval_path,
            split_path,
            input_dir / "extraction_manifest.csv",
        ],
        {"mode": "grid_search", "features": FEATURE_NAMES},
    )
    checkpoint_dir = repository_path(args.checkpoint_dir)
    model_dir = repository_path(args.model_dir)
    all_results: list[dict[str, object]] = []
    tuned_summary: list[dict[str, object]] = []
    tuned_per_class: list[dict[str, object]] = []
    tuned_confusion: list[dict[str, object]] = []
    for ablation, indices in ABLATIONS.items():
        for fold in sorted({block.fold for block in blocks}):
            subsets = split_windows(windows, blocks, fold)
            x_train, y_train = feature_matrix(subsets["train"], indices)
            x_validation, y_validation = feature_matrix(subsets["validation"], indices)
            for model_name in MODEL_NAMES:
                candidates = parameter_grid(experiment["grid_search"][model_name])
                combination_results: list[dict[str, object]] = []
                for number, parameters in enumerate(candidates, start=1):
                    identifier = candidate_id(parameters)
                    checkpoint = (
                        checkpoint_dir
                        / model_name
                        / ablation
                        / f"fold_{fold}__{identifier}.json"
                    )
                    result = None if args.no_resume else load_candidate_result(
                        checkpoint, fingerprint
                    )
                    if result is None:
                        print(
                            f"Grid {model_name} | {ablation} | fold {fold} | "
                            f"{number}/{len(candidates)}",
                            flush=True,
                        )
                        model = build_model(
                            model_name,
                            seed=seed,
                            xgb_device=xgb_device,
                            parameters=parameters,
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
                        predicted = predict_model(model_name, model, x_validation)
                        macro_f1, balanced_accuracy = validation_metrics(
                            y_validation, predicted
                        )
                        result = {
                            "fingerprint": fingerprint,
                            "model": model_name,
                            "ablation": ablation,
                            "fold": fold,
                            "candidate_id": identifier,
                            "parameters": parameters,
                            "validation_macro_f1": macro_f1,
                            "validation_balanced_accuracy": balanced_accuracy,
                            "train_seconds": time.perf_counter() - started,
                            "selected_best": False,
                        }
                        save_candidate_result(checkpoint, result)
                    combination_results.append(result)
                best = max(
                    combination_results,
                    key=lambda row: (
                        float(row["validation_macro_f1"]),
                        float(row["validation_balanced_accuracy"]),
                    ),
                )
                best["selected_best"] = True
                all_results.extend(combination_results)
                best_parameters = best["parameters"]
                best_model = build_model(
                    model_name,
                    seed=seed,
                    xgb_device=xgb_device,
                    parameters=best_parameters,
                )
                best_model = fit_model(
                    model_name,
                    best_model,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                )
                save_checkpoint(
                    best_model,
                    model_dir / f"{model_name}__{ablation}__fold_{fold}.joblib",
                    {
                        "fingerprint": fingerprint,
                        "model": model_name,
                        "ablation": ablation,
                        "fold": fold,
                        "parameters": best_parameters,
                        "validation_macro_f1": best["validation_macro_f1"],
                    },
                )
                for subset in ("validation", "test"):
                    x_values, expected = feature_matrix(subsets[subset], indices)
                    predicted = predict_model(model_name, best_model, x_values)
                    summary, per_class, confusion = evaluate_predictions(
                        model_name=model_name,
                        ablation=ablation,
                        fold=fold,
                        subset=subset,
                        expected=expected,
                        predicted=predicted,
                        train_seconds=float(best["train_seconds"]),
                        resumed=False,
                    )
                    summary["candidate_id"] = best["candidate_id"]
                    summary["parameters"] = json.dumps(best_parameters, sort_keys=True)
                    tuned_summary.append(summary)
                    tuned_per_class.extend(per_class)
                    tuned_confusion.extend(confusion)
    serializable = [
        {
            **row,
            "parameters": json.dumps(row["parameters"], sort_keys=True),
        }
        for row in all_results
    ]
    output_dir = repository_path(args.output_dir)
    _write_csv(output_dir / "classical_grid_search.csv", serializable)
    selected = [row for row in serializable if row["selected_best"]]
    _write_csv(output_dir / "classical_grid_search_best.csv", selected)
    _write_csv(output_dir / "classical_tuned_summary.csv", tuned_summary)
    _write_csv(output_dir / "classical_tuned_per_class.csv", tuned_per_class)
    _write_csv(output_dir / "classical_tuned_confusion.csv", tuned_confusion)
    write_grid_report(
        output_dir / "classical_grid_search.md", selected, tuned_summary
    )
    figure_dir = repository_path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_grid_scores(all_results, figure_dir / "grid_score_distributions.png")
    plot_best_by_fold(all_results, figure_dir / "best_validation_by_fold.png")
    plot_model_boxplot(all_results, figure_dir / "model_score_boxplot.png")
    plot_metric_comparison(tuned_summary, figure_dir / "tuned_metrics_comparison.png")
    plot_fold_macro_f1(tuned_summary, figure_dir / "tuned_macro_f1_by_fold.png")
    plot_per_class_f1(tuned_per_class, figure_dir / "tuned_per_class_f1.png")
    for model_name in MODEL_NAMES:
        plot_confusion(
            tuned_confusion,
            model_name,
            figure_dir / f"tuned_confusion_{model_name}.png",
        )
    print(f"Grid search concluído: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
