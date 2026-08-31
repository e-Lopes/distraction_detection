"""Executa a classificacao hierarquica validation-only congelada na G4.5C."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from .classical_baselines import evaluate_predictions, experiment_fingerprint
from .dummy_baseline import CLASSES, load_series
from .temporal_data import SequenceSplit, build_sequence_fold, limit_sequence_split
from .temporal_engine import class_weights, predict_split, save_run_result, train_model


def level_labels(labels: np.ndarray, level: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if np.any((labels < 0) | (labels >= len(CLASSES))):
        raise ValueError("Rotulo multiclasses invalido")
    if level == 1:
        return (labels != CLASSES.index("alert")).astype(int)
    if level == 2:
        if np.any(labels == CLASSES.index("alert")):
            raise ValueError("Nivel 2 aceita somente exemplos Non-Alert")
        return (labels == CLASSES.index("distraction")).astype(int)
    raise ValueError("level deve ser 1 ou 2")


def make_level_split(
    split: SequenceSplit, level: int, *, require_all_classes: bool = True
) -> SequenceSplit:
    mask = np.ones(len(split.labels), dtype=bool)
    if level == 2:
        mask = split.labels != CLASSES.index("alert")
    labels = level_labels(split.labels[mask], level)
    # Prova operacional de que cada estimador possui as duas classes requeridas.
    if require_all_classes:
        class_weights(labels, num_classes=2)
    indices = np.flatnonzero(mask)
    return SequenceSplit(split.values[indices], labels, tuple(split.metadata[i] for i in indices))


def compose_probabilities(level_1: np.ndarray, level_2: np.ndarray) -> np.ndarray:
    first = np.asarray(level_1, dtype=float)
    second = np.asarray(level_2, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
        raise ValueError("Probabilidades dos niveis devem possuir shape (N, 2)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError("Probabilidades hierarquicas contem NaN ou Inf")
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("Probabilidades hierarquicas devem ser nao negativas")
    first = first / first.sum(axis=1, keepdims=True)
    second = second / second.sum(axis=1, keepdims=True)
    result = np.column_stack((first[:, 0], first[:, 1] * second[:, 0], first[:, 1] * second[:, 1]))
    result /= result.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-7):
        raise FloatingPointError("Composicao hierarquica invalida")
    return result


def save_hierarchy(path: Path, payload: object) -> None:
    """Persistencia atomica usada pelos dois niveis classicos."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def load_hierarchy(path: Path) -> object:
    import joblib

    return joblib.load(path)


def flatten_sequence(split: SequenceSplit) -> tuple[np.ndarray, np.ndarray]:
    return split.values.reshape(len(split.values), -1), np.asarray(split.labels, dtype=int)


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


def _predict_lstm(
    model: torch.nn.Module, split: SequenceSplit, config: Mapping[str, object], device: torch.device
) -> np.ndarray:
    _, _, probabilities = predict_split(
        model, split, batch_size=int(config["batch_size"]), device=device, amp=bool(config["amp"])
    )
    return probabilities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config", default="fase_2/configs/experiment/g45_hierarchical_w60.yaml"
    )
    parser.add_argument("--model", choices=("svm", "lstm"), action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples-per-subset", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    config_path = repository_path(args.experiment_config)
    config = load_yaml(config_path)
    models = args.model or list(config["models"])
    folds = args.fold or [1, 2, 3, 4]
    suffix = f"__smoke_{args.max_samples_per_subset}" if args.max_samples_per_subset else ""
    matrix = [
        {
            "run_id": f"G45__hierarchical_b_w60__{model}__r0__w60__fold_{fold}__seed_42{suffix}",
            "model": model,
            "fold": fold,
            "estimators": 2,
        }
        for model in models
        for fold in folds
    ]
    if len(matrix) > args.max_runs:
        raise ValueError(f"Matriz com {len(matrix)} pipelines excede --max-runs={args.max_runs}")
    print(
        json.dumps(
            {
                "planned_pipeline_count": len(matrix),
                "planned_estimator_count": 2 * len(matrix),
                "runs": matrix,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponivel")
    device = torch.device(device_name)
    data_path = root / "fase_2/configs/data/base.yaml"
    preprocessing_path = repository_path(config["preprocessing_config"])
    interval_path = root / "fase_2/data/manifests/annotation_frame_intervals.csv"
    split_path = root / "fase_2/data/manifests/temporal_splits.csv"
    input_dir = root / "fase_2/data/interim/legacy_extraction"
    data = load_yaml(data_path)
    preprocessing = load_yaml(preprocessing_path)
    blocks = [
        SplitBlock(
            int(r["fold"]), r["subset"], r["video_id"], int(r["start_frame"]), int(r["end_frame"])
        )
        for r in _read_csv(split_path)
    ]
    series = load_series(input_dir)
    labels = expand_behavior_labels(
        {key: len(value) for key, value in series.items()}, _read_csv(interval_path)
    )
    training = dict(config["training"])
    if args.max_epochs is not None:
        training["max_epochs"] = args.max_epochs
        training["patience"] = max(args.max_epochs, 1)
        training["checkpoint_every_epochs"] = 0
    summaries: list[dict[str, object]] = []
    per_classes: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    for fold in folds:
        splits, _, _ = build_sequence_fold(
            series,
            labels,
            blocks,
            preprocessing,
            fold=fold,
            size_frames=60,
            stride_frames=int(data["windowing"]["stride_frames"]),
            minimum_proportion=float(data["windowing"]["minimum_target_proportion"]),
            representation="R0",
        )
        if args.max_samples_per_subset:
            splits = {
                name: limit_sequence_split(value, args.max_samples_per_subset, seed=42 + fold)
                for name, value in splits.items()
            }
        level_splits = {
            level: {
                name: make_level_split(value, level, require_all_classes=name == "train")
                for name, value in splits.items()
            }
            for level in (1, 2)
        }
        fingerprint = experiment_fingerprint(
            [
                config_path,
                data_path,
                preprocessing_path,
                interval_path,
                split_path,
                input_dir / "extraction_manifest.csv",
            ],
            {
                "generation": "G45",
                "intervention": "hierarchical",
                "fold": fold,
                "seed": 42,
                "max_samples": args.max_samples_per_subset,
                "max_epochs": args.max_epochs,
            },
        )
        for model_name in models:
            run_id = f"G45__hierarchical_b_w60__{model_name}__r0__w60__fold_{fold}__seed_42{suffix}"
            print(
                f"\n=== {model_name.upper()} hierárquico | fold {fold}/{max(folds)} ===",
                flush=True,
            )
            started = time.perf_counter()
            histories: dict[str, object] = {}
            if model_name == "svm":
                estimators = []
                for level in (1, 2):
                    level_name = "Alert × Non-Alert" if level == 1 else "Fatigue × Distraction"
                    print(f"  Treinando nível {level}: {level_name}...", flush=True)
                    x_train, y_train = flatten_sequence(level_splits[level]["train"])
                    weights = class_weights(y_train, num_classes=2)
                    estimator = Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                SVC(
                                    C=1,
                                    kernel="linear",
                                    probability=True,
                                    random_state=42,
                                    class_weight={0: float(weights[0]), 1: float(weights[1])},
                                ),
                            ),
                        ]
                    )
                    estimator.fit(x_train, y_train)
                    estimators.append(estimator)
                restored_estimators = []
                for level, estimator in enumerate(estimators, start=1):
                    checkpoint = root / f"fase_2/outputs/models/G45/{run_id}__level_{level}.joblib"
                    save_hierarchy(
                        checkpoint,
                        {"fingerprint": fingerprint, "level": level, "model": estimator},
                    )
                    restored_estimators.append(load_hierarchy(checkpoint)["model"])
                estimators = restored_estimators
                x_val, _ = flatten_sequence(splits["validation"])
                p1 = estimators[0].predict_proba(x_val)
                p2 = estimators[1].predict_proba(x_val)
            else:
                trained = []
                for level in (1, 2):
                    level_fingerprint = experiment_fingerprint(
                        [
                            config_path,
                            data_path,
                            preprocessing_path,
                            interval_path,
                            split_path,
                            input_dir / "extraction_manifest.csv",
                        ],
                        {"parent": fingerprint, "level": level},
                    )
                    result = train_model(
                        model_name="lstm",
                        model_parameters=config["parameters"]["lstm"],
                        splits=level_splits[level],
                        training=training,
                        seed=42,
                        fold=fold,
                        device=device,
                        checkpoint_dir=root / f"fase_2/outputs/models/G45/{run_id}__level_{level}",
                        fingerprint=level_fingerprint,
                        resume=not args.no_resume,
                        feature_names=("ear", "mar", "pitch", "yaw", "roll"),
                        num_classes=2,
                        progress_label=(
                            f"LSTM | fold {fold} | nível {level} "
                            f"({'Alert×Non-Alert' if level == 1 else 'Fatigue×Distraction'})"
                        ),
                    )
                    trained.append(result)
                    histories[f"level_{level}"] = result.history
                p1 = _predict_lstm(trained[0].model, splits["validation"], training, device)
                p2 = _predict_lstm(trained[1].model, splits["validation"], training, device)
            probabilities = compose_probabilities(p1, p2)
            predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
            expected = np.asarray(CLASSES)[splits["validation"].labels]
            seconds = time.perf_counter() - started
            summary, rows, confusion = evaluate_predictions(
                model_name=model_name,
                ablation="hierarchical_r0",
                fold=fold,
                subset="validation",
                expected=expected,
                predicted=predicted,
                train_seconds=seconds,
                resumed=False,
            )
            context = {
                "run_id": run_id,
                "generation": "G45",
                "configuration_id": "hierarchical_b_w60",
                "representation": "R0",
                "window_size_frames": 60,
                "seed": 42,
                "estimators": 2,
            }
            summary.update(context)
            [row.update(context) for row in rows]
            [row.update(context) for row in confusion]
            predictions = [
                {
                    "video_id": meta.video_id,
                    "start_frame": meta.start_frame,
                    "end_frame": meta.end_frame,
                    "actual": actual,
                    "predicted": pred,
                    **{
                        f"prob_{name}": float(probabilities[i, j]) for j, name in enumerate(CLASSES)
                    },
                }
                for i, (meta, actual, pred) in enumerate(
                    zip(splits["validation"].metadata, expected, predicted, strict=True)
                )
            ]
            _write_csv(
                root / f"fase_2/outputs/predictions/G45/{run_id}__validation.csv", predictions
            )
            save_run_result(
                root / f"fase_2/outputs/logs/G45/{run_id}.json",
                {
                    "completed": True,
                    "fingerprint": fingerprint,
                    "run_id": run_id,
                    "model": model_name,
                    "fold": fold,
                    "seed": 42,
                    "level_2_train_only_non_alert": True,
                    "summary": summary,
                    "history": histories,
                },
            )
            summaries.append(summary)
            per_classes.extend(rows)
            confusions.extend(confusion)
            print(
                f"  ✓ fold {fold} concluído | Macro F1 "
                f"{float(summary['macro_f1_all_classes']):.4f} | "
                f"Balanced accuracy {float(summary['balanced_accuracy']):.4f} | "
                f"{seconds:.1f}s",
                flush=True,
            )
    _write_csv(root / "fase_2/outputs/metrics/G45/g45c_runs.csv", summaries)
    _write_csv(root / "fase_2/outputs/metrics/G45/g45c_per_class.csv", per_classes)
    _write_csv(root / "fase_2/outputs/metrics/G45/g45c_confusion.csv", confusions)
    print(f"G4.5C concluida: {len(summaries)} pipelines hierarquicos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
