"""Impacto downstream dos indicadores G4.7 no SVM R0/60 congelado."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.config import load_yaml
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from ..training.classical_baselines import build_model, evaluate_predictions, fit_model, predict_model
from ..training.dummy_baseline import CLASSES
from ..training.g1_baselines import flatten_r0
from ..training.temporal_data import build_sequence_fold

BACKENDS = ("mp", "yolo", "yolo_filtered")
METRICS = ("ear", "mar", "pitch", "yaw", "roll")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def paired_to_series(frame: pd.DataFrame, backend: str) -> list[dict[str, str]]:
    if backend not in BACKENDS:
        raise ValueError(f"Backend inválido: {backend}")
    columns = [f"{backend}_{metric}" for metric in METRICS]
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Colunas ausentes para {backend}: {sorted(missing)}")
    detected_column = "mp_detected" if backend == "mp" else "yolo_detected"
    output: list[dict[str, str]] = []
    for expected_index, (_, source) in enumerate(frame.iterrows()):
        if int(source["frame_index"]) != expected_index:
            raise ValueError("Série G4.7 possui índices descontínuos")
        values = [pd.to_numeric(source[column], errors="coerce") for column in columns]
        valid = bool(int(source[detected_column])) and all(np.isfinite(value) for value in values)
        output.append(
            {
                "video_id": str(source["video_id"]),
                "frame_index": str(expected_index),
                "timestamp_seconds": str(source["timestamp_seconds"]),
                **{
                    metric: str(float(value)) if valid else ""
                    for metric, value in zip(METRICS, values, strict=True)
                },
                "face_detected": "1" if valid else "0",
                "operational_state": "valid" if valid else "face_missing",
            }
        )
    return output


def load_paired_series(input_dir: Path, backend: str) -> dict[str, list[dict[str, str]]]:
    series: dict[str, list[dict[str, str]]] = {}
    for path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if not {"video_id", "frame_index", f"{backend}_ear"} <= set(frame):
            continue
        video_ids = frame["video_id"].astype(str).unique()
        if len(video_ids) != 1 or video_ids[0] in series:
            raise ValueError(f"Arquivo G4.7 duplicado/multivídeo: {path}")
        series[video_ids[0]] = paired_to_series(frame, backend)
    if len(series) != 4:
        raise FileNotFoundError(f"Esperadas quatro séries pareadas em {input_dir}; encontradas {len(series)}")
    return series


def _metric_rows(
    *, backend: str, fold: int, subset: str, expected: np.ndarray, predicted: np.ndarray, elapsed: float
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    summary, per_class, confusion = evaluate_predictions(
        model_name="svm",
        ablation=backend,
        fold=fold,
        subset=subset,
        expected=expected,
        predicted=predicted,
        train_seconds=elapsed,
        resumed=False,
    )
    context = {
        "generation": "G47",
        "backend": backend,
        "window_size_frames": 60,
        "seed": 42,
    }
    summary.update(context)
    for row in (*per_class, *confusion):
        row.update(context)
    return summary, per_class, confusion


def evaluate_backend(
    *,
    series: dict[str, list[dict[str, str]]],
    labels: dict[str, list[str | None]],
    blocks: list[SplitBlock],
    backend: str,
    include_test: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    preprocessing = load_yaml("fase_2/configs/preprocessing/baseline_zero_fill.yaml")
    data_config = load_yaml("fase_2/configs/data/base.yaml")
    experiment = load_yaml("fase_2/configs/experiment/g1_baselines.yaml")
    summaries: list[dict[str, object]] = []
    classes: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    for fold in sorted({block.fold for block in blocks}):
        splits, _, _ = build_sequence_fold(
            series,
            labels,
            blocks,
            preprocessing,
            fold=fold,
            size_frames=60,
            stride_frames=int(data_config["windowing"]["stride_frames"]),
            minimum_proportion=float(data_config["windowing"]["minimum_target_proportion"]),
            representation="R0",
        )
        x_train, y_train = flatten_r0(splits["train"])
        x_validation, y_validation = flatten_r0(splits["validation"])
        model = build_model(
            "svm",
            seed=42,
            xgb_device="cpu",
            parameters=dict(experiment["parameters"]["svm"]),
        )
        started = time.perf_counter()
        fit_model(
            "svm", model, x_train, y_train, x_validation, y_validation, balancing="none"
        )
        elapsed = time.perf_counter() - started
        for subset in (("validation", "test") if include_test else ("validation",)):
            x_values, expected = flatten_r0(splits[subset])
            predicted = predict_model("svm", model, x_values)
            summary, per_class, confusion = _metric_rows(
                backend=backend,
                fold=fold,
                subset=subset,
                expected=expected,
                predicted=predicted,
                elapsed=elapsed,
            )
            summaries.append(summary)
            classes.extend(per_class)
            confusions.extend(confusion)
    return summaries, classes, confusions


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Sem linhas para {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("fase_2/outputs/metrics/G47/downstream"))
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backends = args.backend or list(BACKENDS)
    print(
        json.dumps(
            {
                "generation": "G47",
                "backends": backends,
                "folds": [1, 2, 3, 4],
                "window": 60,
                "seed": 42,
                "subsets": ["validation", "test"] if args.include_test else ["validation"],
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    block_rows = _read_csv(Path("fase_2/data/manifests/temporal_splits.csv"))
    blocks = [
        SplitBlock(
            int(row["fold"]), row["subset"], row["video_id"], int(row["start_frame"]), int(row["end_frame"])
        )
        for row in block_rows
    ]
    all_summary: list[dict[str, object]] = []
    all_class: list[dict[str, object]] = []
    all_confusion: list[dict[str, object]] = []
    for backend in backends:
        series = load_paired_series(args.input_dir, backend)
        labels = expand_behavior_labels(
            {video_id: len(rows) for video_id, rows in series.items()},
            _read_csv(Path("fase_2/data/manifests/annotation_frame_intervals.csv")),
        )
        summary, classes, confusion = evaluate_backend(
            series=series,
            labels=labels,
            blocks=blocks,
            backend=backend,
            include_test=args.include_test,
        )
        all_summary.extend(summary)
        all_class.extend(classes)
        all_confusion.extend(confusion)
    _write(args.output_dir / "g47_svm_runs.csv", all_summary)
    _write(args.output_dir / "g47_svm_per_class.csv", all_class)
    _write(args.output_dir / "g47_svm_confusion.csv", all_confusion)
    aggregate = (
        pd.DataFrame(all_summary)
        .groupby(["backend", "subset"], as_index=False)["macro_f1_all_classes"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "g47_svm_aggregate.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
