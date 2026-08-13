"""Dummy baseline sobre janelas faciais com splits temporais congelados."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock, window_subset
from ..data.windowing import build_windows
from ..preprocessing.missingness import expand_behavior_labels


CLASSES = ("alert", "fatigue", "distraction")
METRICS = ("ear", "mar", "pitch", "yaw", "roll")
FEATURE_NAMES = tuple(
    [name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")]
    + ["face_detected_rate", "missing_ratio", "gap_count", "longest_gap"]
)


@dataclass(frozen=True)
class FeatureWindow:
    video_id: str
    start_frame: int
    end_frame: int
    size_frames: int
    label: str
    values: tuple[float, ...]


def _gap_statistics(detected: Sequence[bool]) -> tuple[int, int]:
    gaps: list[int] = []
    current = 0
    for value in detected:
        if value:
            if current:
                gaps.append(current)
                current = 0
        else:
            current += 1
    if current:
        gaps.append(current)
    return len(gaps), max(gaps, default=0)


def aggregate_window(rows: Sequence[dict[str, str]]) -> tuple[float, ...]:
    """Zero-fill ocorre somente aqui, após missingness estar explicitamente codificada."""
    features: list[float] = []
    for metric in METRICS:
        values = np.array([float(row[metric]) if row[metric] else 0.0 for row in rows])
        features.extend((float(values.mean()), float(values.std())))
    detected = [row["face_detected"] == "1" for row in rows]
    detection_rate = sum(detected) / len(detected)
    gap_count, longest_gap = _gap_statistics(detected)
    features.extend((detection_rate, 1 - detection_rate, float(gap_count), float(longest_gap)))
    return tuple(features)


def load_series(input_dir: Path) -> dict[str, list[dict[str, str]]]:
    series: dict[str, list[dict[str, str]]] = {}
    for path in sorted(input_dir.glob("video_*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Série vazia: {path}")
        video_id = rows[0]["video_id"]
        if any(int(row["frame_index"]) != index for index, row in enumerate(rows)):
            raise ValueError(f"Índices descontínuos: {path}")
        series[video_id] = rows
    if not series:
        raise FileNotFoundError(f"Nenhuma série encontrada em {input_dir}")
    return series


def build_feature_windows(
    series: dict[str, list[dict[str, str]]],
    labels_by_video: dict[str, list[str | None]],
    *,
    size_frames: int,
    stride_frames: int,
    minimum_proportion: float,
) -> list[FeatureWindow]:
    windows = build_windows(
        labels_by_video,
        size_frames=size_frames,
        stride_frames=stride_frames,
        behavior_classes=set(CLASSES),
        minimum_proportion=minimum_proportion,
    )
    features: list[FeatureWindow] = []
    for window in windows:
        if window.label == "mixed":
            continue
        rows = series[window.video_id][window.start_frame : window.end_frame + 1]
        features.append(
            FeatureWindow(
                video_id=window.video_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                size_frames=size_frames,
                label=window.label,
                values=aggregate_window(rows),
            )
        )
    return features


def evaluate_dummy(
    windows: Sequence[FeatureWindow], blocks: Sequence[SplitBlock]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    folds = sorted({block.fold for block in blocks})
    sizes = sorted({window.size_frames for window in windows})
    for size in sizes:
        sized = [window for window in windows if window.size_frames == size]
        for fold in folds:
            by_subset: dict[str, list[FeatureWindow]] = {
                name: [] for name in ("train", "validation", "test")
            }
            for window in sized:
                subset = window_subset(
                    video_id=window.video_id,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    blocks=blocks,
                    fold=fold,
                )
                if subset:
                    by_subset[subset].append(window)
            train = by_subset["train"]
            if not train:
                raise ValueError(f"Fold {fold}, janela {size}: treino vazio")
            model = DummyClassifier(strategy="most_frequent")
            model.fit(
                np.array([window.values for window in train]),
                [window.label for window in train],
            )
            for subset in ("validation", "test"):
                samples = by_subset[subset]
                if not samples:
                    raise ValueError(f"Fold {fold}, janela {size}: {subset} vazio")
                x_values = np.array([window.values for window in samples])
                expected = np.array([window.label for window in samples])
                predicted = model.predict(x_values)
                present = [label for label in CLASSES if label in set(expected)]
                summary_rows.append(
                    {
                        "fold": fold,
                        "window_size_frames": size,
                        "subset": subset,
                        "num_windows": len(samples),
                        "train_majority_class": model.classes_[int(np.argmax(model.class_prior_))],
                        "accuracy": accuracy_score(expected, predicted),
                        "balanced_accuracy": balanced_accuracy_score(expected, predicted),
                        "macro_f1_present_classes": f1_score(
                            expected, predicted, labels=present, average="macro", zero_division=0
                        ),
                        "macro_f1_all_classes": f1_score(
                            expected, predicted, labels=CLASSES, average="macro", zero_division=0
                        ),
                        "present_classes": json.dumps(present),
                        "absent_classes": json.dumps([c for c in CLASSES if c not in present]),
                    }
                )
                precision, recall, f1_values, support = precision_recall_fscore_support(
                    expected, predicted, labels=CLASSES, zero_division=0
                )
                for index, label in enumerate(CLASSES):
                    class_rows.append(
                        {
                            "fold": fold,
                            "window_size_frames": size,
                            "subset": subset,
                            "label": label,
                            "precision": precision[index] if support[index] else "",
                            "recall": recall[index] if support[index] else "",
                            "f1": f1_values[index] if support[index] else "",
                            "support": int(support[index]),
                        }
                    )
                counts = Counter(zip(expected, predicted))
                for actual in CLASSES:
                    for prediction in CLASSES:
                        confusion_rows.append(
                            {
                                "fold": fold,
                                "window_size_frames": size,
                                "subset": subset,
                                "actual": actual,
                                "predicted": prediction,
                                "count": counts[(actual, prediction)],
                            }
                        )
    return summary_rows, class_rows, confusion_rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Nenhuma linha para gravar em {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(materialized[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(materialized)


def write_summary_markdown(path: Path, rows: Sequence[dict[str, object]]) -> None:
    test_rows = [row for row in rows if row["subset"] == "test"]
    sizes = sorted({int(row["window_size_frames"]) for row in test_rows})
    lines = [
        "# Dummy baseline",
        "",
        "Baseline `most_frequent` ajustado separadamente em cada fold. Em todos os casos, "
        "a classe majoritária de treino foi `alert`.",
        "",
        "| Janela | Acurácia média | Balanced accuracy média | Macro F1 (3 classes) |",
        "|---:|---:|---:|---:|",
    ]
    for size in sizes:
        selected = [row for row in test_rows if int(row["window_size_frames"]) == size]
        lines.append(
            f"| {size} | {np.mean([float(row['accuracy']) for row in selected]):.4f} | "
            f"{np.mean([float(row['balanced_accuracy']) for row in selected]):.4f} | "
            f"{np.mean([float(row['macro_f1_all_classes']) for row in selected]):.4f} |"
        )
    lines.extend(
        [
            "",
            "A acurácia elevada reflete o desbalanceamento e não desempenho útil nas classes "
            "minoritárias. Os folds 2 e 4 não possuem fadiga no teste para janelas de 60 e "
            "150 frames; métricas por classe sem suporte ficam vazias no CSV correspondente.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fase_2/configs/data/base.yaml")
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    parser.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml(args.config)
    windowing = config["windowing"]
    series = load_series(repository_path(args.input_dir))
    with repository_path(args.frame_intervals).open(newline="", encoding="utf-8") as stream:
        frame_intervals = list(csv.DictReader(stream))
    labels = expand_behavior_labels(
        {video_id: len(rows) for video_id, rows in series.items()}, frame_intervals
    )
    all_windows: list[FeatureWindow] = []
    for size in windowing["sizes_frames"]:
        all_windows.extend(
            build_feature_windows(
                series,
                labels,
                size_frames=int(size),
                stride_frames=int(windowing["stride_frames"]),
                minimum_proportion=float(windowing["minimum_target_proportion"]),
            )
        )
    with repository_path(args.splits).open(newline="", encoding="utf-8") as stream:
        blocks = [
            SplitBlock(
                fold=int(row["fold"]),
                subset=row["subset"],
                video_id=row["video_id"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
            )
            for row in csv.DictReader(stream)
        ]
    summary, per_class, confusion = evaluate_dummy(all_windows, blocks)
    output_dir = repository_path(args.output_dir)
    _write_csv(output_dir / "dummy_baseline_summary.csv", summary)
    _write_csv(output_dir / "dummy_baseline_per_class.csv", per_class)
    _write_csv(output_dir / "dummy_baseline_confusion.csv", confusion)
    write_summary_markdown(output_dir / "dummy_baseline.md", summary)
    print(f"Dummy baseline concluído: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
