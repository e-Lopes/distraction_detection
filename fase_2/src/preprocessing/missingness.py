"""Diagnóstico de falhas consecutivas na detecção facial."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..data.splits import SplitBlock, window_subset
from ..data.windowing import build_windows


@dataclass(frozen=True)
class MissingnessSummary:
    video_id: str
    total_frames: int
    detected_frames: int
    missing_frames: int
    missing_rate: float
    gap_count: int
    short_gap_count: int
    long_gap_count: int
    shortest_gap_frames: int
    median_gap_frames: float
    longest_gap_frames: int


@dataclass(frozen=True)
class GroupedMissingness:
    video_id: str
    label: str
    total_frames: int
    missing_frames: int
    missing_rate: float


@dataclass(frozen=True)
class WindowMissingness:
    fold: int | None
    subset: str
    video_id: str
    window_size_frames: int
    label: str
    num_windows: int
    total_window_frames: int
    missing_frames: int
    mean_missing_rate: float
    fully_observed_windows: int
    fully_missing_windows: int


def consecutive_false_lengths(values: Iterable[bool]) -> list[int]:
    """Retorna comprimentos de todas as sequências falsas, inclusive nas bordas."""
    gaps: list[int] = []
    current = 0
    for value in values:
        if value:
            if current:
                gaps.append(current)
                current = 0
        else:
            current += 1
    if current:
        gaps.append(current)
    return gaps


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def summarize_detection(
    video_id: str, detected: list[bool], *, short_gap_max_frames: int
) -> MissingnessSummary:
    if short_gap_max_frames < 0:
        raise ValueError("short_gap_max_frames não pode ser negativo")
    gaps = consecutive_false_lengths(detected)
    detected_frames = sum(detected)
    missing_frames = len(detected) - detected_frames
    return MissingnessSummary(
        video_id=video_id,
        total_frames=len(detected),
        detected_frames=detected_frames,
        missing_frames=missing_frames,
        missing_rate=missing_frames / len(detected) if detected else 0.0,
        gap_count=len(gaps),
        short_gap_count=sum(length <= short_gap_max_frames for length in gaps),
        long_gap_count=sum(length > short_gap_max_frames for length in gaps),
        shortest_gap_frames=min(gaps, default=0),
        median_gap_frames=_median(gaps),
        longest_gap_frames=max(gaps, default=0),
    )


def load_detection_series(path: Path) -> tuple[str, list[bool]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Série vazia: {path}")
    required = {"video_id", "frame_index", "face_detected", "ear", "mar", "pitch", "yaw", "roll"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing_columns)}")
    video_ids = {row["video_id"] for row in rows}
    if len(video_ids) != 1:
        raise ValueError(f"Esperado um video_id em {path}, encontrados: {sorted(video_ids)}")
    detected: list[bool] = []
    metrics = ("ear", "mar", "pitch", "yaw", "roll")
    previous_timestamp = float("-inf")
    for expected_index, row in enumerate(rows):
        if int(row["frame_index"]) != expected_index:
            raise ValueError(f"Índice descontínuo em {path}: linha {expected_index + 2}")
        timestamp = float(row["timestamp_seconds"])
        if timestamp < previous_timestamp:
            raise ValueError(f"Timestamp não monotônico em {path}: frame {expected_index}")
        previous_timestamp = timestamp
        face_detected = row["face_detected"] == "1"
        if face_detected and any(not row[field] for field in metrics):
            raise ValueError(f"Métrica ausente com face_detected=1 no frame {expected_index}")
        if not face_detected and any(row[field] for field in metrics):
            raise ValueError(f"Métrica preenchida com face_detected=0 no frame {expected_index}")
        detected.append(face_detected)
    return next(iter(video_ids)), detected


def diagnose_file(path: Path, *, short_gap_max_frames: int) -> MissingnessSummary:
    video_id, detected = load_detection_series(path)
    return summarize_detection(video_id, detected, short_gap_max_frames=short_gap_max_frames)


def expand_behavior_labels(
    video_lengths: dict[str, int], intervals: Iterable[dict[str, str]]
) -> dict[str, list[str | None]]:
    labels = {video_id: [None] * length for video_id, length in video_lengths.items()}
    for interval in intervals:
        video_id = interval["video_id"]
        if video_id not in labels:
            raise ValueError(f"Anotação referencia vídeo desconhecido: {video_id}")
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        if start < 0 or end < start or end >= len(labels[video_id]):
            raise ValueError(f"Intervalo inválido para {video_id}: {start}-{end}")
        value = interval["behavior_label"] or None
        for index in range(start, end + 1):
            if labels[video_id][index] is not None:
                raise ValueError(f"Sobreposição comportamental em {video_id}, frame {index}")
            labels[video_id][index] = value
    return labels


def summarize_by_class(
    labels_by_video: dict[str, list[str | None]],
    detected_by_video: dict[str, list[bool]],
    *,
    behavior_classes: set[str],
) -> list[GroupedMissingness]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for video_id, labels in labels_by_video.items():
        detected = detected_by_video[video_id]
        if len(labels) != len(detected):
            raise ValueError(f"Comprimentos divergentes em {video_id}")
        for label, face_detected in zip(labels, detected):
            if label in behavior_classes:
                counts[(video_id, label)][0] += 1
                counts[(video_id, label)][1] += int(not face_detected)
                counts[("dataset", label)][0] += 1
                counts[("dataset", label)][1] += int(not face_detected)
    return [
        GroupedMissingness(
            video_id=video_id,
            label=label,
            total_frames=values[0],
            missing_frames=values[1],
            missing_rate=values[1] / values[0],
        )
        for (video_id, label), values in sorted(counts.items())
    ]


def summarize_windows(
    labels_by_video: dict[str, list[str | None]],
    detected_by_video: dict[str, list[bool]],
    *,
    sizes_frames: Iterable[int],
    stride_frames: int,
    behavior_classes: set[str],
    minimum_proportion: float,
    split_blocks: list[SplitBlock] | None = None,
) -> tuple[list[WindowMissingness], list[WindowMissingness]]:
    prefixes: dict[str, list[int]] = {}
    for video_id, detected in detected_by_video.items():
        prefix = [0]
        for value in detected:
            prefix.append(prefix[-1] + int(not value))
        prefixes[video_id] = prefix

    per_video: dict[tuple[object, ...], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    per_fold: dict[tuple[object, ...], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    folds = sorted({block.fold for block in split_blocks or []})
    for size in sizes_frames:
        windows = build_windows(
            labels_by_video,
            size_frames=size,
            stride_frames=stride_frames,
            behavior_classes=behavior_classes,
            minimum_proportion=minimum_proportion,
        )
        for window in windows:
            prefix = prefixes[window.video_id]
            missing = prefix[window.end_frame + 1] - prefix[window.start_frame]
            video_key = (None, "all", window.video_id, size, window.label)
            _accumulate_window(per_video[video_key], size, missing)
            dataset_key = (None, "all", "dataset", size, window.label)
            _accumulate_window(per_video[dataset_key], size, missing)
            for fold in folds:
                subset = window_subset(
                    video_id=window.video_id,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    blocks=split_blocks or [],
                    fold=fold,
                )
                if subset:
                    fold_key = (fold, subset, window.video_id, size, window.label)
                    _accumulate_window(per_fold[fold_key], size, missing)
                    fold_dataset_key = (fold, subset, "dataset", size, window.label)
                    _accumulate_window(per_fold[fold_dataset_key], size, missing)
    return _window_rows(per_video), _window_rows(per_fold)


def _accumulate_window(values: list[int], size: int, missing: int) -> None:
    values[0] += 1
    values[1] += missing
    values[2] += int(missing == 0)
    values[3] += int(missing == size)


def _window_rows(groups: dict[tuple[object, ...], list[int]]) -> list[WindowMissingness]:
    rows: list[WindowMissingness] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            int(item[0][0] or 0),
            str(item[0][1]),
            str(item[0][2]),
            int(item[0][3]),
            str(item[0][4]),
        ),
    )
    for key, values in ordered_groups:
        fold, subset, video_id, size, label = key
        num_windows, missing, observed, fully_missing = values
        total = num_windows * int(size)
        rows.append(
            WindowMissingness(
                fold=fold if isinstance(fold, int) else None,
                subset=str(subset),
                video_id=str(video_id),
                window_size_frames=int(size),
                label=str(label),
                num_windows=num_windows,
                total_window_frames=total,
                missing_frames=missing,
                mean_missing_rate=missing / total,
                fully_observed_windows=observed,
                fully_missing_windows=fully_missing,
            )
        )
    return rows


def write_dataclass_csv(path: Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def append_grouped_report(
    path: Path,
    class_rows: Iterable[GroupedMissingness],
    window_rows: Iterable[WindowMissingness],
) -> None:
    dataset_classes = [row for row in class_rows if row.video_id == "dataset"]
    dataset_windows = [
        row
        for row in window_rows
        if row.video_id == "dataset" and row.label != "mixed"
    ]
    lines = [
        "",
        "## Missingness por classe anotada",
        "",
        "| Classe | Frames | Missing |",
        "|---|---:|---:|",
    ]
    for row in dataset_classes:
        lines.append(
            f"| {row.label} | {row.total_frames} | {100 * row.missing_rate:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Missingness médio por janela",
            "",
            "| Janela | Classe | Janelas | Missing médio |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in dataset_windows:
        lines.append(
            f"| {row.window_size_frames} | {row.label} | {row.num_windows} | "
            f"{100 * row.mean_missing_rate:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Os CSVs complementares preservam os recortes por vídeo e por fold. "
            "Janelas `mixed` permanecem nos arquivos para auditoria, mas não integram "
            "o treinamento principal.",
        ]
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def write_missingness_report(
    summaries: list[MissingnessSummary],
    *,
    csv_path: Path,
    markdown_path: Path,
    short_gap_max_frames: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(MissingnessSummary.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    total = sum(summary.total_frames for summary in summaries)
    missing = sum(summary.missing_frames for summary in summaries)
    lines = [
        "# Diagnóstico de missingness facial",
        "",
        f"- Frames auditados: {total:,}".replace(",", "."),
        f"- Frames sem face: {missing:,} ({100 * missing / total:.2f}%)".replace(",", "."),
        "- Missing é definido exclusivamente por `face_detected=0`; "
        "zeros não são usados como sentinela.",
        f"- Gap curto: até {short_gap_max_frames} frames, definido antes dos folds de teste.",
        "",
        "| Vídeo | Frames | Detecção | Missing | Gaps | Curtos | Longos | Maior gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.video_id} | {summary.total_frames} | "
            f"{100 * (1 - summary.missing_rate):.2f}% | {100 * summary.missing_rate:.2f}% | "
            f"{summary.gap_count} | {summary.short_gap_count} | {summary.long_gap_count} | "
            f"{summary.longest_gap_frames} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
