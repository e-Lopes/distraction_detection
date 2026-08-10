"""Leitura e validação das anotações temporais manuais."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class AnnotationInterval:
    video_id: str
    start_second: int
    end_second: int
    behavior_label: str | None
    operational_state: str
    source_label: str
    annotation_version: str
    source_row: int


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    video_id: str | None = None
    source_row: int | None = None


REQUIRED_COLUMNS = {
    "video_id",
    "start_time",
    "end_time",
    "behavior_label",
    "operational_state",
    "source_label",
    "annotation_version",
}


def parse_time(value: str) -> int:
    """Converte MM:SS ou HH:MM:SS em segundos."""
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Tempo inválido: {value!r}")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Tempo inválido: {value!r}") from exc
    if any(number < 0 for number in numbers) or numbers[-1] >= 60:
        raise ValueError(f"Tempo inválido: {value!r}")
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    hours, minutes, seconds = numbers
    if minutes >= 60:
        raise ValueError(f"Tempo inválido: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def load_annotations(path: str | Path) -> tuple[list[AnnotationInterval], list[ValidationIssue]]:
    """Lê CSV sem corrigir silenciosamente linhas inválidas."""
    intervals: list[AnnotationInterval] = []
    issues: list[ValidationIssue] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            return [], [
                ValidationIssue(
                    "error", "missing_columns", f"Colunas obrigatórias ausentes: {missing_columns}"
                )
            ]
        for row_number, row in enumerate(reader, start=2):
            missing = [
                name
                for name in REQUIRED_COLUMNS - {"behavior_label"}
                if not (row.get(name) or "").strip()
            ]
            if missing:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_required_value",
                        f"Campos obrigatórios vazios: {sorted(missing)}",
                        (row.get("video_id") or None),
                        row_number,
                    )
                )
                continue
            try:
                start_second = parse_time(row["start_time"])
                end_second = parse_time(row["end_time"])
            except ValueError as exc:
                issues.append(
                    ValidationIssue(
                        "error", "invalid_time", str(exc), row.get("video_id"), row_number
                    )
                )
                continue
            intervals.append(
                AnnotationInterval(
                    video_id=row["video_id"].strip(),
                    start_second=start_second,
                    end_second=end_second,
                    behavior_label=row["behavior_label"].strip() or None,
                    operational_state=row["operational_state"].strip(),
                    source_label=row["source_label"].strip(),
                    annotation_version=row["annotation_version"].strip(),
                    source_row=row_number,
                )
            )
    return intervals, issues


def validate_annotations(
    intervals: Iterable[AnnotationInterval],
    *,
    expected_video_ids: set[str],
    behavior_classes: set[str],
    operational_states: set[str],
    video_duration_seconds: Mapping[str, float] | None = None,
) -> list[ValidationIssue]:
    """Valida taxonomia, intervalos, continuidade e limites conhecidos."""
    issues: list[ValidationIssue] = []
    by_video: dict[str, list[AnnotationInterval]] = defaultdict(list)
    intervals = list(intervals)

    for interval in intervals:
        row = interval.source_row
        if interval.video_id not in expected_video_ids:
            issues.append(
                ValidationIssue(
                    "error", "unknown_video", "video_id não configurado", interval.video_id, row
                )
            )
        if interval.start_second > interval.end_second:
            issues.append(
                ValidationIssue(
                    "error", "inverted_interval", "Início maior que o fim", interval.video_id, row
                )
            )
        if interval.behavior_label and interval.behavior_label not in behavior_classes:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_behavior_class",
                    f"Classe inválida: {interval.behavior_label}",
                    interval.video_id,
                    row,
                )
            )
        if interval.operational_state not in operational_states:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_operational_state",
                    f"Estado operacional inválido: {interval.operational_state}",
                    interval.video_id,
                    row,
                )
            )
        if interval.operational_state == "valid" and interval.behavior_label is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_behavior",
                    "Estado valid exige classe comportamental",
                    interval.video_id,
                    row,
                )
            )
        if interval.operational_state != "valid" and interval.behavior_label is not None:
            issues.append(
                ValidationIssue(
                    "error",
                    "behavior_during_operational_failure",
                    "Condição operacional não deve receber target comportamental automaticamente",
                    interval.video_id,
                    row,
                )
            )
        duration = (video_duration_seconds or {}).get(interval.video_id)
        if duration is not None:
            if interval.start_second >= duration:
                issues.append(
                    ValidationIssue(
                        "error",
                        "outside_video",
                        f"Início {interval.start_second}s fora da duração {duration:.3f}s",
                        interval.video_id,
                        row,
                    )
                )
            elif interval.end_second >= duration:
                severity = "warning" if interval.end_second - duration < 1 else "error"
                code = "end_clipped_to_video" if severity == "warning" else "outside_video"
                issues.append(
                    ValidationIssue(
                        severity,
                        code,
                        f"Fim inclusivo {interval.end_second}s excede duração {duration:.3f}s",
                        interval.video_id,
                        row,
                    )
                )
        by_video[interval.video_id].append(interval)

    for video_id in sorted(expected_video_ids):
        video_intervals = sorted(by_video.get(video_id, []), key=lambda item: item.start_second)
        if not video_intervals:
            issues.append(
                ValidationIssue(
                    "error", "missing_video_annotations", "Vídeo sem anotações", video_id
                )
            )
            continue
        if video_intervals[0].start_second > 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "unlabeled_start",
                    f"Trecho 0..{video_intervals[0].start_second - 1}s sem rótulo",
                    video_id,
                )
            )
        for previous, current in pairwise(video_intervals):
            if current.start_second <= previous.end_second:
                issues.append(
                    ValidationIssue(
                        "error",
                        "overlap",
                        f"Sobreposição entre linhas {previous.source_row} e {current.source_row}",
                        video_id,
                        current.source_row,
                    )
                )
            elif current.start_second > previous.end_second + 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "unlabeled_gap",
                        f"Trecho {previous.end_second + 1}..{current.start_second - 1}s sem rótulo",
                        video_id,
                        current.source_row,
                    )
                )
        duration = (video_duration_seconds or {}).get(video_id)
        if duration is not None:
            last_covered_second = video_intervals[-1].end_second
            final_second = max(0, math.ceil(duration) - 1)
            if last_covered_second < final_second:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "unlabeled_tail",
                        f"Trecho {last_covered_second + 1}..{final_second}s sem rótulo",
                        video_id,
                    )
                )
    return issues


def annotation_summary(intervals: Iterable[AnnotationInterval]) -> list[dict[str, object]]:
    """Agrega segundos inclusivos por vídeo e classe/estado."""
    totals: Counter[tuple[str, str]] = Counter()
    for interval in intervals:
        label = interval.behavior_label or interval.operational_state
        totals[(interval.video_id, label)] += interval.end_second - interval.start_second + 1
    return [
        {"video_id": video_id, "label": label, "duration_seconds": duration}
        for (video_id, label), duration in sorted(totals.items())
    ]


def issue_rows(issues: Iterable[ValidationIssue]) -> list[dict[str, object]]:
    return [asdict(issue) for issue in issues]


def intervals_to_frames(
    intervals: Iterable[AnnotationInterval],
    video_metadata: Mapping[str, Mapping[str, float | int]],
) -> list[dict[str, object]]:
    """Converte segundos inclusivos para frames usando FPS médio e clipping no vídeo."""
    rows: list[dict[str, object]] = []
    for interval in intervals:
        metadata = video_metadata[interval.video_id]
        fps = float(metadata["fps"])
        num_frames = int(metadata["num_frames"])
        start_frame = math.ceil(interval.start_second * fps)
        end_exclusive = math.ceil((interval.end_second + 1) * fps)
        end_frame = min(num_frames - 1, end_exclusive - 1)
        if start_frame > end_frame:
            continue
        rows.append(
            {
                "video_id": interval.video_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "behavior_label": interval.behavior_label or "",
                "operational_state": interval.operational_state,
                "annotation_version": interval.annotation_version,
                "conversion_fps": fps,
                "conversion_note": "average_container_fps; inclusive_seconds; clipped_to_video",
            }
        )
    return rows
