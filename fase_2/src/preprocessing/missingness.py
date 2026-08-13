"""Diagnóstico de falhas consecutivas na detecção facial."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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


def diagnose_file(path: Path, *, short_gap_max_frames: int) -> MissingnessSummary:
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
    return summarize_detection(
        next(iter(video_ids)), detected, short_gap_max_frames=short_gap_max_frames
    )


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
