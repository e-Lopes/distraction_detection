"""Rotulagem e construção de janelas temporais."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRecord:
    video_id: str
    start_frame: int
    end_frame: int
    size_frames: int
    label: str
    dominant_proportion: float
    labeled_frames: int


def label_window(
    labels: Sequence[str | None],
    *,
    behavior_classes: set[str],
    minimum_proportion: float = 0.60,
) -> tuple[str, float, int]:
    """Rotula pela maioria dos frames comportamentais rotulados."""
    if not 0 < minimum_proportion <= 1:
        raise ValueError("minimum_proportion deve estar em (0, 1]")
    counts = Counter(label for label in labels if label in behavior_classes)
    labeled_frames = sum(counts.values())
    if labeled_frames == 0:
        return "mixed", 0.0, 0
    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]
    proportion = highest / labeled_frames
    if len(winners) != 1 or proportion < minimum_proportion:
        return "mixed", proportion, labeled_frames
    return winners[0], proportion, labeled_frames


def build_windows(
    labels_by_video: Mapping[str, Sequence[str | None]],
    *,
    size_frames: int,
    stride_frames: int,
    behavior_classes: set[str],
    minimum_proportion: float = 0.60,
) -> list[WindowRecord]:
    """Cria somente janelas completas e isoladas dentro de cada vídeo."""
    if size_frames <= 0 or stride_frames <= 0:
        raise ValueError("size_frames e stride_frames devem ser positivos")
    records: list[WindowRecord] = []
    for video_id, labels in labels_by_video.items():
        for start in range(0, len(labels) - size_frames + 1, stride_frames):
            end = start + size_frames - 1
            label, proportion, labeled = label_window(
                labels[start : end + 1],
                behavior_classes=behavior_classes,
                minimum_proportion=minimum_proportion,
            )
            records.append(
                WindowRecord(video_id, start, end, size_frames, label, proportion, labeled)
            )
    return records
