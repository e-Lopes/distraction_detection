"""Contrato normalizado mínimo da G48A — smoke de frameworks faciais."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..features.g47_schema import FacialIndicators, compute_indicators


@dataclass(frozen=True)
class FaceLandmarkResult:
    video_id: str
    frame_id: int
    timestamp_ms: float
    extractor: str
    extractor_version: str
    face_detected: bool
    face_bbox: tuple[float, float, float, float] | None
    landmarks_available: bool
    ear_left: float
    ear_right: float
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    ear_valid: bool
    mar_valid: bool
    head_pose_valid: bool
    confidence: float
    failure_reason: str
    inference_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_result(
    *,
    video_id: str,
    frame_id: int,
    timestamp_ms: float,
    extractor: str,
    extractor_version: str,
    inference_time_ms: float,
    failure_reason: str,
) -> FaceLandmarkResult:
    return FaceLandmarkResult(
        video_id,
        frame_id,
        timestamp_ms,
        extractor,
        extractor_version,
        False,
        None,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        failure_reason,
        inference_time_ms,
    )


def result_from_landmarks(
    *,
    points: np.ndarray,
    width: int,
    height: int,
    video_id: str,
    frame_id: int,
    timestamp_ms: float,
    extractor: str,
    extractor_version: str,
    face_bbox: tuple[float, float, float, float] | None,
    confidence: float,
    inference_time_ms: float,
) -> FaceLandmarkResult:
    values: FacialIndicators = compute_indicators(points, width, height)
    ear_valid = math.isfinite(values.ear)
    mar_valid = math.isfinite(values.mar)
    pose_valid = all(math.isfinite(value) for value in (values.pitch, values.yaw, values.roll))
    failures = []
    if not ear_valid:
        failures.append("ear_invalid")
    if not mar_valid:
        failures.append("mar_invalid")
    if not pose_valid:
        failures.append("head_pose_invalid")
    return FaceLandmarkResult(
        video_id,
        frame_id,
        timestamp_ms,
        extractor,
        extractor_version,
        True,
        face_bbox,
        True,
        values.ear_left,
        values.ear_right,
        values.ear,
        values.mar,
        values.pitch,
        values.yaw,
        values.roll,
        ear_valid,
        mar_valid,
        pose_valid,
        confidence,
        ";".join(failures),
        inference_time_ms,
    )


def map_to_canonical(points: np.ndarray, indices: list[int] | tuple[int, ...]) -> np.ndarray:
    source = np.asarray(points, dtype=float)
    if source.ndim != 2 or source.shape[1] not in {2, 3}:
        raise ValueError("Landmarks fonte devem possuir shape (N, 2) ou (N, 3)")
    if len(indices) != 22 or min(indices) < 0 or max(indices) >= len(source):
        raise ValueError("Mapeamento canônico exige 22 índices válidos")
    selected = source[np.asarray(indices)].copy()
    if selected.shape[1] == 2:
        selected = np.column_stack((selected, np.ones(22, dtype=float)))
    return selected


def select_operator_face(
    boxes: np.ndarray,
    *,
    width: int,
    height: int,
    anchor: tuple[float, float] = (0.65, 0.25),
) -> int:
    """Seleciona a face mais próxima da posição esperada do operador dentro da ROI."""
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("Bounding boxes devem possuir shape (N, 4) não vazio")
    centers = np.column_stack(((values[:, 0] + values[:, 2]) / 2, (values[:, 1] + values[:, 3]) / 2))
    normalized = centers / np.asarray([width, height], dtype=float)
    plausible = (
        (normalized[:, 0] >= 0.50) & (normalized[:, 0] <= 0.90)
        & (normalized[:, 1] >= 0.08) & (normalized[:, 1] <= 0.55)
    )
    if not plausible.any():
        raise ValueError("Nenhuma face na região espacial plausível do operador")
    distances = np.linalg.norm(normalized - np.asarray(anchor), axis=1)
    distances[~plausible] = np.inf
    return int(np.argmin(distances))
