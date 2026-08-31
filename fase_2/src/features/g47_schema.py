"""Contrato geométrico compartilhado pela comparação G4.7.

Esquerda e direita são sempre anatômicas (da pessoa), nunca relativas à imagem.
O módulo não depende de MediaPipe ou Ultralytics e pode ser testado isoladamente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

LANDMARK_NAMES = (
    "left_eye_outer",
    "left_eye_upper_outer",
    "left_eye_upper_inner",
    "left_eye_inner",
    "left_eye_lower_inner",
    "left_eye_lower_outer",
    "right_eye_outer",
    "right_eye_upper_outer",
    "right_eye_upper_inner",
    "right_eye_inner",
    "right_eye_lower_inner",
    "right_eye_lower_outer",
    "mouth_left_corner",
    "mouth_upper_left",
    "mouth_upper_center",
    "mouth_upper_right",
    "mouth_right_corner",
    "mouth_lower_right",
    "mouth_lower_center",
    "mouth_lower_left",
    "nose_tip",
    "chin_center",
)

DEFAULT_FLIP_INDEX = (
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    4,
    5,
    16,
    15,
    14,
    13,
    12,
    19,
    18,
    17,
    20,
    21,
)

# Conversão do Face Mesh para a convenção anatômica acima.
MEDIAPIPE_INDICES = (
    263,
    387,
    385,
    362,
    380,
    373,
    33,
    160,
    158,
    133,
    153,
    144,
    291,
    312,
    13,
    82,
    61,
    87,
    14,
    317,
    1,
    152,
)

LEFT_EYE = np.arange(0, 6)
RIGHT_EYE = np.arange(6, 12)
MOUTH = np.arange(12, 20)
POSE_INDICES = np.asarray([20, 21, 0, 6, 12, 16])

# Modelo genérico na ordem nariz, queixo, olho E, olho D, boca E, boca D.
# Os sinais de X acompanham a convenção anatômica.
HEAD_MODEL_3D = np.asarray(
    [
        [0.000, 0.000, 0.000],
        [0.000, -0.064, -0.013],
        [0.043, 0.033, -0.026],
        [-0.043, 0.033, -0.026],
        [0.029, -0.029, -0.024],
        [-0.029, -0.029, -0.024],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class LandmarkSchema:
    names: tuple[str, ...]
    flip_index: tuple[int, ...]
    mediapipe_indices: tuple[int, ...]
    wflw_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FacialIndicators:
    ear: float
    ear_left: float
    ear_right: float
    mar: float
    pitch: float
    yaw: float
    roll: float


def validate_flip_index(flip_index: tuple[int, ...] | list[int], size: int = 22) -> tuple[int, ...]:
    """Valida permutação de flip e sua propriedade de involução."""
    values = tuple(int(value) for value in flip_index)
    if len(values) != size:
        raise ValueError(f"flip_idx deve conter {size} índices; recebeu {len(values)}")
    if sorted(values) != list(range(size)):
        raise ValueError("flip_idx deve ser uma permutação de 0..21")
    if any(values[values[index]] != index for index in range(size)):
        raise ValueError("flip_idx deve ser involutivo: flip_idx[flip_idx[i]] == i")
    return values


def load_schema(path: str | Path) -> LandmarkSchema:
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError("Schema de landmarks deve ser um objeto YAML")
    shape = payload.get("kpt_shape")
    if shape != [22, 3]:
        raise ValueError("kpt_shape deve ser exatamente [22, 3]")
    names = tuple(payload.get("names", ()))
    if names != LANDMARK_NAMES:
        raise ValueError("Ordem/nome dos 22 landmarks difere do contrato G4.7")
    flip_index = validate_flip_index(payload.get("flip_idx", ()))
    mappings = payload.get("mappings", {})
    if not isinstance(mappings, dict):
        raise TypeError("mappings deve ser um objeto")
    mediapipe = tuple(int(value) for value in mappings.get("mediapipe", ()))
    if mediapipe != MEDIAPIPE_INDICES:
        raise ValueError("Mapeamento MediaPipe difere do contrato validado")
    raw_wflw = mappings.get("wflw")
    wflw = tuple(int(value) for value in raw_wflw) if raw_wflw is not None else None
    if wflw is not None and (len(wflw) != 22 or len(set(wflw)) != 22):
        raise ValueError("Mapeamento WFLW deve conter 22 índices distintos")
    return LandmarkSchema(names, flip_index, mediapipe, wflw)


def flip_landmarks(landmarks: np.ndarray, flip_index: tuple[int, ...]) -> np.ndarray:
    """Espelha coordenadas normalizadas e troca os rótulos anatômicos."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape not in {(22, 2), (22, 3)}:
        raise ValueError("Landmarks devem ter shape (22, 2) ou (22, 3)")
    mapping = validate_flip_index(flip_index)
    mirrored = points.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored[np.asarray(mapping)]


def mediapipe_to_schema(landmarks: Any, width: int, height: int) -> np.ndarray:
    """Converte landmarks MediaPipe normalizados para pontos G4.7 em pixels."""
    if width <= 0 or height <= 0:
        raise ValueError("Dimensões da imagem devem ser positivas")
    output = np.empty((22, 3), dtype=float)
    for target, source in enumerate(MEDIAPIPE_INDICES):
        point = landmarks[source]
        output[target] = (float(point.x) * width, float(point.y) * height, 1.0)
    return output


def _visible(points: np.ndarray, indices: np.ndarray, minimum_confidence: float) -> bool:
    selected = points[indices]
    return bool(
        np.isfinite(selected[:, :2]).all()
        and (selected[:, 2] >= minimum_confidence).all()
    )


def compute_ear(points: np.ndarray, indices: np.ndarray, minimum_confidence: float = 0.5) -> float:
    points = np.asarray(points, dtype=float)
    if points.shape != (22, 3) or not _visible(points, indices, minimum_confidence):
        return math.nan
    eye = points[indices, :2]
    horizontal = np.linalg.norm(eye[0] - eye[3])
    if horizontal <= 1e-9:
        return math.nan
    vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    return float(vertical / (2.0 * horizontal))


def compute_mar(points: np.ndarray, minimum_confidence: float = 0.5) -> float:
    points = np.asarray(points, dtype=float)
    if points.shape != (22, 3) or not _visible(points, MOUTH, minimum_confidence):
        return math.nan
    mouth = points[MOUTH, :2]
    horizontal = np.linalg.norm(mouth[0] - mouth[4])
    if horizontal <= 1e-9:
        return math.nan
    upper = mouth[[1, 2, 3]].mean(axis=0)
    lower = mouth[[5, 6, 7]].mean(axis=0)
    return float(np.linalg.norm(upper - lower) / horizontal)


def camera_matrix(width: int, height: int) -> np.ndarray:
    focal_length = float(width)
    return np.asarray(
        [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )


def compute_head_pose(
    points: np.ndarray,
    width: int,
    height: int,
    minimum_confidence: float = 0.5,
) -> tuple[float, float, float]:
    """Retorna pitch, yaw e roll em graus ou NaNs quando a pose é inválida."""
    points = np.asarray(points, dtype=float)
    missing = (math.nan, math.nan, math.nan)
    if points.shape != (22, 3) or not _visible(points, POSE_INDICES, minimum_confidence):
        return missing
    success, rotation_vector, _ = cv2.solvePnP(
        HEAD_MODEL_3D,
        points[POSE_INDICES, :2].astype(np.float64),
        camera_matrix(width, height),
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return missing
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = (float(value) for value in angles)
    return pitch, yaw, roll


def compute_indicators(
    points: np.ndarray, width: int, height: int, minimum_confidence: float = 0.5
) -> FacialIndicators:
    left = compute_ear(points, LEFT_EYE, minimum_confidence)
    right = compute_ear(points, RIGHT_EYE, minimum_confidence)
    available = [value for value in (left, right) if np.isfinite(value)]
    ear = float(np.mean(available)) if available else math.nan
    mar = compute_mar(points, minimum_confidence)
    pitch, yaw, roll = compute_head_pose(points, width, height, minimum_confidence)
    return FacialIndicators(ear, left, right, mar, pitch, yaw, roll)


def normalized_mean_error(expected: np.ndarray, predicted: np.ndarray) -> float:
    """NME visível normalizado pela distância entre centros dos olhos."""
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if expected.shape != (22, 3) or predicted.shape != (22, 3):
        raise ValueError("NME exige arrays (22, 3)")
    visible = (expected[:, 2] > 0) & (predicted[:, 2] > 0)
    if not visible.any():
        return math.nan
    left_center = expected[LEFT_EYE, :2].mean(axis=0)
    right_center = expected[RIGHT_EYE, :2].mean(axis=0)
    scale = np.linalg.norm(left_center - right_center)
    if not np.isfinite(scale) or scale <= 1e-9:
        return math.nan
    errors = np.linalg.norm(expected[visible, :2] - predicted[visible, :2], axis=1)
    return float(errors.mean() / scale)
