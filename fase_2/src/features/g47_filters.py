"""Filtros causais e diagnósticos de estabilidade da G4.7."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class LandmarkFilter:
    def update(self, points: np.ndarray, timestamp_seconds: float) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class EMAFilter(LandmarkFilter):
    alpha: float = 0.4
    _state: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha da EMA deve estar em (0, 1]")

    def reset(self) -> None:
        self._state = None

    def update(self, points: np.ndarray, timestamp_seconds: float) -> np.ndarray:
        del timestamp_seconds
        values = np.asarray(points, dtype=float)
        if values.shape != (22, 3):
            raise ValueError("Filtro espera landmarks (22, 3)")
        if self._state is None:
            self._state = values[:, :2].copy()
        else:
            visible = values[:, 2] > 0
            self._state[visible] = (
                self.alpha * values[visible, :2]
                + (1.0 - self.alpha) * self._state[visible]
            )
        output = values.copy()
        output[:, :2] = self._state
        return output


def _smoothing_factor(cutoff: float, delta_seconds: float) -> float:
    if cutoff <= 0 or delta_seconds <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / delta_seconds)


@dataclass
class OneEuroFilter(LandmarkFilter):
    minimum_cutoff: float = 1.0
    beta: float = 0.02
    derivative_cutoff: float = 1.0
    _value: np.ndarray | None = None
    _derivative: np.ndarray | None = None
    _timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_cutoff <= 0 or self.derivative_cutoff <= 0 or self.beta < 0:
            raise ValueError("Parâmetros inválidos do One Euro Filter")

    def reset(self) -> None:
        self._value = None
        self._derivative = None
        self._timestamp = None

    def update(self, points: np.ndarray, timestamp_seconds: float) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape != (22, 3):
            raise ValueError("Filtro espera landmarks (22, 3)")
        coordinates = values[:, :2]
        if self._value is None or self._timestamp is None or timestamp_seconds <= self._timestamp:
            self._value = coordinates.copy()
            self._derivative = np.zeros_like(coordinates)
            self._timestamp = float(timestamp_seconds)
            return values.copy()
        delta = float(timestamp_seconds - self._timestamp)
        visible = values[:, 2] > 0
        derivative = (coordinates - self._value) / delta
        derivative_alpha = _smoothing_factor(self.derivative_cutoff, delta)
        self._derivative[visible] = (
            derivative_alpha * derivative[visible]
            + (1.0 - derivative_alpha) * self._derivative[visible]
        )
        cutoff = self.minimum_cutoff + self.beta * np.abs(self._derivative)
        alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * delta))
        self._value[visible] = (
            alpha[visible] * coordinates[visible]
            + (1.0 - alpha[visible]) * self._value[visible]
        )
        self._timestamp = float(timestamp_seconds)
        output = values.copy()
        output[:, :2] = self._value
        return output


def make_filter(name: str) -> LandmarkFilter | None:
    if name == "none":
        return None
    if name == "ema":
        return EMAFilter()
    if name == "one_euro":
        return OneEuroFilter()
    raise ValueError(f"Filtro desconhecido: {name}")


def bland_altman(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return {"count": int(valid.sum()), "bias": math.nan, "lower": math.nan, "upper": math.nan}
    differences = second[valid] - first[valid]
    bias = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    return {
        "count": int(valid.sum()),
        "bias": bias,
        "lower": bias - 1.96 * deviation,
        "upper": bias + 1.96 * deviation,
    }


def temporal_jitter(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    if not valid_pairs.any():
        return math.nan
    return float(np.median(np.abs(np.diff(values)[valid_pairs])))
