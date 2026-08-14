"""Agregação estatística de execuções independentes por seed."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np


_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def t_critical_95(degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        return float("nan")
    eligible = [key for key in _T_975 if key >= degrees_freedom]
    return _T_975[min(eligible)] if eligible else 1.96


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("Não é possível agregar uma lista vazia")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    margin = (
        t_critical_95(len(array) - 1) * standard_deviation / sqrt(len(array))
        if len(array) > 1
        else float("nan")
    )
    return {
        "n": len(array),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "ci95_lower": mean - margin if len(array) > 1 else float("nan"),
        "ci95_upper": mean + margin if len(array) > 1 else float("nan"),
    }
