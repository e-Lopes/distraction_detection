"""Correção interpretável de indicadores faciais condicionada à pose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class PoseCorrection:
    polynomial: PolynomialFeatures
    scaler: StandardScaler
    estimator: HuberRegressor
    closed_ratio: float
    training_rows: int
    training_videos: tuple[str, ...]
    expected_lower: float
    expected_upper: float


def fit_pose_correction(
    rows: pd.DataFrame,
    *,
    closed_ratio: float = 0.70,
) -> PoseCorrection:
    """Ajusta EAR aberto esperado por pitch/yaw somente sobre treino Alert válido."""
    if not 0 < closed_ratio < 1:
        raise ValueError("closed_ratio deve estar em (0, 1)")
    required = {"video_id", "ear", "pitch", "yaw", "face_detected", "behavior_label"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    selected = rows.loc[
        rows["face_detected"].eq(1) & rows["behavior_label"].eq("alert"),
        ["video_id", "ear", "pitch", "yaw"],
    ].copy()
    for column in ("ear", "pitch", "yaw"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna()
    if len(selected) < 30:
        raise ValueError("Treino Alert insuficiente para correção de pose")
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    pose = polynomial.fit_transform(selected[["pitch", "yaw"]].to_numpy())
    scaler = StandardScaler().fit(pose)
    estimator = HuberRegressor(max_iter=500).fit(scaler.transform(pose), selected["ear"].to_numpy())
    return PoseCorrection(
        polynomial=polynomial,
        scaler=scaler,
        estimator=estimator,
        closed_ratio=closed_ratio,
        training_rows=len(selected),
        training_videos=tuple(sorted(selected["video_id"].unique())),
        expected_lower=float(selected["ear"].quantile(0.10)),
        expected_upper=float(selected["ear"].quantile(0.90)),
    )


def apply_pose_correction(rows: pd.DataFrame, correction: PoseCorrection) -> pd.DataFrame:
    result = rows.copy()
    numeric = result[["ear", "pitch", "yaw"]].apply(pd.to_numeric, errors="coerce")
    valid = result["face_detected"].eq(1) & numeric.notna().all(axis=1)
    expected = pd.Series(float("nan"), index=result.index, dtype=float)
    if valid.any():
        pose = correction.polynomial.transform(numeric.loc[valid, ["pitch", "yaw"]].to_numpy())
        predicted = correction.estimator.predict(correction.scaler.transform(pose))
        predicted = np.clip(predicted, correction.expected_lower, correction.expected_upper)
        expected.loc[valid] = predicted
    result["ear_expected_open"] = expected
    result["ear_pose_corrected"] = numeric["ear"] / expected
    result["eye_closed_calibrated"] = pd.Series(
        result["ear_pose_corrected"] < correction.closed_ratio,
        index=result.index,
        dtype="boolean",
    ).where(valid)
    return result


def rolling_perclos(
    rows: pd.DataFrame,
    *,
    window_seconds: float,
    minimum_coverage: float = 0.5,
    closed_column: str = "eye_closed_calibrated",
) -> pd.DataFrame:
    """Calcula PERCLOS causal por tempo, sem transformar missingness em fechamento."""
    if window_seconds <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("Janela/cobertura inválida")
    timestamps = pd.to_numeric(rows["timestamp_seconds"], errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps não monotônicos")
    closed = rows[closed_column].astype("boolean")
    indexed = pd.DataFrame(
        {"valid": closed.notna().astype(float), "closed": closed.fillna(False).astype(float)},
        index=pd.to_timedelta(timestamps, unit="s"),
    )
    window = f"{window_seconds}s"
    valid_count = indexed["valid"].rolling(window, closed="both").sum()
    closed_count = indexed["closed"].rolling(window, closed="both").sum()
    sample_count = pd.Series(1.0, index=indexed.index).rolling(window, closed="both").sum()
    coverage = valid_count / sample_count
    perclos = (100 * closed_count / valid_count.where(valid_count > 0)).where(
        coverage >= minimum_coverage
    )
    result = rows.copy()
    result[f"perclos_{int(window_seconds)}s"] = perclos.to_numpy()
    result[f"coverage_{int(window_seconds)}s"] = coverage.to_numpy()
    return result


def pose_signal_correlations(rows: pd.DataFrame, ear_column: str) -> dict[str, float]:
    numeric = rows[[ear_column, "pitch", "yaw"]].apply(pd.to_numeric, errors="coerce").dropna()
    return {
        pose: float(numeric[ear_column].corr(numeric[pose], method="spearman"))
        for pose in ("pitch", "yaw")
    }
