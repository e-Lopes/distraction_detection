"""Adaptadores pequenos para paradigmas de classificacao temporal.

Imports opcionais de sktime ficam confinados a este modulo e so ocorrem quando um
estimador afetado e construido. O DTW dependente e local para evitar uma segunda
dependencia (tslearn) e para tornar custo e cache explicitos.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..data.windowing import build_windows
from ..features.temporal_window_features import extract_temporal_window_features
from .dummy_baseline import CLASSES, FeatureWindow


INSTALL_COMMAND = 'python -m pip install -e "fase_2[tsc]"'
SKTIME_PARADIGMS = {"shapelet", "transform", "ensemble"}


class OptionalDependencyMissing(RuntimeError):
    pass


class CostLimitExceeded(RuntimeError):
    pass


def build_temporal_feature_windows(series, labels_by_video, *, size_frames: int,
                                   stride_frames: int, minimum_proportion: float,
                                   groups: Sequence[str], thresholds: Mapping[str, float],
                                   fps_by_video: Mapping[str, float]) -> list[FeatureWindow]:
    windows = build_windows(labels_by_video, size_frames=size_frames, stride_frames=stride_frames,
                            behavior_classes=set(CLASSES), minimum_proportion=minimum_proportion)
    result = []
    for window in windows:
        if window.label == "mixed":
            continue
        rows = series[window.video_id][window.start_frame:window.end_frame + 1]
        values = extract_temporal_window_features(rows, groups=groups, thresholds=thresholds,
                                                  fps=fps_by_video[window.video_id])
        result.append(FeatureWindow(window.video_id, window.start_frame, window.end_frame,
                                    size_frames, window.label, values))
    return result


def dependency_available(paradigm: str) -> bool:
    return paradigm not in SKTIME_PARADIGMS or importlib.util.find_spec("sktime") is not None


def require_sktime() -> None:
    if not dependency_available("shapelet"):
        raise OptionalDependencyMissing(f"sktime ausente. Instale com: {INSTALL_COMMAND}")


@dataclass(frozen=True)
class DTWCost:
    train_windows: int
    evaluation_windows: int
    pairs: int
    cell_updates: int
    estimated_cache_bytes: int


def estimate_dtw_cost(train_windows: int, evaluation_windows: int, window: int,
                      channels: int = 5) -> DTWCost:
    pairs = int(train_windows) * int(evaluation_windows)
    return DTWCost(train_windows, evaluation_windows, pairs,
                   pairs * int(window) * int(window) * int(channels), pairs * 8)


def enforce_dtw_limit(cost: DTWCost, maximum_pairs: int, *, allow_expensive: bool = False) -> None:
    if cost.pairs > int(maximum_pairs) and not allow_expensive:
        raise CostLimitExceeded(
            f"DTW bloqueado: {cost.pairs:,} pares excedem {maximum_pairs:,}. "
            "Revise o custo e repita com --allow-expensive se estiver autorizado."
        )


def dependent_dtw(left: np.ndarray, right: np.ndarray, radius: int) -> float:
    """DTW dependente multivariado com banda Sakoe-Chiba fixa."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("DTW requer arrays [tempo, canais] com os mesmos canais")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("DTW recebeu NaN ou Inf")
    radius = max(int(radius), abs(len(left) - len(right)))
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, len(left) + 1):
        current = np.full(len(right) + 1, np.inf)
        for j in range(max(1, i - radius), min(len(right), i + radius) + 1):
            local = float(np.sum((left[i - 1] - right[j - 1]) ** 2))
            current[j] = local + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(np.sqrt(previous[-1]))


class DependentDTW1NN:
    def __init__(self, *, radius: int, cache_path: Path | None = None):
        self.radius = int(radius)
        self.cache_path = cache_path

    def fit(self, values: np.ndarray, labels: Sequence[str]) -> "DependentDTW1NN":
        self.values_ = np.asarray(values, dtype=float)
        self.labels_ = np.asarray(labels)
        if self.values_.ndim != 3:
            raise ValueError("1-NN DTW requer entrada [amostras, tempo, canais]")
        return self

    def _distance_matrix(self, values: np.ndarray) -> np.ndarray:
        if self.cache_path and self.cache_path.is_file():
            cached = np.load(self.cache_path)
            if cached.shape == (len(values), len(self.values_)):
                return cached
        distances = np.empty((len(values), len(self.values_)), dtype=np.float64)
        for i, query in enumerate(values):
            for j, reference in enumerate(self.values_):
                distances[i, j] = dependent_dtw(query, reference, self.radius)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp.npy")
            np.save(temporary, distances)
            temporary.replace(self.cache_path)
        return distances

    def predict(self, values: np.ndarray) -> np.ndarray:
        distances = self._distance_matrix(np.asarray(values, dtype=float))
        return self.labels_[np.argmin(distances, axis=1)]


class XGBoostLabelAdapter:
    def __init__(self, parameters: Mapping[str, object], *, seed: int, balancing: str):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**dict(parameters), random_state=seed)
        self.balancing = balancing

    def fit(self, values: np.ndarray, labels: Sequence[str]) -> "XGBoostLabelAdapter":
        encoded = np.asarray([CLASSES.index(label) for label in labels])
        fit_params = {}
        if self.balancing == "class_weights":
            counts = {label: list(labels).count(label) for label in set(labels)}
            fit_params["sample_weight"] = np.asarray(
                [len(labels) / (len(counts) * counts[label]) for label in labels])
        self.model.fit(values, encoded, **fit_params)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([CLASSES[int(value)] for value in self.model.predict(values)])


def feature_classifier(model: str, parameters: Mapping[str, object], *, seed: int,
                       balancing: str) -> BaseEstimator:
    class_weight = "balanced" if balancing == "class_weights" else None
    params = dict(parameters)
    if model == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
            **params, class_weight=class_weight, random_state=seed))])
    if model == "svm":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(
            **params, class_weight=class_weight, random_state=seed))])
    if model == "random_forest":
        return RandomForestClassifier(**params, class_weight=class_weight, random_state=seed)
    if model == "xgboost":
        return XGBoostLabelAdapter(params, seed=seed, balancing=balancing)
    raise ValueError(f"Modelo de features desconhecido: {model}")


class ShapeletRidgeAdapter:
    def __init__(self, parameters: Mapping[str, object], *, seed: int):
        require_sktime()
        try:
            from sktime.transformations.shapelet_transform import RandomShapeletTransform
        except ImportError:  # compatibilidade com namespace anterior a 1.0
            from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
        params = dict(parameters)
        self.transformer = RandomShapeletTransform(
            n_shapelet_samples=int(params["candidate_budget"]),
            max_shapelets=int(params["max_shapelets"]),
            min_shapelet_length=int(params["min_length"]),
            max_shapelet_length=int(params["max_length"]),
            n_jobs=int(params.get("n_jobs", 1)), random_state=seed,
        )
        self.classifier = RidgeClassifier(class_weight="balanced")

    def fit(self, values: np.ndarray, labels: Sequence[str]) -> "ShapeletRidgeAdapter":
        transformed = self.transformer.fit_transform(np.transpose(values, (0, 2, 1)), labels)
        self.classifier.fit(transformed, labels)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        self.last_transformed_ = np.asarray(
            self.transformer.transform(np.transpose(values, (0, 2, 1))))
        return self.classifier.predict(self.last_transformed_)

    def shapelet_metadata(self) -> list[dict[str, object]]:
        return [{"importance": item[0], "length": item[1], "start": item[2], "channel": item[3],
                 "training_index": item[4], "class": item[5]} for item in self.transformer.shapelets]

    def closest_external_examples(self, metadata: Sequence[object]) -> list[dict[str, object]]:
        result = []
        for shapelet_index in range(self.last_transformed_.shape[1]):
            closest = int(np.argmin(self.last_transformed_[:, shapelet_index]))
            item = metadata[closest]
            result.append({"shapelet_index": shapelet_index,
                           "distance": float(self.last_transformed_[closest, shapelet_index]),
                           "video_id": item.video_id, "start_frame": item.start_frame,
                           "end_frame": item.end_frame})
        return result


class MiniRocketRidgeAdapter:
    def __init__(self, parameters: Mapping[str, object], *, seed: int):
        require_sktime()
        try:
            from sktime.transformations.rocket import MiniRocketMultivariate
        except ImportError:  # compatibilidade com namespace anterior a 1.0
            from sktime.transformations.panel.rocket import MiniRocketMultivariate
        params = dict(parameters)
        self.transformer = MiniRocketMultivariate(
            num_kernels=int(params["num_kernels"]),
            max_dilations_per_kernel=int(params.get("max_dilations_per_kernel", 32)),
            n_jobs=int(params.get("n_jobs", 1)), random_state=seed,
        )
        self.classifier = RidgeClassifierCV(alphas=tuple(float(x) for x in params["alphas"]),
                                            class_weight="balanced")

    def fit(self, values: np.ndarray, labels: Sequence[str]) -> "MiniRocketRidgeAdapter":
        transformed = self.transformer.fit_transform(np.transpose(values, (0, 2, 1)))
        self.classifier.fit(transformed, labels)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.transformer.transform(np.transpose(values, (0, 2, 1))))


def write_shapelet_metadata(path: Path, adapter: ShapeletRidgeAdapter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapter.shapelet_metadata(), indent=2), encoding="utf-8")
