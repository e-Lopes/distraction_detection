from __future__ import annotations

import numpy as np
import pytest

from fase_2.src.training.tsc_adapters import (
    CostLimitExceeded,
    DependentDTW1NN,
    dependent_dtw,
    enforce_dtw_limit,
    estimate_dtw_cost,
    feature_classifier,
)


def test_dependent_multivariate_dtw_and_one_nn():
    alert = np.zeros((4, 2))
    fatigue = np.ones((4, 2))
    assert dependent_dtw(alert, alert, radius=1) == 0
    assert dependent_dtw(alert, fatigue, radius=1) > 0

    model = DependentDTW1NN(radius=1).fit(np.asarray([alert, fatigue]), ["alert", "fatigue"])
    assert model.predict(np.asarray([fatigue])).tolist() == ["fatigue"]


def test_dtw_cost_guard_requires_explicit_authorization():
    cost = estimate_dtw_cost(100, 50, window=60, channels=5)
    assert cost.pairs == 5000
    assert cost.cell_updates == 90_000_000
    with pytest.raises(CostLimitExceeded, match="--allow-expensive"):
        enforce_dtw_limit(cost, maximum_pairs=4999)
    enforce_dtw_limit(cost, maximum_pairs=4999, allow_expensive=True)


def test_synthetic_feature_adapter_to_metric():
    values = np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    labels = np.asarray(["alert", "alert", "fatigue", "fatigue"])
    model = feature_classifier("logistic_regression", {"max_iter": 50}, seed=42,
                               balancing="class_weights")
    model.fit(values, labels)
    assert np.mean(model.predict(values) == labels) == 1.0
