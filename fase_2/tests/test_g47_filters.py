import numpy as np
import pytest

from fase_2.src.features.g47_filters import EMAFilter, OneEuroFilter, bland_altman, temporal_jitter


def _points(value: float) -> np.ndarray:
    points = np.full((22, 3), value, dtype=float)
    points[:, 2] = 1
    return points


def test_ema_is_causal_and_resettable():
    filter_ = EMAFilter(alpha=0.5)
    first = filter_.update(_points(0), 0)
    second = filter_.update(_points(2), 0.1)
    assert first[0, 0] == 0
    assert second[0, 0] == 1
    filter_.reset()
    assert filter_.update(_points(2), 0.2)[0, 0] == 2


def test_one_euro_preserves_shape_confidence_and_finiteness():
    filter_ = OneEuroFilter()
    first = filter_.update(_points(0), 0)
    second = filter_.update(_points(1), 0.1)
    assert second.shape == (22, 3)
    assert np.isfinite(second).all()
    np.testing.assert_array_equal(first[:, 2], second[:, 2])


def test_agreement_and_jitter_ignore_missing_values():
    first = np.array([1.0, 2.0, np.nan, 4.0])
    second = np.array([2.0, 3.0, 5.0, 5.0])
    result = bland_altman(first, second)
    assert result["count"] == 3
    assert result["bias"] == pytest.approx(1)
    assert temporal_jitter(first) == pytest.approx(1)
