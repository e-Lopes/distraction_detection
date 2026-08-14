import pytest

from fase_2.src.training.stability import descriptive_statistics


def test_descriptive_statistics_include_sample_std_and_t_interval():
    result = descriptive_statistics([1, 2, 3, 4, 5])
    assert result["n"] == 5
    assert result["mean"] == 3
    assert result["median"] == 3
    assert result["minimum"] == 1
    assert result["maximum"] == 5
    assert result["standard_deviation"] == pytest.approx(1.58113883)
    assert result["ci95_lower"] == pytest.approx(1.0368, abs=1e-3)
    assert result["ci95_upper"] == pytest.approx(4.9632, abs=1e-3)


def test_descriptive_statistics_reject_empty_input():
    with pytest.raises(ValueError, match="vazia"):
        descriptive_statistics([])
