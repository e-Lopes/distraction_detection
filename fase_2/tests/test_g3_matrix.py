from pathlib import Path

import pytest

from fase_2.src.training.g3_matrix import validate_inheritance


def _config():
    return {
        "configuration_id": "g3",
        "window_size_frames": 60,
        "models": ["tcn"],
        "seeds": [42],
        "training": {"balancing": "none", "max_epochs": 150},
        "parameters": {"tcn": {"channels": [32, 64]}},
    }


def test_g3_inheritance_accepts_identical_training_and_model_parameters():
    config = _config()
    source = {
        "window_size_frames": 60,
        "training": dict(config["training"]),
        "parameters": {"tcn": dict(config["parameters"]["tcn"])},
    }
    validate_inheritance(config, source)


def test_g3_inheritance_rejects_changed_budget():
    config = _config()
    source = {
        "window_size_frames": 60,
        "training": {"balancing": "none", "max_epochs": 149},
        "parameters": config["parameters"],
    }
    with pytest.raises(ValueError, match="Orcamento"):
        validate_inheritance(config, source)
