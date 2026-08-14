from pathlib import Path

import numpy as np
import pytest
import torch

from fase_2.src.training.temporal_data import SequenceMetadata, SequenceSplit
from fase_2.src.training.temporal_engine import (
    load_training_checkpoint,
    predict_split,
    train_model,
)


def _split(repeats: int) -> SequenceSplit:
    labels = np.tile(np.arange(3), repeats)
    values = np.zeros((len(labels), 6, 8), dtype=np.float32)
    for index, label in enumerate(labels):
        values[index, :, label] = 1.0
    metadata = tuple(
        SequenceMetadata("video_01", index * 6, index * 6 + 5, str(label))
        for index, label in enumerate(labels)
    )
    return SequenceSplit(values, labels, metadata)


def test_engine_saves_best_last_periodic_and_records_epochs(tmp_path: Path):
    splits = {"train": _split(4), "validation": _split(2), "test": _split(2)}
    training = {
        "max_epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 2,
        "minimum_delta": 0.0,
        "checkpoint_every_epochs": 1,
        "gradient_clip_norm": 1.0,
        "amp": False,
        "deterministic": True,
        "num_workers": 0,
        "balancing": "none",
    }
    result = train_model(
        model_name="lstm",
        model_parameters={
            "hidden_dim": 8,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "pooling": "last",
        },
        splits=splits,
        training=training,
        seed=42,
        fold=1,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        fingerprint="test",
        resume=True,
        feature_names=("ear", "mar", "pitch", "yaw", "roll", "face_detected", "was_interpolated", "missing_duration_so_far"),
    )
    assert 1 <= result.best_epoch <= result.stopping_epoch == 2
    assert len(result.history) == 2
    assert (tmp_path / "best_macro_f1.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "epoch_001.pt").exists()
    checkpoint = load_training_checkpoint(
        tmp_path / "last.pt", fingerprint="test", device=torch.device("cpu")
    )
    assert checkpoint is not None
    assert checkpoint["seed"] == 42
    assert checkpoint["feature_names"] == [
        "ear", "mar", "pitch", "yaw", "roll", "face_detected", "was_interpolated", "missing_duration_so_far"
    ]
    expected, predicted, probabilities = predict_split(
        result.model,
        splits["test"],
        batch_size=6,
        device=torch.device("cpu"),
        amp=False,
    )
    assert expected.shape == predicted.shape == (6,)
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.peak_gpu_memory_bytes == 0

    training["max_epochs"] = 3
    resumed = train_model(
        model_name="lstm",
        model_parameters={
            "hidden_dim": 8,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "pooling": "last",
        },
        splits=splits,
        training=training,
        seed=42,
        fold=1,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        fingerprint="test",
        resume=True,
    )
    assert resumed.resumed is True
    assert resumed.stopping_epoch == 3
    assert len(resumed.history) == 3


def test_engine_rejects_non_finite_values(tmp_path: Path):
    splits = {"train": _split(2), "validation": _split(1), "test": _split(1)}
    splits["train"].values[0, 0, 0] = np.nan
    training = {
        "max_epochs": 1,
        "batch_size": 3,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "patience": 1,
        "minimum_delta": 0.0,
        "checkpoint_every_epochs": 0,
        "gradient_clip_norm": 1.0,
        "amp": False,
        "deterministic": True,
        "num_workers": 0,
        "balancing": "none",
    }
    with pytest.raises(FloatingPointError, match="NaN ou Inf"):
        train_model(
            model_name="lstm",
            model_parameters={"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
            splits=splits,
            training=training,
            seed=42,
            fold=1,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
            fingerprint="nan-test",
            resume=False,
        )
