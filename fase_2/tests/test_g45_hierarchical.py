import numpy as np
import pytest
from fase_2.src.training.g45_hierarchical import (
    compose_probabilities,
    level_labels,
    load_hierarchy,
    make_level_split,
    save_hierarchy,
)
from fase_2.src.training.temporal_data import SequenceMetadata, SequenceSplit
from fase_2.src.training.temporal_engine import class_weights


def _split(labels):
    labels = np.asarray(labels)
    return SequenceSplit(
        np.arange(len(labels) * 4, dtype=float).reshape(len(labels), 2, 2),
        labels,
        tuple(SequenceMetadata("v", i, i + 1, "x") for i in range(len(labels))),
    )


def test_level_mapping_and_level_2_train_only_filter():
    assert level_labels(np.array([0, 1, 2]), 1).tolist() == [0, 1, 1]
    result = make_level_split(_split([0, 1, 2, 0, 2, 1]), 2)
    assert result.labels.tolist() == [0, 1, 1, 0]
    assert [m.start_frame for m in result.metadata] == [1, 2, 4, 5]


def test_missing_required_binary_class_fails_explicitly():
    with pytest.raises(ValueError, match="sem classe"):
        make_level_split(_split([0, 0, 2]), 2)


def test_binary_weights_are_independent_n_over_2nc():
    assert np.allclose(class_weights(np.array([0, 0, 0, 1]), num_classes=2), [4 / 6, 2])


def test_probability_composition_canonical_and_finite():
    result = compose_probabilities(
        np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([[0.25, 0.75], [0.6, 0.4]])
    )
    assert np.allclose(result, [[0.8, 0.05, 0.15], [0.1, 0.54, 0.36]])
    assert np.all(result >= 0)
    assert np.allclose(result.sum(axis=1), 1)
    with pytest.raises(FloatingPointError):
        compose_probabilities(np.array([[np.nan, 1]]), np.array([[0.5, 0.5]]))


def test_hierarchy_reload_reproduces_predictions(tmp_path):
    payload = {"level_1": np.array([[0.7, 0.3]]), "level_2": np.array([[0.2, 0.8]])}
    path = tmp_path / "hierarchy.joblib"
    save_hierarchy(path, payload)
    restored = load_hierarchy(path)
    assert np.allclose(
        compose_probabilities(level_1=restored["level_1"], level_2=restored["level_2"]),
        [[0.7, 0.06, 0.24]],
    )
