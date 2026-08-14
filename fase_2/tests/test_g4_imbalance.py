import numpy as np

from fase_2.src.training.dummy_baseline import CLASSES
from fase_2.src.training.temporal_data import (
    PHYSICAL_RANGES,
    SequenceMetadata,
    light_augment_training_windows,
)
from fase_2.src.training.temporal_engine import (
    class_weights,
    sample_weights,
    weighted_sample_indices,
)


FEATURES = ("ear", "mar", "pitch", "yaw", "roll")
CONFIG = {
    "minority_classes": ["fatigue", "distraction"],
    "copies_per_minority_window": 1,
    "scale_range": [0.98, 1.02],
    "jitter_std": {"ear": 0.005, "mar": 0.005, "pitch": 0.5, "yaw": 0.5, "roll": 0.5},
    "masking_probability": 0.5,
    "max_mask_frames": 3,
}


def _inputs():
    labels = [CLASSES.index("alert"), CLASSES.index("fatigue"), CLASSES.index("distraction")]
    values = [np.tile(np.asarray([0.3, 0.4, 1.0, 2.0, 3.0], dtype=np.float32), (8, 1)) for _ in labels]
    metadata = [SequenceMetadata(f"v{i}", 0, 7, CLASSES[label]) for i, label in enumerate(labels)]
    return values, labels, metadata


def test_class_weight_formula_uses_training_counts_only():
    labels = np.asarray([0] * 6 + [1] * 2 + [2], dtype=int)
    np.testing.assert_allclose(class_weights(labels), [9 / 18, 9 / 6, 9 / 3])
    np.testing.assert_allclose(sample_weights(labels), class_weights(labels)[labels])


def test_weighted_sampling_is_deterministic_and_materializes_n_train():
    labels = np.asarray([0] * 20 + [1] * 3 + [2] * 2, dtype=int)
    first = weighted_sample_indices(labels, seed=42)
    second = weighted_sample_indices(labels, seed=42)
    assert len(first) == len(labels)
    np.testing.assert_array_equal(first, second)
    assert set(labels[first]) == {0, 1, 2}


def test_augmentation_duplicates_only_minority_and_is_deterministic():
    values, labels, metadata = _inputs(); audit = []
    first = light_augment_training_windows(values, labels, metadata, FEATURES, CONFIG, seed=42, audit_records=audit)
    second = light_augment_training_windows(values, labels, metadata, FEATURES, CONFIG, seed=42)
    assert len(first[0]) == 5
    assert first[1] == labels + [CLASSES.index("fatigue"), CLASSES.index("distraction")]
    np.testing.assert_array_equal(first[0][0], values[0])
    np.testing.assert_array_equal(first[0][-1], second[0][-1])
    assert len(audit) == 2 and {row["class"] for row in audit} == {"fatigue", "distraction"}


def test_augmentation_respects_physical_ranges_and_logs_before_after():
    values, labels, metadata = _inputs(); audit = []
    augmented, _, _ = light_augment_training_windows(values, labels, metadata, FEATURES, CONFIG, seed=42, audit_records=audit)
    for sample in augmented:
        for index, feature in enumerate(FEATURES):
            low, high = PHYSICAL_RANGES[feature]
            assert np.all(sample[:, index] >= low) and np.all(sample[:, index] <= high)
    assert all(row["before_values"] and row["after_values"] for row in audit)
    assert all(row["before_sha256"] != row["after_sha256"] for row in audit)


def test_augmentation_rejects_non_r0_features():
    values, labels, metadata = _inputs()
    try:
        light_augment_training_windows(values, labels, metadata, FEATURES + ("face_detected",), CONFIG, seed=42)
    except ValueError as error:
        assert "R0" in str(error)
    else:
        raise AssertionError("feature nao fisica deveria ser rejeitada")
