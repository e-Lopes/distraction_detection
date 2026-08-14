from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from fase_2.src.training.classical_baselines import (
    ABLATIONS,
    balanced_sample_weights,
    experiment_fingerprint,
    feature_matrix,
    fit_model,
    load_checkpoint,
    save_checkpoint,
)
from fase_2.src.training.dummy_baseline import FeatureWindow


def test_ablation_indices_are_disjoint_where_expected():
    facial = set(ABLATIONS["facial_only"])
    missingness = set(ABLATIONS["missingness_only"])
    assert facial.isdisjoint(missingness)
    assert facial | missingness == set(ABLATIONS["facial_plus_missingness"])


def test_feature_matrix_selects_only_requested_columns():
    window = FeatureWindow("video_01", 0, 3, 4, "alert", tuple(range(14)))
    values, labels = feature_matrix([window], ABLATIONS["missingness_only"])
    assert values.tolist() == [[10, 11, 12, 13]]
    assert labels.tolist() == ["alert"]


def test_balanced_weights_give_each_class_equal_total_weight():
    labels = ["alert", "alert", "alert", "fatigue", "distraction", "distraction"]
    weights = balanced_sample_weights(labels)
    totals = {
        label: float(weights[np.asarray(labels) == label].sum()) for label in set(labels)
    }
    assert len(set(round(value, 8) for value in totals.values())) == 1


def test_checkpoint_resumes_only_with_matching_fingerprint(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    fingerprint = experiment_fingerprint([source], {"seed": 42})
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], ["alert", "fatigue"])
    checkpoint = tmp_path / "model.joblib"
    save_checkpoint(model, checkpoint, {"fingerprint": fingerprint, "train_seconds": 1.5})
    loaded = load_checkpoint(checkpoint, fingerprint)
    assert loaded is not None
    assert loaded[0].predict([[0]]).tolist() == ["alert"]
    assert loaded[1]["train_seconds"] == 1.5
    assert load_checkpoint(checkpoint, "stale") is None


def test_classical_fit_rejects_unknown_balancing():
    model = DummyClassifier(strategy="most_frequent")
    with pytest.raises(ValueError, match="Balanceamento"):
        fit_model(
            "svm",
            model,
            np.asarray([[0.0], [1.0]]),
            np.asarray(["alert", "fatigue"]),
            np.asarray([[2.0]]),
            np.asarray(["alert"]),
            balancing="both",
        )
