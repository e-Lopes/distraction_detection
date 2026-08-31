import pytest

from fase_2.src.data.targets import (
    PERSONALIZED_BINARY_CLASSES,
    map_personalized_binary_labels,
    personalized_binary_label,
    validate_personalized_target_config,
)


def test_personalized_binary_mapping_groups_behavior_without_losing_semantics():
    assert personalized_binary_label("alert") == "normal"
    assert personalized_binary_label("fatigue") == "non_normal"
    assert personalized_binary_label("distraction") == "non_normal"
    assert map_personalized_binary_labels(["alert", "fatigue", "distraction"]) == [
        "normal",
        "non_normal",
        "non_normal",
    ]


@pytest.mark.parametrize(
    "label",
    [None, "", "mixed", "operator_absent", "face_missing", "occlusion", "unavailable"],
)
def test_operational_and_ambiguous_states_never_become_non_normal(label):
    assert personalized_binary_label(label) is None


def test_unknown_behavior_fails_loudly():
    with pytest.raises(ValueError, match="desconhecido"):
        personalized_binary_label("sleeping")


def test_target_config_requires_frozen_class_order_mapping_and_gate_exclusions():
    config = {
        "classes": list(PERSONALIZED_BINARY_CLASSES),
        "behavior_mapping": {
            "alert": "normal",
            "fatigue": "non_normal",
            "distraction": "non_normal",
        },
        "excluded_from_behavior_target": [
            "mixed",
            "operator_absent",
            "face_missing",
            "occlusion",
            "unavailable",
        ],
    }
    validate_personalized_target_config(config)
    with pytest.raises(ValueError, match="Mapeamento esperado"):
        validate_personalized_target_config(
            {**config, "behavior_mapping": {"alert": "normal"}}
        )
    with pytest.raises(ValueError, match="ausentes da exclusão"):
        validate_personalized_target_config(
            {**config, "excluded_from_behavior_target": ["mixed"]}
        )
