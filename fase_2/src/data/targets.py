"""Taxonomias de alvo sem misturar comportamento e disponibilidade operacional."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


BEHAVIOR_CLASSES = ("alert", "fatigue", "distraction")
PERSONALIZED_BINARY_CLASSES = ("normal", "non_normal")
PERSONALIZED_BINARY_MAPPING = {
    "alert": "normal",
    "fatigue": "non_normal",
    "distraction": "non_normal",
}
NON_BEHAVIOR_LABELS = {
    "mixed",
    "operator_absent",
    "face_missing",
    "occlusion",
    "unavailable",
}


ATTENTION_CLASSES = ("attention", "distraction")
ATTENTION_MAPPING = {"alert": "attention", "fatigue": "distraction",
                     "distraction": "distraction"}


def target_classes(config):
    return tuple(config.get("target", {}).get("classes", BEHAVIOR_CLASSES))


def validate_target(config):
    target = config.get("target")
    if not target:
        return
    if (tuple(target.get("classes", ())) != ATTENTION_CLASSES
            or target.get("mapping") != ATTENTION_MAPPING):
        raise ValueError("Target binário exige attention/distraction e alert → attention; "
                         "fatigue + distraction → distraction.")
    if target.get("map_before_windowing") is not True:
        raise ValueError("O target deve ser aplicado antes do janelamento.")


def map_target_labels(labels_by_video, config):
    """Map source annotations before majority voting; never modify source files."""
    validate_target(config)
    if not config.get("target"):
        return labels_by_video
    def convert(label):
        if label is None or label == "" or label in NON_BEHAVIOR_LABELS:
            return None
        if label not in ATTENTION_MAPPING:
            raise ValueError(f"Rótulo de origem desconhecido: {label}")
        return ATTENTION_MAPPING[label]
    return {video: [convert(label) for label in labels]
            for video, labels in labels_by_video.items()}


def personalized_binary_label(label: str | None) -> str | None:
    """Converte um rótulo comportamental para o alvo binário personalizado.

    Estados operacionais e janelas ambíguas não são convertidos em ``non_normal``: eles
    permanecem fora da tarefa comportamental e devem ser tratados pelo gate de disponibilidade.
    """

    if label is None:
        return None
    normalized = str(label).strip().lower()
    if not normalized or normalized in NON_BEHAVIOR_LABELS:
        return None
    try:
        return PERSONALIZED_BINARY_MAPPING[normalized]
    except KeyError as error:
        raise ValueError(f"Rótulo comportamental desconhecido: {label}") from error


def map_personalized_binary_labels(labels: Iterable[str | None]) -> list[str | None]:
    return [personalized_binary_label(label) for label in labels]


def validate_personalized_target_config(config: Mapping[str, object]) -> None:
    """Valida a semântica congelada do target antes de qualquer treinamento."""

    if tuple(config.get("classes", ())) != PERSONALIZED_BINARY_CLASSES:
        raise ValueError(
            f"Ordem esperada das classes personalizadas: {PERSONALIZED_BINARY_CLASSES}"
        )
    configured_mapping = {
        str(source): str(target)
        for source, target in dict(config.get("behavior_mapping", {})).items()
    }
    if configured_mapping != PERSONALIZED_BINARY_MAPPING:
        raise ValueError(
            "Mapeamento esperado: alert->normal e fatigue/distraction->non_normal"
        )
    excluded = {str(value) for value in config.get("excluded_from_behavior_target", ())}
    if not NON_BEHAVIOR_LABELS.issubset(excluded):
        missing = sorted(NON_BEHAVIOR_LABELS - excluded)
        raise ValueError(f"Estados não comportamentais ausentes da exclusão: {missing}")
