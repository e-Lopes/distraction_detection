"""Contratos de proveniência e splits subject-independent para dados públicos."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

REQUIRED_FIELDS = {
    "dataset_id",
    "subject_id",
    "sample_id",
    "relative_path",
    "task",
    "label",
    "license",
}


def validate_public_manifest(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Manifesto público vazio")
    seen = set()
    for row in materialized:
        missing = REQUIRED_FIELDS.difference(row)
        if missing or any(not row[field].strip() for field in REQUIRED_FIELDS):
            raise ValueError(f"Manifesto público incompleto: {sorted(missing)}")
        identity = (row["dataset_id"], row["sample_id"])
        if identity in seen:
            raise ValueError(f"Amostra pública duplicada: {identity}")
        seen.add(identity)
    return materialized


def subject_independent_split(
    rows: Iterable[Mapping[str, str]], *, validation_subjects: set[str]
) -> dict[str, list[dict[str, str]]]:
    materialized = validate_public_manifest(rows)
    known = {row["subject_id"] for row in materialized}
    if not validation_subjects or not validation_subjects < known:
        raise ValueError("Validação deve conter um subconjunto próprio de sujeitos")
    result = {"train": [], "validation": []}
    for row in materialized:
        subset = "validation" if row["subject_id"] in validation_subjects else "train"
        result[subset].append(row)
    train_subjects = {row["subject_id"] for row in result["train"]}
    validation = {row["subject_id"] for row in result["validation"]}
    if train_subjects & validation:
        raise AssertionError("Vazamento de sujeito no dataset público")
    return result
