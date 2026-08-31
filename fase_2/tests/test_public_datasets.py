import pytest
from fase_2.src.data.public_datasets import subject_independent_split


def _row(subject, sample):
    return {
        "dataset_id": "NTHU-DDD",
        "subject_id": subject,
        "sample_id": sample,
        "relative_path": f"{subject}/{sample}.mp4",
        "task": "eye_state",
        "label": "open",
        "license": "academic",
    }


def test_public_split_is_subject_independent():
    split = subject_independent_split(
        [_row("a", "1"), _row("b", "2"), _row("b", "3")], validation_subjects={"b"}
    )
    assert {row["subject_id"] for row in split["train"]} == {"a"}
    assert {row["subject_id"] for row in split["validation"]} == {"b"}


def test_public_manifest_rejects_duplicate_samples():
    with pytest.raises(ValueError, match="duplicada"):
        subject_independent_split(
            [_row("a", "1"), _row("a", "1"), _row("b", "2")],
            validation_subjects={"b"},
        )
