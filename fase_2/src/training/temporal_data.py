"""Construção de sequências por fold com pré-processamento ajustado no treino."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.splits import SplitBlock
from ..data.windowing import build_windows
from ..preprocessing.strategies import fit_training_medians, preprocess_block
from .dummy_baseline import CLASSES


SIGNAL_FEATURES = (
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
)
MISSINGNESS_FEATURES = (
    "face_detected",
    "was_interpolated",
    "missing_duration_so_far",
)
REPRESENTATION_FEATURES = {
    "R0": SIGNAL_FEATURES,
    "R1": SIGNAL_FEATURES,
    "R2": SIGNAL_FEATURES + MISSINGNESS_FEATURES,
}
# Compatibilidade para consumidores anteriores, que usavam R2 implicitamente.
SEQUENCE_FEATURES = REPRESENTATION_FEATURES["R2"]
PHYSICAL_RANGES = {
    "ear": (0.0, 1.0),
    "mar": (0.0, 2.0),
    "pitch": (-180.0, 180.0),
    "yaw": (-180.0, 180.0),
    "roll": (-180.0, 180.0),
}


@dataclass(frozen=True)
class SequenceMetadata:
    video_id: str
    start_frame: int
    end_frame: int
    label: str
    missing_ratio: float = 0.0
    interpolated_ratio: float = 0.0


@dataclass(frozen=True)
class SequenceSplit:
    values: np.ndarray
    labels: np.ndarray
    metadata: tuple[SequenceMetadata, ...]


@dataclass(frozen=True)
class SequenceScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)


class WindowSequenceDataset(Dataset):
    def __init__(self, split: SequenceSplit) -> None:
        self.values = torch.from_numpy(split.values.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(split.labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.values[index], self.labels[index]


def limit_sequence_split(
    split: SequenceSplit, max_samples: int | None, *, seed: int
) -> SequenceSplit:
    """Reduz um split para smoke mantendo, quando possivel, todas as classes observadas."""

    if max_samples is None or len(split.labels) <= max_samples:
        return split
    if max_samples <= 0:
        raise ValueError("max_samples deve ser positivo")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    classes = np.unique(split.labels)
    quota = max(1, max_samples // len(classes))
    for label in classes:
        candidates = np.flatnonzero(split.labels == label)
        selected.extend(rng.choice(candidates, min(quota, len(candidates)), replace=False))
    remaining = max_samples - len(selected)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(split.labels)), np.asarray(selected))
        selected.extend(rng.choice(available, min(remaining, len(available)), replace=False))
    indices = np.sort(np.asarray(selected[:max_samples], dtype=int))
    return SequenceSplit(
        values=split.values[indices],
        labels=split.labels[indices],
        metadata=tuple(split.metadata[index] for index in indices),
    )


def representation_features(representation: str) -> tuple[str, ...]:
    try:
        return REPRESENTATION_FEATURES[representation.upper()]
    except KeyError as error:
        raise ValueError(
            f"Representacao sequencial desconhecida: {representation}. Use R0, R1 ou R2"
        ) from error


def _frame_vector(row: Mapping[str, str], features: Sequence[str]) -> list[float]:
    return [float(row[name]) for name in features]


def light_augment_training_windows(
    values: Sequence[np.ndarray], labels: Sequence[int], metadata: Sequence[SequenceMetadata],
    features: Sequence[str], config: Mapping[str, object], *, seed: int,
    audit_records: list[dict[str, object]] | None = None,
) -> tuple[list[np.ndarray], list[int], list[SequenceMetadata]]:
    """Materializa uma copia leve por janela minoritaria antes da padronizacao."""
    class_to_index = {label: index for index, label in enumerate(CLASSES)}
    minority = {class_to_index[str(name)] for name in config.get("minority_classes", ())}
    if not minority or int(config.get("copies_per_minority_window", 1)) != 1:
        raise ValueError("G4 exige uma copia por janela minoritaria")
    scale_low, scale_high = (float(x) for x in config["scale_range"])
    jitter = {str(k): float(v) for k, v in dict(config["jitter_std"]).items()}
    mask_probability = float(config["masking_probability"])
    max_mask = int(config["max_mask_frames"])
    if any(name not in PHYSICAL_RANGES or name not in jitter for name in features):
        raise ValueError("augmentation G4 aceita somente features fisicas R0")
    rng = np.random.default_rng(seed)
    out_values = [np.asarray(value, dtype=np.float32) for value in values]
    out_labels = list(labels)
    out_metadata = list(metadata)
    for source_index, (source, label, item) in enumerate(zip(values, labels, metadata, strict=True)):
        if int(label) not in minority:
            continue
        before = np.asarray(source, dtype=np.float32)
        factors = rng.uniform(scale_low, scale_high, size=len(features)).astype(np.float32)
        noise = np.column_stack([
            rng.normal(0.0, jitter[name], size=len(before)) for name in features
        ]).astype(np.float32)
        after = before * factors + noise
        mask_start, mask_length = -1, 0
        if rng.random() < mask_probability:
            mask_length = int(rng.integers(1, min(max_mask, len(after)) + 1))
            mask_start = int(rng.integers(0, len(after) - mask_length + 1))
            after[mask_start : mask_start + mask_length] = 0.0
        for index, name in enumerate(features):
            after[:, index] = np.clip(after[:, index], *PHYSICAL_RANGES[name])
        after = after.astype(np.float32)
        out_values.append(after)
        out_labels.append(int(label))
        out_metadata.append(item)
        if audit_records is not None:
            audit_records.append({
                "source_index": source_index, "class": CLASSES[int(label)],
                "video_id": item.video_id, "start_frame": item.start_frame,
                "end_frame": item.end_frame, "seed": seed,
                "operations": "scaling+jitter" + ("+masking" if mask_length else ""),
                "scale_factors": json.dumps(factors.tolist(), separators=(",", ":")),
                "mask_start": mask_start, "mask_length": mask_length,
                "before_sha256": hashlib.sha256(np.ascontiguousarray(before).tobytes()).hexdigest(),
                "after_sha256": hashlib.sha256(np.ascontiguousarray(after).tobytes()).hexdigest(),
                "before_values": json.dumps(before.tolist(), separators=(",", ":")),
                "after_values": json.dumps(after.tolist(), separators=(",", ":")),
            })
    return out_values, out_labels, out_metadata


def build_sequence_fold(
    series: Mapping[str, Sequence[dict[str, str]]],
    labels_by_video: Mapping[str, Sequence[str | None]],
    blocks: Sequence[SplitBlock],
    preprocessing: Mapping[str, object],
    *,
    fold: int,
    size_frames: int,
    stride_frames: int,
    minimum_proportion: float,
    representation: str = "R2",
    training_augmentation: Mapping[str, object] | None = None,
    augmentation_seed: int = 42,
    augmentation_audit: list[dict[str, object]] | None = None,
) -> tuple[dict[str, SequenceSplit], SequenceScaler, dict[str, float]]:
    representation = representation.upper()
    features = representation_features(representation)
    short_gap_max = int(preprocessing.get("short_gap_max_frames", 0))
    if representation == "R0" and short_gap_max != 0:
        raise ValueError("R0 exige zero-fill sem interpolacao de gaps curtos")
    if representation == "R1" and short_gap_max <= 0:
        raise ValueError("R1 exige tratamento configurado para gaps curtos")
    if representation == "R2":
        configured_flags = tuple(str(value) for value in preprocessing.get("flags", ()))
        if configured_flags != MISSINGNESS_FEATURES:
            raise ValueError(
                "R2 exige flags na ordem face_detected, was_interpolated, "
                "missing_duration_so_far"
            )
    fold_blocks = [block for block in blocks if block.fold == fold]
    use_medians = str(preprocessing.get("long_gap_fill", "zero")) == "training_median"
    medians = fit_training_medians(series, fold_blocks) if use_medians else {}
    causal_durations: dict[tuple[str, int, int], np.ndarray] = {}
    by_video: dict[str, list[SplitBlock]] = {}
    for block in fold_blocks:
        original = series[block.video_id][block.start_frame : block.end_frame + 1]
        duration_rows = preprocess_block(
            original,
            {"short_gap_max_frames": 0, "long_gap_fill": "zero"},
        )
        causal_durations[(block.video_id, block.start_frame, block.end_frame)] = np.asarray(
            [float(row["missing_duration_so_far"]) for row in duration_rows],
            dtype=np.float32,
        )
        by_video.setdefault(block.video_id, []).append(block)

    windows = build_windows(
        labels_by_video,
        size_frames=size_frames,
        stride_frames=stride_frames,
        behavior_classes=set(CLASSES),
        minimum_proportion=minimum_proportion,
    )
    raw: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "validation", "test")}
    labels: dict[str, list[int]] = {name: [] for name in raw}
    metadata: dict[str, list[SequenceMetadata]] = {name: [] for name in raw}
    class_to_index = {label: index for index, label in enumerate(CLASSES)}
    for window in windows:
        if window.label == "mixed":
            continue
        matches = [
            block
            for block in by_video.get(window.video_id, [])
            if window.start_frame >= block.start_frame and window.end_frame <= block.end_frame
        ]
        if len(matches) > 1:
            raise ValueError("Janela atribuída a múltiplos blocos")
        if not matches:
            continue
        block = matches[0]
        original = series[window.video_id][window.start_frame : window.end_frame + 1]
        rows = preprocess_block(
            original,
            preprocessing,
            training_medians=medians if use_medians else None,
        )
        relative = window.start_frame - block.start_frame
        durations = causal_durations[(block.video_id, block.start_frame, block.end_frame)][
            relative : relative + size_frames
        ]
        for row, duration in zip(rows, durations, strict=True):
            row["missing_duration_so_far"] = str(float(duration))
        sequence = np.asarray(
            [_frame_vector(row, features) for row in rows], dtype=np.float32
        )
        raw[block.subset].append(sequence)
        labels[block.subset].append(class_to_index[window.label])
        metadata[block.subset].append(
            SequenceMetadata(
                window.video_id,
                window.start_frame,
                window.end_frame,
                window.label,
                missing_ratio=sum(row["face_detected"] != "1" for row in original)
                / len(original),
                interpolated_ratio=sum(row["was_interpolated"] == "1" for row in rows)
                / len(rows),
            )
        )
    if any(not values for values in raw.values()):
        raise ValueError(f"Fold {fold} contém subconjunto temporal vazio")

    if training_augmentation is not None:
        if representation != "R0":
            raise ValueError("augmentation G4 foi congelada somente para R0")
        raw["train"], labels["train"], metadata["train"] = light_augment_training_windows(
            raw["train"], labels["train"], metadata["train"], features,
            training_augmentation, seed=augmentation_seed,
            audit_records=augmentation_audit,
        )

    train_values = np.asarray(raw["train"], dtype=np.float32)
    if not np.isfinite(train_values).all():
        raise ValueError(f"Fold {fold}/{representation} contem NaN ou Inf no treino")
    flattened = train_values.reshape(-1, train_values.shape[-1]).astype(np.float64)
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    if representation == "R2":
        for flag in ("face_detected", "was_interpolated"):
            index = features.index(flag)
            mean[index] = 0.0
            scale[index] = 1.0
    scale[scale == 0] = 1.0
    scaler = SequenceScaler(tuple(mean), tuple(scale))
    result = {
        subset: SequenceSplit(
            values=scaler.transform(np.asarray(raw[subset], dtype=np.float32)).astype(
                np.float32
            ),
            labels=np.asarray(labels[subset], dtype=np.int64),
            metadata=tuple(metadata[subset]),
        )
        for subset in raw
    }
    for subset, split in result.items():
        if not np.isfinite(split.values).all():
            raise ValueError(
                f"Fold {fold}/{representation} contem NaN ou Inf em {subset}"
            )
    return result, scaler, medians
