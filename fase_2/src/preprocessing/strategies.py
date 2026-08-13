"""Estratégias de missingness ajustadas e aplicadas sem atravessar blocos temporais."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..data.splits import SplitBlock


METRICS = ("ear", "mar", "pitch", "yaw", "roll")


def fit_training_medians(
    series: Mapping[str, Sequence[dict[str, str]]],
    blocks: Sequence[SplitBlock],
) -> dict[str, float]:
    """Ajusta medianas somente nos frames detectados dos blocos de treino."""

    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    train_blocks = [block for block in blocks if block.subset == "train"]
    if not train_blocks:
        raise ValueError("Nenhum bloco de treino para ajustar as medianas")
    for block in train_blocks:
        rows = series[block.video_id][block.start_frame : block.end_frame + 1]
        for row in rows:
            if row["face_detected"] != "1":
                continue
            for metric in METRICS:
                if row[metric] != "":
                    values[metric].append(float(row[metric]))
    missing = [metric for metric, observed in values.items() if not observed]
    if missing:
        raise ValueError(f"Sem valores de treino para calcular medianas: {missing}")
    return {metric: float(np.median(observed)) for metric, observed in values.items()}


def _missing_runs(detected: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_detected in enumerate(detected):
        if not is_detected and start is None:
            start = index
        elif is_detected and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(detected) - 1))
    return runs


def preprocess_block(
    rows: Sequence[dict[str, str]],
    config: Mapping[str, object],
    *,
    training_medians: Mapping[str, float] | None = None,
) -> list[dict[str, str]]:
    """Transforma um único bloco; portanto, interpolação nunca cruza um split."""

    if not rows:
        return []
    transformed = [dict(row) for row in rows]
    detected = [row["face_detected"] == "1" for row in rows]
    short_gap_max = int(config.get("short_gap_max_frames", 0))
    method = str(config.get("short_gap_method", "linear"))
    if short_gap_max and method != "linear":
        raise ValueError(f"Método de interpolação não suportado: {method}")

    interpolated = [False] * len(rows)
    for start, end in _missing_runs(detected):
        length = end - start + 1
        bounded = start > 0 and end + 1 < len(rows)
        if not bounded or length > short_gap_max:
            continue
        if not all(rows[start - 1][metric] and rows[end + 1][metric] for metric in METRICS):
            continue
        for offset, index in enumerate(range(start, end + 1), start=1):
            fraction = offset / (length + 1)
            for metric in METRICS:
                left = float(rows[start - 1][metric])
                right = float(rows[end + 1][metric])
                transformed[index][metric] = str(left + fraction * (right - left))
            interpolated[index] = True

    long_gap_fill = str(
        config.get("long_gap_fill", config.get("fill_long_gaps", "zero"))
    )
    if long_gap_fill not in {"zero", "training_median"}:
        raise ValueError(f"Preenchimento de gap longo não suportado: {long_gap_fill}")
    if long_gap_fill == "training_median" and training_medians is None:
        raise ValueError("training_medians é obrigatório para preencher gaps longos")

    missing_duration = 0
    for index, row in enumerate(transformed):
        if detected[index]:
            missing_duration = 0
        else:
            missing_duration += 1
        for metric in METRICS:
            if row[metric] == "":
                fill = 0.0 if long_gap_fill == "zero" else float(training_medians[metric])
                row[metric] = str(fill)
        row["was_interpolated"] = "1" if interpolated[index] else "0"
        row["missing_duration_so_far"] = str(missing_duration)
    return transformed


def strategy_feature_names(config: Mapping[str, object]) -> tuple[str, ...]:
    names = [name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")]
    names.extend(str(name) for name in config.get("window_missingness_features", []))
    flags = [str(flag) for flag in config.get("flags", [])] if config.get("add_flags") else []
    if "face_detected" in flags:
        names.append("face_detected_rate")
    if "was_interpolated" in flags:
        names.append("was_interpolated_rate")
    if "missing_duration_so_far" in flags:
        names.extend(("missing_duration_mean", "missing_duration_max"))
    if len(names) != len(set(names)):
        raise ValueError(f"Features duplicadas na estratégia {config.get('name')}: {names}")
    return tuple(names)


def aggregate_preprocessed_window(
    rows: Sequence[dict[str, str]], config: Mapping[str, object]
) -> tuple[float, ...]:
    if not rows:
        raise ValueError("Janela vazia")
    values: dict[str, float] = {}
    for metric in METRICS:
        metric_values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        values[f"{metric}_mean"] = float(metric_values.mean())
        values[f"{metric}_std"] = float(metric_values.std())

    detected = [row["face_detected"] == "1" for row in rows]
    runs = _missing_runs(detected)
    values["missing_ratio"] = 1 - sum(detected) / len(detected)
    values["gap_count"] = float(len(runs))
    values["longest_gap"] = float(max((end - start + 1 for start, end in runs), default=0))
    values["face_detected_rate"] = sum(detected) / len(detected)
    values["was_interpolated_rate"] = sum(
        row["was_interpolated"] == "1" for row in rows
    ) / len(rows)
    durations = np.asarray([float(row["missing_duration_so_far"]) for row in rows])
    values["missing_duration_mean"] = float(durations.mean())
    values["missing_duration_max"] = float(durations.max())
    return tuple(values[name] for name in strategy_feature_names(config))
