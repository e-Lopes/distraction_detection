"""Atributos comportamentais e dinâmicos calculados sobre janelas temporais.

O extrator é independente do rótulo da janela. Valores faciais ausentes não entram nas
estatísticas dos sinais e interrompem eventos, enquanto a ausência permanece representada
explicitamente pelo grupo ``missingness``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


SIGNALS = ("ear", "mar", "pitch", "yaw", "roll")
AVAILABLE_GROUPS = (
    "signal_distribution",
    "signal_dynamics",
    "ocular",
    "oral",
    "head_pose",
    "multivariate",
    "missingness",
)

DEFAULT_THRESHOLDS = {
    "ear_closed": 0.25,
    "mar_open": 0.55,
    "pitch_center": 0.0,
    "pitch_deviation": 15.0,
    "yaw_center": 0.0,
    "yaw_deviation": 20.0,
}


def _validated_groups(groups: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(group) for group in groups)
    if not selected:
        raise ValueError("Ao menos um grupo de atributos temporais é obrigatório")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Grupos temporais duplicados: {selected}")
    unknown = [group for group in selected if group not in AVAILABLE_GROUPS]
    if unknown:
        raise ValueError(f"Grupos temporais desconhecidos: {unknown}")
    return selected


def temporal_feature_names(groups: Sequence[str] = AVAILABLE_GROUPS) -> tuple[str, ...]:
    """Retorna a ordem estável das colunas produzidas pelo extrator."""

    names: list[str] = []
    for group in _validated_groups(groups):
        if group == "signal_distribution":
            for signal in SIGNALS:
                names.extend(
                    f"{signal}_{suffix}"
                    for suffix in (
                        "mean",
                        "std",
                        "min",
                        "max",
                        "q10",
                        "q25",
                        "median",
                        "q75",
                        "q90",
                        "range",
                        "iqr",
                        "slope",
                        "start_end_delta",
                        "autocorrelation_lag1",
                        "line_length",
                        "peak_count",
                        "mean_crossings",
                    )
                )
        elif group == "signal_dynamics":
            for signal in SIGNALS:
                names.extend(
                    f"{signal}_{suffix}" for suffix in (
                        "delta_mean", "delta_std", "mean_abs_delta", "max_abs_delta",
                        "second_delta_mean", "second_delta_std", "mean_abs_velocity",
                        "max_abs_velocity", "mean_abs_acceleration", "max_abs_acceleration",
                        "total_variation",
                    )
                )
        elif group == "ocular":
            names.extend(
                (
                    "perclos",
                    "eye_closure_event_count",
                    "eye_closure_rate_per_minute",
                    "longest_eye_closure_fraction",
                    "mean_eye_closure_seconds",
                    "max_eye_closure_seconds",
                )
            )
        elif group == "oral":
            names.extend(
                (
                    "mouth_open_ratio",
                    "mouth_open_event_count",
                    "mouth_open_rate_per_minute",
                    "longest_mouth_open_fraction",
                    "mean_mouth_open_seconds",
                    "max_mouth_open_seconds",
                )
            )
        elif group == "head_pose":
            names.extend(
                (
                    "pose_away_ratio",
                    "pose_away_event_count",
                    "pose_away_rate_per_minute",
                    "longest_pose_away_fraction",
                    "mean_pose_away_seconds",
                    "max_pose_away_seconds",
                )
            )
        elif group == "multivariate":
            for pair in ("ear_mar", "ear_pitch", "mar_pitch"):
                names.extend((f"{pair}_correlation", f"{pair}_covariance"))
        elif group == "missingness":
            names.extend(
                ("face_detected_rate", "missing_ratio", "interpolated_ratio", "gap_count",
                 "longest_gap_fraction", "longest_gap_frames", "face_loss_count",
                 "face_recovery_count")
            )
    return tuple(names)


def _numeric_signal(
    rows: Sequence[Mapping[str, str]], signal: str
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    values: list[float] = []
    for index, row in enumerate(rows):
        value = row.get(signal, "")
        if row.get("face_detected") != "1" or value in ("", None):
            continue
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"Valor não finito em {signal}, frame relativo {index}")
        indices.append(index)
        values.append(parsed)
    return np.asarray(indices, dtype=float), np.asarray(values, dtype=float)


def _runs(active: Sequence[bool]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in active:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _event_features(active: Sequence[bool], *, duration_seconds: float) -> tuple[float, ...]:
    runs = _runs(active)
    count = len(runs)
    rate = 60.0 * count / duration_seconds if duration_seconds > 0 else 0.0
    longest_fraction = max(runs, default=0) / len(active)
    return float(count), float(rate), float(longest_fraction)


def _event_durations(active: Sequence[bool], sample_rate: float) -> tuple[float, float]:
    durations = [length / sample_rate for length in _runs(active)]
    return (float(np.mean(durations)), float(max(durations))) if durations else (0.0, 0.0)


def _sample_rate(rows: Sequence[Mapping[str, str]], fps: float | None) -> float:
    if fps is not None:
        return float(fps)
    timestamps = [float(row.get("timestamp_seconds", row.get("timestamp_s", 0))) for row in rows]
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    return 1.0 / float(np.median(positive)) if len(positive) else 1.0


def _safe_pair_stats(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 2:
        return 0.0, 0.0
    covariance = float(np.cov(left, right, ddof=0)[0, 1])
    correlation = 0.0 if np.std(left) == 0 or np.std(right) == 0 else float(np.corrcoef(left, right)[0, 1])
    return correlation, covariance


def _window_duration_seconds(rows: Sequence[Mapping[str, str]], fps: float | None) -> float:
    if fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps deve ser positivo e finito")
        return len(rows) / float(fps)
    timestamps: list[float] = []
    for row in rows:
        raw = row.get("timestamp_seconds", row.get("timestamp_s", ""))
        if raw not in ("", None):
            timestamps.append(float(raw))
    if len(timestamps) == len(rows) and len(rows) > 1:
        differences = np.diff(np.asarray(timestamps, dtype=float))
        positive = differences[differences > 0]
        if len(positive):
            return len(rows) * float(np.median(positive))
    # Fallback explícito para dados sintéticos/legados sem timestamp. Contagens e razões
    # continuam válidas; a taxa passa a ser expressa assumindo uma amostra por segundo.
    return float(len(rows))


def extract_temporal_window_features(
    rows: Sequence[Mapping[str, str]],
    *,
    groups: Sequence[str] = AVAILABLE_GROUPS,
    thresholds: Mapping[str, float] | None = None,
    fps: float | None = None,
) -> tuple[float, ...]:
    """Extrai atributos de uma janela usando apenas observações contidas nela."""

    if not rows:
        raise ValueError("Janela temporal vazia")
    selected = _validated_groups(groups)
    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    if limits["ear_closed"] <= 0 or limits["mar_open"] <= 0:
        raise ValueError("Limiar de EAR/MAR deve ser positivo")
    if limits["pitch_deviation"] <= 0 or limits["yaw_deviation"] <= 0:
        raise ValueError("Limiar de desvio de pose deve ser positivo")

    duration_seconds = _window_duration_seconds(rows, fps)
    sample_rate = _sample_rate(rows, fps)
    detected = [row.get("face_detected") == "1" for row in rows]
    signal_values = {signal: _numeric_signal(rows, signal) for signal in SIGNALS}
    values: dict[str, float] = {}

    if "signal_distribution" in selected:
        for signal, (indices, observed) in signal_values.items():
            if len(observed):
                slope = float(np.polyfit(indices, observed, 1)[0]) if len(observed) > 1 else 0.0
                q10, q25, median, q75, q90 = np.quantile(observed, (0.1, 0.25, 0.5, 0.75, 0.9))
                adjacent = np.diff(indices) == 1
                paired_left = observed[:-1][adjacent]
                paired_right = observed[1:][adjacent]
                autocorrelation = _safe_pair_stats(paired_left, paired_right)[0]
                line_length = float(np.abs(paired_right - paired_left).sum())
                peaks = sum(indices[i - 1] + 1 == indices[i] == indices[i + 1] - 1
                            and observed[i] > observed[i - 1] and observed[i] > observed[i + 1]
                            for i in range(1, len(observed) - 1))
                centered = observed - observed.mean()
                crossings = int(np.sum((centered[:-1] * centered[1:] < 0) & adjacent))
                summary = (
                    observed.mean(), observed.std(), observed.min(), observed.max(),
                    q10, q25, median, q75, q90, observed.max() - observed.min(), q75 - q25,
                    slope, observed[-1] - observed[0], autocorrelation, line_length, peaks, crossings,
                )
            else:
                summary = (0.0,) * 17
            for suffix, value in zip(
                ("mean", "std", "min", "max", "q10", "q25", "median", "q75", "q90",
                 "range", "iqr", "slope", "start_end_delta", "autocorrelation_lag1",
                 "line_length", "peak_count", "mean_crossings"),
                summary,
                strict=True,
            ):
                values[f"{signal}_{suffix}"] = float(value)

    if "signal_dynamics" in selected:
        for signal, (indices, observed) in signal_values.items():
            adjacent = np.diff(indices) == 1
            deltas = np.diff(observed)[adjacent] if len(observed) > 1 else np.asarray([])
            second = np.diff(deltas) if len(deltas) > 1 else np.asarray([])
            velocity = deltas * sample_rate
            acceleration = second * sample_rate * sample_rate
            summary = (
                deltas.mean(), deltas.std(), np.abs(deltas).mean(), np.abs(deltas).max(),
                second.mean() if len(second) else 0.0, second.std() if len(second) else 0.0,
                np.abs(velocity).mean(), np.abs(velocity).max(),
                np.abs(acceleration).mean() if len(acceleration) else 0.0,
                np.abs(acceleration).max() if len(acceleration) else 0.0,
                np.abs(deltas).sum(),
            ) if len(deltas) else (0.0,) * 11
            for suffix, value in zip(
                ("delta_mean", "delta_std", "mean_abs_delta", "max_abs_delta",
                 "second_delta_mean", "second_delta_std", "mean_abs_velocity",
                 "max_abs_velocity", "mean_abs_acceleration", "max_abs_acceleration",
                 "total_variation"),
                summary,
                strict=True,
            ):
                values[f"{signal}_{suffix}"] = float(value)

    def condition(signal: str, predicate: Callable[[float], bool]) -> list[bool]:
        result: list[bool] = []
        for row in rows:
            raw = row.get(signal, "")
            valid = row.get("face_detected") == "1" and raw not in ("", None)
            result.append(bool(valid and predicate(float(raw))))
        return result

    valid_count = sum(detected)
    if "ocular" in selected:
        active = condition("ear", lambda value: value < limits["ear_closed"])
        values["perclos"] = sum(active) / valid_count if valid_count else 0.0
        event_count, rate, longest = _event_features(active, duration_seconds=duration_seconds)
        mean_duration, max_duration = _event_durations(active, sample_rate)
        values.update(
            eye_closure_event_count=event_count,
            eye_closure_rate_per_minute=rate,
            longest_eye_closure_fraction=longest,
            mean_eye_closure_seconds=mean_duration,
            max_eye_closure_seconds=max_duration,
        )
    if "oral" in selected:
        active = condition("mar", lambda value: value > limits["mar_open"])
        values["mouth_open_ratio"] = sum(active) / valid_count if valid_count else 0.0
        event_count, rate, longest = _event_features(active, duration_seconds=duration_seconds)
        mean_duration, max_duration = _event_durations(active, sample_rate)
        values.update(
            mouth_open_event_count=event_count,
            mouth_open_rate_per_minute=rate,
            longest_mouth_open_fraction=longest,
            mean_mouth_open_seconds=mean_duration,
            max_mouth_open_seconds=max_duration,
        )
    if "head_pose" in selected:
        active = []
        for row in rows:
            pitch, yaw = row.get("pitch", ""), row.get("yaw", "")
            valid = (
                row.get("face_detected") == "1"
                and pitch not in ("", None)
                and yaw not in ("", None)
            )
            active.append(
                bool(
                    valid
                    and (
                        abs(float(pitch) - limits["pitch_center"]) > limits["pitch_deviation"]
                        or abs(float(yaw) - limits["yaw_center"]) > limits["yaw_deviation"]
                    )
                )
            )
        values["pose_away_ratio"] = sum(active) / valid_count if valid_count else 0.0
        event_count, rate, longest = _event_features(active, duration_seconds=duration_seconds)
        mean_duration, max_duration = _event_durations(active, sample_rate)
        values.update(
            pose_away_event_count=event_count,
            pose_away_rate_per_minute=rate,
            longest_pose_away_fraction=longest,
            mean_pose_away_seconds=mean_duration,
            max_pose_away_seconds=max_duration,
        )
    if "multivariate" in selected:
        for left_name, right_name in (("ear", "mar"), ("ear", "pitch"), ("mar", "pitch")):
            paired = [(float(row[left_name]), float(row[right_name])) for row in rows
                      if row.get("face_detected") == "1" and row.get(left_name) not in ("", None)
                      and row.get(right_name) not in ("", None)]
            left = np.asarray([item[0] for item in paired])
            right = np.asarray([item[1] for item in paired])
            correlation, covariance = _safe_pair_stats(left, right)
            values[f"{left_name}_{right_name}_correlation"] = correlation
            values[f"{left_name}_{right_name}_covariance"] = covariance
    if "missingness" in selected:
        missing = [not value for value in detected]
        missing_runs = _runs(missing)
        interpolated = [row.get("was_interpolated") in {"1", "1.0", "true", "True"} for row in rows]
        losses = sum(detected[index - 1] and not detected[index] for index in range(1, len(rows)))
        recoveries = sum(not detected[index - 1] and detected[index] for index in range(1, len(rows)))
        values.update(
            face_detected_rate=valid_count / len(rows),
            missing_ratio=1.0 - valid_count / len(rows),
            interpolated_ratio=sum(interpolated) / len(rows),
            gap_count=float(len(missing_runs)),
            longest_gap_fraction=max(missing_runs, default=0) / len(rows),
            longest_gap_frames=float(max(missing_runs, default=0)),
            face_loss_count=float(losses),
            face_recovery_count=float(recoveries),
        )

    ordered = tuple(values[name] for name in temporal_feature_names(selected))
    if not np.isfinite(np.asarray(ordered, dtype=float)).all():
        raise ValueError("Atributos temporais contêm NaN ou infinito")
    return ordered
