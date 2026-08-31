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
                        "median",
                        "q90",
                        "range",
                        "slope",
                    )
                )
        elif group == "signal_dynamics":
            for signal in SIGNALS:
                names.extend(
                    f"{signal}_{suffix}"
                    for suffix in ("delta_mean", "delta_std", "mean_abs_delta", "max_abs_delta")
                )
        elif group == "ocular":
            names.extend(
                (
                    "perclos",
                    "eye_closure_event_count",
                    "eye_closure_rate_per_minute",
                    "longest_eye_closure_fraction",
                )
            )
        elif group == "oral":
            names.extend(
                (
                    "mouth_open_ratio",
                    "mouth_open_event_count",
                    "mouth_open_rate_per_minute",
                    "longest_mouth_open_fraction",
                )
            )
        elif group == "head_pose":
            names.extend(
                (
                    "pose_away_ratio",
                    "pose_away_event_count",
                    "pose_away_rate_per_minute",
                    "longest_pose_away_fraction",
                )
            )
        elif group == "missingness":
            names.extend(
                ("face_detected_rate", "missing_ratio", "gap_count", "longest_gap_fraction")
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
    detected = [row.get("face_detected") == "1" for row in rows]
    signal_values = {signal: _numeric_signal(rows, signal) for signal in SIGNALS}
    values: dict[str, float] = {}

    if "signal_distribution" in selected:
        for signal, (indices, observed) in signal_values.items():
            if len(observed):
                slope = float(np.polyfit(indices, observed, 1)[0]) if len(observed) > 1 else 0.0
                q10, median, q90 = np.quantile(observed, (0.1, 0.5, 0.9))
                summary = (
                    observed.mean(), observed.std(), observed.min(), observed.max(),
                    q10, median, q90, observed.max() - observed.min(), slope,
                )
            else:
                summary = (0.0,) * 9
            for suffix, value in zip(
                ("mean", "std", "min", "max", "q10", "median", "q90", "range", "slope"),
                summary,
                strict=True,
            ):
                values[f"{signal}_{suffix}"] = float(value)

    if "signal_dynamics" in selected:
        for signal, (indices, observed) in signal_values.items():
            adjacent = np.diff(indices) == 1
            deltas = np.diff(observed)[adjacent] if len(observed) > 1 else np.asarray([])
            summary = (
                deltas.mean(), deltas.std(), np.abs(deltas).mean(), np.abs(deltas).max()
            ) if len(deltas) else (0.0,) * 4
            for suffix, value in zip(
                ("delta_mean", "delta_std", "mean_abs_delta", "max_abs_delta"),
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
        values.update(
            eye_closure_event_count=event_count,
            eye_closure_rate_per_minute=rate,
            longest_eye_closure_fraction=longest,
        )
    if "oral" in selected:
        active = condition("mar", lambda value: value > limits["mar_open"])
        values["mouth_open_ratio"] = sum(active) / valid_count if valid_count else 0.0
        event_count, rate, longest = _event_features(active, duration_seconds=duration_seconds)
        values.update(
            mouth_open_event_count=event_count,
            mouth_open_rate_per_minute=rate,
            longest_mouth_open_fraction=longest,
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
        values.update(
            pose_away_event_count=event_count,
            pose_away_rate_per_minute=rate,
            longest_pose_away_fraction=longest,
        )
    if "missingness" in selected:
        missing = [not value for value in detected]
        missing_runs = _runs(missing)
        values.update(
            face_detected_rate=valid_count / len(rows),
            missing_ratio=1.0 - valid_count / len(rows),
            gap_count=float(len(missing_runs)),
            longest_gap_fraction=max(missing_runs, default=0) / len(rows),
        )

    ordered = tuple(values[name] for name in temporal_feature_names(selected))
    if not np.isfinite(np.asarray(ordered, dtype=float)).all():
        raise ValueError("Atributos temporais contêm NaN ou infinito")
    return ordered
