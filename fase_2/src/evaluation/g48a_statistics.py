"""Estatísticas pareadas para a extração integral da G48A."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

EXTRACTORS = ("mediapipe", "insightface", "openface")
MEASURES = ("ear", "mar", "pitch", "yaw", "roll")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return center - margin, center + margin


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def longest_false_run_masked(values: np.ndarray, eligible: np.ndarray) -> int:
    longest = current = 0
    for value, include in zip(values.astype(bool), eligible.astype(bool), strict=True):
        current = 0 if not include or value else current + 1
        longest = max(longest, current)
    return longest


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * covariance / denominator) if denominator else math.nan


def circular_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(length / block_length)
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets) % length).ravel()[:length]


def block_bootstrap_interval(values: np.ndarray, block_length: int, *, samples: int = 1000,
                             seed: int = 42) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).any():
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = data[circular_block_indices(len(data), block_length, rng)]
        estimates[index] = np.nanmean(selected)
    return tuple(float(value) for value in np.nanpercentile(estimates, [2.5, 97.5]))


def load_frames(root: Path, video_id: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for extractor in EXTRACTORS:
        path = root / "metrics" / extractor / f"{video_id}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame["frame_id"].duplicated().any():
            raise ValueError(f"frame_id duplicado em {path}")
        frames[extractor] = frame.sort_values("frame_id").set_index("frame_id", drop=False)
    if "mediapipe" not in frames:
        raise FileNotFoundError("A baseline MediaPipe é obrigatória")
    return frames


def calculate(frames: dict[str, pd.DataFrame], fps: float,
              operational_state: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    for name, frame in frames.items():
        detected = frame["face_detected"].astype(bool).to_numpy()
        successes, total = int(detected.sum()), len(frame)
        low, high = wilson_interval(successes, total)
        latency = pd.to_numeric(frame["inference_time_ms"], errors="coerce").dropna()
        state = (operational_state.reindex(frame.index) if operational_state is not None
                 else pd.Series("valid", index=frame.index))
        present = state.eq("valid").to_numpy()
        absent = state.eq("operator_absent").to_numpy()
        present_total, absent_total = int(present.sum()), int(absent.sum())
        present_successes = int(np.sum(detected & present))
        absent_successes = int(np.sum(detected & absent))
        present_low, present_high = wilson_interval(present_successes, present_total)
        absent_low, absent_high = wilson_interval(absent_successes, absent_total)
        summaries.append({
            "extractor": name, "frames": total, "detected": successes,
            "detection_rate": successes / total, "detection_ci95_low": low,
            "detection_ci95_high": high,
            "longest_missing_run_frames": longest_false_run(detected),
            "longest_missing_run_seconds": longest_false_run(detected) / fps,
            "operator_present_frames": present_total,
            "detected_operator_present": present_successes,
            "detection_rate_operator_present": (float(np.mean(detected[present]))
                                                if present_total else math.nan),
            "detection_present_ci95_low": present_low,
            "detection_present_ci95_high": present_high,
            "operator_absent_frames": absent_total,
            "detected_operator_absent": absent_successes,
            "false_detection_rate_operator_absent": (float(np.mean(detected[absent]))
                                                      if absent_total else math.nan),
            "false_detection_absent_ci95_low": absent_low,
            "false_detection_absent_ci95_high": absent_high,
            "longest_missing_run_present_frames": longest_false_run_masked(detected, present),
            "longest_missing_run_present_seconds": longest_false_run_masked(detected, present) / fps,
            "latency_median_ms": (latency.median() if len(latency) else math.nan),
            "latency_p95_ms": (latency.quantile(0.95) if len(latency) else math.nan),
        })

    availability, agreement = [], []
    baseline = frames["mediapipe"]
    for candidate_name, candidate in frames.items():
        if candidate_name == "mediapipe":
            continue
        common = baseline.index.intersection(candidate.index)
        base = baseline.loc[common]
        cand = candidate.loc[common]
        base_detected = base["face_detected"].astype(bool).to_numpy()
        cand_detected = cand["face_detected"].astype(bool).to_numpy()
        base_only = int(np.sum(base_detected & ~cand_detected))
        candidate_only = int(np.sum(~base_detected & cand_detected))
        detection_delta = cand_detected.astype(float) - base_detected.astype(float)
        state = (operational_state.reindex(common) if operational_state is not None
                 else pd.Series("valid", index=common))
        present = state.eq("valid").to_numpy()
        absent = state.eq("operator_absent").to_numpy()
        delta_low, delta_high = block_bootstrap_interval(detection_delta, max(round(fps * 5), 1))
        delta_present = np.where(present, detection_delta, np.nan)
        delta_absent = np.where(absent, detection_delta, np.nan)
        present_low, present_high = block_bootstrap_interval(
            delta_present, max(round(fps * 5), 1), seed=43
        )
        absent_low, absent_high = block_bootstrap_interval(
            delta_absent, max(round(fps * 5), 1), seed=44
        )
        discordant = base_only + candidate_only
        availability.append({
            "candidate": candidate_name, "paired_frames": len(common),
            "both_detected": int(np.sum(base_detected & cand_detected)),
            "mediapipe_only": base_only, "candidate_only": candidate_only,
            "neither_detected": int(np.sum(~base_detected & ~cand_detected)),
            "detection_rate_delta": float(np.mean(detection_delta)),
            "detection_rate_delta_operator_present": (float(np.mean(detection_delta[present]))
                                                       if present.any() else math.nan),
            "delta_present_block_bootstrap_ci95_low": present_low,
            "delta_present_block_bootstrap_ci95_high": present_high,
            "detection_rate_delta_operator_absent": (float(np.mean(detection_delta[absent]))
                                                      if absent.any() else math.nan),
            "delta_absent_block_bootstrap_ci95_low": absent_low,
            "delta_absent_block_bootstrap_ci95_high": absent_high,
            "delta_block_bootstrap_ci95_low": delta_low,
            "delta_block_bootstrap_ci95_high": delta_high,
            "mcnemar_exact_p": (binomtest(candidate_only, discordant, 0.5).pvalue
                                if discordant else 1.0),
        })
        for measure in MEASURES:
            x = pd.to_numeric(base[measure], errors="coerce").to_numpy()
            y = pd.to_numeric(cand[measure], errors="coerce").to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            valid &= present
            x_valid, y_valid = x[valid], y[valid]
            if measure in {"pitch", "yaw", "roll"}:
                difference_valid = (y_valid - x_valid + 180.0) % 360.0 - 180.0
                y_for_agreement = x_valid + difference_valid
            else:
                difference_valid = y_valid - x_valid
                y_for_agreement = y_valid
            differences = np.full(len(common), np.nan)
            differences[valid] = difference_valid
            n = int(valid.sum())
            bias = float(np.mean(difference_valid)) if n else math.nan
            sd = float(np.std(difference_valid, ddof=1)) if n > 1 else math.nan
            bias_low, bias_high = block_bootstrap_interval(
                differences, max(round(fps * 5), 1), seed=42 + MEASURES.index(measure)
            )
            agreement.append({
                "candidate": candidate_name, "measure": measure, "paired_valid": n,
                "bias_candidate_minus_mediapipe": bias,
                "bias_block_bootstrap_ci95_low": bias_low,
                "bias_block_bootstrap_ci95_high": bias_high,
                "mae": float(np.mean(np.abs(difference_valid))) if n else math.nan,
                "absolute_error_p95": (float(np.quantile(np.abs(difference_valid), 0.95))
                                       if n else math.nan),
                "rmse": float(np.sqrt(np.mean(difference_valid ** 2))) if n else math.nan,
                "pearson_r": (float(np.corrcoef(x_valid, y_for_agreement)[0, 1])
                              if n > 1 else math.nan),
                "ccc": concordance_correlation(x_valid, y_for_agreement),
                "bland_altman_low": bias - 1.96 * sd,
                "bland_altman_high": bias + 1.96 * sd,
            })
    return pd.DataFrame(summaries), pd.DataFrame(availability), pd.DataFrame(agreement)


def load_operational_state(path: Path, video_id: str, frame_index: pd.Index) -> pd.Series:
    intervals = pd.read_csv(path).query("video_id == @video_id")
    state = pd.Series(index=frame_index, dtype="object")
    for row in intervals.itertuples(index=False):
        state.loc[int(row.start_frame):int(row.end_frame)] = row.operational_state
    if state.isna().any():
        raise ValueError(f"Anotação operacional incompleta para {video_id}")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("fase_2/outputs/G48A_all_frames"))
    parser.add_argument("--video-id", default="video_04")
    parser.add_argument("--fps", type=float, default=17.813525615558454)
    parser.add_argument("--annotations", type=Path,
                        default=Path("fase_2/data/manifests/annotation_frame_intervals.csv"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/metrics/G48A"))
    args = parser.parse_args()
    frames = load_frames(args.root, args.video_id)
    state = load_operational_state(args.annotations, args.video_id, frames["mediapipe"].index)
    summary, availability, agreement = calculate(frames, args.fps, state)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / f"g48a_{args.video_id}"
    summary.to_csv(f"{prefix}_extractor_summary.csv", index=False)
    availability.to_csv(f"{prefix}_pairwise_availability.csv", index=False)
    agreement.to_csv(f"{prefix}_agreement.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Artefatos gravados com prefixo: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
