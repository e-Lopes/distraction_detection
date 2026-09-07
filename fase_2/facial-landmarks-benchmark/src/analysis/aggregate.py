"""Aggregate weighted metrics, confidence intervals and paired comparisons."""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NUMERIC_COLS = ["repetition", "n_frames", "n_frames_detected", "duration_s", "mean_fps",
                "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "max_latency_ms",
                "avg_cpu_pct", "peak_rss_mb", "avg_gpu_util_pct", "peak_gpu_mem_mb"]
KEY = ["run_id", "repetition", "framework", "backend", "dataset", "video_id"]

def load_summary(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "aggregate_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} nao encontrado")
    df = pd.read_csv(path)
    missing = set(KEY) - set(df.columns)
    if missing:
        raise ValueError(f"Schema antigo/invalido; colunas ausentes: {sorted(missing)}")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    duplicates = df.duplicated(KEY, keep=False)
    if duplicates.any():
        raise ValueError(f"Resultados duplicados para a mesma execucao:\n{df.loc[duplicates, KEY].to_string(index=False)}")
    df["configuration"] = df["framework"] + "/" + df["backend"]
    return df

def bootstrap_mean_ci(values, seed=20260907, samples=10000):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))

def build_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (framework, backend, dataset), group in df.groupby(["framework", "backend", "dataset"], sort=True):
        fps_low, fps_high = bootstrap_mean_ci(group["mean_fps"])
        det_low, det_high = bootstrap_mean_ci(group["detection_rate_pct"])
        frames, detected, duration = group["n_frames"].sum(), group["n_frames_detected"].sum(), group["duration_s"].sum()
        rows.append({
            "framework": framework, "backend": backend, "configuration": f"{framework}/{backend}",
            "dataset": dataset, "n_observations": len(group), "n_videos": group["video_id"].nunique(),
            "n_repetitions": group["repetition"].nunique(), "n_frames": int(frames),
            "throughput_fps": frames / duration if duration else np.nan,
            "fps_mean": group["mean_fps"].mean(), "fps_std": group["mean_fps"].std(),
            "fps_median": group["mean_fps"].median(), "fps_ci95_low": fps_low, "fps_ci95_high": fps_high,
            "detection_weighted_pct": 100.0 * detected / frames if frames else np.nan,
            "detection_mean_pct": group["detection_rate_pct"].mean(),
            "detection_ci95_low": det_low, "detection_ci95_high": det_high,
            "p50_latency_ms_mean": group["p50_latency_ms"].mean(),
            "p95_latency_ms_mean": group["p95_latency_ms"].mean(),
            "avg_cpu_pct_mean": group["avg_cpu_pct"].mean(),
            "peak_rss_mb_max": group["peak_rss_mb"].max(),
            "avg_gpu_util_pct_mean": group["avg_gpu_util_pct"].mean(),
            "peak_gpu_mem_mb_max": group["peak_gpu_mem_mb"].max(),
        })
    return pd.DataFrame(rows)

def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted

def build_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    join = ["run_id", "repetition", "dataset", "video_id"]
    for dataset, data in df.groupby("dataset"):
        configurations = sorted(data["configuration"].unique())
        for a, b in itertools.combinations(configurations, 2):
            left = data[data.configuration == a][join + ["mean_fps", "detection_rate_pct"]]
            right = data[data.configuration == b][join + ["mean_fps", "detection_rate_pct"]]
            paired = left.merge(right, on=join, suffixes=("_a", "_b"))
            if paired.empty:
                continue
            ratio = paired.mean_fps_a / paired.mean_fps_b.replace(0, np.nan)
            diff = paired.detection_rate_pct_a - paired.detection_rate_pct_b
            fps_delta = paired.mean_fps_a - paired.mean_fps_b
            try:
                pvalue = wilcoxon(fps_delta).pvalue if np.any(fps_delta != 0) else 1.0
            except ValueError:
                pvalue = np.nan
            rows.append({"dataset": dataset, "configuration_a": a, "configuration_b": b,
                         "n_pairs": len(paired), "fps_ratio_a_over_b_geomean": np.exp(np.log(ratio.dropna()).mean()),
                         "fps_delta_mean": fps_delta.mean(), "detection_delta_pct_points": diff.mean(),
                         "wilcoxon_fps_p_raw": pvalue})
    result = pd.DataFrame(rows)
    if not result.empty:
        valid = result["wilcoxon_fps_p_raw"].notna()
        result.loc[valid, "wilcoxon_fps_p_holm"] = holm_adjust(result.loc[valid, "wilcoxon_fps_p_raw"])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/results")
    args = parser.parse_args()
    df = load_summary(args.results_dir)
    grouped = build_grouped_stats(df)
    paired = build_paired_comparisons(df)
    grouped.to_csv(os.path.join(args.results_dir, "grouped_stats.csv"), index=False)
    paired.to_csv(os.path.join(args.results_dir, "paired_comparisons.csv"), index=False)
    print(grouped.to_string(index=False))

if __name__ == "__main__":
    main()
