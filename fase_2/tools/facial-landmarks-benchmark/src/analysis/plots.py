"""
Gera os gráficos comparativos a partir de /results/aggregate_summary.csv e dos
CSVs por-frame em /results/<framework>/<dataset>/*__frames.csv.

Uso:
    python -m src.analysis.plots --results-dir /results --out-dir /results/plots

Gráficos gerados:
    1. fps_by_framework.png        - FPS médio por framework/dataset (barras)
    2. latency_boxplot.png         - distribuição de latência por frame (box)
    3. detection_rate.png          - taxa de detecção (%) por framework/dataset
    4. resource_usage.png          - CPU médio (%) e pico de RAM (MB)
    5. gpu_utilization.png         - utilização média de GPU (%), se houver dados
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.aggregate import load_summary, build_grouped_stats

FRAMEWORK_ORDER = ["mediapipe", "openface", "insightface"]
FRAMEWORK_COLORS = {"mediapipe": "#2E86AB", "openface": "#A23B72", "insightface": "#F18F01"}


def _ordered(frameworks):
    present = [f for f in FRAMEWORK_ORDER if f in frameworks]
    extra = [f for f in frameworks if f not in FRAMEWORK_ORDER]
    return present + extra


def plot_fps(grouped: pd.DataFrame, out_dir: str):
    datasets = sorted(grouped["dataset"].unique())
    frameworks = _ordered(grouped["framework"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(frameworks), 1)
    x = range(len(datasets))

    for i, fw in enumerate(frameworks):
        sub = grouped[grouped["framework"] == fw].set_index("dataset").reindex(datasets)
        vals = sub["mean_fps_mean"].fillna(0).values
        errs = sub["mean_fps_std"].fillna(0).values
        positions = [xi + i * width for xi in x]
        ax.bar(positions, vals, width=width, yerr=errs, capsize=3, label=fw, color=FRAMEWORK_COLORS.get(fw))

    ax.set_xticks([xi + width * (len(frameworks) - 1) / 2 for xi in x])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("FPS médio")
    ax.set_title("FPS médio por framework e dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fps_by_framework.png"), dpi=150)
    plt.close(fig)


def plot_latency_boxplot(results_dir: str, out_dir: str):
    data = []
    labels = []
    colors = []
    for fw in _ordered({os.path.basename(p) for p in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(p)}):
        pattern = os.path.join(results_dir, fw, "*", "*__frames.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        dfs = [pd.read_csv(f, usecols=["inference_ms"]) for f in files]
        if not dfs:
            continue
        combined = pd.concat(dfs)["inference_ms"].dropna()
        if combined.empty:
            continue
        data.append(combined.values)
        labels.append(fw)
        colors.append(FRAMEWORK_COLORS.get(fw, "#888888"))

    if not data:
        print("[plots] Sem dados por-frame para o boxplot de latência.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        # matplotlib < 3.9 não tem 'tick_labels' ainda
        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Latência por frame (ms)")
    ax.set_title("Distribuição de latência de inferência por framework\n(OpenFace: aproximação vídeo-nível, ver README)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_boxplot.png"), dpi=150)
    plt.close(fig)


def plot_detection_rate(grouped: pd.DataFrame, out_dir: str):
    datasets = sorted(grouped["dataset"].unique())
    frameworks = _ordered(grouped["framework"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(frameworks), 1)
    x = range(len(datasets))

    for i, fw in enumerate(frameworks):
        sub = grouped[grouped["framework"] == fw].set_index("dataset").reindex(datasets)
        vals = sub["detection_rate_pct_mean"].fillna(0).values
        positions = [xi + i * width for xi in x]
        ax.bar(positions, vals, width=width, label=fw, color=FRAMEWORK_COLORS.get(fw))

    ax.set_xticks([xi + width * (len(frameworks) - 1) / 2 for xi in x])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Taxa de detecção (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Taxa de detecção de face por framework e dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detection_rate.png"), dpi=150)
    plt.close(fig)


def plot_resource_usage(grouped: pd.DataFrame, out_dir: str):
    frameworks = _ordered(grouped["framework"].unique())
    cpu_by_fw = grouped.groupby("framework")["avg_cpu_pct_mean"].mean().reindex(frameworks)
    mem_by_fw = grouped.groupby("framework")["peak_rss_mb_max"].mean().reindex(frameworks)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = [FRAMEWORK_COLORS.get(fw, "#888888") for fw in frameworks]

    axes[0].bar(frameworks, cpu_by_fw.values, color=colors)
    axes[0].set_ylabel("CPU médio (%)")
    axes[0].set_title("Uso médio de CPU")

    axes[1].bar(frameworks, mem_by_fw.values, color=colors)
    axes[1].set_ylabel("Pico de RAM (MB)")
    axes[1].set_title("Pico de memória residente (RSS)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "resource_usage.png"), dpi=150)
    plt.close(fig)


def plot_gpu_utilization(grouped: pd.DataFrame, out_dir: str):
    if "avg_gpu_util_pct_mean" not in grouped.columns or grouped["avg_gpu_util_pct_mean"].dropna().empty:
        print("[plots] Sem dados de GPU (NVML indisponível ou execução em CPU). Pulando gpu_utilization.png.")
        return

    frameworks = _ordered(grouped["framework"].unique())
    gpu_by_fw = grouped.groupby("framework")["avg_gpu_util_pct_mean"].mean().reindex(frameworks)
    colors = [FRAMEWORK_COLORS.get(fw, "#888888") for fw in frameworks]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(frameworks, gpu_by_fw.values, color=colors)
    ax.set_ylabel("Utilização média de GPU (%)")
    ax.set_title("Utilização de GPU por framework")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gpu_utilization.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/results")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary(args.results_dir)
    grouped = build_grouped_stats(df)

    plot_fps(grouped, out_dir)
    plot_latency_boxplot(args.results_dir, out_dir)
    plot_detection_rate(grouped, out_dir)
    plot_resource_usage(grouped, out_dir)
    plot_gpu_utilization(grouped, out_dir)

    print(f"[plots] Gráficos salvos em {out_dir}")


if __name__ == "__main__":
    main()
