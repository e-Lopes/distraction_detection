"""Calcula e visualiza PERCLOS móvel com os thresholds fornecidos pelo pesquisador."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EAR_CLOSED = 0.25
THRESHOLDS = (3.75, 7.5, 11.25, 15.0)


def calculate_perclos(
    frame: pd.DataFrame,
    *,
    window_seconds: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    """PERCLOS = frames EAR<0,25 / frames faciais válidos na janela."""
    result = frame.copy()
    timestamps = pd.to_numeric(result["timestamp_seconds"], errors="coerce")
    duration = max(float(timestamps.max() - timestamps.min()), 1.0)
    fps = max((len(result) - 1) / duration, 1.0)
    window_frames = max(1, round(window_seconds * fps))
    detected = pd.to_numeric(result["face_detected"], errors="coerce").fillna(0).eq(1)
    ear = pd.to_numeric(result["ear"], errors="coerce")
    valid = detected & ear.notna()
    closed = valid & ear.lt(EAR_CLOSED)
    valid_count = valid.astype(float).rolling(window_frames, min_periods=1).sum()
    closed_count = closed.astype(float).rolling(window_frames, min_periods=1).sum()
    expected_count = pd.Series(1.0, index=result.index).rolling(window_frames, min_periods=1).sum()
    result["perclos_percent"] = 100.0 * closed_count / valid_count.where(valid_count > 0)
    result["facial_coverage"] = valid_count / expected_count
    result.loc[result["facial_coverage"] < minimum_coverage, "perclos_percent"] = float("nan")
    return result


def category(value: float) -> str:
    if pd.isna(value):
        return "indisponivel"
    if value <= 3.75:
        return "baixo_1"
    if value <= 7.5:
        return "baixo_2"
    if value <= 11.25:
        return "moderado_1"
    if value <= 15.0:
        return "moderado_2"
    return "severo"


def plot_perclos(frame: pd.DataFrame, output: Path, *, window_seconds: float) -> None:
    time_minutes = pd.to_numeric(frame["timestamp_seconds"], errors="coerce") / 60.0
    video_id = str(frame["video_id"].iloc[0])
    figure, (axis, coverage_axis) = plt.subplots(
        2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": (4, 1)}
    )
    axis.axhspan(0, 7.5, color="#2ca02c", alpha=0.10, label="Baixa (≤ 7,5%)")
    axis.axhspan(7.5, 15, color="#ffd700", alpha=0.16, label="Moderada (7,5–15%)")
    upper = max(25.0, float(frame["perclos_percent"].max(skipna=True)) * 1.05)
    axis.axhspan(15, upper, color="#d62728", alpha=0.10, label="Severa (> 15%)")
    axis.plot(time_minutes, frame["perclos_percent"], color="#1f77b4", linewidth=1.1)
    for threshold in THRESHOLDS:
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=0.7)
    axis.set_ylim(0, upper)
    axis.set_ylabel("PERCLOS (%)")
    axis.set_title(
        f"PERCLOS ao longo do tempo — {video_id}\n"
        f"janela móvel {window_seconds:g} s | olho fechado: EAR < {EAR_CLOSED}"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", ncol=3, fontsize=9)
    coverage_axis.fill_between(
        time_minutes,
        100.0 * frame["facial_coverage"],
        color="#777777",
        alpha=0.5,
    )
    coverage_axis.axhline(50, color="#d62728", linestyle="--", linewidth=0.8)
    coverage_axis.set_ylim(0, 100)
    coverage_axis.set_ylabel("Cobertura (%)")
    coverage_axis.set_xlabel("Tempo do vídeo (minutos)")
    coverage_axis.grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument("--output-dir", default="fase_2/outputs/figures/perclos")
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.window_seconds <= 0 or not 0 < args.minimum_coverage <= 1:
        raise ValueError("Janela deve ser positiva e cobertura deve pertencer a (0, 1]")
    sources = sorted(Path(args.input_dir).glob("video_*.csv"))
    if not sources:
        raise FileNotFoundError(f"Nenhuma série encontrada em {args.input_dir}")
    output_dir = Path(args.output_dir)
    summary = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] Calculando PERCLOS de {source.stem}...", flush=True)
        values = calculate_perclos(
            pd.read_csv(source),
            window_seconds=args.window_seconds,
            minimum_coverage=args.minimum_coverage,
        )
        values["perclos_category"] = values["perclos_percent"].map(category)
        plot_perclos(
            values,
            output_dir / f"{source.stem}_perclos_over_time.png",
            window_seconds=args.window_seconds,
        )
        counts = values["perclos_category"].value_counts()
        summary.append(
            {
                "video_id": source.stem,
                "window_seconds": args.window_seconds,
                "minimum_coverage": args.minimum_coverage,
                "ear_closed_threshold": EAR_CLOSED,
                **{
                    name: int(counts.get(name, 0))
                    for name in (
                        "baixo_1",
                        "baixo_2",
                        "moderado_1",
                        "moderado_2",
                        "severo",
                        "indisponivel",
                    )
                },
            }
        )
    pd.DataFrame(summary).to_csv(output_dir / "perclos_summary.csv", index=False)
    print(f"Análise PERCLOS concluída em {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
