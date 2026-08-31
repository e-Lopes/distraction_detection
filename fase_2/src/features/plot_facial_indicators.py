"""Gera gráficos dos indicadores faciais ao longo do tempo para inspeção."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDICATORS = ("ear", "mar", "pitch", "yaw", "roll")
COLORS = {
    "ear": "#1f77b4",
    "mar": "#d62728",
    "pitch": "#2ca02c",
    "yaw": "#9467bd",
    "roll": "#ff7f0e",
}
LABELS = {
    "ear": "EAR — abertura dos olhos",
    "mar": "MAR — abertura da boca",
    "pitch": "Pitch (graus)",
    "yaw": "Yaw (graus)",
    "roll": "Roll (graus)",
}


def plot_video(source: Path, output: Path, *, smoothing_seconds: float = 5.0) -> None:
    frame = pd.read_csv(source)
    required = {"video_id", "timestamp_seconds", "face_detected", *INDICATORS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {source}: {sorted(missing)}")
    time_minutes = pd.to_numeric(frame["timestamp_seconds"], errors="coerce") / 60.0
    detected = pd.to_numeric(frame["face_detected"], errors="coerce").fillna(0).eq(1)
    fps = len(frame) / max(float(time_minutes.max()) * 60.0, 1.0)
    rolling_frames = max(1, round(smoothing_seconds * fps))
    figure, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    video_id = str(frame["video_id"].iloc[0])
    for axis, indicator in zip(axes, INDICATORS, strict=True):
        values = pd.to_numeric(frame[indicator], errors="coerce").where(detected)
        # Série bruta preservada em baixa opacidade; mediana móvel apenas auxilia a leitura.
        axis.plot(time_minutes, values, color=COLORS[indicator], alpha=0.18, linewidth=0.45)
        smoothed = values.rolling(rolling_frames, center=True, min_periods=1).median()
        axis.plot(
            time_minutes,
            smoothed,
            color=COLORS[indicator],
            linewidth=1.15,
            label=f"Mediana móvel ({smoothing_seconds:g} s)",
        )
        if indicator == "ear":
            axis.axhline(0.25, color="#444444", linestyle="--", linewidth=0.8, label="EAR 0,25")
        elif indicator == "mar":
            axis.axhline(0.55, color="#444444", linestyle="--", linewidth=0.8, label="MAR 0,55")
        axis.set_ylabel(LABELS[indicator])
        axis.grid(alpha=0.22)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Tempo do vídeo (minutos)")
    detection_rate = 100.0 * float(detected.mean())
    figure.suptitle(
        f"Indicadores faciais ao longo do tempo — {video_id}\n"
        f"detecção facial: {detection_rate:.1f}% | linha clara: valor bruto | linha forte: mediana móvel",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fase_2/data/interim/legacy_extraction")
    parser.add_argument("--output-dir", default="fase_2/outputs/figures/facial_indicators")
    parser.add_argument("--smoothing-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.smoothing_seconds <= 0:
        raise ValueError("--smoothing-seconds deve ser positivo")
    input_dir = Path(args.input_dir)
    sources = sorted(input_dir.glob("video_*.csv"))
    if not sources:
        raise FileNotFoundError(f"Nenhuma série facial encontrada em {input_dir}")
    output_dir = Path(args.output_dir)
    for index, source in enumerate(sources, start=1):
        output = output_dir / f"{source.stem}_indicators_over_time.png"
        print(f"[{index}/{len(sources)}] Gerando {output}...", flush=True)
        plot_video(source, output, smoothing_seconds=args.smoothing_seconds)
    print(f"Gráficos concluídos em {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
