"""Gera gráficos dos indicadores faciais ao longo do tempo para inspeção."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

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


def _moving_median(values: np.ndarray, size: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = size // 2
    for index in range(len(values)):
        selected = values[max(0, index-half):min(len(values), index+half+1)]
        if np.isfinite(selected).any():
            result[index] = np.nanmedian(selected)
    return result


def plot_video(source: Path, output: Path, *, smoothing_seconds: float = 5.0) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as stream:
        frame = list(csv.DictReader(stream))
    if not frame:
        raise ValueError(f"Série vazia: {source}")
    required = {"video_id", "timestamp_seconds", "face_detected", *INDICATORS}
    missing = required.difference(frame[0])
    if missing:
        raise ValueError(f"Colunas ausentes em {source}: {sorted(missing)}")
    time_minutes = np.asarray([float(row["timestamp_seconds"]) for row in frame]) / 60.0
    detected = np.asarray([row["face_detected"] == "1" for row in frame])
    fps = len(frame) / max(float(np.nanmax(time_minutes)) * 60.0, 1.0)
    rolling_frames = max(1, round(smoothing_seconds * fps))
    figure, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    video_id = frame[0]["video_id"]
    individual = []
    for axis, indicator in zip(axes, INDICATORS, strict=True):
        values = np.asarray([float(row[indicator]) if detected[index] and row[indicator]
                             else np.nan for index, row in enumerate(frame)])
        # Série bruta preservada em baixa opacidade; mediana móvel apenas auxilia a leitura.
        axis.plot(time_minutes, values, color=COLORS[indicator], alpha=0.18, linewidth=0.45)
        smoothed = _moving_median(values, rolling_frames)
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
        detail, detail_axis = plt.subplots(figsize=(15, 4))
        detail_axis.plot(time_minutes, values, color=COLORS[indicator], alpha=0.22, linewidth=0.5,
                         label="Valor medido")
        detail_axis.plot(time_minutes, smoothed, color=COLORS[indicator], linewidth=1.2,
                         label=f"Tendência de {smoothing_seconds:g} s")
        detail_axis.set(title=f"{LABELS[indicator]} ao longo do vídeo — {video_id}",
                        xlabel="Tempo do vídeo (minutos)", ylabel=LABELS[indicator])
        detail_axis.grid(alpha=0.22); detail_axis.legend()
        detail.tight_layout()
        detail_path = output.with_name(f"{video_id}_{indicator}.png")
        detail.savefig(detail_path, dpi=170, bbox_inches="tight")
        detail.savefig(detail_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(detail)
        individual.append(detail_path)
    axes[-1].set_xlabel("Tempo do vídeo (minutos)")
    detection_rate = 100.0 * float(detected.mean())
    figure.suptitle(
        f"Indicadores faciais ao longo do tempo — {video_id}\n"
        f"detecção facial: {detection_rate:.1f}% | linha clara: valor bruto | linha forte: mediana móvel",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    return [output, *individual]


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
