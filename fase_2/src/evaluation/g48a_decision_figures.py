"""Gera figuras de apresentação para a decisão MediaPipe versus OpenFace."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#15803D"
INK = "#0F172A"
MUTED = "#475569"
LIGHT = "#F8FAFC"


def pt(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _finish(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def evidence_figure(summary: pd.DataFrame, output: Path) -> None:
    data = summary.set_index("extractor").loc[["mediapipe", "openface"]]
    labels = ["MediaPipe", "OpenFace"]
    colors = [BLUE, ORANGE]
    present = data["detection_rate_operator_present"].to_numpy() * 100
    present_low = data["detection_present_ci95_low"].to_numpy() * 100
    present_high = data["detection_present_ci95_high"].to_numpy() * 100
    false_absent = data["false_detection_rate_operator_absent"].to_numpy() * 100
    longest_gap = data["longest_missing_run_present_seconds"].to_numpy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    x = np.arange(2)

    axes[0].bar(x, present, color=colors, width=0.58)
    axes[0].errorbar(x, present, yerr=[present - present_low, present_high - present],
                     fmt="none", ecolor=INK, capsize=5, linewidth=1.5)
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Cobertura com operador", fontweight="bold")
    axes[0].set_ylabel("Frames com face detectada (%)")
    for index, value in enumerate(present):
        axes[0].text(index, value + 3, f"{pt(value, 2)}%", ha="center", fontweight="bold")

    axes[1].bar(x, false_absent, color=colors, width=0.58)
    axes[1].set_ylim(0, max(false_absent) * 1.55)
    axes[1].set_title("Falso positivo sem operador", fontweight="bold")
    axes[1].set_ylabel("Frames com detecção incorreta (%)")
    for index, value in enumerate(false_absent):
        axes[1].text(index, value + max(false_absent) * 0.08, f"{pt(value, 2)}%",
                     ha="center", fontweight="bold")

    axes[2].bar(x, longest_gap, color=colors, width=0.58)
    axes[2].set_ylim(0, max(longest_gap) * 1.3)
    axes[2].set_title("Maior falha contínua", fontweight="bold")
    axes[2].set_ylabel("Segundos sem detectar a face")
    for index, value in enumerate(longest_gap):
        axes[2].text(index, value + max(longest_gap) * 0.05, f"{pt(value)} s",
                     ha="center", fontweight="bold")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("O OpenFace melhora a cobertura no vídeo 4", fontsize=19,
                    fontweight="bold", color=INK)
    figure.text(0.5, 0.015,
                "Evidência medida em 31.790 frames. Cobertura maior não implica indicadores equivalentes.",
                ha="center", color=MUTED, fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.9))
    _finish(figure, output)


def decision_matrix(output: Path) -> None:
    rows = [
        ("Cobertura no vídeo 4", "79,81%", "88,32%", "OpenFace"),
        ("Execução", "Python direto", "Contêiner Docker", "MediaPipe"),
        ("Pipeline atual", "Já integrado", "Exige adaptação", "MediaPipe"),
        ("Modelos existentes", "Compatíveis", "Recalibrar/retreinar", "MediaPipe"),
        ("Latência comparável", "5,25 ms mediana", "Ainda não medida", "Inconclusivo"),
        ("Evidência no projeto", "Histórico nos 4 vídeos", "Comparação em 1 vídeo", "MediaPipe"),
    ]
    figure, axis = plt.subplots(figsize=(14, 7.5))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.5, 0.94, "Matriz de decisão para o estágio atual", ha="center",
              fontsize=21, fontweight="bold", color=INK)
    axis.text(0.5, 0.895, "Dados medidos + custo prático de mudança", ha="center",
              fontsize=11, color=MUTED)

    columns = [(0.04, 0.31, "Critério"), (0.32, 0.20, "MediaPipe"),
               (0.53, 0.20, "OpenFace"), (0.74, 0.22, "Vantagem atual")]
    header_y, row_height = 0.78, 0.10
    for left, width, title in columns:
        axis.add_patch(FancyBboxPatch((left, header_y), width, 0.07,
                                     boxstyle="round,pad=0.008", facecolor=INK,
                                     edgecolor="none"))
        axis.text(left + width / 2, header_y + 0.035, title, color="white", ha="center",
                  va="center", fontweight="bold")

    for index, row in enumerate(rows):
        y = header_y - (index + 1) * row_height
        background = "#F1F5F9" if index % 2 == 0 else "white"
        axis.add_patch(FancyBboxPatch((0.04, y), 0.92, 0.085,
                                     boxstyle="round,pad=0.006", facecolor=background,
                                     edgecolor="#E2E8F0"))
        values = [row[0], row[1], row[2], row[3]]
        centers = [0.175, 0.42, 0.63, 0.85]
        for col, (center, value) in enumerate(zip(centers, values, strict=True)):
            color = GREEN if col == 3 and value != "Inconclusivo" else INK
            axis.text(center, y + 0.043, value, ha="center", va="center", fontsize=11,
                      color=color, fontweight="bold" if col in {0, 3} else "normal")

    axis.text(0.5, 0.035,
              "Conclusão: OpenFace vence em cobertura; MediaPipe vence no equilíbrio geral hoje.",
              ha="center", fontsize=14, fontweight="bold", color=GREEN)
    _finish(figure, output)


def recommendation_slide(summary: pd.DataFrame, availability: pd.DataFrame, output: Path) -> None:
    data = summary.set_index("extractor")
    delta = availability.loc[availability["candidate"].eq("openface"),
                             "detection_rate_delta_operator_present"].iloc[0] * 100
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.06, 0.9, "DECISÃO RECOMENDADA", fontsize=14, fontweight="bold", color=BLUE)
    axis.text(0.06, 0.79, "MediaPipe permanece\ncomo extrator principal", fontsize=32,
              fontweight="bold", color=INK, va="top")
    axis.text(0.06, 0.58,
              "Melhor equilíbrio entre leveza, integração\ne continuidade dos resultados já obtidos.",
              fontsize=17, color=MUTED, va="top", linespacing=1.45)

    cards = [
        (0.06, "PRONTO", "Pipeline e modelos atuais"),
        (0.265, "5,25 ms", "Latência mediana medida"),
        (0.47, "4 vídeos", "Histórico no projeto"),
    ]
    for x, value, label in cards:
        axis.add_patch(FancyBboxPatch((x, 0.29), 0.18, 0.15, boxstyle="round,pad=0.012",
                                     facecolor=LIGHT, edgecolor="#CBD5E1", linewidth=1.2))
        axis.text(x + 0.09, 0.38, value, ha="center", fontsize=19, fontweight="bold", color=BLUE)
        axis.text(x + 0.09, 0.325, label, ha="center", fontsize=10, color=MUTED)

    axis.add_patch(FancyBboxPatch((0.70, 0.20), 0.25, 0.64, boxstyle="round,pad=0.02",
                                 facecolor="#FFF7ED", edgecolor="#FDBA74", linewidth=1.5))
    axis.text(0.825, 0.76, "OpenFace", ha="center", fontsize=24, fontweight="bold", color=ORANGE)
    axis.text(0.825, 0.66, f"+{pt(delta, 2)} p.p.", ha="center", fontsize=34,
              fontweight="bold", color=ORANGE)
    axis.text(0.825, 0.59, "de cobertura no vídeo 4", ha="center", fontsize=12, color=MUTED)
    axis.plot([0.74, 0.91], [0.53, 0.53], color="#FDBA74", linewidth=1)
    axis.text(0.825, 0.46, "Alternativa promissora", ha="center", fontsize=14,
              fontweight="bold", color=INK)
    axis.text(0.825, 0.37,
              "Ainda requer:\n• comparação nos demais vídeos\n• recalibração dos indicadores\n• retreinamento dos modelos",
              ha="center", va="center", fontsize=12, color=MUTED, linespacing=1.5)
    axis.text(0.06, 0.12,
              "OpenFace vence em cobertura. MediaPipe vence no trade-off geral do projeto atual.",
              fontsize=15, fontweight="bold", color=GREEN)
    _finish(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path,
                        default=Path("fase_2/outputs/metrics/G48A"))
    parser.add_argument("--video-id", default="video_04")
    parser.add_argument("--output", type=Path,
                        default=Path("fase_2/outputs/figures/G48A/decision"))
    args = parser.parse_args()

    prefix = args.metrics / f"g48a_{args.video_id}"
    summary = pd.read_csv(f"{prefix}_extractor_summary.csv")
    availability = pd.read_csv(f"{prefix}_pairwise_availability.csv")
    required = {"mediapipe", "openface"}
    if not required.issubset(set(summary["extractor"])):
        raise ValueError("As métricas precisam conter MediaPipe e OpenFace")

    evidence_figure(summary, args.output / "01_evidencia_medida.png")
    decision_matrix(args.output / "02_matriz_decisao.png")
    recommendation_slide(summary, availability, args.output / "03_recomendacao.png")
    print(f"3 figuras geradas em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
