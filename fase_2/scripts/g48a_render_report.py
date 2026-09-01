#!/usr/bin/env python3
"""Gera painéis de validação visual e HTML resumido da G1 G48A."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EXTRACTORS = ("mediapipe_face_mesh", "insightface_2d106", "openface_68")
COLORS = ((0, 255, 0), (0, 200, 255), (255, 100, 30))


def _panel(frame: np.ndarray, name: str, points: list[list[float]] | None, detected: bool) -> np.ndarray:
    canvas = frame.copy()
    if points:
        for index, point in enumerate(points):
            center = tuple(int(round(value)) for value in point[:2])
            cv2.circle(canvas, center, 3, COLORS[index // 6 if index < 12 else 2], -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (center[0] + 3, center[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, f"{name}: {'OK' if detected else 'N/D'}", (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 80) if detected else (80, 80, 255), 1, cv2.LINE_AA)
    return canvas


def _video_panel(
    frame: np.ndarray,
    name: str,
    points: list[list[float]] | None,
    row: pd.Series,
) -> np.ndarray:
    canvas = _panel(frame, name, points, bool(row.face_detected))
    footer = np.zeros((66, canvas.shape[1], 3), dtype=np.uint8)
    if bool(row.face_detected):
        line1 = f"EAR {row.ear:.3f} | MAR {row.mar:.3f}"
        line2 = f"pitch {row.pitch:+.1f} | yaw {row.yaw:+.1f} | roll {row.roll:+.1f}"
    else:
        line1 = "EAR N/D | MAR N/D"
        line2 = f"pose N/D | {row.failure_reason}"
    cv2.putText(footer, line1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(footer, line2, (8, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (230, 230, 230), 1, cv2.LINE_AA)
    return cv2.vconcat((canvas, footer))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke"))
    parser.add_argument("--manifest", type=Path, default=Path("fase_2/data/manifests/g48a_smoke_30.csv"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    metrics = {name: pd.read_csv(args.root / "metrics" / f"{name}.csv") for name in EXTRACTORS}
    points = {name: json.loads((args.root / "metrics" / f"{name}_landmarks.json").read_text())
              for name in EXTRACTORS}
    figures = args.root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for difficulty, subset in manifest.groupby("difficulty", sort=False):
        tiles = []
        for target in subset.itertuples(index=False):
            frame = cv2.imread(str(args.root / "frames" / f"{target.sample_id}.jpg"))
            panels = []
            for name in EXTRACTORS:
                row = metrics[name].query("video_id == @target.video_id and frame_id == @target.frame_id").iloc[0]
                panels.append(_panel(frame, name, points[name].get(target.sample_id), bool(row.face_detected)))
            tile = cv2.hconcat(panels)
            cv2.rectangle(tile, (0, tile.shape[0] - 24), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
            cv2.putText(tile, f"{target.sample_id} | {target.temporal_class} | {difficulty}",
                        (8, tile.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(tile)
        blank = np.zeros_like(tiles[0])
        while len(tiles) % 2:
            tiles.append(blank)
        sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(figures / f"landmarks_{difficulty}.jpg"), sheet)

    # Slideshow sincronizado: cada alvo permanece 1,5 s na tela, nos três extratores.
    fps = 10
    frames_per_target = 15
    first = cv2.imread(str(args.root / "frames" / f"{manifest.iloc[0].sample_id}.jpg"))
    output_size = (first.shape[1] * 3, first.shape[0] + 66 + 34)
    writer = cv2.VideoWriter(
        str(figures / "comparison_3_extractors.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Não foi possível criar o vídeo comparativo")
    for target in manifest.itertuples(index=False):
        frame = cv2.imread(str(args.root / "frames" / f"{target.sample_id}.jpg"))
        panels = []
        for name in EXTRACTORS:
            row = metrics[name].query("video_id == @target.video_id and frame_id == @target.frame_id").iloc[0]
            panels.append(_video_panel(frame, name, points[name].get(target.sample_id), row))
        comparison = cv2.hconcat(panels)
        header = np.zeros((34, comparison.shape[1], 3), dtype=np.uint8)
        label = f"{target.sample_id} | classe {target.temporal_class} | dificuldade {target.difficulty}"
        cv2.putText(header, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
        comparison = cv2.vconcat((header, comparison))
        for _ in range(frames_per_target):
            writer.write(comparison)
    writer.release()
    summary = []
    for name, frame in metrics.items():
        summary.append({"extractor": name, "faces": int(frame.face_detected.sum()),
                        "ear_valid": int(frame.ear_valid.sum()), "mar_valid": int(frame.mar_valid.sum()),
                        "pose_valid": int(frame.head_pose_valid.sum()),
                        "median_ms": float(frame.inference_time_ms.median())})
    table = pd.DataFrame(summary).to_html(index=False, float_format=lambda value: f"{value:.2f}")
    links = "".join(f'<li><a href="../figures/landmarks_{html.escape(level)}.jpg">{html.escape(level)}</a></li>'
                    for level in manifest.difficulty.unique())
    report = args.root / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>G48A G1 smoke</title>"
        "<h1>G48A — G1 Smoke Test</h1><p>Contagens descritivas; não constituem ranking.</p>"
        + table
        + '<h2>Visualização simultânea</h2><video controls width="100%" '
          'src="../figures/comparison_3_extractors.mp4"></video>'
        + "<h2>Painéis dos 22 pontos canônicos</h2><ul>" + links + "</ul>", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
