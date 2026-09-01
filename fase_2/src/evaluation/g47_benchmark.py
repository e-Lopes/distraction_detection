"""Benchmark pareado MediaPipe Face Mesh x YOLO26-Pose facial da G4.7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

from ..features.extract_facial_series import load_roi
from ..features.g47_extractors import MediaPipeFaceMeshExtractor, YOLOFacePoseExtractor
from ..features.g47_filters import LandmarkFilter, bland_altman, make_filter, temporal_jitter
from ..features.g47_schema import (
    FacialIndicators,
    compute_coco_head_pose,
    compute_indicators,
    load_schema,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INDICATORS = ("ear", "mar", "pitch", "yaw", "roll")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _crop(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI inválida")
    if x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("ROI ultrapassa as dimensões do vídeo")
    return frame[y : y + height, x : x + width]


def _empty_indicators() -> FacialIndicators:
    return FacialIndicators(*(math.nan for _ in range(7)))


def _yolo_indicators(points: np.ndarray, width: int, height: int) -> FacialIndicators:
    if points.shape == (22, 3):
        return compute_indicators(points, width, height)
    if points.shape == (17, 3):
        pitch, yaw, roll = compute_coco_head_pose(points, width, height, 0.05)
        return FacialIndicators(math.nan, math.nan, math.nan, math.nan, pitch, yaw, roll)
    return _empty_indicators()


def _indicator_dict(prefix: str, values: FacialIndicators) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in asdict(values).items()}


def _percentiles(values: pd.Series) -> dict[str, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return {"mean": math.nan, "variance": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan}
    return {
        "mean": float(finite.mean()),
        "variance": float(finite.var(ddof=1)) if len(finite) > 1 else 0.0,
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
    }


def _gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    first = completed.stdout.strip().splitlines()[0].split(",")
    if len(first) != 4:
        return {}
    return {
        "gpu_name": first[0].strip(),
        "gpu_memory_used_mb": float(first[1]),
        "gpu_utilization_percent": float(first[2]),
        "gpu_power_watts": float(first[3]),
    }


def _warmup(capture: cv2.VideoCapture, extractors: Sequence[object], count: int, roi: object) -> int:
    warmed = 0
    while warmed < count:
        success, frame = capture.read()
        if not success:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        region = _crop(frame, roi)
        for extractor in extractors:
            extractor.detect(region)
        warmed += 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return warmed


def run_benchmark(
    *,
    video_path: Path,
    video_id: str,
    media_pipe: MediaPipeFaceMeshExtractor,
    yolo: YOLOFacePoseExtractor,
    output_csv: Path,
    roi: tuple[int, int, int, int] | None = None,
    warmup_frames: int = 200,
    start_frame: int = 0,
    max_frames: int | None = None,
    progress_every: int = 500,
    landmark_filter: LandmarkFilter | None = None,
    filter_reset_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Executa comparação pareada; frames de warmup nunca entram na saída."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if warmup_frames < 0 or start_frame < 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("warmup deve ser não negativo e max_frames deve ser positivo")
    if filter_reset_seconds <= 0:
        raise ValueError("filter_reset_seconds deve ser positivo")
    rows: list[dict[str, object]] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    try:
        if warmup_frames:
            _warmup(capture, (media_pipe, yolo), warmup_frames, roi)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if landmark_filter:
            landmark_filter.reset()
        last_yolo_detection: float | None = None
        frame_index = start_frame
        processed = 0
        while max_frames is None or processed < max_frames:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps if fps > 0 else float(frame_index)
            region = _crop(frame, roi)
            height, width = region.shape[:2]

            # Alternar a ordem reduz viés térmico/ordem sem alterar a associação por frame.
            detections: dict[str, object] = {}
            totals: dict[str, float] = {}
            order = (("mp", media_pipe), ("yolo", yolo))
            if frame_index % 2:
                order = tuple(reversed(order))
            indicators: dict[str, FacialIndicators] = {}
            for name, extractor in order:
                total_started = time.perf_counter_ns()
                detection = extractor.detect(region)
                if detection.points is None:
                    values = _empty_indicators()
                elif name == "yolo":
                    values = _yolo_indicators(detection.points, width, height)
                else:
                    values = compute_indicators(detection.points, width, height)
                totals[name] = (time.perf_counter_ns() - total_started) / 1e6
                detections[name] = detection
                indicators[name] = values

            yolo_detection = detections["yolo"]
            filtered = _empty_indicators()
            if yolo_detection.points is not None:
                if (
                    landmark_filter
                    and last_yolo_detection is not None
                    and timestamp - last_yolo_detection > filter_reset_seconds
                ):
                    landmark_filter.reset()
                last_yolo_detection = timestamp
                if landmark_filter and yolo_detection.points.shape == (22, 3):
                    filtered_points = landmark_filter.update(yolo_detection.points, timestamp)
                    filtered = compute_indicators(filtered_points, width, height)

            mp_detection = detections["mp"]
            row: dict[str, object] = {
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "mp_detected": int(mp_detection.points is not None),
                "yolo_detected": int(yolo_detection.points is not None),
                "mp_confidence": mp_detection.confidence,
                "yolo_confidence": yolo_detection.confidence,
                "mp_failure": mp_detection.failure,
                "yolo_failure": yolo_detection.failure,
                "mp_inference_ms": mp_detection.inference_ms,
                "yolo_inference_ms": yolo_detection.inference_ms,
                "mp_total_ms": totals["mp"],
                "yolo_total_ms": totals["yolo"],
            }
            row.update(_indicator_dict("mp", indicators["mp"]))
            row.update(_indicator_dict("yolo", indicators["yolo"]))
            row.update(_indicator_dict("yolo_filtered", filtered))
            rows.append(row)
            frame_index += 1
            processed += 1
            if progress_every and processed % progress_every == 0:
                elapsed = time.perf_counter() - started_wall
                coverage_mp = np.mean([item["mp_detected"] for item in rows])
                coverage_yolo = np.mean([item["yolo_detected"] for item in rows])
                print(
                    f"{video_id}: {processed} frames (fonte {frame_index}/{source_frames}) | "
                    f"{processed / elapsed:.2f} FPS | "
                    f"MP {coverage_mp:.1%} | YOLO {coverage_yolo:.1%}",
                    flush=True,
                )
    finally:
        capture.release()

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Benchmark não processou frames")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output_csv)
    wall_seconds = time.perf_counter() - started_wall
    summary: dict[str, object] = {
        "video_id": video_id,
        "source_frames": source_frames,
        "processed_frames": len(frame),
        "source_fps": fps,
        "wall_seconds": wall_seconds,
        "cpu_seconds": time.process_time() - started_cpu,
        "throughput_fps": len(frame) / wall_seconds,
        "mp_coverage": float(frame["mp_detected"].mean()),
        "yolo_coverage": float(frame["yolo_detected"].mean()),
        "mp_inference": _percentiles(frame["mp_inference_ms"]),
        "yolo_inference": _percentiles(frame["yolo_inference_ms"]),
        "mp_total": _percentiles(frame["mp_total_ms"]),
        "yolo_total": _percentiles(frame["yolo_total_ms"]),
    }
    summary.update(_gpu_snapshot())
    return frame, summary


def agreement_metrics(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for indicator in INDICATORS:
        first = pd.to_numeric(frame[f"mp_{indicator}"], errors="coerce").to_numpy()
        second = pd.to_numeric(frame[f"yolo_{indicator}"], errors="coerce").to_numpy()
        valid = np.isfinite(first) & np.isfinite(second)
        agreement = bland_altman(first, second)
        rows.append(
            {
                "indicator": indicator,
                **agreement,
                "mae": float(np.mean(np.abs(second[valid] - first[valid]))) if valid.any() else math.nan,
                "rmse": (
                    float(np.sqrt(np.mean(np.square(second[valid] - first[valid]))))
                    if valid.any()
                    else math.nan
                ),
                "mp_jitter": temporal_jitter(first),
                "yolo_jitter": temporal_jitter(second),
            }
        )
    return rows


def plot_results(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    time_axis = frame["timestamp_seconds"]
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)
    axes[0].plot(time_axis, frame["mp_ear"], label="MediaPipe EAR", alpha=0.8)
    axes[0].plot(time_axis, frame["yolo_ear"], label="YOLO EAR", alpha=0.65)
    axes[0].plot(time_axis, frame["mp_mar"], label="MediaPipe MAR", alpha=0.7)
    axes[0].plot(time_axis, frame["yolo_mar"], label="YOLO MAR", alpha=0.55)
    axes[0].set_ylabel("Razão")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.2)
    colors = {"pitch": "tab:blue", "yaw": "tab:orange", "roll": "tab:green"}
    for indicator, color in colors.items():
        axes[1].plot(time_axis, frame[f"mp_{indicator}"], color=color, label=f"MP {indicator}")
        axes[1].plot(
            time_axis,
            frame[f"yolo_{indicator}"],
            color=color,
            linestyle="--",
            alpha=0.65,
            label=f"YOLO {indicator}",
        )
    axes[1].set_ylabel("Ângulo (graus)")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.2)
    latency = frame[["mp_total_ms", "yolo_total_ms"]].rename(
        columns={"mp_total_ms": "MediaPipe", "yolo_total_ms": "YOLO26"}
    )
    means = latency.mean()
    errors = latency.std(ddof=1)
    axes[2].bar(means.index, means.values, yerr=errors.values, capsize=5)
    axes[2].set_ylabel("Latência total (ms)")
    axes[2].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    bland, bland_axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, indicator in zip(bland_axes, ("pitch", "yaw", "roll"), strict=True):
        first = pd.to_numeric(frame[f"mp_{indicator}"], errors="coerce").to_numpy()
        second = pd.to_numeric(frame[f"yolo_{indicator}"], errors="coerce").to_numpy()
        valid = np.isfinite(first) & np.isfinite(second)
        means_pair = (first[valid] + second[valid]) / 2
        differences = second[valid] - first[valid]
        stats = bland_altman(first, second)
        axis.scatter(means_pair, differences, s=3, alpha=0.15)
        for value, style in ((stats["bias"], "-"), (stats["lower"], "--"), (stats["upper"], "--")):
            axis.axhline(value, color="tab:red", linestyle=style)
        axis.set_title(indicator)
        axis.set_xlabel("Média dos métodos (°)")
        axis.grid(alpha=0.2)
    bland_axes[0].set_ylabel("YOLO − MediaPipe (°)")
    bland.tight_layout()
    bland.savefig(output.with_name(output.stem + "_bland_altman.png"), dpi=180, bbox_inches="tight")
    plt.close(bland)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument(
        "--schema", type=Path, default=Path("fase_2/configs/features/g47_face_landmarks.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("fase_2/outputs/metrics/G47"))
    parser.add_argument("--figure-dir", type=Path, default=Path("fase_2/outputs/figures/G47"))
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--filter", choices=("none", "ema", "one_euro"), default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_schema(args.schema)
    if not args.video.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {args.video}")
    output_csv = args.output_dir / f"{args.video_id}__{args.model.stem}__{args.device.replace(':', '_')}.csv"
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Saída existente: {output_csv}. Use --overwrite.")
    roi = load_roi(args.roi_config)
    with ExitStack() as stack:
        media_pipe = stack.enter_context(MediaPipeFaceMeshExtractor())
        yolo = stack.enter_context(
            YOLOFacePoseExtractor(args.model, device=args.device, image_size=args.imgsz)
        )
        frame, summary = run_benchmark(
            video_path=args.video,
            video_id=args.video_id,
            media_pipe=media_pipe,
            yolo=yolo,
            output_csv=output_csv,
            roi=roi,
            warmup_frames=args.warmup_frames,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            progress_every=args.progress_every,
            landmark_filter=make_filter(args.filter),
        )
    summary.update(
        {
            "model": args.model.as_posix(),
            "model_sha256": _sha256(args.model),
            "device": args.device,
            "filter": args.filter,
            "python": platform.python_version(),
            "schema_sha256": _sha256(args.schema),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_csv.stem
    (args.output_dir / f"{stem}__summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(agreement_metrics(frame)).to_csv(
        args.output_dir / f"{stem}__agreement.csv", index=False
    )
    plot_results(frame, args.figure_dir / f"{stem}.png")
    print(f"Benchmark concluído: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
