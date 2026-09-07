#!/usr/bin/env python3
"""Executa OpenFace em todos os frames, por lotes PNG lossless e sem display."""
from __future__ import annotations

import argparse, csv, json, math, os, shutil, subprocess, time
from pathlib import Path
import cv2
import numpy as np
from fase_2.src.evaluation.g48a_contract import (
    OPENFACE_MAPPING, empty_result, map_to_canonical, result_from_landmarks,
)

IMAGE = "algebr/openface@sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197"
VERSION = "2.0-era/sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197"

def _valid_openface_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aceita a saída de imagem (sem ``success``) e a saída de tracking (com a coluna)."""
    return [row for row in rows if "success" not in row or int(float(row["success"])) == 1]

def _count_rows(path: Path) -> int:
    if not path.is_file(): return 0
    with path.open(encoding="utf-8", newline="") as stream:
        return max(sum(1 for _ in stream) - 1, 0)

def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}

def _run_batch(frames_dir: Path, raw_dir: Path, log_path: Path, verbose: bool) -> None:
    command = [
        "docker", "run", "--rm",
        "--entrypoint", "/home/openface-build/build/bin/FaceLandmarkImg",
        "-v", f"{frames_dir.resolve()}:/input:ro", "-v", f"{raw_dir.resolve()}:/output",
        IMAGE, "-fdir", "/input", "-out_dir", "/output", "-2Dfp", "-pose",
    ]
    if verbose:
        subprocess.run(command, check=True)
        return
    with log_path.open("a", encoding="utf-8") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"OpenFace falhou; log completo: {log_path}", flush=True)
            raise

def _result(raw_path: Path, video_id: str, frame_id: int, fps: float, width: int, height: int):
    timestamp_ms = frame_id * 1000.0 / fps
    if raw_path.is_file():
        with raw_path.open(encoding="utf-8-sig", newline="") as stream:
            valid = _valid_openface_rows(list(csv.DictReader(stream, skipinitialspace=True)))
    else: valid = []
    if not valid:
        reason = "openface_tracking_failed" if raw_path.is_file() else "openface_output_missing"
        return empty_result(video_id=video_id, frame_id=frame_id, timestamp_ms=timestamp_ms,
                            extractor="openface_68", extractor_version=VERSION,
                            inference_time_ms=math.nan, failure_reason=reason)
    row = max(valid, key=lambda item: float(item.get("confidence", 0)))
    points = np.column_stack(([float(row[f"x_{i}"]) for i in range(68)],
                              [float(row[f"y_{i}"]) for i in range(68)]))
    canonical = map_to_canonical(points, OPENFACE_MAPPING)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    return result_from_landmarks(points=canonical, width=width, height=height,
        video_id=video_id, frame_id=frame_id, timestamp_ms=timestamp_ms, extractor="openface_68",
        extractor_version=VERSION, face_bbox=tuple(float(v) for v in (*mins, *maxs)),
        confidence=float(row["confidence"]), inference_time_ms=math.nan)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("fase_2/data/manifests/videos.csv"))
    parser.add_argument("--roi", type=Path, default=Path("fase_2/configs/preprocessing/legacy_roi.json"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/G48A_all_frames"))
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--docker-verbose", action="store_true",
                        help="Exibe também o log interno detalhado do OpenFace.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-detections", action="store_true",
                        help="Recalcula somente frames positivos do CSV completo existente.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0: parser.error("--max-frames deve ser positivo")
    if args.batch_size <= 0: parser.error("--batch-size deve ser positivo")
    with args.videos.open(encoding="utf-8", newline="") as stream: videos = list(csv.DictReader(stream))
    if args.video_id: videos = [row for row in videos if row["video_id"] in args.video_id]
    x, y, width, height = map(int, json.loads(args.roi.read_text())["roi_cadeira"])
    metrics_dir, cache_root = args.output / "metrics/openface", args.output / "cache/openface_batches"
    log_dir = args.output / "logs/openface"
    metrics_dir.mkdir(parents=True, exist_ok=True); cache_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for meta in videos:
        video_id, fps = meta["video_id"], float(meta["fps"])
        expected = min(int(meta["num_frames"]), args.max_frames or int(meta["num_frames"]))
        destination = metrics_dir / f"{video_id}.csv"
        reused: dict[int, dict[str, str]] = {}
        if args.reuse_detections:
            if _count_rows(destination) != expected:
                raise ValueError(f"{destination} precisa conter {expected} linhas para reutilização")
            with destination.open(encoding="utf-8", newline="") as stream:
                reused = {int(row["frame_id"]): row for row in csv.DictReader(stream)}
        elif not args.overwrite and _count_rows(destination) == expected:
            print(f"{video_id}: completo, reutilizado ({expected} frames)", flush=True); continue
        work = cache_root / video_id; frames_dir, raw_dir = work / "frames", work / "raw"
        shutil.rmtree(work, ignore_errors=True); frames_dir.mkdir(parents=True); raw_dir.mkdir(parents=True)
        temporary = destination.with_suffix(".csv.tmp")
        capture = cv2.VideoCapture(meta["relative_path"])
        if not capture.isOpened(): raise FileNotFoundError(meta["relative_path"])
        started, detected = time.monotonic(), 0
        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = None
                for start in range(0, expected, args.batch_size):
                    end = min(start + args.batch_size, expected)
                    selected: set[int] = set()
                    for frame_id in range(start, end):
                        ok, frame = capture.read()
                        if not ok: raise RuntimeError(f"{video_id}: leitura terminou em {frame_id}/{expected}")
                        if reused and not _csv_bool(reused[frame_id]["face_detected"]):
                            continue
                        selected.add(frame_id)
                        path = frames_dir / f"frame_{frame_id:09d}.png"
                        if not cv2.imwrite(str(path), frame[y:y+height, x:x+width]): raise RuntimeError(str(path))
                    batch_number = start // args.batch_size + 1
                    total_batches = (expected + args.batch_size - 1) // args.batch_size
                    print(f"{video_id}: lote {batch_number}/{total_batches} no OpenFace "
                          f"(frames {start}-{end - 1})", flush=True)
                    if selected:
                        _run_batch(frames_dir, raw_dir, log_dir / f"{video_id}.log", args.docker_verbose)
                    for frame_id in range(start, end):
                        if reused and frame_id not in selected:
                            row = reused[frame_id]
                        else:
                            row = _result(raw_dir / f"frame_{frame_id:09d}.csv", video_id, frame_id,
                                          fps, width, height).to_dict()
                        detected += int(_csv_bool(row["face_detected"]))
                        if writer is None:
                            writer = csv.DictWriter(output, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row)
                    output.flush()
                    rate = end / max(time.monotonic() - started, 1e-9); eta = (expected-end)/rate
                    print(f"{video_id}: {end}/{expected} ({end/expected:6.2%}) | {rate:6.2f} frames/s | ETA {eta/60:6.1f} min", flush=True)
                    for folder in (frames_dir, raw_dir):
                        for path in folder.iterdir(): path.unlink()
            os.replace(temporary, destination)
            print(f"{video_id}: {detected}/{expected} faces", flush=True)
        finally:
            capture.release(); shutil.rmtree(work, ignore_errors=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
