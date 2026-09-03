#!/usr/bin/env python3
"""Executa um extrator G48A sequencialmente em todos os frames dos vídeos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from fase_2.src.evaluation.g48a_contract import (
    empty_result,
    map_to_canonical,
    result_from_landmarks,
    select_operator_face,
)
from fase_2.src.features.g47_extractors import MediaPipeFaceMeshExtractor

INSIGHTFACE_MAPPING = [93, 96, 95, 89, 90, 91, 35, 41, 42, 39, 37, 36, 67, 68, 71, 64, 52, 55, 53, 58, 86, 0]


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    mins, maxs = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return tuple(float(value) for value in (*mins, *maxs))


def _insightface(model_root: Path, device: str):
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDA solicitada, mas CUDAExecutionProvider não está disponível no ambiente "
            f"g48_insightface (providers: {available})."
        )
    use_cuda = device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in available)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda
                 else ["CPUExecutionProvider"])
    print(f"InsightFace providers: {providers}", flush=True)

    app = FaceAnalysis(name="buffalo_l", root=str(model_root),
                       allowed_modules=["detection", "landmark_2d_106"],
                       providers=providers)
    app.prepare(ctx_id=0 if use_cuda else -1, det_thresh=0.5, det_size=(640, 640))
    return app


def _is_complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        return max(sum(1 for _ in stream) - 1, 0) == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", required=True, choices=("mediapipe", "insightface"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Provider do InsightFace; MediaPipe Face Mesh Python usa CPU.")
    parser.add_argument("--videos", type=Path, default=Path("fase_2/data/manifests/videos.csv"))
    parser.add_argument("--roi", type=Path, default=Path("fase_2/configs/preprocessing/legacy_roi.json"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/G48A_all_frames"))
    parser.add_argument("--model-root", type=Path,
                        default=Path("fase_2/outputs/cache/G48A_framework_smoke/insightface"))
    parser.add_argument("--video-id", action="append", help="Restringe a um vídeo; repetível.")
    parser.add_argument("--max-frames", type=int, help="Limite por vídeo para smoke/diagnóstico.")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Intervalo de atualização do progresso; zero desativa.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames deve ser positivo")
    if args.progress_every < 0:
        parser.error("--progress-every não pode ser negativo")

    with args.videos.open(encoding="utf-8", newline="") as stream:
        videos = [SimpleNamespace(**row) for row in csv.DictReader(stream)]
    if args.video_id:
        videos = [video for video in videos if video.video_id in args.video_id]
        missing = sorted(set(args.video_id) - {video.video_id for video in videos})
        if missing:
            parser.error(f"video_id inexistente: {', '.join(missing)}")
    roi = json.loads(args.roi.read_text(encoding="utf-8"))["roi_cadeira"]
    roi_x, roi_y, roi_width, roi_height = (int(value) for value in roi)
    out_dir = args.output / "metrics" / args.extractor
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = (MediaPipeFaceMeshExtractor(minimum_confidence=0.5)
                 if args.extractor == "mediapipe" else _insightface(args.model_root, args.device))

    try:
        for meta in videos:
            expected = min(int(meta.num_frames), args.max_frames or int(meta.num_frames))
            destination = out_dir / f"{meta.video_id}.csv"
            if not args.overwrite and _is_complete(destination, expected):
                print(f"{meta.video_id}: completo, reutilizado ({expected} frames)", flush=True)
                continue
            temporary = destination.with_suffix(".csv.tmp")
            capture = cv2.VideoCapture(str(meta.relative_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Vídeo não pôde ser aberto: {meta.relative_path}")
            detected = 0
            started_video = time.monotonic()
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = None
                for frame_id in range(expected):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"{meta.video_id}: leitura terminou no frame {frame_id}/{expected}")
                    frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                    timestamp_ms = frame_id * 1000.0 / float(meta.fps)
                    if args.extractor == "mediapipe":
                        detection = extractor.detect(frame)
                        if detection.points is None:
                            result = empty_result(video_id=meta.video_id, frame_id=frame_id,
                                                  timestamp_ms=timestamp_ms, extractor="mediapipe_face_mesh",
                                                  extractor_version="0.10.21", inference_time_ms=detection.inference_ms,
                                                  failure_reason=detection.failure or "face_missing")
                        else:
                            result = result_from_landmarks(points=detection.points, width=roi_width, height=roi_height,
                                                           video_id=meta.video_id, frame_id=frame_id,
                                                           timestamp_ms=timestamp_ms, extractor="mediapipe_face_mesh",
                                                           extractor_version="0.10.21", face_bbox=_bbox(detection.points),
                                                           confidence=detection.confidence,
                                                           inference_time_ms=detection.inference_ms)
                    else:
                        began = time.perf_counter_ns()
                        faces = extractor.get(frame)
                        elapsed = (time.perf_counter_ns() - began) / 1e6
                        try:
                            index = select_operator_face(np.asarray([face.bbox for face in faces]),
                                                         width=roi_width, height=roi_height)
                        except ValueError:
                            result = empty_result(video_id=meta.video_id, frame_id=frame_id,
                                                  timestamp_ms=timestamp_ms, extractor="insightface_2d106",
                                                  extractor_version="0.7.3/buffalo_l", inference_time_ms=elapsed,
                                                  failure_reason="face_missing" if not faces else "operator_face_missing")
                        else:
                            face = faces[index]
                            points = map_to_canonical(np.asarray(face.landmark_2d_106, dtype=float), INSIGHTFACE_MAPPING)
                            result = result_from_landmarks(points=points, width=roi_width, height=roi_height,
                                                           video_id=meta.video_id, frame_id=frame_id,
                                                           timestamp_ms=timestamp_ms, extractor="insightface_2d106",
                                                           extractor_version="0.7.3/buffalo_l",
                                                           face_bbox=tuple(float(v) for v in face.bbox),
                                                           confidence=float(face.det_score), inference_time_ms=elapsed)
                    row = result.to_dict()
                    detected += int(row["face_detected"])
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    processed = frame_id + 1
                    if args.progress_every and processed % args.progress_every == 0:
                        stream.flush()
                        elapsed_progress = max(time.monotonic() - started_video, 1e-9)
                        rate = processed / elapsed_progress
                        eta = (expected - processed) / rate
                        print(
                            f"{meta.video_id}: {processed}/{expected} "
                            f"({processed / expected:6.2%}) | {rate:6.2f} frames/s | "
                            f"ETA {eta / 60:6.1f} min",
                            flush=True,
                        )
            capture.release()
            os.replace(temporary, destination)
            elapsed_video = time.monotonic() - started_video
            print(f"{meta.video_id}: {detected}/{expected} faces; {elapsed_video:.1f}s", flush=True)
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
