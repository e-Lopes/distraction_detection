"""InsightFace detector + 2D-106 landmark benchmark on CPU or CUDA."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import cv2
import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis

sys.path.insert(0, "/app")
from src.common.metrics import discover_videos, percentile, timer_ms
from src.common.resource_monitor import ResourceMonitor
from src.common.schema import (FrameCSVWriter, FrameResult, VideoSummary,
                               append_summary, result_stem, write_run_metadata)

FRAMEWORK = "insightface"

def verify_providers(app: FaceAnalysis, backend: str) -> dict:
    loaded = {}
    for task, model in app.models.items():
        session = getattr(model, "session", None)
        loaded[task] = session.get_providers() if session else []
    if backend == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA solicitado, mas CUDAExecutionProvider nao esta disponivel")
        failed = [task for task, providers in loaded.items()
                  if not providers or providers[0] != "CUDAExecutionProvider"]
        if failed:
            raise RuntimeError(f"Modelos nao executando prioritariamente em CUDA: {failed}; {loaded}")
    return loaded

def process_video(app, video_id, video_path, args, backend, model_load_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[insightface] AVISO: nao foi possivel abrir {video_path}")
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    for _ in range(args.warmup_frames):
        ok, frame = cap.read()
        if not ok:
            break
        app.get(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stem = result_stem(args.output_dir, FRAMEWORK, backend, args.run_id,
                       args.repetition, args.dataset, video_id)
    latencies, n_detected, frame_idx = [], 0, 0
    monitor = ResourceMonitor(interval_s=args.monitor_interval)
    monitor.start()
    t_start = time.perf_counter()
    with FrameCSVWriter(stem + "__frames.csv") as writer:
        while not args.max_frames or frame_idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            with timer_ms() as measured:
                faces = app.get(frame)
            latency = measured["ms"]
            latencies.append(latency)
            detected, confidence, n_landmarks = bool(faces), None, 0
            if detected:
                face = max(faces, key=lambda item: float(getattr(item, "det_score", 0.0)))
                confidence = float(getattr(face, "det_score", 0.0))
                points = getattr(face, "landmark_2d_106", None)
                n_landmarks = int(points.shape[0]) if points is not None else 0
                # Detection without the requested landmark output is a failure here.
                detected = n_landmarks == 106
            n_detected += int(detected)
            writer.write(FrameResult(args.run_id, args.repetition, FRAMEWORK, backend,
                args.dataset, video_id, frame_idx, frame_idx / fps, round(latency, 4),
                detected, n_landmarks, confidence))
            frame_idx += 1
    duration = time.perf_counter() - t_start
    resources = monitor.stop()
    cap.release()
    return VideoSummary(args.run_id, args.repetition, FRAMEWORK, backend, args.dataset,
        video_id, "frame", "inference_only", frame_idx, n_detected, round(duration, 4),
        round(frame_idx / duration, 3) if duration else 0.0,
        round(percentile(latencies, 50), 4), round(percentile(latencies, 90), 4),
        round(percentile(latencies, 95), 4), round(max(latencies), 4) if latencies else 0.0,
        resources.avg_cpu_pct, resources.peak_rss_mb, resources.avg_gpu_util_pct,
        resources.peak_gpu_mem_mb, resources.gpu_available, round(model_load_s, 4),
        width, height, round(fps, 4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--model-pack", default="buffalo_l")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()
    videos = discover_videos(args.input)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise SystemExit(f"Nenhum video encontrado em {args.input}")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.backend == "cuda"
                 else ["CPUExecutionProvider"])
    load_start = time.perf_counter()
    # Excludes recognition, age/gender and other buffalo_l modules from timing.
    app = FaceAnalysis(name=args.model_pack, providers=providers,
                       allowed_modules=["detection", "landmark_2d_106"])
    app.prepare(ctx_id=0 if args.backend == "cuda" else -1,
                det_size=(args.det_size, args.det_size))
    model_load_s = time.perf_counter() - load_start
    actual = verify_providers(app, args.backend)
    write_run_metadata(args.output_dir, FRAMEWORK, args.backend, args.run_id, args.repetition,
        {"insightface_version": insightface.__version__, "onnxruntime_version": ort.__version__,
         "available_providers": ort.get_available_providers(), "model_providers": actual,
         "allowed_modules": ["detection", "landmark_2d_106"], "model_pack": args.model_pack,
         "det_size": args.det_size, "warmup_frames": args.warmup_frames,
         "python": platform.python_version()})
    for index, (video_id, path) in enumerate(videos, 1):
        print(f"[insightface/{args.backend}] {index}/{len(videos)} {video_id}")
        summary = process_video(app, video_id, path, args, args.backend, model_load_s)
        if summary:
            append_summary(os.path.join(args.output_dir, "aggregate_summary.csv"), summary)

if __name__ == "__main__":
    main()
