"""MediaPipe Tasks FaceLandmarker benchmark (478 landmarks, CPU)."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, "/app")
from src.common.metrics import discover_videos, percentile, timer_ms
from src.common.resource_monitor import ResourceMonitor
from src.common.schema import (FrameCSVWriter, FrameResult, VideoSummary,
                               append_summary, result_stem, write_run_metadata)

FRAMEWORK = "mediapipe"
BACKEND = "cpu"
DEFAULT_MODEL_PATH = "/app/models/face_landmarker.task"

def create_landmarker(model_path: str):
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

def process_video(landmarker, video_id, video_path, args, model_load_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[mediapipe] AVISO: nao foi possivel abrir {video_path}")
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Warm-up is excluded from all measurements. Rewind preserves identical input.
    warmup_done = 0
    while warmup_done < args.warmup_frames:
        ok, frame = cap.read()
        if not ok:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_for_video(image, int(round(warmup_done * 1000.0 / fps)))
        warmup_done += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stem = result_stem(args.output_dir, FRAMEWORK, BACKEND, args.run_id,
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
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(round((warmup_done + 1 + frame_idx) * 1000.0 / fps))
            with timer_ms() as measured:
                result = landmarker.detect_for_video(image, timestamp_ms)
            latency = measured["ms"]
            latencies.append(latency)
            detected = bool(result.face_landmarks)
            n_detected += int(detected)
            writer.write(FrameResult(args.run_id, args.repetition, FRAMEWORK, BACKEND,
                args.dataset, video_id, frame_idx, frame_idx / fps, round(latency, 4),
                detected, len(result.face_landmarks[0]) if detected else 0))
            frame_idx += 1
    duration = time.perf_counter() - t_start
    resources = monitor.stop()
    cap.release()
    return VideoSummary(args.run_id, args.repetition, FRAMEWORK, BACKEND, args.dataset,
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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()
    videos = discover_videos(args.input)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise SystemExit(f"Nenhum video encontrado em {args.input}")
    write_run_metadata(args.output_dir, FRAMEWORK, BACKEND, args.run_id, args.repetition,
        {"mediapipe_version": mp.__version__, "model_path": args.model_path,
         "warmup_frames": args.warmup_frames, "python": platform.python_version()})
    for index, (video_id, path) in enumerate(videos, 1):
        load_start = time.perf_counter()
        landmarker = create_landmarker(args.model_path)
        load_s = time.perf_counter() - load_start
        print(f"[mediapipe] {index}/{len(videos)} {video_id}")
        summary = process_video(landmarker, video_id, path, args, load_s)
        landmarker.close()
        if summary:
            append_summary(os.path.join(args.output_dir, "aggregate_summary.csv"), summary)

if __name__ == "__main__":
    main()
