"""OpenFace 2.2.0 CE-CLM benchmark (68 landmarks, CPU)."""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import tempfile
import time

import cv2

sys.path.insert(0, "/app")
from src.common.metrics import discover_videos
from src.common.resource_monitor import ResourceMonitor
from src.common.schema import (FrameCSVWriter, FrameResult, VideoSummary,
                               append_summary, result_stem, write_run_metadata)

FRAMEWORK, BACKEND = "openface", "cpu"
DEFAULT_BINARY = "/openface/build/bin/FeatureExtraction"

def trim_video(video_path: str, frames: int, output: str) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    count = 0
    while count < frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return count

def execute(binary: str, video_path: str, output_dir: str, monitor_interval: float, monitor: bool):
    cmd = [binary, "-f", video_path, "-out_dir", output_dir, "-2Dfp", "-q"]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sampler = ResourceMonitor(interval_s=monitor_interval, pid=proc.pid) if monitor else None
    if sampler:
        sampler.start()
    log, _ = proc.communicate()
    resources = sampler.stop() if sampler else None
    duration = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(f"OpenFace falhou ({proc.returncode}):\n{log[-4000:]}")
    return duration, resources

def process_video(binary, video_id, video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[openface] AVISO: nao foi possivel abrir {video_path}")
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    with tempfile.TemporaryDirectory() as temp:
        if args.warmup_frames:
            warmup = os.path.join(temp, "warmup.avi")
            if trim_video(video_path, args.warmup_frames, warmup):
                warmout = os.path.join(temp, "warmout")
                os.makedirs(warmout)
                execute(binary, warmup, warmout, args.monitor_interval, False)
        measured_input = video_path
        if args.max_frames:
            # Smoke-test only. Published runs must use max_frames=0 because this
            # MJPEG transcode changes the input compared with the other runners.
            measured_input = os.path.join(temp, "smoke_trimmed.avi")
            trim_video(video_path, args.max_frames, measured_input)
        raw_dir = os.path.join(temp, "raw")
        os.makedirs(raw_dir)
        duration, resources = execute(binary, measured_input, raw_dir, args.monitor_interval, True)
        raw_csv = os.path.join(raw_dir, os.path.splitext(os.path.basename(measured_input))[0] + ".csv")
        if not os.path.isfile(raw_csv):
            raise RuntimeError(f"OpenFace nao gerou {raw_csv}")
        rows = []
        with open(raw_csv, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            for original in reader:
                rows.append({key.strip(): (value.strip() if value else value)
                             for key, value in original.items()})
        n_frames = len(rows)
        n_detected = sum(row.get("success", "0") == "1" for row in rows)
        average_ms = duration * 1000.0 / n_frames if n_frames else 0.0
        stem = result_stem(args.output_dir, FRAMEWORK, BACKEND, args.run_id,
                           args.repetition, args.dataset, video_id)
        with FrameCSVWriter(stem + "__frames.csv") as writer:
            for fallback, row in enumerate(rows):
                success = row.get("success", "0") == "1"
                writer.write(FrameResult(args.run_id, args.repetition, FRAMEWORK, BACKEND,
                    args.dataset, video_id, int(float(row.get("frame", fallback + 1))) - 1,
                    float(row.get("timestamp", fallback / fps)), round(average_ms, 4), success,
                    68 if success else 0, float(row.get("confidence", 0.0) or 0.0)))
        return VideoSummary(args.run_id, args.repetition, FRAMEWORK, BACKEND, args.dataset,
            video_id, "video", "end_to_end_video_average", n_frames, n_detected,
            round(duration, 4), round(n_frames / duration, 3) if duration else 0.0,
            round(average_ms, 4), round(average_ms, 4), round(average_ms, 4),
            round(average_ms, 4), resources.avg_cpu_pct, resources.peak_rss_mb,
            resources.avg_gpu_util_pct, resources.peak_gpu_mem_mb, resources.gpu_available,
            0.0, width, height, round(fps, 4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/results")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    args = parser.parse_args()
    videos = discover_videos(args.input)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise SystemExit(f"Nenhum video encontrado em {args.input}")
    if not os.path.isfile(args.binary):
        raise SystemExit(f"Binario ausente: {args.binary}")
    write_run_metadata(args.output_dir, FRAMEWORK, BACKEND, args.run_id, args.repetition,
        {"openface_version": "2.2.0", "binary": args.binary, "git_tag": "OpenFace_2.2.0",
         "warmup_frames": args.warmup_frames, "python": platform.python_version(),
         "latency_warning": "Only end-to-end video average is available."})
    for index, (video_id, path) in enumerate(videos, 1):
        print(f"[openface] {index}/{len(videos)} {video_id}")
        summary = process_video(args.binary, video_id, path, args)
        if summary:
            append_summary(os.path.join(args.output_dir, "aggregate_summary.csv"), summary)

if __name__ == "__main__":
    main()
