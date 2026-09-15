"""Common, append-only result schema used by every benchmark runner."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

FRAME_CSV_FIELDS = [
    "run_id", "repetition", "framework", "backend", "dataset", "video_id",
    "frame_idx", "timestamp_s", "inference_ms", "detected", "n_landmarks", "confidence",
]
SUMMARY_CSV_FIELDS = [
    "run_id", "repetition", "framework", "backend", "dataset", "video_id",
    "granularity", "latency_kind", "n_frames", "n_frames_detected",
    "detection_rate_pct", "duration_s", "mean_fps", "p50_latency_ms",
    "p90_latency_ms", "p95_latency_ms", "max_latency_ms", "avg_cpu_pct",
    "peak_rss_mb", "avg_gpu_util_pct", "peak_gpu_mem_mb", "gpu_available",
    "model_load_s", "width", "height", "source_fps",
]

@dataclass
class FrameResult:
    run_id: str
    repetition: int
    framework: str
    backend: str
    dataset: str
    video_id: str
    frame_idx: int
    timestamp_s: float
    inference_ms: float
    detected: bool
    n_landmarks: int
    confidence: Optional[float] = None

    def as_row(self) -> dict:
        row = asdict(self)
        row["detected"] = int(self.detected)
        return row

@dataclass
class VideoSummary:
    run_id: str
    repetition: int
    framework: str
    backend: str
    dataset: str
    video_id: str
    granularity: str
    latency_kind: str
    n_frames: int
    n_frames_detected: int
    duration_s: float
    mean_fps: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    avg_cpu_pct: Optional[float]
    peak_rss_mb: Optional[float]
    avg_gpu_util_pct: Optional[float]
    peak_gpu_mem_mb: Optional[float]
    gpu_available: bool
    model_load_s: float = 0.0
    width: int = 0
    height: int = 0
    source_fps: float = 0.0

    @property
    def detection_rate_pct(self) -> float:
        return 100.0 * self.n_frames_detected / self.n_frames if self.n_frames else 0.0

    def as_row(self) -> dict:
        row = asdict(self)
        row["detection_rate_pct"] = round(self.detection_rate_pct, 4)
        return {key: row.get(key) for key in SUMMARY_CSV_FIELDS}

class FrameCSVWriter:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FRAME_CSV_FIELDS)
        self._writer.writeheader()

    def write(self, row: FrameResult):
        self._writer.writerow(row.as_row())

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def append_summary(summary_path: str, summary: VideoSummary):
    """Append one result, rejecting an incompatible pre-v2 result file."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    exists = os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
    if exists:
        with open(summary_path, newline="", encoding="utf-8") as source:
            existing = next(csv.reader(source), [])
        if existing != SUMMARY_CSV_FIELDS:
            raise RuntimeError(f"Schema antigo em {summary_path}. Mova/apague o arquivo antes desta bateria.")
    with open(summary_path, "a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=SUMMARY_CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(summary.as_row())

def result_stem(output_dir: str, framework: str, backend: str, run_id: str,
                repetition: int, dataset: str, video_id: str) -> str:
    directory = os.path.join(output_dir, framework, backend, run_id, f"rep_{repetition:02d}", dataset)
    return os.path.join(directory, video_id)

def write_run_metadata(output_dir: str, framework: str, backend: str, run_id: str,
                       repetition: int, extra: Optional[dict] = None):
    meta = {"schema_version": 2, "run_id": run_id, "repetition": repetition,
            "framework": framework, "backend": backend}
    if extra:
        meta.update(extra)
    path = os.path.join(output_dir, framework, backend, run_id, f"rep_{repetition:02d}", "run_metadata.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as target:
        json.dump(meta, target, indent=2, ensure_ascii=False)
