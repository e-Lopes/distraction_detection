"""Inspeção não destrutiva de metadados dos vídeos."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST_COLUMNS = [
    "video_id",
    "relative_path",
    "session_id",
    "fps",
    "width",
    "height",
    "num_frames",
    "duration_seconds",
    "annotation_version",
    "metadata_backend",
    "metadata_limitation",
]


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator)


def _inspect_ffprobe(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fps = _fraction(stream["avg_frame_rate"])
    duration = float(stream.get("duration") or payload["format"]["duration"])
    raw_frames = stream.get("nb_frames")
    num_frames = int(raw_frames) if raw_frames not in {None, "N/A"} else round(duration * fps)
    limitation = "num_frames estimado por duração×FPS" if raw_frames in {None, "N/A"} else ""
    return {
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "num_frames": num_frames,
        "duration_seconds": duration,
        "metadata_backend": "ffprobe",
        "metadata_limitation": limitation,
    }


def _inspect_opencv(path: Path) -> dict[str, Any] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV não conseguiu abrir o vídeo: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        num_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or num_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Metadados inválidos retornados pelo OpenCV: {path}")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "duration_seconds": num_frames / fps,
        "metadata_backend": "opencv",
        "metadata_limitation": "FPS e frame count são metadados declarados pelo container",
    }


def inspect_video(path: str | Path) -> dict[str, Any]:
    """Tenta ffprobe e depois OpenCV, sempre em modo de leitura."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    for inspector in (_inspect_ffprobe, _inspect_opencv):
        metadata = inspector(video_path)
        if metadata is not None:
            return metadata
    raise RuntimeError(
        "Não há backend de metadados disponível. Instale ffprobe ou as dependências do "
        "projeto (opencv-python-headless)."
    )


def build_manifest_rows(
    video_specs: Iterable[dict[str, str]], *, videos_dir: Path, repository_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in video_specs:
        path = videos_dir / spec["filename"]
        metadata = inspect_video(path)
        rows.append(
            {
                "video_id": spec["video_id"],
                "relative_path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
                "session_id": spec["session_id"],
                **metadata,
                "annotation_version": spec["annotation_version"],
            }
        )
    return rows


def write_manifest(rows: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
