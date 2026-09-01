"""Preparação reproduzível da amostra manual de landmarks da G4.8."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from ..features.extract_facial_series import load_roi
from ..features.g47_schema import LANDMARK_NAMES
from ..preprocessing.missingness import expand_behavior_labels

VALID_LABELS = {"alert", "fatigue", "distraction"}


def _pose_bin(pitch: object, yaw: object) -> str:
    pitch_value = pd.to_numeric(pitch, errors="coerce")
    yaw_value = pd.to_numeric(yaw, errors="coerce")
    if not np.isfinite(pitch_value) or not np.isfinite(yaw_value):
        return "unknown"
    magnitude = max(abs(float(pitch_value)), abs(float(yaw_value)))
    if magnitude < 10:
        return "near"
    if magnitude < 25:
        return "medium"
    return "large"


def select_manifest_rows(
    series: pd.DataFrame,
    labels: Sequence[str | None],
    *,
    video_id: str,
    fps: float,
    count: int,
    minimum_gap_seconds: float,
    seed: int,
) -> pd.DataFrame:
    """Seleciona frames válidos em round-robin por classe/detecção/pose."""
    if len(series) != len(labels):
        raise ValueError("Série e labels possuem comprimentos diferentes")
    if fps <= 0 or count <= 0 or minimum_gap_seconds < 0:
        raise ValueError("fps/count devem ser positivos e gap não negativo")
    buckets: dict[tuple[str, str, str], list[int]] = {}
    strata: dict[int, tuple[str, str, str]] = {}
    for frame_index, (label, (_, row)) in enumerate(zip(labels, series.iterrows(), strict=True)):
        if label not in VALID_LABELS:
            continue
        detected = "detected" if int(row["face_detected"]) else "missing"
        key = (label, detected, _pose_bin(row.get("pitch"), row.get("yaw")))
        buckets.setdefault(key, []).append(frame_index)
        strata[frame_index] = key
    rng = random.Random(seed)
    for candidates in buckets.values():
        rng.shuffle(candidates)
    keys = sorted(buckets, key=lambda key: (len(buckets[key]), key))
    gap_frames = max(1, round(minimum_gap_seconds * fps))
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            candidates = buckets[key]
            while candidates:
                candidate = candidates.pop()
                if any(abs(candidate - previous) < gap_frames for previous in selected):
                    continue
                selected.append(candidate)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Amostra insuficiente em {video_id}: solicitados {count}, obtidos {len(selected)}")
    rows = []
    for frame_index in sorted(selected):
        label, detected, pose = strata[frame_index]
        rows.append(
            {
                "sample_id": f"{video_id}__frame_{frame_index:06d}",
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "behavior_label": label,
                "face_detected_baseline": int(detected == "detected"),
                "pose_bin_baseline": pose,
                "selection_seed": seed,
                "annotation_status": "pending",
                "review_status": "pending",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    *,
    videos_manifest: Path,
    intervals_manifest: Path,
    series_dir: Path,
    output: Path,
    samples_per_video: int = 100,
    minimum_gap_seconds: float = 2.0,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Manifesto existente: {output}. Use --overwrite.")
    videos = pd.read_csv(videos_manifest)
    intervals = pd.read_csv(intervals_manifest).to_dict("records")
    series_by_video: dict[str, pd.DataFrame] = {}
    for video_id in videos["video_id"]:
        path = series_dir / f"{video_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Série facial ausente: {path}")
        series_by_video[video_id] = pd.read_csv(path)
    labels = expand_behavior_labels(
        {video_id: len(series) for video_id, series in series_by_video.items()}, intervals
    )
    parts = []
    for row in videos.itertuples(index=False):
        parts.append(
            select_manifest_rows(
                series_by_video[row.video_id],
                labels[row.video_id],
                video_id=row.video_id,
                fps=float(row.fps),
                count=samples_per_video,
                minimum_gap_seconds=minimum_gap_seconds,
                seed=seed,
            )
        )
    manifest = pd.concat(parts, ignore_index=True).sort_values(["video_id", "frame_index"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_cvat_package(
    *,
    sample_manifest: Path,
    videos_manifest: Path,
    roi_config: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Extrai as ROIs congeladas e cria metadados locais para anotação CVAT."""
    images_dir = output_dir / "images"
    package_manifest = output_dir / "package_manifest.csv"
    labels_path = output_dir / "cvat_labels.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Pacote CVAT existente: {output_dir}. Use --overwrite.")
    images_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_manifest)
    videos = pd.read_csv(videos_manifest).set_index("video_id")
    roi = load_roi(roi_config)
    if roi is None:
        raise ValueError("A exportação G4.8 exige ROI explícita")
    x, y, width, height = roi
    rows: list[dict[str, object]] = []
    for video_id, group in sample.groupby("video_id", sort=True):
        if video_id not in videos.index:
            raise ValueError(f"Vídeo ausente do manifesto: {video_id}")
        video_path = Path(str(videos.loc[video_id, "relative_path"]))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir {video_path}")
        try:
            for item in group.itertuples(index=False):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(item.frame_index))
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f"Falha ao ler {video_id}/frame {item.frame_index}")
                if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
                    raise ValueError(f"ROI inválida para {video_id}: {roi} em {frame.shape[:2]}")
                crop = frame[y : y + height, x : x + width]
                filename = f"{item.sample_id}.jpg"
                target = images_dir / filename
                if not cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Falha ao gravar {target}")
                rows.append(
                    {
                        "sample_id": item.sample_id,
                        "image": f"images/{filename}",
                        "video_id": video_id,
                        "frame_index": int(item.frame_index),
                        "width": crop.shape[1],
                        "height": crop.shape[0],
                        "sha256": _sha256(target),
                    }
                )
        finally:
            capture.release()
    package = pd.DataFrame(rows).sort_values(["video_id", "frame_index"])
    package.to_csv(package_manifest, index=False)
    labels = {
        "label": "face_22",
        "type": "skeleton",
        "keypoints_order": list(LANDMARK_NAMES),
        "coordinate_convention": "anatomical_subject_left_right",
        "occlusion_rule": "mark outside only when the anatomical point cannot be located",
    }
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("sample-manifest", "export-cvat"), default="sample-manifest")
    parser.add_argument("--videos-manifest", type=Path, default=Path("fase_2/data/manifests/videos.csv"))
    parser.add_argument(
        "--intervals-manifest",
        type=Path,
        default=Path("fase_2/data/manifests/annotation_frame_intervals.csv"),
    )
    parser.add_argument("--series-dir", type=Path, default=Path("fase_2/data/interim/geometry_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fase_2/data/manifests/g48_annotation_sample.csv"),
    )
    parser.add_argument("--samples-per-video", type=int, default=100)
    parser.add_argument("--minimum-gap-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-manifest", type=Path, default=Path("fase_2/data/manifests/g48_annotation_sample.csv"))
    parser.add_argument("--roi-config", type=Path, default=Path("fase_2/configs/preprocessing/legacy_roi.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("fase_2/data/external/G48/cvat_sample"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-cvat":
        package = export_cvat_package(
            sample_manifest=args.sample_manifest,
            videos_manifest=args.videos_manifest,
            roi_config=args.roi_config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Pacote CVAT criado: {len(package)} imagens em {args.output_dir}")
        return 0
    manifest = build_manifest(
        videos_manifest=args.videos_manifest,
        intervals_manifest=args.intervals_manifest,
        series_dir=args.series_dir,
        output=args.output,
        samples_per_video=args.samples_per_video,
        minimum_gap_seconds=args.minimum_gap_seconds,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Manifesto G4.8 criado: {len(manifest)} frames")
    print("Por vídeo:", dict(Counter(manifest["video_id"])))
    print("Por classe:", dict(Counter(manifest["behavior_label"])))
    print("Baseline:", dict(Counter(manifest["face_detected_baseline"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
