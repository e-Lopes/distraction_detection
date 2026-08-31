"""Preparação de dados faciais WFLW/CVAT/YOLO para a G4.7."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..features.g47_schema import (
    LANDMARK_NAMES,
    LandmarkSchema,
    flip_landmarks,
    load_schema,
    normalized_mean_error,
)
from ..preprocessing.missingness import expand_behavior_labels


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _pose_bin(pitch: float, yaw: float) -> str:
    magnitude = max(abs(pitch), abs(yaw))
    if magnitude < 10:
        return "near"
    if magnitude < 25:
        return "medium"
    return "large"


def stratified_sample_indices(
    frame: pd.DataFrame,
    labels: Sequence[str | None],
    *,
    count: int,
    minimum_gap_frames: int,
    seed: int,
) -> list[int]:
    """Amostra round-robin por classe, detecção e magnitude de pose."""
    if count <= 0 or minimum_gap_frames < 0:
        raise ValueError("count deve ser positivo e minimum_gap_frames não negativo")
    if len(frame) != len(labels):
        raise ValueError("Série e labels têm comprimentos diferentes")
    candidates: dict[tuple[str, str, str], list[int]] = {}
    for index, row in frame.iterrows():
        label = labels[index] or "unlabeled"
        detected = "detected" if int(row["face_detected"]) else "missing"
        pitch = pd.to_numeric(row.get("pitch"), errors="coerce")
        yaw = pd.to_numeric(row.get("yaw"), errors="coerce")
        pose = (
            _pose_bin(float(pitch), float(yaw))
            if np.isfinite(pitch) and np.isfinite(yaw)
            else "unknown"
        )
        candidates.setdefault((label, detected, pose), []).append(int(index))
    rng = random.Random(seed)
    for values in candidates.values():
        rng.shuffle(values)
    selected: list[int] = []
    selected_set: set[int] = set()
    keys = sorted(candidates, key=lambda key: (len(candidates[key]), key))
    while len(selected) < count:
        progress = False
        for key in keys:
            bucket = candidates[key]
            while bucket:
                candidate = bucket.pop()
                if any(abs(candidate - previous) < minimum_gap_frames for previous in selected_set):
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                progress = True
                break
            if len(selected) == count:
                break
        if not progress:
            break
    if len(selected) != count:
        raise ValueError(
            f"Não foi possível selecionar {count} frames com gap {minimum_gap_frames}; "
            f"obtidos {len(selected)}"
        )
    return sorted(selected)


def build_local_sample(
    *,
    videos: Mapping[str, Path],
    series_dir: Path,
    intervals_path: Path,
    output_dir: Path,
    samples_per_video: int = 200,
    minimum_gap_seconds: float = 2.0,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    intervals = _read_csv(intervals_path)
    frames_by_video: dict[str, pd.DataFrame] = {}
    for video_id in videos:
        path = series_dir / f"{video_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Série facial ausente: {path}")
        frames_by_video[video_id] = pd.read_csv(path)
    labels = expand_behavior_labels(
        {video_id: len(frame) for video_id, frame in frames_by_video.items()}, intervals
    )
    image_dir = output_dir / "images"
    manifest_path = output_dir / "sample_manifest.csv"
    if (image_dir.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"Amostra G4.7 já existe em {output_dir}; use --overwrite")
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for video_id, video_path in sorted(videos.items()):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            capture.release()
            raise ValueError(f"FPS inválido em {video_path}")
        selected = stratified_sample_indices(
            frames_by_video[video_id],
            labels[video_id],
            count=samples_per_video,
            minimum_gap_frames=max(1, round(minimum_gap_seconds * fps)),
            seed=seed,
        )
        try:
            for frame_index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, image = capture.read()
                if not success:
                    raise RuntimeError(f"Falha ao ler {video_id}/frame {frame_index}")
                filename = f"{video_id}__frame_{frame_index:06d}.jpg"
                target = image_dir / filename
                if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Falha ao gravar {target}")
                source = frames_by_video[video_id].iloc[frame_index]
                pitch = pd.to_numeric(source.get("pitch"), errors="coerce")
                yaw = pd.to_numeric(source.get("yaw"), errors="coerce")
                rows.append(
                    {
                        "image": f"images/{filename}",
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                        "behavior_label": labels[video_id][frame_index] or "",
                        "face_detected_baseline": int(source["face_detected"]),
                        "pose_bin": (
                            _pose_bin(float(pitch), float(yaw))
                            if np.isfinite(pitch) and np.isfinite(yaw)
                            else "unknown"
                        ),
                        "annotation_status": "pending",
                    }
                )
        finally:
            capture.release()
    manifest = pd.DataFrame(rows).sort_values(["video_id", "frame_index"])
    manifest.to_csv(manifest_path, index=False)
    return manifest


def parse_wflw_line(line: str) -> tuple[np.ndarray, list[str], str]:
    """Lê o formato oficial: 196 coords, 4 bbox, 6 atributos e caminho."""
    parts = line.strip().split()
    if len(parts) < 207:
        raise ValueError("Linha WFLW incompleta")
    coordinates = np.asarray([float(value) for value in parts[:196]], dtype=float).reshape(98, 2)
    metadata = parts[196:-1]
    return coordinates, metadata, parts[-1]


def convert_wflw(
    *,
    annotation_file: Path,
    image_root: Path,
    output_root: Path,
    schema: LandmarkSchema,
    overwrite: bool = False,
) -> int:
    if schema.wflw_indices is None:
        raise ValueError("Mapeamento WFLW ainda não foi preenchido no schema YAML")
    labels_dir = output_root / "labels"
    images_dir = output_root / "images"
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Destino não vazio: {output_root}")
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        points, _, relative = parse_wflw_line(line)
        source = image_root / relative
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(f"Imagem WFLW não encontrada: {source}")
        height, width = image.shape[:2]
        selected = points[np.asarray(schema.wflw_indices)]
        minimum = selected.min(axis=0)
        maximum = selected.max(axis=0)
        center = (minimum + maximum) / 2
        box_size = maximum - minimum
        normalized = selected / np.asarray([width, height])
        values = [
            "0",
            f"{center[0] / width:.8f}",
            f"{center[1] / height:.8f}",
            f"{box_size[0] / width:.8f}",
            f"{box_size[1] / height:.8f}",
        ]
        for x_value, y_value in normalized:
            values.extend((f"{x_value:.8f}", f"{y_value:.8f}", "2"))
        flat_name = relative.replace("/", "__").replace("\\", "__")
        image_target = images_dir / flat_name
        if not image_target.exists():
            image_target.write_bytes(source.read_bytes())
        (labels_dir / f"{Path(flat_name).stem}.txt").write_text(
            " ".join(values) + "\n", encoding="utf-8"
        )
        count += 1
    return count


def _coco_keypoints(annotation: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(annotation.get("keypoints", ()), dtype=float)
    if values.size != 66:
        raise ValueError("Anotação CVAT/COCO deve conter 22 keypoints x 3")
    return values.reshape(22, 3)


def convert_cvat_coco(
    *, annotations_path: Path, images_dir: Path, output_dir: Path, schema: LandmarkSchema
) -> int:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if len(categories) != 1 or tuple(categories[0].get("keypoints", ())) != schema.names:
        raise ValueError("Ordem de keypoints do CVAT/COCO difere do schema G4.7")
    images = {int(item["id"]): item for item in payload.get("images", [])}
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for annotation in payload.get("annotations", []):
        image = images[int(annotation["image_id"])]
        width, height = float(image["width"]), float(image["height"])
        points = _coco_keypoints(annotation)
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        values = [
            "0",
            f"{(x + box_width / 2) / width:.8f}",
            f"{(y + box_height / 2) / height:.8f}",
            f"{box_width / width:.8f}",
            f"{box_height / height:.8f}",
        ]
        for point_x, point_y, visibility in points:
            values.extend(
                (
                    f"{point_x / width:.8f}",
                    f"{point_y / height:.8f}",
                    str(int(np.clip(visibility, 0, 2))),
                )
            )
        image_path = images_dir / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem CVAT ausente: {image_path}")
        (labels_dir / f"{image_path.stem}.txt").write_text(" ".join(values) + "\n", encoding="utf-8")
        count += 1
    return count


def interannotator_nme(first_path: Path, second_path: Path) -> pd.DataFrame:
    def load(path: Path) -> dict[int, np.ndarray]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(item["image_id"]): _coco_keypoints(item) for item in payload["annotations"]}

    first, second = load(first_path), load(second_path)
    shared = sorted(set(first) & set(second))
    if not shared:
        raise ValueError("Exportações não possuem imagens anotadas em comum")
    rows = [{"image_id": image_id, "nme": normalized_mean_error(first[image_id], second[image_id])} for image_id in shared]
    return pd.DataFrame(rows)


def read_yolo_pose_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([float(value) for value in path.read_text(encoding="utf-8").split()])
    if values.size != 5 + 22 * 3:
        raise ValueError("Label YOLO deve conter classe, caixa e 22 keypoints x 3")
    return values[1:5], values[5:].reshape(22, 3)


def audit_flip(image_path: Path, label_path: Path, output_path: Path, schema: LandmarkSchema) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    _, points = read_yolo_pose_label(label_path)
    mirrored = flip_landmarks(points, schema.flip_index)
    images = (image.copy(), cv2.flip(image, 1))
    for canvas, values in zip(images, (points, mirrored), strict=True):
        height, width = canvas.shape[:2]
        for index, (x_value, y_value, visibility) in enumerate(values):
            if visibility <= 0:
                continue
            center = (round(x_value * width), round(y_value * height))
            cv2.circle(canvas, center, 3, (0, 255, 0), -1)
            cv2.putText(canvas, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(images, axis=1))


def _video_mapping(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        video_id, separator, path = value.partition("=")
        if not separator or not video_id or video_id in output:
            raise ValueError("Use --video video_id=caminho.mp4 sem IDs duplicados")
        output[video_id] = Path(path)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-schema")
    validate.add_argument("--schema", type=Path, required=True)
    sample = subparsers.add_parser("sample-local")
    sample.add_argument("--video", action="append", required=True)
    sample.add_argument("--series-dir", type=Path, required=True)
    sample.add_argument("--intervals", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--samples-per-video", type=int, default=200)
    sample.add_argument("--minimum-gap-seconds", type=float, default=2.0)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--overwrite", action="store_true")
    wflw = subparsers.add_parser("convert-wflw")
    wflw.add_argument("--annotations", type=Path, required=True)
    wflw.add_argument("--image-root", type=Path, required=True)
    wflw.add_argument("--output-dir", type=Path, required=True)
    wflw.add_argument("--schema", type=Path, required=True)
    wflw.add_argument("--overwrite", action="store_true")
    cvat = subparsers.add_parser("convert-cvat")
    cvat.add_argument("--annotations", type=Path, required=True)
    cvat.add_argument("--images", type=Path, required=True)
    cvat.add_argument("--output-dir", type=Path, required=True)
    cvat.add_argument("--schema", type=Path, required=True)
    agreement = subparsers.add_parser("agreement")
    agreement.add_argument("--first", type=Path, required=True)
    agreement.add_argument("--second", type=Path, required=True)
    agreement.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("audit-flip")
    audit.add_argument("--image", type=Path, required=True)
    audit.add_argument("--label", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--schema", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-schema":
        schema = load_schema(args.schema)
        print(f"Schema válido: {len(schema.names)} pontos; flip involutivo")
    elif args.command == "sample-local":
        manifest = build_local_sample(
            videos=_video_mapping(args.video),
            series_dir=args.series_dir,
            intervals_path=args.intervals,
            output_dir=args.output_dir,
            samples_per_video=args.samples_per_video,
            minimum_gap_seconds=args.minimum_gap_seconds,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"Amostra criada: {len(manifest)} frames; {Counter(manifest['video_id'])}")
    elif args.command == "convert-wflw":
        count = convert_wflw(
            annotation_file=args.annotations,
            image_root=args.image_root,
            output_root=args.output_dir,
            schema=load_schema(args.schema),
            overwrite=args.overwrite,
        )
        print(f"WFLW convertido: {count} imagens")
    elif args.command == "convert-cvat":
        count = convert_cvat_coco(
            annotations_path=args.annotations,
            images_dir=args.images,
            output_dir=args.output_dir,
            schema=load_schema(args.schema),
        )
        print(f"CVAT convertido: {count} anotações")
    elif args.command == "agreement":
        frame = interannotator_nme(args.first, args.second)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        median, p95 = frame["nme"].median(), frame["nme"].quantile(0.95)
        print(f"NME mediano={median:.4f}; p95={p95:.4f}; aprovado={median <= 0.03 and p95 <= 0.08}")
    elif args.command == "audit-flip":
        audit_flip(args.image, args.label, args.output, load_schema(args.schema))
        print(f"Auditoria gravada: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
