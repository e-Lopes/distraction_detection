"""Extrai séries faciais por frame para o pipeline temporal da Fase 2.

O extrator deriva do protótipo histórico da Fase 1, mas produz uma tabela
canônica, não gera rótulos de treinamento e nunca troca dados reais por dados
sintéticos silenciosamente.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP = [82, 13, 312]
MOUTH_BOTTOM = [87, 14, 317]
MOUTH_LR = [78, 308]
HEAD_POSE_IDS = [1, 152, 33, 263, 61, 291]
HEAD_POSE_3D = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.000, -0.064, -0.013],
        [-0.043, 0.033, -0.026],
        [0.043, 0.033, -0.026],
        [-0.029, -0.029, -0.024],
        [0.029, -0.029, -0.024],
    ],
    dtype=np.float64,
)

EAR_THRESHOLD = 0.25
EAR_CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 0.55
PITCH_THRESHOLD = -15.0

DEFAULT_VIDEOS = (
    ("video_01", "1.mp4"),
    ("video_02", "2.mp4"),
    ("video_03", "3.mp4"),
    ("video_04", "4.mp4"),
)

SERIES_FIELDS = (
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "face_detected",
    "operational_state",
    "legacy_heuristic_state",
)


@dataclass(frozen=True)
class ExtractionSummary:
    video_id: str
    source_filename: str
    source_sha256: str
    output_filename: str
    output_sha256: str
    fps: float
    width: int
    height: int
    source_num_frames: int
    processed_frames: int
    face_detected_frames: int
    face_detection_rate: float
    complete: bool
    roi: str


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def compute_ear(
    landmarks: Sequence[object], indices: Sequence[int], width: int, height: int
) -> float:
    points = np.array(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in indices]
    )
    numerator = _distance(points[1], points[5]) + _distance(points[2], points[4])
    denominator = 2.0 * _distance(points[0], points[3]) + 1e-6
    return numerator / denominator


def compute_mar(landmarks: Sequence[object], width: int, height: int) -> float:
    top = np.mean(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in MOUTH_TOP],
        axis=0,
    )
    bottom = np.mean(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in MOUTH_BOTTOM],
        axis=0,
    )
    left = np.array([landmarks[MOUTH_LR[0]].x * width, landmarks[MOUTH_LR[0]].y * height])
    right = np.array(
        [landmarks[MOUTH_LR[1]].x * width, landmarks[MOUTH_LR[1]].y * height]
    )
    return _distance(top, bottom) / (_distance(left, right) + 1e-6)


def compute_head_pose(
    landmarks: Sequence[object], width: int, height: int
) -> tuple[float, float, float]:
    image_points = np.array(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in HEAD_POSE_IDS],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        HEAD_POSE_3D,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("Não foi possível estimar a pose da cabeça")
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
    if sy > 1e-6:
        yaw = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        roll = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    else:
        yaw = 0.0
        roll = math.degrees(math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
    return pitch, yaw, roll


def legacy_heuristic_state(ear: float, mar: float, pitch: float, ear_streak: int) -> str:
    """Reproduz a regra histórica somente para auditoria, nunca como target."""
    if ear_streak >= EAR_CONSECUTIVE_FRAMES or mar > MAR_THRESHOLD:
        return "fatigue"
    if pitch < PITCH_THRESHOLD:
        return "distraction"
    return "alert"


def load_roi(path: Path | None) -> tuple[int, int, int, int] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    value = payload.get("roi_cadeira")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Configuração ROI inválida em {path}: esperado roi_cadeira=[x,y,w,h]")
    roi = tuple(int(item) for item in value)
    if any(item < 0 for item in roi[:2]) or any(item <= 0 for item in roi[2:]):
        raise ValueError("ROI exige x/y não negativos e largura/altura positivas")
    return roi


def validate_roi(roi: tuple[int, int, int, int] | None, width: int, height: int) -> None:
    if roi is None:
        return
    x, y, roi_width, roi_height = roi
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(f"ROI {roi} ultrapassa o frame {width}x{height}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_video_specs(values: Sequence[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_VIDEOS)
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Vídeo inválido '{value}'; use video_id=arquivo.mp4")
        video_id, filename = (part.strip() for part in value.split("=", 1))
        if not video_id or not filename or video_id in seen:
            raise ValueError(f"Especificação de vídeo inválida ou duplicada: '{value}'")
        if Path(filename).name != filename:
            raise ValueError("O nome do vídeo não pode conter diretórios")
        seen.add(video_id)
        specs.append((video_id, filename))
    return specs


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def _atomic_target(path: Path, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Saída já existe: {path}. Use --overwrite para substituí-la.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(path.suffix + ".tmp")


def extract_video(
    *,
    video_id: str,
    video_path: Path,
    output_path: Path,
    roi: tuple[int, int, int, int] | None,
    face_mesh: object,
    overwrite: bool = False,
    max_frames: int | None = None,
    progress_every: int = 1000,
    hash_video: bool = True,
) -> ExtractionSummary:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    validate_roi(roi, width, height)
    temporary_path = _atomic_target(output_path, overwrite)
    temporary_path.unlink(missing_ok=True)

    processed_frames = 0
    detected_frames = 0
    ear_streak = 0
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SERIES_FIELDS)
            writer.writeheader()
            while True:
                if max_frames is not None and processed_frames >= max_frames:
                    break
                success, frame = capture.read()
                if not success:
                    break
                frame_index = processed_frames
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if timestamp <= 0 and frame_index > 0 and fps > 0:
                    timestamp = frame_index / fps

                if roi is None:
                    region = frame
                else:
                    x, y, roi_width, roi_height = roi
                    region = frame[y : y + roi_height, x : x + roi_width]
                region_height, region_width = region.shape[:2]
                result = face_mesh.process(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

                face_detected = bool(result.multi_face_landmarks)
                ear = mar = pitch = yaw = roll = None
                if face_detected:
                    landmarks = result.multi_face_landmarks[0].landmark
                    ear = (
                        compute_ear(landmarks, RIGHT_EYE, region_width, region_height)
                        + compute_ear(landmarks, LEFT_EYE, region_width, region_height)
                    ) / 2.0
                    mar = compute_mar(landmarks, region_width, region_height)
                    try:
                        pitch, yaw, roll = compute_head_pose(
                            landmarks, region_width, region_height
                        )
                    except RuntimeError:
                        pitch = yaw = roll = None
                    detected_frames += 1
                    ear_streak = ear_streak + 1 if ear < EAR_THRESHOLD else 0
                else:
                    ear_streak = 0

                legacy_state = legacy_heuristic_state(
                    ear or 0.0, mar or 0.0, pitch or 0.0, ear_streak
                )
                writer.writerow(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": _format_metric(timestamp),
                        "ear": _format_metric(ear),
                        "mar": _format_metric(mar),
                        "pitch": _format_metric(pitch),
                        "yaw": _format_metric(yaw),
                        "roll": _format_metric(roll),
                        "face_detected": int(face_detected),
                        "operational_state": "valid" if face_detected else "face_missing",
                        "legacy_heuristic_state": legacy_state,
                    }
                )
                processed_frames += 1
                if progress_every > 0 and processed_frames % progress_every == 0:
                    print(f"{video_id}: {processed_frames}/{source_num_frames} frames")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    complete = processed_frames == source_num_frames
    return ExtractionSummary(
        video_id=video_id,
        source_filename=video_path.name,
        source_sha256=sha256_file(video_path) if hash_video else "",
        output_filename=output_path.name,
        output_sha256=sha256_file(output_path),
        fps=fps,
        width=width,
        height=height,
        source_num_frames=source_num_frames,
        processed_frames=processed_frames,
        face_detected_frames=detected_frames,
        face_detection_rate=detected_frames / processed_frames if processed_frames else 0.0,
        complete=complete,
        roi=json.dumps(roi) if roi else "full_frame",
    )


def write_manifest(path: Path, summaries: Iterable[ExtractionSummary]) -> None:
    rows = [asdict(summary) for summary in summaries]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ExtractionSummary.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def write_demo(output_dir: Path, overwrite: bool) -> None:
    path = output_dir / "demo_facial_series.csv"
    temporary_path = _atomic_target(path, overwrite)
    rows = [
        {
            "video_id": "demo",
            "frame_index": index,
            "timestamp_seconds": _format_metric(index / 10),
            "ear": _format_metric(0.28 if index != 2 else None),
            "mar": _format_metric(0.12 if index != 2 else None),
            "pitch": _format_metric(-4.0 if index != 2 else None),
            "yaw": _format_metric(1.0 if index != 2 else None),
            "roll": _format_metric(0.5 if index != 2 else None),
            "face_detected": int(index != 2),
            "operational_state": "valid" if index != 2 else "face_missing",
            "legacy_heuristic_state": "alert",
        }
        for index in range(5)
    ]
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SERIES_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Demonstração explícita gerada: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, help="Diretório dos vídeos reais")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fase_2/data/interim/legacy_extraction"),
        help="Diretório de saída não versionado",
    )
    parser.add_argument("--roi-config", type=Path, help="JSON opcional com roi_cadeira")
    parser.add_argument(
        "--video",
        action="append",
        help="Mapeamento video_id=arquivo.mp4; pode ser repetido",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Limite para teste parcial")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--skip-video-hash", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Gera somente uma pequena série sintética identificada como demo",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames deve ser positivo")
    if args.demo:
        if args.video_dir or args.video:
            raise ValueError("--demo não pode ser combinado com vídeos reais")
        write_demo(args.output_dir, args.overwrite)
        return 0
    if args.video_dir is None:
        raise ValueError("--video-dir é obrigatório fora do modo --demo")

    specs = parse_video_specs(args.video)
    missing = [filename for _, filename in specs if not (args.video_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError("Vídeos não encontrados: " + ", ".join(missing))
    roi = load_roi(args.roi_config)

    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe não instalado. Instale a dependência opcional: "
            "pip install -e 'fase_2[extraction]'"
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ExtractionSummary] = []
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for video_id, filename in specs:
            print(f"Extraindo {video_id}: {filename}")
            summaries.append(
                extract_video(
                    video_id=video_id,
                    video_path=args.video_dir / filename,
                    output_path=args.output_dir / f"{video_id}.csv",
                    roi=roi,
                    face_mesh=face_mesh,
                    overwrite=args.overwrite,
                    max_frames=args.max_frames,
                    progress_every=args.progress_every,
                    hash_video=not args.skip_video_hash,
                )
            )
    write_manifest(args.output_dir / "extraction_manifest.csv", summaries)
    print(f"Extração concluída: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
