"""
extract_mediapipe_features.py

Extrai landmarks faciais, EAR, MAR e Head Pose (Pitch, Yaw, Roll).

A pessoa e localizada com MediaPipe Pose dentro da mesma ROI usada em
fase_1/analise_comportamento2.py. O Face Mesh roda apenas nessa ROI, evitando
que outro rosto presente no quadro seja usado na extracao das features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("Instale o mediapipe executando: pip install mediapipe opencv-python") from e

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ============================================================================
# CONFIGURAÇÃO DE CAMINHOS DOS VÍDEOS
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATHS = {
    "video_01": REPO_ROOT / "fase_2/data/raw/1.mp4",
    "video_02": REPO_ROOT / "fase_2/data/raw/2.mp4",
    "video_03": REPO_ROOT / "fase_2/data/raw/3.mp4",
    "video_04": REPO_ROOT / "fase_2/data/raw/4.mp4",
}

ROI_CONFIG_PATH = REPO_ROOT / "fase_1/roi_config.json"
UNIFIED_CLASSES_CSV = TEST_DIR / "results/classificacoes_frames_exatos.csv"
OUTPUT_MERGED_CSV = TEST_DIR / "results/classificacoes_com_mediapipe.csv"

# Frequência de log no terminal (ex: a cada 300 frames)
LOG_EVERY_N_FRAMES = 300

# Índices do Face Mesh para EAR / MAR / Head Pose
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = [82, 13, 312]
MOUTH_BOTTOM = [87, 14, 317]
MOUTH_LR = [78, 308]

FACE_3D = np.array([
    [0.000, 0.000, 0.000],
    [0.000, -0.064, -0.013],
    [-0.043, 0.033, -0.026],
    [0.043, 0.033, -0.026],
    [-0.029, -0.029, -0.024],
    [0.029, -0.029, -0.024],
], dtype=np.float64)

FACE_2D_INDICES = [1, 152, 33, 263, 61, 291]


def load_roi(config_path: Path = ROI_CONFIG_PATH) -> tuple[int, int, int, int]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuracao de ROI nao encontrada: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    roi = config.get("roi_cadeira")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError(f"'roi_cadeira' invalida em {config_path}: {roi!r}")

    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensoes invalidas para a ROI: {roi!r}")
    return x, y, width, height


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = roi
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + width)
    y2 = min(frame_height, y + height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"ROI {roi!r} fora do frame com dimensoes {frame_width}x{frame_height}."
        )
    return frame[y1:y2, x1:x2]


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0


def calculate_mar(
    mouth_top: np.ndarray,
    mouth_bottom: np.ndarray,
    mouth_corners: np.ndarray,
) -> float:
    v = np.linalg.norm(mouth_top.mean(axis=0) - mouth_bottom.mean(axis=0))
    h = np.linalg.norm(mouth_corners[0] - mouth_corners[1])
    return float(v / h) if h > 0 else 0.0


def estimate_head_pose(landmarks_2d: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    h, w, _ = frame_shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(FACE_3D, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0

    rotation, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rotation[2, 0], sy))
    if sy > 1e-6:
        yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        roll = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    else:
        yaw = 0.0
        roll = math.degrees(math.atan2(-rotation[1, 2], rotation[1, 1]))
    return pitch, yaw, roll


def extract_features_from_video(
    video_path: str | Path,
    video_id: str,
    roi: tuple[int, int, int, int] | None = None,
    max_frames: int | None = None,
) -> pd.DataFrame:
    roi = load_roi() if roi is None else roi
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    rows = []
    frame_index = 0

    # Mantem os parametros de Pose validados em analise_comportamento2.py.
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as pose, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            if max_frames is not None and frame_index >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            person_frame = crop_roi(frame, roi)
            h, w = person_frame.shape[:2]
            rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)

            pose_detected = int(pose_results.pose_landmarks is not None)
            face_detected = 0
            ear, mar, pitch, yaw, roll = np.nan, np.nan, np.nan, np.nan, np.nan

            if pose_detected and face_results.multi_face_landmarks:
                face_detected = 1
                lm = face_results.multi_face_landmarks[0].landmark
                
                pts_2d = np.array([(lm[i].x * w, lm[i].y * h) for i in FACE_2D_INDICES], dtype=np.float64)
                left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
                right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])
                mouth_top = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_TOP])
                mouth_bottom = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_BOTTOM])
                mouth_corners = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_LR])

                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                mar = calculate_mar(mouth_top, mouth_bottom, mouth_corners)
                pitch, yaw, roll = estimate_head_pose(pts_2d, (h, w, 3))

            rows.append({
                "video_id": video_id,
                "frame_index": frame_index,
                "ear": ear,
                "mar": mar,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "face_detected": face_detected,
                "pose_detected": pose_detected,
            })

            # Feedback periódico no terminal (tratamento seguro de strings)
            if frame_index % LOG_EVERY_N_FRAMES == 0:
                ear_str = f"{ear:.3f}" if face_detected and not np.isnan(ear) else "N/A (Face não achada)"
                mar_str = f"{mar:.3f}" if face_detected and not np.isnan(mar) else "N/A"
                print(
                    f"[{video_id}] Frame {frame_index} | Pessoa: {pose_detected} | "
                    f"Face: {face_detected} | EAR: {ear_str} | MAR: {mar_str}"
                )

            frame_index += 1

    cap.release()
    return pd.DataFrame(rows)


def main():
    print("=== Iniciando Extração de Landmarks com MediaPipe (Com Monitoramento) ===")
    roi = load_roi()
    print(f"ROI da cadeira: {roi}")
    all_features = []

    for video_id, path in VIDEO_PATHS.items():
        if not Path(path).exists():
            print(f"[Aviso] Vídeo não encontrado no caminho: {path}. Pulando...")
            continue
            
        print(f"\nProcessando {video_id} em: {path}...")
        df_feats = extract_features_from_video(path, video_id, roi)
        all_features.append(df_feats)
        print(f"  -> Concluído {video_id}: {len(df_feats)} frames processados.")

    if not all_features:
        print("[Erro] Nenhum vídeo foi processado com sucesso. Verifique os caminhos em VIDEO_PATHS.")
        return

    df_media = pd.concat(all_features, ignore_index=True)

    print(f"\nUnindo com as classes de '{UNIFIED_CLASSES_CSV}'...")
    if not Path(UNIFIED_CLASSES_CSV).exists():
        raise FileNotFoundError(f"O arquivo '{UNIFIED_CLASSES_CSV}' não foi encontrado.")

    df_classes = pd.read_csv(UNIFIED_CLASSES_CSV)
    feature_columns = [
        "ear", "mar", "pitch", "yaw", "roll", "face_detected", "pose_detected"
    ]
    old_features = [column for column in feature_columns if column in df_classes.columns]
    if old_features:
        print(f"Substituindo features existentes no CSV-base: {old_features}")
        df_classes = df_classes.drop(columns=old_features)

    df_final = pd.merge(
        df_classes,
        df_media,
        on=["video_id", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    
    Path(OUTPUT_MERGED_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_MERGED_CSV, index=False)
    print(f"Dataset completo salvo com sucesso em: {OUTPUT_MERGED_CSV}")


if __name__ == "__main__":
    main()
