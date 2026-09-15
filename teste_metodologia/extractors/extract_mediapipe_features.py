"""
extract_mediapipe_features.py

Extrai landmarks faciais, EAR, MAR e Head Pose (Pitch, Yaw, Roll) usando MediaPipe Face Mesh,
com feedback impresso no terminal a cada X frames, e os une ao CSV unificado de classificações.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("Instale o mediapipe executando: pip install mediapipe opencv-python") from e

mp_face_mesh = mp.solutions.face_mesh

# ============================================================================
# CONFIGURAÇÃO DE CAMINHOS DOS VÍDEOS
# ============================================================================
VIDEO_PATHS = {
    "video_01": "fase_2/data/raw/1.mp4",
    "video_02": "fase_2/data/raw/2.mp4",
    "video_03": "fase_2/data/raw/3.mp4",
    "video_04": "fase_2/data/raw/4.mp4",
}

UNIFIED_CLASSES_CSV = "classificacoes_frames_exatos.csv"
OUTPUT_MERGED_CSV = "classificacoes_com_mediapipe.csv"

# Frequência de log no terminal (ex: a cada 300 frames)
LOG_EVERY_N_FRAMES = 300

# Índices do Face Mesh para EAR / MAR / Head Pose
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14, 82, 312, 87, 317]

FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # Nariz
    [0.0, -330.0, -65.0],     # Queixo
    [-225.0, 170.0, -135.0],  # Olho esquerdo
    [225.0, 170.0, -135.0],   # Olho direito
    [-150.0, -150.0, -125.0], # Boca esq
    [150.0, -150.0, -125.0]   # Boca dir
], dtype=np.float64)

FACE_2D_INDICES = [1, 152, 33, 263, 61, 291]


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    v = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
    h = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    return float(v / h) if h > 0 else 0.0


def estimate_head_pose(landmarks_2d: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    h, w, _ = frame_shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(FACE_3D, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return float(euler_angle[0]), float(euler_angle[1]), float(euler_angle[2])


def extract_features_from_video(video_path: str, video_id: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    rows = []
    frame_index = 0

    # Limiar reduzido para 0.2 para evitar falsos negativos em ângulos desafiadores
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            face_detected = 0
            ear, mar, pitch, yaw, roll = np.nan, np.nan, np.nan, np.nan, np.nan

            if results.multi_face_landmarks:
                face_detected = 1
                lm = results.multi_face_landmarks[0].landmark
                
                pts_2d = np.array([(lm[i].x * w, lm[i].y * h) for i in FACE_2D_INDICES], dtype=np.float64)
                left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
                right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])
                mouth = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH])

                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                mar = calculate_mar(mouth)
                pitch, yaw, roll = estimate_head_pose(pts_2d, (h, w, 3))

            rows.append({
                "video_id": video_id,
                "frame_index": frame_index,
                "ear": ear,
                "mar": mar,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "face_detected": face_detected
            })

            # Feedback periódico no terminal (tratamento seguro de strings)
            if frame_index % LOG_EVERY_N_FRAMES == 0:
                ear_str = f"{ear:.3f}" if face_detected and not np.isnan(ear) else "N/A (Face não achada)"
                mar_str = f"{mar:.3f}" if face_detected and not np.isnan(mar) else "N/A"
                print(f"[{video_id}] Frame {frame_index} | Detectado: {face_detected} | EAR: {ear_str} | MAR: {mar_str}")

            frame_index += 1

    cap.release()
    return pd.DataFrame(rows)


def main():
    print("=== Iniciando Extração de Landmarks com MediaPipe (Com Monitoramento) ===")
    all_features = []

    for video_id, path in VIDEO_PATHS.items():
        if not Path(path).exists():
            print(f"[Aviso] Vídeo não encontrado no caminho: {path}. Pulando...")
            continue
            
        print(f"\nProcessando {video_id} em: {path}...")
        df_feats = extract_features_from_video(path, video_id)
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

    df_final = pd.merge(df_classes, df_media, on=["video_id", "frame_index"], how="left")
    
    Path(OUTPUT_MERGED_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_MERGED_CSV, index=False)
    print(f"Dataset completo salvo com sucesso em: {OUTPUT_MERGED_CSV}")


if __name__ == "__main__":
    main()