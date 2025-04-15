import torch
import json
import cv2
from ultralytics import YOLO
import os
from collections import deque
import mediapipe as mp
import numpy as np

# -------------------------------
# Configurações
# -------------------------------
CONFIG_PATH = "roi_config.json"
OUTPUT_LOG = "distraction_log_ranges.txt"
MODEL_PATH = "yolov11x.pt"

# Configurações de dispositivo
torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

# Limiares de detecção
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320
PROCESS_EVERY_N_FRAMES = 2
DISCTRACTION_BUFFER_SIZE = 7
START_TIME_SECONDS = 240

# Controle de tempo para transições
CELULAR_TIMEOUT_SECONDS = 6
COOLDOWN_ATENCAO = 1.0
WRIST_PROXIMITY_THRESHOLD = 0.33  # 1/3 da largura da ROI
MIN_POSE_CONFIDENCE = 0.7

# Inicializa apenas o módulo de pose do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=MIN_POSE_CONFIDENCE,
    min_tracking_confidence=MIN_POSE_CONFIDENCE,
    model_complexity=1  # Reduz complexidade do modelo
)

# -------------------------------
# Funções auxiliares
# -------------------------------

def determine_state(results):
    found_person = any(int(box.cls) == 0 for r in results for box in r.boxes)
    found_phone = any(int(box.cls) == 67 for r in results for box in r.boxes)
    return found_person, found_phone

def write_log(file, start, end, state):
    duration = end - start
    file.write(f"{start:.1f},{end:.1f},{state},{duration:.1f}\n")

def check_wrists_close(pose_landmarks, roi_width):
    """Verifica se os pulsos estão próximos (menos de 1/3 da largura da ROI)"""
    if not pose_landmarks:
        return False
    
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Calcula distância normalizada entre os pulsos
    distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
    return distance < WRIST_PROXIMITY_THRESHOLD

# -------------------------------
# Processamento de vídeo (simplificado)
# -------------------------------

def process_video(config, model):
    video_path = config["video_path"]
    roi_chair = config["roi_cadeira"]

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(START_TIME_SECONDS * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    distraction_buffer = deque(maxlen=DISCTRACTION_BUFFER_SIZE)
    current_state = None
    state_start_time = START_TIME_SECONDS
    state_last_time = START_TIME_SECONDS
    last_attention_time = START_TIME_SECONDS
    celular_exit_timer = None

    with open(OUTPUT_LOG, "w") as log_file:
        log_file.write("Inicio(s),Fim(s),Estado,Duração(s)\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if current_state:
                        write_log(log_file, state_start_time, state_last_time, current_state)
                    break

                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                timestamp = frame_count / fps
                x, y, w, h = roi_chair
                roi = frame[y:y+h, x:x+w]

                # Processa apenas a pose (para detectar pulsos)
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_roi)
                
                # Verifica pulsos próximos (1/3 da largura da ROI)
                wrists_close = False
                wrist_positions = []
                if pose_results.pose_landmarks:
                    wrists_close = check_wrists_close(pose_results.pose_landmarks, w)
                    
                    # Armazena posições dos pulsos para visualização
                    for wrist_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        landmark = pose_results.pose_landmarks.landmark[wrist_id]
                        wrist_positions.append((
                            int(landmark.x * w) + x,
                            int(landmark.y * h) + y
                        ))

                # Detecção YOLO
                results = model.predict(
                    roi,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                    half=USE_HALF,
                    imgsz=IMAGE_SIZE,
                    device=DEVICE
                )

                found_person, found_phone = determine_state(results)

                # Lógica de transição de estado
                new_state = current_state

                if not found_person:
                    new_state = "FORA_DA_CADEIRA"
                    celular_exit_timer = None
                elif found_phone:
                    new_state = "DESATENTO (CELULAR)"
                    celular_exit_timer = None
                elif wrists_close:
                    new_state = "DESATENTO (MAOS FORA DOS CONTROLES)"
                    celular_exit_timer = None
                elif current_state == "DESATENTO (CELULAR)":
                    if celular_exit_timer is None:
                        celular_exit_timer = timestamp
                    elif timestamp - celular_exit_timer >= CELULAR_TIMEOUT_SECONDS:
                        new_state = "ATENTO"
                        celular_exit_timer = None
                else:
                    if current_state == "ATENTO" and (timestamp - last_attention_time) < COOLDOWN_ATENCAO:
                        new_state = "ATENTO"
                    else:
                        new_state = "ATENTO"
                        last_attention_time = timestamp

                # Troca de estado
                if new_state != current_state:
                    if current_state:
                        write_log(log_file, state_start_time, timestamp, current_state)
                    current_state = new_state
                    state_start_time = timestamp

                state_last_time = timestamp

                # Visualização simplificada
                # ROI e estado
                color_map = {
                    "ATENTO": (0, 255, 0),
                    "DESATENTO (CELULAR)": (0, 0, 255),
                    "DESATENTO (MAOS FORA DOS CONTROLES)": (0, 0, 255),#(255, 255, 0),
                    "FORA_DA_CADEIRA": (255, 0, 0)
                }
                roi_color = color_map.get(current_state, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)
                cv2.putText(frame, f"Estado: {current_state}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, roi_color, 2)

                # Bounding boxes do celular
                for r in results:
                    for box in r.boxes:
                        if int(box.cls) == 67:
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = b
                            x1 += x
                            x2 += x
                            y1 += y
                            y2 += y
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "Celular", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Mostra apenas os pulsos (sem o resto da pose)
                for i, (wx, wy) in enumerate(wrist_positions):
                    color = (0, 255, 255) if wrists_close else (0, 255, 0)
                    cv2.circle(frame, (wx, wy), 10, color, -1)
                    cv2.putText(frame, f"Pulso {'E' if i == 0 else 'D'}", (wx-25, wy-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Mostra distância quando relevante
                if len(wrist_positions) == 2 and wrists_close:
                    wx1, wy1 = wrist_positions[0]
                    wx2, wy2 = wrist_positions[1]
                    distance = np.sqrt((wx1-wx2)**2 + (wy1-wy2)**2)
                    cv2.line(frame, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{distance:.1f}px", ((wx1+wx2)//2, (wy1+wy2)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Monitoramento Simplificado", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            pose.close()

# -------------------------------
# Execução principal
# -------------------------------

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = YOLO(MODEL_PATH).to(DEVICE)
    process_video(config, model)