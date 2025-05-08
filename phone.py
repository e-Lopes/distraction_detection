import torch
import json
import cv2
from ultralytics import YOLO
import os
from collections import deque
import time

# -------------------------------
# 🔧 Configurações
# -------------------------------
CONFIG_PATH = "roi_config.json"
OUTPUT_LOG = "distraction_log_ranges.txt"
MODEL_PATH = "yolov11x.pt"  # ou yolov11m.pt, como preferir

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320
PROCESS_EVERY_N_FRAMES = 2
DISCTRACTION_BUFFER_SIZE = 7
START_TIME_SECONDS = 240  # Começar aos 4 minutos
CLASS_NAMES = {0: "person", 67: "cell phone"}

# Controle de tempo para transições
CELULAR_TIMEOUT_SECONDS = 6

# -------------------------------
# 🔍 Funções auxiliares
# -------------------------------

def determine_state(results):
    found_person = any(int(box.cls) == 0 for r in results for box in r.boxes)
    found_phone = any(int(box.cls) == 67 for r in results for box in r.boxes)
    return found_person, found_phone

def write_log(file, start, end, state):
    duration = end - start
    file.write(f"{start:.1f},{end:.1f},{state},{duration:.1f}\n")

# -------------------------------
# 🎞️ Processamento de vídeo
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

    celular_exit_timer = None  # Temporizador para sair do estado "CELULAR"

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
                    new_state = "PHONE"
                    celular_exit_timer = None
                elif current_state == "PHONE":
                    if celular_exit_timer is None:
                        celular_exit_timer = timestamp
                    elif timestamp - celular_exit_timer >= CELULAR_TIMEOUT_SECONDS:
                        new_state = "ATENTO"
                        celular_exit_timer = None
                else:
                    new_state = "ATENTO"
                    celular_exit_timer = None

                # Troca de estado
                if new_state != current_state:
                    if current_state:
                        write_log(log_file, state_start_time, timestamp, current_state)
                    current_state = new_state
                    state_start_time = timestamp

                state_last_time = timestamp

                # 🖼️ Mostrar ROI e estado
                roi_color = (0, 255, 0) if current_state == "ATENTO" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)
                cv2.putText(frame, f"Estado: {current_state}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 🔴 Desenhar bounding boxes do celular dentro da ROI
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
                            cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 💬 Mensagem extra se celular reaparece durante o timeout
                # 💬 Contagem regressiva para sair do estado "CELULAR"
                if current_state == "PHONE" and celular_exit_timer:
                    remaining = CELULAR_TIMEOUT_SECONDS - (timestamp - celular_exit_timer)
                    if remaining > 0:
                        msg = f"Voltando ao estado ATENTO em: {remaining:.1f}s"
                        cv2.putText(frame, msg, (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


                cv2.imshow("Monitoramento", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

# -------------------------------
# 🚀 Execução principal
# -------------------------------

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = YOLO(MODEL_PATH)
    process_video(config, model)
