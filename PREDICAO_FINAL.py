import cv2
import numpy as np
import json
import math
import mediapipe as mp
from collections import deque
import torch
from ultralytics import YOLO
import os

#USAR o interpreter 3.11.13 (yolo_env)

# ========= CONFIGURAÇÕES =========
PATH_VIDEOS = "/home/edu/Desktop/distraction_detection/distraction_detection/videos"
FRAMES_SKIP = 2  # Pula 2 frames a cada 1 processado (3x mais rápido)
SHOW_ROI_ONLY = True  # Mostra apenas a área dentro da ROI

# Configurações de detecção
ANGULO_ALERTA = 17
MIN_POSE_CONFIDENCE = 0.7
CONFIDENCE_THRESHOLD = 0.6
PROCESS_EVERY_N_FRAMES = 2
GENERAL_COOLDOWN = 1.0
CELULAR_COOLDOWN = 10.0

# Cores para diferentes estados
COR_NAO_DETECTADO = (255, 0, 0)
COR_MAOS_FORA = (0, 165, 255)
COR_POSTURA_RUIM = (0, 0, 255)
COR_CELULAR = (255, 0, 255)
COR_NORMAL = (0, 255, 0)

# ========= INICIALIZAÇÃO =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=MIN_POSE_CONFIDENCE, 
                    min_tracking_confidence=MIN_POSE_CONFIDENCE)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("/home/edu/Desktop/distraction_detection/distraction_detection/yolov11x.pt").to(device)

# ========= FUNÇÕES =========
def load_config():
    try:
        with open("/home/edu/Desktop/distraction_detection/distraction_detection/roi_config.json") as f:
            config = json.load(f)
        return config['roi_cadeira']
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return None

def calcular_inclinacao(landmarks):
    try:
        orelha_esq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        orelha_dir = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        vetor = np.array([orelha_dir.x - orelha_esq.x, orelha_dir.y - orelha_esq.y])
        angulo = math.degrees(math.atan2(abs(vetor[1]), abs(vetor[0])))
        return min(90, angulo)
    except Exception as e:
        print(f"Erro no cálculo do ângulo: {e}")
        return 0

def check_wrists_close(pose_landmarks, min_confidence=0.5):
    """Verifica se os pulsos estão próximos usando distância euclidiana 2D e normalização pela largura dos ombros."""
    if not pose_landmarks:
        return False, 0  # Retorna também a distância normalizada

    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if (left_wrist.visibility < min_confidence or 
        right_wrist.visibility < min_confidence):
        return False, 0

    # Distância euclidiana 2D entre os pulsos
    dist = np.linalg.norm(
        np.array([left_wrist.x, left_wrist.y]) - 
        np.array([right_wrist.x, right_wrist.y])
    )

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width < 0.1:
        return False, 0

    normalized_distance = dist / shoulder_width

    return normalized_distance < 1.3 or (normalized_distance > 2.3), normalized_distance

def check_wrists_low(pose_landmarks, roi_height, min_confidence=0.5):
    """Verifica se os pulsos estão na metade inferior da ROI e visíveis"""
    if not pose_landmarks:
        return False
    
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    # Verificar visibilidade
    if (left_wrist.visibility < min_confidence or 
        right_wrist.visibility < min_confidence):
        return False
    
    return (left_wrist.y > 0.5 and right_wrist.y > 0.5)

def filter_phone_detections(boxes, roi_area):
    valid_phones = []
    for box in boxes:
        if int(box.cls) == 67 and box.conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            if 0.01 < (box_area / roi_area) < 0.3:
                valid_phones.append(box)
    return valid_phones

def process_video(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return None

    video_name = os.path.basename(video_path)
    x, y, w, h = roi
    roi_area = w * h

    # Configuração da janela
    cv2.namedWindow(f'Monitoramento - {video_name}', cv2.WINDOW_NORMAL)
    if SHOW_ROI_ONLY:
        cv2.resizeWindow(f'Monitoramento - {video_name}', w, h)
    else:
        half_w, half_h = w//2, h//2
        cv2.resizeWindow(f'Monitoramento - {video_name}', half_w, half_h)

    frames_por_estado = {
        "NAO_DETECTADO": 0,
        "MAOS_FORA": 0,
        "POSTURA_RUIM": 0,
        "CELULAR": 0,
        "NORMAL": 0
    }
    frames_totais = 0
    last_state_change = 0
    last_celular_time = None
    current_state = "NORMAL"

    while cap.isOpened():
        # Pular frames conforme definido
        for _ in range(FRAMES_SKIP + 1):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break

        frames_totais += 1
        
        # Obter apenas a ROI
        if SHOW_ROI_ONLY:
            display_frame = frame[y:y+h, x:x+w].copy()
            roi_frame = display_frame
        else:
            display_frame = cv2.resize(frame, (w//2, h//2))
            roi_frame = frame[y:y+h, x:x+w]

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        time_since_last_change = timestamp - last_state_change

        # Processar detecções apenas na ROI
        results = model.predict(roi_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_roi)
        
        # Determinar estado
        angulo = 0
        wrists_close = False
        wrists_low = False
        found_phone = False
        wrists_visible = False
        normalized_distance = -1  # Valor padrão para indicar não calculado
        pessoa_detectada = pose_results.pose_landmarks is not None  # Nova variável para verificar se há pessoa detectada

        # Verificar celular
        all_boxes = [box for r in results for box in r.boxes]
        valid_phones = filter_phone_detections(all_boxes, roi_area)
        found_phone = len(valid_phones) > 0
        
        # Verificar postura e mãos
        if pessoa_detectada:
            angulo = calcular_inclinacao(pose_results.pose_landmarks)
            
            # Verificar visibilidade dos pulsos
            left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrists_visible = (left_wrist.visibility >= 0.5 and 
                            right_wrist.visibility >= 0.5)
            
            if wrists_visible:
                # Calcula distância normalizada
                wrists_close, normalized_distance = check_wrists_close(pose_results.pose_landmarks)
                wrists_low = check_wrists_low(pose_results.pose_landmarks, h)
        
        # Lógica de hierarquia de estados - PRIORIDADE MAIS ALTA PRIMEIRO
        new_state = current_state
        
        if not pessoa_detectada:  # Primeiro verificamos se não há pessoa detectada
            if current_state != "NAO_DETECTADO" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "NAO_DETECTADO"
        elif found_phone: # Encontrou celular
            if current_state != "CELULAR" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "CELULAR"
                last_celular_time = timestamp
        elif wrists_visible and wrists_close and wrists_low: # Maos muito proximas na parte inferior
            if current_state != "MAOS_FORA" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "MAOS_FORA"
        elif current_state == "CELULAR":
            if last_celular_time and (timestamp - last_celular_time) < CELULAR_COOLDOWN:
                new_state = "CELULAR"
            elif time_since_last_change >= GENERAL_COOLDOWN:
                if angulo > ANGULO_ALERTA:
                    new_state = "POSTURA_RUIM"
                elif not wrists_visible or (wrists_close and not wrists_low):
                    new_state = "MAOS_FORA"
                else:
                    new_state = "NORMAL"
        elif angulo > ANGULO_ALERTA:
            if current_state != "POSTURA_RUIM" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "POSTURA_RUIM"
        elif not wrists_visible or (wrists_close and not wrists_low):
            if current_state != "MAOS_FORA" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "MAOS_FORA"
        else:
            if current_state != "NORMAL" and time_since_last_change >= GENERAL_COOLDOWN:
                new_state = "NORMAL"
        
        # Atualizar estado
        if new_state != current_state:
            current_state = new_state
            last_state_change = timestamp
        
        frames_por_estado[current_state] += 1

        # Visualização
        color_map = {
            "NAO_DETECTADO": COR_NAO_DETECTADO,
            "MAOS_FORA": COR_MAOS_FORA,
            "POSTURA_RUIM": COR_POSTURA_RUIM,
            "CELULAR": COR_CELULAR,
            "NORMAL": COR_NORMAL
        }
        
        estado_texto = {
            "NAO_DETECTADO": "NAO DETECTADO",
            "MAOS_FORA": "MAOS FORA",
            "POSTURA_RUIM": "POSTURA RUIM",
            "CELULAR": "CELULAR DETECTADO",
            "NORMAL": "NORMAL"
        }
        
        cor = color_map[current_state]
        
        if SHOW_ROI_ONLY:
            # Desenhar diretamente na ROI
            cv2.rectangle(display_frame, (0, 0), (w, h), cor, 3)
            #cv2.putText(display_frame, f"{estado_texto[current_state]} ({angulo:.1f} graus)", 
            #           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            q
            # Mostrar distância sempre (com formatação condicional)
            dist_text = "Dist. pulsos: N/A" if normalized_distance < 0 else f"Dist. pulsos: {normalized_distance:.2f}"
            #cv2.putText(display_frame, dist_text, 
             #          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            
            # Desenhar bounding boxes dos celulares
            for box in valid_phones:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
             #   cv2.putText(display_frame, "CELULAR", (x1, y1-10),
             #              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
        cv2.imshow(f'Monitoramento - {video_name}', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f'Monitoramento - {video_name}')
    
    return {
        "video": video_name,
        "total_frames": frames_totais,
        "estados": {k: v for k, v in frames_por_estado.items()},
        "porcentagens": {k: (v / frames_totais * 100) if frames_totais > 0 else 0 
                        for k, v in frames_por_estado.items()}
    }

def main():
    roi = load_config()
    if roi is None:
        return

    # Verificar vídeos na pasta especificada
    video_files = []
    for i in range(1, 5):
        video_path = os.path.join(PATH_VIDEOS, f"{i}.mp4")
        if os.path.exists(video_path):
            video_files.append(video_path)
    
    if not video_files:
        print(f"Nenhum vídeo (1.mp4 a 4.mp4) encontrado em: {PATH_VIDEOS}")
        return

    relatorios = []
    
    for video_file in video_files:
        print(f"\nProcessando: {video_file}")
        relatorio = process_video(video_file, roi)
        if relatorio:
            relatorios.append(relatorio)
    
    pose.close()
    cv2.destroyAllWindows()

    # Relatório final
    print("\n=== RELATÓRIO FINAL ===")
    for rel in relatorios:
        print(f"\nVídeo: {rel['video']}")
        print(f"Total de frames: {rel['total_frames']}")
        for estado, porcent in rel['porcentagens'].items():
            print(f"{estado}: {porcent:.1f}%")

if __name__ == "__main__":
    main()