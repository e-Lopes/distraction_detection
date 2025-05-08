import os
import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe.python.solutions import pose as mp_pose
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from io import BytesIO

# Configurações globais
MODEL_PATH = "yolov11x.pt"
DEVICE = "cuda"  # ou "cpu"
CONFIDENCE_THRESHOLD = 0.4
PROCESS_EVERY_N_FRAMES = 5
IMAGE_SIZE = 640
INITIAL_STATE = "ATENTO"
COR_NORMAL = (0, 255, 0)
COR_ALERTA = (0, 0, 255)
HISTORICO_GRAFICO = 30
OUTPUT_LOG_PREFIX = "log_"
TIMELINE_GRAPH_PREFIX = "timeline_"
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def write_log(log_file, start, end, state):
    dur = round(end - start, 2)
    log_file.write(f"{start:.2f},{end:.2f},{state},{dur:.2f}\n")

def calcular_inclinacao(landmarks):
    try:
        ombro_esq = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        ombro_dir = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        delta_y = ombro_esq.y - ombro_dir.y
        delta_x = ombro_esq.x - ombro_dir.x
        angulo = np.degrees(np.arctan2(delta_y, delta_x))
        return angulo
    except:
        return 0

def determine_state(results, landmarks, h, w):
    pessoas = [d for d in results.boxes.cls if int(d) == 0]
    if not pessoas:
        return "SEM_PESSOA", None
    if landmarks:
        angulo = calcular_inclinacao(landmarks)
        if abs(angulo) > 15:
            return "DESATENTO", angulo
    return "ATENTO", None

def setup_grafico_inclinacao():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(-30, 30)
    ax.set_xlim(0, HISTORICO_GRAFICO)
    ax.set_title("Inclinação dos Ombros")
    ax.grid()
    fig.tight_layout()
    return fig, ax, line

def update_grafico_inclinacao(fig, ax, line, data):
    line.set_ydata(data)
    line.set_xdata(range(len(data)))
    canvas = FigureCanvas(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def generate_timeline_chart(csv_path, output_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = {"ATENTO": "green", "DESATENTO": "red", "SEM_PESSOA": "gray"}
    for _, row in df.iterrows():
        ax.axvspan(row["start"], row["end"], color=colors.get(row["state"], "blue"), alpha=0.5)
    ax.set_xlabel("Tempo (s)")
    ax.set_title("Linha do Tempo de Estado")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def process_video(video_path, roi_chair, output_suffix=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO(MODEL_PATH).to(DEVICE)
    model.fuse()
    
    current_state = INITIAL_STATE
    state_start_time = 0
    frame_count = 0
    x, y, w, h = roi_chair
    valid_phones = []
    
    # Configurações do gráfico postural
    fig, ax, line = setup_grafico_inclinacao()
    historico_inclinacao = deque([0]*HISTORICO_GRAFICO, maxlen=HISTORICO_GRAFICO)
    ultimo_angulo = 0
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_log = f"{OUTPUT_LOG_PREFIX}{video_name}{output_suffix}.txt"
    
    with open(output_log, "w") as log_file:
        log_file.write("Inicio(s),Fim(s),Estado,Duração(s)\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Registrar o último estado antes de sair
                if current_state:
                    write_log(log_file, state_start_time, frame_count/fps, current_state)
                break
                
            frame_count += 1
            timestamp = frame_count / fps
            
            # Inicializa pose_results como None a cada frame
            pose_results = None
            
            # Processar apenas alguns frames para otimização
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                roi = frame[y:y+h, x:x+w]
                
                # Detecção de objetos
                results = model(roi, imgsz=IMAGE_SIZE, verbose=False, device=DEVICE)
                
                # Detecção de pose (em todos os frames processados)
                pose_results = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                
                new_state, valid_phones = determine_state(
                    results, 
                    pose_results.pose_landmarks if pose_results else None, 
                    h, w
                )
                
                if new_state != current_state:
                    if current_state:
                        write_log(log_file, state_start_time, timestamp, current_state)
                    current_state = new_state
                    state_start_time = timestamp
            
            # Atualização contínua do gráfico postural (em todos os frames)
            if frame_count % 2 == 0:  # Atualiza a cada 2 frames para performance
                # Usa pose_results se disponível, senão usa None
                current_pose = pose_results.pose_landmarks if pose_results else None
                if current_pose:
                    ultimo_angulo = calcular_inclinacao(current_pose)
                else:
                    ultimo_angulo = 0  # Valor padrão quando não detectado
                
                historico_inclinacao.append(ultimo_angulo)
                
                # Atualiza o gráfico
                grafico_img = update_grafico_inclinacao(fig, ax, line, historico_inclinacao)
                grafico_img = cv2.resize(grafico_img, (400, 200))
                frame[10:210, 10:410] = grafico_img
            
            # Restante do código permanece igual...
            # Visualização (em todos os frames)
            color_map = {
                "ATENTO": (0, 255, 0),
                "DESATENTO (CELULAR)": (0, 0, 255),
                "MAOS FORA DOS CONTROLES": (0, 165, 255),
                "FORA_DA_CADEIRA": (255, 0, 0)
            }
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_map.get(current_state, (255,255,255)), 2)
            cv2.putText(frame, f"Estado: {current_state}", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(current_state), 2)
            
            # Mostra celulares detectados
            for box in valid_phones:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(frame, (x1+x, y1+y), (x2+x, y2+y), (0, 0, 255), 2)
                cv2.putText(frame, f"CELULAR {box.conf:.2f}", (x1+x, y1+y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Monitoramento Postural e de Atenção", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Registrar estado atual se sair manualmente
                if current_state:
                    write_log(log_file, state_start_time, timestamp, current_state)
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Gerar gráfico de timeline com verificação
    if os.path.exists(output_log) and os.path.getsize(output_log) > 20:  # Mais que apenas o cabeçalho
        generate_timeline_chart(output_log, f"{TIMELINE_GRAPH_PREFIX}{video_name}.png")
    else:
        print(f"Aviso: Arquivo de log vazio ou muito pequeno - {output_log}")
    
    print(f"\nProcessamento concluído para {video_name}")