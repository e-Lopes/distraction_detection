import torch
import json
import cv2
from ultralytics import YOLO
import os
from collections import deque
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import urllib.request

# -------------------------------
# Configurações
# -------------------------------
CONFIG_PATH = "roi_config.json"
OUTPUT_LOG_PREFIX = "distraction_log_"
MODEL_PATH = "yolov11x.pt"
TIMELINE_GRAPH_PREFIX = "attention_timeline_"

MODEL_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
# Configurações de dispositivo
torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

# Parâmetros de detecção
CONFIDENCE_THRESHOLD = 0.65
MIN_PERSON_CONFIDENCE = 0.6
IMAGE_SIZE = 640
PROCESS_EVERY_N_FRAMES = 2
START_TIME_SECONDS = 0
MIN_CONSECUTIVE_FRAMES = 5
PHONE_MIN_AREA = 0.02
PHONE_MAX_AREA = 0.25
MIN_PHONE_ASPECT_RATIO = 0.5
MAX_PHONE_ASPECT_RATIO = 2.0

# Controle de tempo
CELULAR_COOLDOWN = 10.0
GENERAL_COOLDOWN = 2.0
WRIST_PROXIMITY_THRESHOLD = 0.33
MIN_POSE_CONFIDENCE = 0.7

# Estado inicial
INITIAL_STATE = "ATENTO"

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=MIN_POSE_CONFIDENCE,
    min_tracking_confidence=MIN_POSE_CONFIDENCE,
    model_complexity=1
)

# -------------------------------
# Funções auxiliares
# -------------------------------

def ensure_model_exists(model_path, download_url): # Função para garantir o modelo do yolov11x
    if not os.path.isfile(model_path):
        print(f"Modelo não encontrado em '{model_path}'. Baixando de {download_url}...")
        urllib.request.urlretrieve(download_url, model_path)
        print("Download concluído.")

def check_wrists_close(pose_landmarks, roi_width):
    """Verifica se os pulsos estão próximos (menos de 1/3 da largura da ROI)"""
    if not pose_landmarks:
        return False
    
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    distance = np.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
    return distance < WRIST_PROXIMITY_THRESHOLD

def check_wrists_low(pose_landmarks, roi_height):
    """Verifica se ambos os pulsos estão na metade inferior da ROI"""
    if not pose_landmarks:
        return False
    
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    
    return left_wrist.y > 0.5 and right_wrist.y > 0.5

def filter_phone_detections(boxes, roi_area):
    """Filtra detecções de celular baseado em tamanho, posição e proporção"""
    valid_phones = []
    for box in boxes:
        if int(box.cls) == 67 and box.conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            box_area = width * height
            relative_area = box_area / roi_area
            aspect_ratio = width / height if height > 0 else 0
            
            if (PHONE_MIN_AREA < relative_area < PHONE_MAX_AREA and
                MIN_PHONE_ASPECT_RATIO < aspect_ratio < MAX_PHONE_ASPECT_RATIO):
                valid_phones.append(box)
    return valid_phones

def determine_state(results, consecutive_empty_frames, roi_area, pose_landmarks, roi_height, roi_width):
    """Determina o estado com verificações adicionais para pulsos"""
    found_person = any(int(box.cls) == 0 and box.conf > MIN_PERSON_CONFIDENCE 
                     for r in results for box in r.boxes)
    
    if found_person:
        consecutive_empty_frames = 0
    else:
        consecutive_empty_frames += 1
    
    person_present = consecutive_empty_frames < MIN_CONSECUTIVE_FRAMES
    wrists_close = check_wrists_close(pose_landmarks, roi_width) if pose_landmarks else False
    wrists_low = check_wrists_low(pose_landmarks, roi_height) if pose_landmarks else False
    
    all_boxes = [box for r in results for box in r.boxes]
    valid_phones = filter_phone_detections(all_boxes, roi_area)
    found_phone = len(valid_phones) > 0
    
    return person_present, found_phone, consecutive_empty_frames, valid_phones, wrists_close, wrists_low

def write_log(file, start, end, state):
    duration = end - start
    file.write(f"{start:.1f},{end:.1f},{state},{duration:.1f}\n")

def generate_timeline_chart(log_data, output_file):
    """Gera um gráfico de linha temporal com os estados e suas durações"""
    state_colors = {
        "ATENTO": "green",
        "DESATENTO (CELULAR)": "red",
        "DESATENTO (MAOS FORA DOS CONTROLES)": "orange",
        "FORA_DA_CADEIRA": "blue"
    }
    
    intervals = []
    with open(log_data, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                start, end, state, duration = parts
                intervals.append((float(start), float(end), state, float(duration)))
    
    if not intervals:
        print(f"Nenhum dado encontrado em {log_data}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (start, end, state, duration) in enumerate(intervals):
        ax.broken_barh([(start, duration)], (0, 1), facecolors=state_colors.get(state, "gray"))
    
    ax.set_yticks([])
    ax.set_xlabel('Tempo (segundos)')
    ax.set_title('Linha Temporal dos Estados de Atenção')
    
    legend_elements = [Patch(facecolor=color, label=state) 
                      for state, color in state_colors.items()]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo como {output_file}")

# -------------------------------
# Interface Gráfica
# -------------------------------

class VideoSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seletor de Vídeo para Monitoramento")
        self.root.geometry("500x300")
        
        self.video_files = []
        self.selected_videos = []
        
        self.create_widgets()
        self.load_config()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Selecione os arquivos .mp4 para serem análisados", font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.video_listbox = tk.Listbox(main_frame, selectmode=tk.MULTIPLE, height=6)
        self.video_listbox.pack(fill=tk.BOTH, expand=True, pady=10)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Selecionar Pasta", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Processar Selecionados", command=self.process_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Processar Todos", command=self.process_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Sair", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
    def load_config(self):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            video_dir = os.path.dirname(config["video_path"])
            self.scan_videos(video_dir)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar a configuração: {str(e)}")
    
    def select_folder(self):
        folder = filedialog.askdirectory(title="Selecione a pasta com os vídeos")
        if folder:
            self.scan_videos(folder)
    
    def scan_videos(self, folder):
        self.video_files = glob.glob(os.path.join(folder, "*.mp4"))
        
        self.video_listbox.delete(0, tk.END)
        for video in self.video_files:
            self.video_listbox.insert(tk.END, os.path.basename(video))
    
    def process_selected(self):
        selections = self.video_listbox.curselection()
        if not selections:
            messagebox.showwarning("Aviso", "Nenhum vídeo selecionado!")
            return
            
        self.selected_videos = [self.video_files[i] for i in selections]
        self.root.destroy()
    
    def process_all(self):
        if not self.video_files:
            messagebox.showwarning("Aviso", "Nenhum vídeo encontrado!")
            return
            
        self.selected_videos = self.video_files
        self.root.destroy()

# -------------------------------
# Processamento de vídeo
# -------------------------------

def process_video(video_path, roi_chair, output_suffix=""):
    ensure_model_exists(MODEL_PATH, MODEL_DOWNLOAD_URL)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(START_TIME_SECONDS * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_log = f"{OUTPUT_LOG_PREFIX}{video_name}{output_suffix}.txt"
    timeline_graph = f"{TIMELINE_GRAPH_PREFIX}{video_name}{output_suffix}.png"

    current_state = INITIAL_STATE
    state_start_time = START_TIME_SECONDS
    state_last_time = START_TIME_SECONDS
    last_state_change = START_TIME_SECONDS
    last_celular_time = None
    consecutive_empty_frames = 0
    x, y, w, h = roi_chair
    roi_area = w * h

    model = YOLO(MODEL_PATH).to(DEVICE)

    # Configuração da janela para abrir centralizada e com tamanho adequado
    cv2.namedWindow("Monitoramento", cv2.WINDOW_NORMAL)
    
    # Obtém a resolução do monitor principal
    screen_width = 1920  # Valor padrão caso não consiga obter
    screen_height = 1080  # Valor padrão caso não consiga obter
    
    try:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        pass
    
    # Define o tamanho da janela para 80% da resolução do monitor
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    cv2.resizeWindow("Monitoramento", window_width, window_height)
    
    # Centraliza a janela
    cv2.moveWindow("Monitoramento", 
                  (screen_width - window_width) // 2, 
                  (screen_height - window_height) // 2)

    with open(output_log, "w") as log_file:
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
                roi = frame[y:y+h, x:x+w]

                # Processa pose
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_roi)
                
                # Detecção YOLO
                results = model.predict(roi, conf=CONFIDENCE_THRESHOLD, verbose=False, 
                                      half=USE_HALF, imgsz=IMAGE_SIZE, device=DEVICE)
                
                # Determina estado
                person_present, found_phone, consecutive_empty_frames, valid_phones, wrists_close, wrists_low = determine_state(
                    results, consecutive_empty_frames, roi_area, pose_results.pose_landmarks, h, w)

                # Lógica de transição modificada
                new_state = current_state
                time_since_last_change = timestamp - last_state_change

                if not person_present:
                    if current_state != "FORA_DA_CADEIRA" and time_since_last_change >= GENERAL_COOLDOWN:
                        new_state = "FORA_DA_CADEIRA"
                        last_celular_time = None
                else:
                    if found_phone:
                        if current_state != "DESATENTO (CELULAR)":
                            if time_since_last_change >= GENERAL_COOLDOWN:
                                new_state = "DESATENTO (CELULAR)"
                                last_celular_time = timestamp
                        else:
                            new_state = "DESATENTO (CELULAR)"
                    elif current_state == "DESATENTO (CELULAR)":
                        if last_celular_time and (timestamp - last_celular_time) < CELULAR_COOLDOWN:
                            new_state = "DESATENTO (CELULAR)"
                        elif time_since_last_change >= GENERAL_COOLDOWN:
                            new_state = "DESATENTO (MAOS FORA DOS CONTROLES)" if wrists_close else "ATENTO"
                    elif wrists_close:
                        if current_state != "DESATENTO (MAOS FORA DOS CONTROLES)" and time_since_last_change >= GENERAL_COOLDOWN:
                            new_state = "DESATENTO (MAOS FORA DOS CONTROLES)"
                    else:
                        # Nova condição para ATENTO - pulsos devem estar na parte inferior
                        if current_state != "ATENTO" and time_since_last_change >= GENERAL_COOLDOWN and wrists_low:
                            new_state = "ATENTO"
                        elif not wrists_low and person_present:
                            new_state = "DESATENTO (MAOS FORA DOS CONTROLES)"

                # Atualiza estado
                if new_state != current_state:
                    if current_state:
                        write_log(log_file, state_start_time, timestamp, current_state)
                    current_state = new_state
                    state_start_time = timestamp
                    last_state_change = timestamp

                state_last_time = timestamp

                # Visualização
                color_map = {
                    "ATENTO": (0, 255, 0),
                    "DESATENTO (CELULAR)": (0, 0, 255),
                    "DESATENTO (MAOS FORA DOS CONTROLES)": (0, 165, 255),
                    "FORA_DA_CADEIRA": (255, 0, 0)
                }
                roi_color = color_map.get(current_state, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), roi_color, 2)
                cv2.putText(frame, f"Estado: {current_state}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, roi_color, 2)

                # Mostra apenas celulares válidos
                for box in valid_phones:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1+x, y1+y), (x2+x, y2+y), (0, 0, 255), 2)
                    cv2.putText(frame, f"CELULAR {box.conf:.2f}", (x1+x, y1+y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Visualização adicional para debug
                if pose_results.pose_landmarks:
                    # Linha divisória da metade inferior
                    cv2.line(frame, (x, y + h//2), (x + w, y + h//2), (255, 255, 0), 1)
                    
                    # Pulsos
                    for wrist_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                        landmark = pose_results.pose_landmarks.landmark[wrist_id]
                        wx, wy = int(landmark.x * w) + x, int(landmark.y * h) + y
                        color = (0, 255, 0) if landmark.y > 0.5 else (0, 0, 255)
                        cv2.circle(frame, (wx, wy), 8, color, -1)

                # Redimensiona o frame para caber na janela
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                if frame_width > window_width or frame_height > window_height:
                    if frame_width / window_width > frame_height / window_height:
                        new_width = window_width
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = window_height
                        new_width = int(new_height * aspect_ratio)
                    
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imshow("Monitoramento", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
    
    generate_timeline_chart(output_log, timeline_graph)

# -------------------------------
# Execução principal
# -------------------------------

if __name__ == "__main__":
    #Faz o download do modelo do yolo se ele nao estiver na pasta
    ensure_model_exists(MODEL_PATH, MODEL_DOWNLOAD_URL)

    # Cria e executa a interface gráfica
    root = tk.Tk()
    app = VideoSelectorApp(root)
    root.mainloop()
    
    # Se vídeos foram selecionados, processa eles
    if hasattr(app, 'selected_videos') and app.selected_videos:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        
        for video_path in app.selected_videos:
            print(f"\nProcessando vídeo: {video_path}")
            process_video(video_path, config["roi_cadeira"])
    
    cv2.destroyAllWindows()
    pose.close()