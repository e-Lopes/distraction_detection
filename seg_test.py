import torch
import json
import cv2
import os
from collections import deque
import numpy as np
from ultralytics import YOLO

# Configurações
CONFIG_PATH = "roi_config.json"
OUTPUT_LOG = "segmentation_log.txt"
SEGMENTATION_MODEL_PATH = "yolo11x-seg.pt"  # Modelo de segmentação

# Dispositivo (GPU/CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo de segmentação
model = YOLO(SEGMENTATION_MODEL_PATH).to(DEVICE)

def process_video(config):
    video_path = config["video_path"]
    roi_chair = config["roi_cadeira"]
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(240 * fps)  # Começa aos 4 minutos
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Aplica segmentação apenas na ROI
        x, y, w, h = roi_chair
        roi = frame[y:y+h, x:x+w]
        
        # Executa segmentação com YOLOv11x-Seg
        results = model.predict(
            roi, 
            conf=0.4, 
            classes=[67],  # Filtra apenas celulares
            device=DEVICE,
            retina_masks=True  # Máscaras de alta qualidade
        )
        
        # Sobreponha a máscara segmentada
        for result in results:
            if result.masks is not None:
                # Combina todas as máscaras de celular
                combined_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                for mask in result.masks.data:
                    mask = mask.cpu().numpy().astype(np.uint8)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                # Aplica a máscara colorida (ex: vermelho transparente)
                colored_mask = np.zeros_like(roi)
                colored_mask[combined_mask == 1] = [0, 0, 255]  # Vermelho
                frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.7, colored_mask, 0.3, 0)
        
        # Mostra o resultado
        cv2.imshow("Segmentação de Celular", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    process_video(config)