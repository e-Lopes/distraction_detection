import cv2
import numpy as np
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Configurações
CONFIG_FILE = 'roi_config.json'
TEMP_VIDEO_FRAME = 'temp_frame.jpg'
COR_CONFIRMACAO = (0, 255, 0)  # Verde para ROI confirmado

def select_video_file():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    return filedialog.askopenfilename(
        title="Selecione o vídeo para análise",
        filetypes=[("Vídeos", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*")]
    )

def draw_roi(image, roi, color=(0, 255, 255), thickness=2):
    if roi:
        x, y, w, h = roi
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image

def setup_roi(video_path):
    current_roi = None
    roi_confirmed = None
    drawing = False
    
    # Capturar frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Erro ao ler vídeo")
        return

    cv2.imwrite(TEMP_VIDEO_FRAME, frame)
    image = cv2.imread(TEMP_VIDEO_FRAME)
    clone = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_roi, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_roi = [x, y, 0, 0]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_roi[2] = x - current_roi[0]
                current_roi[3] = y - current_roi[1]
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_roi[2] = x - current_roi[0]
            current_roi[3] = y - current_roi[1]
            # Garantir dimensões positivas
            if current_roi[2] < 0:
                current_roi[0] += current_roi[2]
                current_roi[2] = abs(current_roi[2])
            if current_roi[3] < 0:
                current_roi[1] += current_roi[3]
                current_roi[3] = abs(current_roi[3])

    cv2.namedWindow("Definir ROI da Cadeira")
    cv2.setMouseCallback("Definir ROI da Cadeira", mouse_callback)

    print("Instruções:")
    print("1. Desenhe o ROI ao redor da cadeira/operador")
    print("2. Pressione ESPAÇO para confirmar")
    print("3. Pressione 's' para salvar ou 'q' para sair")

    while True:
        display = clone.copy()
        
        # Desenhar ROI atual ou confirmado
        if roi_confirmed:
            draw_roi(display, roi_confirmed, COR_CONFIRMACAO, 3)
        elif current_roi:
            draw_roi(display, current_roi, (0, 255, 255), 2)  # Amarelo para não confirmado
        
        # Texto de ajuda
        cv2.putText(display, "Defina a ROI e pressione ESPACO para confirmar, depois pressione a tecla 's' para salvar e", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and current_roi:  # Confirmar ROI
            roi_confirmed = current_roi.copy()
            print(f"ROI confirmado: {roi_confirmed}")
            
        elif key == ord('s'):  # Salvar
            if roi_confirmed:
                config = {
                    'video_path': video_path,
                    'roi_cadeira': roi_confirmed
                }
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"Configuração salva em {CONFIG_FILE}")
                break
            else:
                print("Confirme o ROI antes de salvar")
                
        elif key == ord('q'):  # Sair
            print("Configuração cancelada")
            break
            
        cv2.imshow("Definir ROI da Cadeira", display)

    cv2.destroyAllWindows()
    Path(TEMP_VIDEO_FRAME).unlink(missing_ok=True)

if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        setup_roi(video_path)
    else:
        print("Nenhum vídeo selecionado")