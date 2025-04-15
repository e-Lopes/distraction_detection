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
    scale_x, scale_y = 1.0, 1.0  # Fatores de escala para mapear coordenadas
    
    # Capturar frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Erro ao ler vídeo")
        return

    cv2.imwrite(TEMP_VIDEO_FRAME, frame)
    original_image = cv2.imread(TEMP_VIDEO_FRAME)
    original_height, original_width = original_image.shape[:2]
    clone = original_image.copy()

    # Configuração da janela para abrir centralizada e com tamanho adequado
    cv2.namedWindow("Definir ROI da Cadeira", cv2.WINDOW_NORMAL)
    
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
    cv2.resizeWindow("Definir ROI da Cadeira", window_width, window_height)
    
    # Centraliza a janela
    cv2.moveWindow("Definir ROI da Cadeira", 
                  (screen_width - window_width) // 2, 
                  (screen_height - window_height) // 2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_roi, drawing, scale_x, scale_y
        
        # Ajusta as coordenadas para a imagem original
        x_orig = int(x / scale_x)
        y_orig = int(y / scale_y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_roi = [x_orig, y_orig, 0, 0]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_roi[2] = x_orig - current_roi[0]
                current_roi[3] = y_orig - current_roi[1]
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_roi[2] = x_orig - current_roi[0]
            current_roi[3] = y_orig - current_roi[1]
            # Garantir dimensões positivas
            if current_roi[2] < 0:
                current_roi[0] += current_roi[2]
                current_roi[2] = abs(current_roi[2])
            if current_roi[3] < 0:
                current_roi[1] += current_roi[3]
                current_roi[3] = abs(current_roi[3])

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
        
        # Redimensiona o frame para caber na janela
        resized_display = display.copy()
        frame_height, frame_width = resized_display.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        # Calcula os fatores de escala
        if frame_width > window_width or frame_height > window_height:
            if frame_width / window_width > frame_height / window_height:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = window_height
                new_width = int(new_height * aspect_ratio)
            
            resized_display = cv2.resize(resized_display, (new_width, new_height))
            scale_x = new_width / frame_width
            scale_y = new_height / frame_height
        else:
            scale_x = scale_y = 1.0
        
        # Texto de ajuda (desenhar depois do redimensionamento para manter legibilidade)
        cv2.putText(resized_display, "Defina a ROI e pressione ESPACO para confirmar, depois pressione a tecla 's' para salvar", (10, 30),
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
            
        cv2.imshow("Definir ROI da Cadeira", resized_display)

    cv2.destroyAllWindows()
    Path(TEMP_VIDEO_FRAME).unlink(missing_ok=True)

if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        setup_roi(video_path)
    else:
        print("Nenhum vídeo selecionado")