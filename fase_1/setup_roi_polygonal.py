import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from tkinter import Tk, filedialog

CONFIG_FILE = 'roi_config.json'
MODEL_PATH = 'yolov11x-seg.pt'  # Caminho do modelo de segmentação


def select_video_file():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    return filedialog.askopenfilename(
        title="Selecione o vídeo para análise",
        filetypes=[("Vídeos", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*")]
    )

def get_combined_polygon_from_clicks(frame, masks, classes, names):
    selected = []
    display = frame.copy()

    # Desenha todas as máscaras detectadas com cores diferentes
    for i, seg in enumerate(masks):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.fillPoly(display, [np.array(seg, dtype=np.int32)], color)
        x, y = int(seg[0][0]), int(seg[0][1])
        label = f"{i}: {names[int(classes[i])]}"
        cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print("Clique nos objetos que pertencem ao ROI. Pressione ENTER para finalizar.")

    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, seg in enumerate(masks):
                if cv2.pointPolygonTest(np.array(seg, dtype=np.int32), (x, y), False) >= 0:
                    if i not in selected:
                        selected.append(i)
                        print(f"Objeto {i} ({names[int(classes[i])]}) selecionado")

    cv2.namedWindow("Selecione partes do ROI")
    cv2.setMouseCallback("Selecione partes do ROI", click_callback)

    while True:
        temp = display.copy()
        for i in selected:
            cv2.polylines(temp, [np.array(masks[i], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.imshow("Selecione partes do ROI", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break

    cv2.destroyAllWindows()

    if not selected:
        print("Nenhuma região selecionada.")
        return None

    # Combina as máscaras selecionadas
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for i in selected:
        cv2.fillPoly(combined_mask, [np.array(masks[i], dtype=np.int32)], 255)

    # Encontrar contorno externo da região combinada
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return largest[:, 0, :].tolist()
    return None

def extract_polygonal_roi(video_path):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro ao ler o primeiro frame do vídeo")
        return None

    results = model.predict(source=frame, conf=0.4, task='segment', verbose=False)

    for r in results:
        if not hasattr(r, 'masks') or r.masks is None:
            continue
        masks = r.masks.xy
        classes = r.boxes.cls
        names = r.names

        return get_combined_polygon_from_clicks(frame, masks, classes, names)

    print("Nenhum objeto detectado no frame.")
    return None

def draw_polygon(image, polygon, color=(0, 255, 0)):
    if polygon is not None:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    return image

def main():
    video_path = select_video_file()
    if not video_path:
        print("Nenhum vídeo selecionado")
        return

    polygon = extract_polygonal_roi(video_path)
    if polygon is None:
        print("ROI poligonal não encontrada.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        display = draw_polygon(frame.copy(), polygon)
        cv2.imshow("ROI Poligonal Final", display)
        print("Pressione qualquer tecla para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    config = {
        'video_path': video_path,
        'roi_cadeira_polygon': polygon
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuração salva em {CONFIG_FILE}")

if __name__ == "__main__":
    main()
