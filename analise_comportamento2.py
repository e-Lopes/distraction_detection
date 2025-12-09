import cv2
import numpy as np
import json
import math
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path

videos = '/home/edu/Desktop/distraction_detection/distraction_detection/videos/'

# Configurações
ANGULO_ALERTA = 20
TEMPO_MINIMO_ALERTA = 1.5
HISTORICO_GRAFICO = 60
COR_ALERTA = (0, 0, 255)
COR_NORMAL = (0, 255, 0)
COR_NAO_DETECTADO = (255, 0, 0)

# Configurar backend do matplotlib para 'Agg' (sem interface gráfica)
plt.switch_backend('agg')

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def load_config():
    try:
        with open('roi_config.json') as f:
            config = json.load(f)
        return config['roi_cadeira'], config['video_path']
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return None, None

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

def setup_grafico():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_ylim(0, 90)
    ax.axhline(y=ANGULO_ALERTA, color='r', linestyle='--')
    ax.set_title('Head Tilt (Last 60 Frames)')
    ax.set_ylabel('Degrees')
    line, = ax.plot([], [], 'b-')
    fig.tight_layout()
    return fig, ax, line

def update_grafico(fig, ax, line, data):
    line.set_data(range(len(data)), data)
    ax.set_xlim(0, HISTORICO_GRAFICO)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

def main():
    roi, video_path = load_config()
    if roi is None:
        return

    inicio_segundos = 0  # <<< Tempo de início em segundos
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, inicio_segundos * 1000)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = Path('videos')
    output_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_dir/'output_analisado.mp4'), fourcc,
                         fps*2, (width, height))

    fig, ax, line = setup_grafico()
    historico = deque(maxlen=HISTORICO_GRAFICO)
    angulos_totais = []
    timestamps = []
    frames_alerta = 0
    frames_totais = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_totais += 1
        x, y, w, h = roi
        angulo = 0
        estado = "NAO DETECTADO"
        cor = COR_NAO_DETECTADO

        try:
            results = pose.process(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                angulo = calcular_inclinacao(results.pose_landmarks)
                estado = "NORMAL"
                cor = COR_NORMAL

                if angulo > ANGULO_ALERTA:
                    estado = "ALERTA"
                    cor = COR_ALERTA
                    frames_alerta += 1

                mp_drawing.draw_landmarks(
                    frame[y:y+h, x:x+w],
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
        except Exception as e:
            print(f"Erro no processamento: {e}")

        historico.append(angulo)
        angulos_totais.append(angulo)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        # Atualizar gráfico
        if len(historico) > 0:
            grafico_img = update_grafico(fig, ax, line, historico)
            grafico_img = cv2.resize(grafico_img, (400, 200))
            frame[10:210, 10:410] = grafico_img

        # UI
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 3)
        cv2.putText(frame, f"{estado} ({angulo:.1f}°)", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        porcentagem = (frames_alerta / frames_totais) * 100
        cv2.putText(frame, f"Alerta: {porcentagem:.1f}%", 
                   (width-300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('Monitoramento Postural', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()

    # Salvar gráfico final
    if angulos_totais:
        plt.figure(figsize=(10,5))
        plt.plot(timestamps, angulos_totais, 'b-', label='Degrees')
        plt.axhline(ANGULO_ALERTA, color='r', linestyle='--', label='Limit')
        plt.title('Head Tilt')
        plt.ylabel('Degrees')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.savefig(str(output_dir/'grafico_completo.png'))
        plt.close()

    print(f"\nRelatório Final:")
    print(f"Frames analisados: {frames_totais}")
    print(f"Tempo em alerta: {porcentagem:.1f}%")

if __name__ == "__main__":
    main()
