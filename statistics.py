import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Configurações
FRAMES_DIR = "frames_por_estado"
STATES = ["NORMAL", "POSTURA_RUIM", "MAOS_FORA", "NAO_DETECTADO"]

def analyze_frame_distribution():
    """Analisa a distribuição dos frames por estado"""
    frame_counts = defaultdict(int)
    total_frames = 0
    
    print("\n=== DISTRIBUIÇÃO DOS FRAMES POR ESTADO ===")
    for state in STATES:
        state_dir = os.path.join(FRAMES_DIR, state)
        if os.path.exists(state_dir):
            count = len([f for f in os.listdir(state_dir) if f.endswith('.jpg')])
            frame_counts[state] = count
            total_frames += count
    
    if total_frames == 0:
        print("Nenhum frame encontrado. Verifique o diretório.")
        return
    
    # Imprimir distribuição
    for state, count in frame_counts.items():
        print(f"{state}: {count} frames ({count/total_frames:.1%})")
    
    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(frame_counts.keys(), frame_counts.values())
    plt.title("Distribuição de Frames por Estado")
    plt.xlabel("Estado")
    plt.ylabel("Número de Frames")
    plt.show()

def sample_frames_per_state(n=5):
    """Mostra exemplos de frames de cada estado"""
    for state in STATES:
        state_dir = os.path.join(FRAMES_DIR, state)
        if os.path.exists(state_dir):
            frames = [f for f in os.listdir(state_dir) if f.endswith('.jpg')]
            print(f"\n=== AMOSTRA DE {state} ({min(n, len(frames))} frames) ===")
            
            for i, frame_file in enumerate(frames[:n]):
                frame_path = os.path.join(state_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                # Redimensionar para visualização
                frame = cv2.resize(frame, (400, 300))
                
                # Mostrar frame
                cv2.imshow(f"{state} - {frame_file}", frame)
                cv2.waitKey(500)  # Mostra cada frame por 0.5 segundos
                cv2.destroyAllWindows()

def main():
    analyze_frame_distribution()
    sample_frames_per_state()

if __name__ == "__main__":
    main()