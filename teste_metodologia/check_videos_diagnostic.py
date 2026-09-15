"""
check_videos_diagnostic.py

Inspeciona detalhadamente cada vídeo no CSV unificado para garantir 
que a distribuição de frames, detecções faciais e estados estão corretas.
"""

import pandas as pd
from pathlib import Path

UNIFIED_CSV = "/home/eduardo/Code/distraction_detection/teste_metodologia/results/classificacoes_frames_exatos.csv"

def main():
    if not Path(UNIFIED_CSV).exists():
        print(f"[Erro] Arquivo não encontrado: {UNIFIED_CSV}")
        return

    df = pd.read_csv(UNIFIED_CSV)
    print(f"=== Diagnóstico Geral do Dataset ===")
    print(f"Total de linhas no CSV: {len(df)}")
    print(f"Vídeos encontrados: {df['video_id'].unique().tolist()}\n")

    for video_id, group in df.groupby("video_id"):
        total_frames = len(group)
        face_detected_count = (group["face_detected"] == 1).sum()
        detection_rate = (face_detected_count / total_frames) * 100 if total_frames > 0 else 0
        
        print(f"--------------------------------------------------")
        print(f" Vídeo ID: {video_id}")
        print(f"--------------------------------------------------")
        print(f"  -> Total de frames: {total_frames}")
        print(f"  -> Frames com face detectada (1): {face_detected_count} ({detection_rate:.1f}%)")
        print(f"  -> Frames sem face detectada (0): {total_frames - face_detected_count} ({100 - detection_rate:.1f}%)")
        
        print(f"  -> Distribuição de Estados:")
        state_counts = group["state"].value_counts(dropna=False)
        for state, count in state_counts.items():
            pct = (count / total_frames) * 100
            print(f"     - {state}: {count} frames ({pct:.1f}%)")
        
        # Verifica se há valores nulos nas features nos frames onde a face foi detectada
        valid_face_rows = group[group["face_detected"] == 1]
        if not valid_face_rows.empty:
            null_features = valid_face_rows[["ear", "mar", "pitch", "yaw", "roll"]].isna().sum().sum()
            print(f"  -> Valores NaN nas 5 features (onde face_detected=1): {null_features}")
        else:
            print(f"  -> [Aviso] Nenhuma face detectada neste vídeo!")
        print()

if __name__ == "__main__":
    main()