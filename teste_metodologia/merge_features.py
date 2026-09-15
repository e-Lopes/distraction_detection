"""
merge_features.py

Lê os arquivos de vídeo individuais (video_01.csv a video_04.csv) e transfere
as colunas de landmarks e detecção para o CSV unificado (classificacoes_frames_exatos.csv).
"""

from pathlib import Path
import pandas as pd

# Caminhos dos arquivos
UNIFIED_CSV = "/home/eduardo/Code/distraction_detection/teste_metodologia/results/classificacoes_frames_exatos.csv"
VIDEO_CSV_LIST = [
    "fase_2/data/interim/legacy_extraction/video_01.csv",
    "fase_2/data/interim/legacy_extraction/video_02.csv",
    "fase_2/data/interim/legacy_extraction/video_03.csv",
    "fase_2/data/interim/legacy_extraction/video_04.csv"
]

def main():
    print("=== Iniciando mesclagem das features para o CSV unificado ===")
    
    if not Path(UNIFIED_CSV).exists():
        raise FileNotFoundError(f"Arquivo unificado não encontrado: {UNIFIED_CSV}")

    # 1. Carrega o CSV unificado base
    df_unified = pd.read_csv(UNIFIED_CSV)
    print(f"Linhas no CSV unificado antes do merge: {len(df_unified)}")

    # 2. Carrega e concatena os CSVs individuais dos vídeos
    feature_dfs = []
    columns_to_bring = ["video_id", "frame_index", "ear", "mar", "pitch", "yaw", "roll", "face_detected"]

    for path in VIDEO_CSV_LIST:
        if Path(path).exists():
            df_v = pd.read_csv(path)
            # Garante que as colunas necessárias existem no arquivo
            missing = [c for c in columns_to_bring if c not in df_v.columns]
            if missing:
                print(f"[Aviso] O arquivo {path} não possui as colunas: {missing}")
                continue
            
            feature_dfs.append(df_v[columns_to_bring])
            print(f"  -> Carregado {path}: {len(df_v)} linhas")
        else:
            print(f"[Aviso] Arquivo de vídeo não encontrado: {path}")

    if not feature_dfs:
        raise ValueError("Nenhum arquivo de vídeo individual válido foi encontrado para a mesclagem.")

    df_features = pd.concat(feature_dfs, ignore_index=True)

    # 3. Remove colunas de features antigas do unificado caso já existissem (para evitar duplicação _x/_y)
    cols_to_drop = [c for c in ["ear", "mar", "pitch", "yaw", "roll", "face_detected"] if c in df_unified.columns]
    if cols_to_drop:
        df_unified = df_unified.drop(columns=cols_to_drop)

    # 4. Faz o merge (esquerda) usando video_id e frame_index como chaves exatas
    df_merged = pd.merge(
        df_unified,
        df_features,
        on=["video_id", "frame_index"],
        how="left"
    )

    print(f"Linhas no CSV unificado após o merge: {len(df_merged)}")
    
    # 5. Salva o resultado de volta no CSV unificado
    df_merged.to_csv(UNIFIED_CSV, index=False)
    print(f"\nSucesso! Arquivo '{UNIFIED_CSV}' atualizado com as features reais.")
    print("Colunas finais presentes no arquivo:")
    print(df_merged.columns.tolist())

if __name__ == "__main__":
    main()