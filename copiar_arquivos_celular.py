import os
import shutil
import pandas as pd

# Caminhos
base_dir = "/home/edu/frames_por_estado"
csv_path = "/home/edu/Desktop/distraction_detection/distraction_detection/validacao_manual.csv"
output_dir = "/home/edu/DATASET_CELULAR"

# Cria a pasta de saída, se não existir
os.makedirs(output_dir, exist_ok=True)

# Lê o CSV
df = pd.read_csv(csv_path)

# Filtra apenas as linhas cujo estado_real == CELULAR
df_celular = df[df["estado_real"] == "CELULAR"]

# Percorre as imagens filtradas
for _, row in df_celular.iterrows():
    arquivo = row["arquivo"]
    estado_predito = row["estado_predito"]

    # Caminho original do arquivo
    src = os.path.join(base_dir, estado_predito, arquivo)
    # Caminho de destino
    dst = os.path.join(output_dir, arquivo)

    # Copia o arquivo se existir
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"[AVISO] Arquivo não encontrado: {src}")
