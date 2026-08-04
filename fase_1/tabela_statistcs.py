import pandas as pd
from sklearn.metrics import classification_report
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configuração do caminho
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'validacao_manual.csv')

# Mapeamento para os nomes de exibição
label_display_names = {
    'NORMAL': 'Normal Operation',
    'POSTURA_RUIM': 'Posture Deviation',
    'MAOS_FORA': 'Hands Off Controls',
    'NAO_DETECTADO': 'Operator Absence'
}

# 1. Carregar dados do CSV
try:
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ['estado_real', 'estado_predito']):
        raise ValueError("CSV deve conter colunas 'estado_real' e 'estado_predito'")
except Exception as e:
    print(f"Erro ao ler CSV: {e}")
    exit()

# 2. Pré-processamento
df['estado_real'] = df['estado_real'].str.upper().str.strip()
df['estado_predito'] = df['estado_predito'].str.upper().str.strip()

# Substituir 'CELULAR' por 'MAOS_FORA'
df['estado_real'] = df['estado_real'].replace('CELULAR', 'MAOS_FORA')
df['estado_predito'] = df['estado_predito'].replace('CELULAR', 'MAOS_FORA')

# 3. Gerar relatório
try:
    report = classification_report(
        df['estado_real'],
        df['estado_predito'],
        labels=['NORMAL', 'POSTURA_RUIM', 'MAOS_FORA', 'NAO_DETECTADO'],
        target_names=['NORMAL', 'POSTURA_RUIM', 'MAOS_FORA', 'NAO_DETECTADO'],
        output_dict=True
    )
except Exception as e:
    print(f"Erro ao gerar relatório: {e}")
    print("\nValores únicos em estado_real:", df['estado_real'].unique())
    print("Valores únicos em estado_predito:", df['estado_predito'].unique())
    exit()

# 4. Formatando a saída
formatted_report = {}
for label, metrics in report.items():
    if label in label_display_names:
        formatted_report[label_display_names[label]] = {
            'Precision': round(metrics['precision'], 2),
            'Recall': round(metrics['recall'], 2),
            'F1-score': round(metrics['f1-score'], 2),
            'Support': int(metrics['support'])
        }

# 5. Exibir resultados
print("\n{:<20} {:<10} {:<10} {:<10} {:<10}".format(
    'Class', 'Precision', 'Recall', 'F1-score', 'Support'))
print("-" * 60)

for class_name, metrics in formatted_report.items():
    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
        class_name,
        metrics['Precision'],
        metrics['Recall'],
        metrics['F1-score'],
        metrics['Support']))

# Métricas adicionais
print("\nAdditional Metrics:")
print(f"Overall Accuracy: {report['accuracy']:.2f}")
print(f"Macro F1-score: {report['macro avg']['f1-score']:.2f}")
print(f"Weighted F1-score: {report['weighted avg']['f1-score']:.2f}")

# 6. Salvar resultados em um arquivo
output_path = os.path.join(base_dir, 'classification_report.txt')
with open(output_path, 'w') as f:
    f.write("Classification Report (com 'CELULAR' tratado como 'MAOS_FORA')\n")
    f.write("-" * 60 + "\n")
    for class_name, metrics in formatted_report.items():
        f.write(f"{class_name:<20} {metrics['Precision']:<10.2f} {metrics['Recall']:<10.2f} "
                f"{metrics['F1-score']:<10.2f} {metrics['Support']:<10}\n")
    f.write("\nAdditional Metrics:\n")
    f.write(f"Overall Accuracy: {report['accuracy']:.2f}\n")
    f.write(f"Macro F1-score: {report['macro avg']['f1-score']:.2f}\n")
    f.write(f"Weighted F1-score: {report['weighted avg']['f1-score']:.2f}\n")
    f.write(f"\nTotal de registros analisados: {len(df)}\n")

print(f"\nRelatório salvo em: {output_path}")

# 7. Gerar matriz de confusão normalizada
labels = ['NORMAL', 'POSTURA_RUIM', 'MAOS_FORA', 'NAO_DETECTADO']
display_labels = ['Normal Operation', 'Posture Deviation', 'Hands Off Controls', 'Operator Absence']

# Matriz de confusão (normalizada linha a linha)
cm = confusion_matrix(df['estado_real'], df['estado_predito'], labels=labels, normalize='true')

# Plotando a matriz
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix', fontsize=20, pad=20)

# Adicionar valores na matriz
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)

plt.xticks(np.arange(len(display_labels)), display_labels, rotation=45, fontsize=16, ha='right')
plt.yticks(np.arange(len(display_labels)), display_labels, fontsize=16)
plt.ylabel('True Label', fontsize=18, labelpad=20)
plt.xlabel('Predicted Label', fontsize=18, labelpad=20)
plt.tight_layout()

# Salvar imagem
image_path = os.path.join(base_dir, 'confusion_matrix.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight', transparent=False)
plt.close()

print(f"Imagem da matriz de confusão salva em: {image_path}")
