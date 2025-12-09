import numpy as np
import matplotlib.pyplot as plt

# Matriz de confusão normalizada
cm_normalized = np.array([
    [0.92, 0.00, 0.08, 0.00],
    [0.01, 0.97, 0.02, 0.00],
    [0.21, 0.00, 0.72, 0.07],
    [0.01, 0.11, 0.09, 0.79]
])

# Nomes das classes
classes = ['Normal Operation', 'Posture Deviation', 'Hands Off Controls', 'Operator Absence']

# Configurar o tamanho da fonte globalmente (opcional)
plt.rcParams.update({'font.size': 16})

# Criar a figura com tamanho ajustado para a fonte grande
plt.figure(figsize=(10, 8))  # Aumente o tamanho para evitar sobreposição

# Plotar a matriz
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix', fontsize=20, pad=20)  # Título com fonte 20
#plt.colorbar().ax.tick_params(labelsize=20)  # Barra de cores com fonte 20

# Adicionar os valores nas células
thresh = cm_normalized.max() / 2.
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black",
                 fontsize=20)  # Fonte dos valores

# Configurar eixos
plt.xticks(np.arange(len(classes)), classes, rotation=45, fontsize=16,ha='right')
plt.yticks(np.arange(len(classes)), classes, fontsize=16)
plt.tight_layout()
plt.ylabel('True Label', fontsize=18,labelpad=20)
plt.xlabel('Predicted Label', fontsize=18,labelpad=20)

# Salvar a imagem
plt.savefig('/home/edu/Downloads/confusion_matrix.png', 
            dpi=300, 
            bbox_inches='tight',
            transparent=False)  # Fundo branco
plt.close()

print("Imagem salva em '/home/edu/Downloads/confusion_matrix.png'")