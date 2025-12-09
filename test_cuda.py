#import tensorflow as tf
#print(tf.__version__)

import torch

print("\n--- Verificação Pós-Instalação ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU devices: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA não disponível - problemas persistentes")
print("----------------------------------\n")

# Teste prático
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor = torch.randn(2,3).to(device)
print(f"Tensor criado em: {tensor.device}")