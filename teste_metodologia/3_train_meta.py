"""
train_meta.py

Loop de Meta-Treino para Prototypical Networks utilizando as janelas com features reais.
"""

import os
import torch
import torch.optim as optim

from load_own_data import load_windows
from episode_sampler import index_by_task_class, EpisodeSampler
from prototypical_network import TemporalEncoder1D, PrototypicalLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device para meta-treino: {device}")
    if device.type == "cuda":
        print(f"Placa de Vídeo: {torch.cuda.get_device_name(0)}")

    possible_paths = ["results/windows_binary.pkl", "teste_metodologia/results/windows_binary.pkl"]
    windows_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if not windows_path:
        raise FileNotFoundError("Arquivo windows_binary.pkl não encontrado. Rode o run_episode_sampler.py primeiro.")

    print(f"Carregando janelas de: {windows_path}")
    windows = load_windows(windows_path)

    task_index = index_by_task_class(windows)
    sampler = EpisodeSampler(windows, task_index, n_way=2, k_shot=3, q_query=5, seed=42)

    encoder = TemporalEncoder1D(n_features=5, embedding_dim=64).to(device)
    criterion = PrototypicalLoss(encoder)
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)

    print(f"Iniciando Meta-Treino com tasks: {sampler.task_ids}")
    
    encoder.train()
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        episode = sampler.sample_episode(task_id=None)
        loss, acc, _ = criterion(episode)
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Época [{epoch}/{num_epochs}] | Task: {episode.task_id} | Loss: {loss.item():.4f} | Acurácia Query: {acc.item()*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    torch.save(encoder.state_dict(), "results/temporal_encoder_meta.pth")
    print("\nMeta-treino finalizado na GPU e modelo salvo com sucesso!")

if __name__ == "__main__":
    main()