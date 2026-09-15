"""
run_validation_episode.py

Testa a Pipeline A carregando as janelas salvas e rodando um episódio de avaliação.
"""

import torch
from load_own_data import load_windows
from episode_sampler import index_by_task_class, EpisodeSampler
from prototypical_network import TemporalEncoder1D, PrototypicalLoss

def main():
    # 1. Carrega as janelas salvas anteriormente
    windows_path = "teste_metodologia/results/windows_binary.pkl"
    print(f"Carregando janelas de: {windows_path}")
    windows = load_windows(windows_path)

    # 2. Configura o sampler para o vídeo de teste (ex: 'video_01')
    task_index = index_by_task_class(windows)
    sampler = EpisodeSampler(windows, task_index, n_way=2, k_shot=5, q_query=10, seed=42)

    # 3. Instancia o Encoder e a loss Prototípica
    encoder = TemporalEncoder1D(n_features=5, embedding_dim=64)
    proto_loss = PrototypicalLoss(encoder)

    # 4. Amostra um episódio do 'video_01' e avalia
    target_task = "video_01"
    episode = sampler.sample_episode(task_id=target_task)
    
    encoder.eval()
    with torch.no_grad():
        loss, acc, preds = proto_loss(episode)

    print(f"\n--- Validação da Pipeline A (Task: {target_task}) ---")
    print(f"Ordem das classes: {episode.class_order}")
    print(f"Perda Prototípica: {loss.item():.4f}")
    print(f"Acurácia no Query Set: {acc.item() * 100:.2f}%")
    print(f"Predições vs Real (Query):")
    print(f"  Preds: {preds.tolist()}")
    print(f"  Real:  {episode.query_y.tolist()}")

if __name__ == "__main__":
    main()