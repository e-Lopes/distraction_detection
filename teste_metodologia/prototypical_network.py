from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalEncoder1D(nn.Module):
    def __init__(self, n_features: int = 5, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Garante que o tensor de entrada esteja no mesmo device do modelo
        x = x.to(device)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        x = self.fc(x)
        return x

class PrototypicalLoss(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder.to(device)

    def forward(self, episode):
        support_x = episode.support_x.float().to(device)
        support_y = episode.support_y.long().to(device)
        query_x = episode.query_x.float().to(device)
        query_y = episode.query_y.long().to(device)

        z_support = self.encoder(support_x)
        z_query = self.encoder(query_x)

        n_way = len(episode.class_order)
        embedding_dim = z_support.size(-1)

        prototypes = torch.zeros(n_way, embedding_dim, device=z_support.device)
        for c in range(n_way):
            class_mask = support_y == c
            if class_mask.sum() > 0:
                prototypes[c] = z_support[class_mask].mean(dim=0)

        dists = torch.cdist(z_query.unsqueeze(1), prototypes.unsqueeze(0)).squeeze(1)
        logits = -dists

        loss = F.cross_entropy(logits, query_y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_y).float().mean()

        return loss, acc, preds