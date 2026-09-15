"""
episode_sampler.py

Sampler de episódios para o experimento de few-shot / meta-learning
(estilo Prototypical Networks) adaptado para:
  - 2 classes binárias: Alert vs Non-Alert (Fatigue + Distraction)
  - Ignorar estados Absent / Unknown (tratados como ausência de face/dados válidos)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Esquema binário: Alert vs Non-Alert (Fatigue + Distraction agrupadas)
CLASSES_BINARY: List[str] = ["Alert", "Non-Alert"]
BINARY_LABEL_MAP: Dict[str, str] = {
    "alert": "Alert",
    "fatigue": "Non-Alert",
    "distraction": "Non-Alert",
}

# Colunas de indicadores por frame, na ordem em que viram as features da janela
FEATURE_COLUMNS: List[str] = ["EAR", "MAR", "Pitch", "Yaw", "Roll"]


def merge_labels(frame_labels: Sequence[str], label_map: Dict[str, str]) -> List[str]:
    """
    Remapeia rótulos por frame usando `label_map`. Ignora rótulos que não
    estiverem mapeados (como absent/unknown).
    """
    return [label_map[lbl] for lbl in frame_labels if lbl in label_map]


@dataclass
class Window:
    """Uma janela temporal rotulada, pertencente a uma task (vídeo/sujeito)."""

    task_id: str
    label: str
    features: np.ndarray  # shape [window_len, n_features]


def build_windows(
    frame_labels: Sequence[str],
    frame_detected: Sequence[bool],
    frame_features: np.ndarray,
    task_id: str,
    window_len: int,
    stride: Optional[int] = None,
    min_valid_ratio: float = 0.5,
) -> List[Window]:
    """
    Converte indicadores por frame de UM vídeo em janelas fixas rotuladas.
    Descarta frames 'absent'/'unknown' e valida a proporção mínima de frames úteis.
    """
    n_frames = len(frame_labels)
    assert len(frame_detected) == n_frames
    assert frame_features.shape[0] == n_frames

    stride = stride or window_len
    windows: List[Window] = []

    for start in range(0, n_frames - window_len + 1, stride):
        end = start + window_len
        valid = np.asarray(frame_detected[start:end], dtype=bool)
        valid_ratio = valid.mean() if len(valid) else 0.0
        if valid_ratio < min_valid_ratio:
            continue

        # Coleta apenas os rótulos válidos e mapeados dentro da janela
        labels_in_window = [
            BINARY_LABEL_MAP[lbl.lower()]
            for lbl, ok in zip(frame_labels[start:end], valid)
            if ok and lbl.lower() in BINARY_LABEL_MAP
        ]
        
        if not labels_in_window:
            continue
            
        # rótulo majoritário entre os frames válidos da janela
        majority_label = max(set(labels_in_window), key=labels_in_window.count)

        windows.append(
            Window(
                task_id=task_id,
                label=majority_label,
                features=frame_features[start:end].astype(np.float32),
            )
        )

    return windows


def index_by_task_class(
    windows: Sequence[Window],
) -> Dict[str, Dict[str, List[int]]]:
    """
    Organiza uma lista de janelas em {task_id: {classe: [índices em `windows`]}}.
    """
    index: Dict[str, Dict[str, List[int]]] = {}
    for i, w in enumerate(windows):
        index.setdefault(w.task_id, {}).setdefault(w.label, []).append(i)
    return index


@dataclass
class Episode:
    support_x: torch.Tensor  # [n_way * k_shot, window_len, n_features]
    support_y: torch.Tensor  # [n_way * k_shot]
    query_x: torch.Tensor    # [n_way * q_query, window_len, n_features]
    query_y: torch.Tensor    # [n_way * q_query]
    task_id: str
    class_order: List[str] = field(default_factory=list)


class EpisodeSampler:
    """
    Amostra episódios N-way K-shot dentro de uma única task por vez para o esquema binário.
    """

    def __init__(
        self,
        windows: Sequence[Window],
        task_index: Dict[str, Dict[str, List[int]]],
        n_way: Optional[int] = 2,
        k_shot: int = 5,
        q_query: int = 10,
        seed: Optional[int] = None,
    ):
        self.windows = windows
        self.n_way = n_way if n_way is not None else 2
        self.k_shot = k_shot
        self.q_query = q_query
        self._rng = random.Random(seed)

        needed = k_shot + q_query
        self.usable_tasks: Dict[str, Dict[str, List[int]]] = {}
        for task_id, class_map in task_index.items():
            usable_classes = {
                c: idxs for c, idxs in class_map.items() if len(idxs) >= needed
            }
            if len(usable_classes) >= self.n_way:
                self.usable_tasks[task_id] = usable_classes

        if not self.usable_tasks:
            raise ValueError(
                f"Nenhuma task tem >= {self.n_way} classes com >= {needed} "
                f"janelas cada (k_shot={k_shot} + q_query={q_query})."
            )

    @property
    def task_ids(self) -> List[str]:
        return list(self.usable_tasks.keys())

    def sample_episode(self, task_id: Optional[str] = None) -> Episode:
        if task_id is None:
            task_id = self._rng.choice(self.task_ids)
        elif task_id not in self.usable_tasks:
            raise ValueError(
                f"Task '{task_id}' não tem classes/janelas suficientes para "
                f"n_way={self.n_way}, k_shot={self.k_shot}, q_query={self.q_query}."
            )

        class_map = self.usable_tasks[task_id]
        # Garante ordem consistente das classes binárias se existirem
        chosen_classes = sorted(list(class_map.keys()))
        if len(chosen_classes) > self.n_way:
            chosen_classes = self._rng.sample(chosen_classes, self.n_way)

        support_x, support_y, query_x, query_y = [], [], [], []
        for local_label, cls in enumerate(chosen_classes):
            idxs = class_map[cls][:]
            self._rng.shuffle(idxs)
            support_idxs = idxs[: self.k_shot]
            query_idxs = idxs[self.k_shot : self.k_shot + self.q_query]

            for i in support_idxs:
                support_x.append(self.windows[i].features)
                support_y.append(local_label)
            for i in query_idxs:
                query_x.append(self.windows[i].features)
                query_y.append(local_label)

        return Episode(
            support_x=torch.as_tensor(np.stack(support_x)),
            support_y=torch.as_tensor(support_y, dtype=torch.long),
            query_x=torch.as_tensor(np.stack(query_x)),
            query_y=torch.as_tensor(query_y, dtype=torch.long),
            task_id=task_id,
            class_order=chosen_classes,
        )

    def episode_loader(self, num_episodes: int, task_id: Optional[str] = None):
        for _ in range(num_episodes):
            yield self.sample_episode(task_id=task_id)