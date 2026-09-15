"""
episode_sampler.py

Sampler de episódios para o experimento de few-shot / meta-learning
(estilo Prototypical Networks) comparando:
  Pipeline A: landmarks -> 5 séries temporais (EAR, MAR, Pitch, Yaw, Roll)
  Pipeline B: frames/vídeo bruto

Ideia central do desenho (diferente do few-shot clássico de imagem):
- As CLASSES são fixas e compartilhadas entre todas as tasks: Alert,
  Fatigue, Distraction. Não estamos generalizando para classes novas.
- Cada TASK é um domínio distinto: um sujeito/vídeo do NTHU-DDD ou
  UTA-RLDD durante o meta-treino, ou um dos seus vídeos reais de
  teleoperação durante a meta-avaliação.
- O objetivo é generalizar para uma TASK (domínio/câmera/sujeito) nova
  usando poucos exemplos rotulados por classe: "few-shot domain adaptation".

Fluxo:
  1. build_windows(...)        -> converte séries temporais brutas de UM
                                   vídeo em janelas fixas rotuladas.
  2. index_by_task_class(...)  -> organiza janelas de vários vídeos em
                                   {task_id: {classe: [índices]}}.
  3. EpisodeSampler             -> amostra episódios (support/query) a
                                   partir dessa estrutura.

Este arquivo já roda de ponta a ponta com dados sintéticos (ver o bloco
`if __name__ == "__main__"`) para você validar a lógica antes de plugar
os dados reais.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Classes originais do protocolo (ver protocolo-comparacao-algoritmos)
CLASSES: List[str] = ["Alert", "Fatigue", "Distraction"]

# Esquema binário: Alert vs Not-Alert (Fatigue + Distraction agrupadas)
CLASSES_BINARY: List[str] = ["Alert", "Not-Alert"]
BINARY_LABEL_MAP: Dict[str, str] = {
    "Alert": "Alert",
    "Fatigue": "Not-Alert",
    "Distraction": "Not-Alert",
}

# Colunas de indicadores por frame, na ordem em que viram as features da janela
FEATURE_COLUMNS: List[str] = ["EAR", "MAR", "Pitch", "Yaw", "Roll"]


def merge_labels(frame_labels: Sequence[str], label_map: Dict[str, str]) -> List[str]:
    """
    Remapeia rótulos por frame usando `label_map` (ex.: BINARY_LABEL_MAP para
    colapsar Fatigue+Distraction em "Not-Alert"). Aplique isso ANTES de
    build_windows, para que o voto majoritário da janela já considere os
    rótulos fundidos.
    """
    try:
        return [label_map[lbl] for lbl in frame_labels]
    except KeyError as e:
        raise KeyError(
            f"Rótulo {e} não está em label_map — inclua-o no mapeamento "
            "(todo rótulo original precisa de um destino)."
        ) from e


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

    frame_labels:    rótulo por frame (ex.: "Alert"/"Fatigue"/"Distraction"),
                      já vindo da sua anotação.
    frame_detected:  booleano por frame indicando se houve detecção facial
                      válida naquele frame (liga com a discussão de
                      presença vs ausência do operador).
    frame_features:  array [n_frames, n_features] com EAR/MAR/Pitch/Yaw/Roll.
    task_id:         identificador do vídeo/sujeito (ex.: "video_01",
                      "nthu_subj03").
    window_len:      tamanho da janela em frames (ex.: 5s * fps).
    stride:          passo entre janelas; default = window_len (sem overlap).
    min_valid_ratio: fração mínima de frames com detecção válida dentro da
                      janela para ela ser aproveitada; caso contrário é
                      descartada (evita janelas dominadas por ausência real
                      do operador ou falha de detecção).

    Retorna uma lista de Window. O rótulo da janela é o majoritário entre os
    frames válidos.
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

        labels_in_window = [
            lbl for lbl, ok in zip(frame_labels[start:end], valid) if ok
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
    Organiza uma lista de janelas (possivelmente de vários vídeos) em
    {task_id: {classe: [índices em `windows`]}}, para consumo do
    EpisodeSampler.
    """
    index: Dict[str, Dict[str, List[int]]] = {}
    for i, w in enumerate(windows):
        index.setdefault(w.task_id, {}).setdefault(w.label, []).append(i)
    return index


@dataclass
class Episode:
    support_x: torch.Tensor  # [n_way * k_shot, window_len, n_features]
    support_y: torch.Tensor  # [n_way * k_shot]  (rótulos locais 0..n_way-1)
    query_x: torch.Tensor  # [n_way * q_query, window_len, n_features]
    query_y: torch.Tensor  # [n_way * q_query]
    task_id: str
    class_order: List[str] = field(default_factory=list)  # index -> nome da classe


class EpisodeSampler:
    """
    Amostra episódios N-way K-shot dentro de uma única task por vez.

    Uso típico:
      - Meta-treino: task_id=None a cada chamada -> sorteia uma task
        (sujeito) por vez entre as tasks disponíveis (ex.: NTHU-DDD,
        UTA-RLDD), simulando "cada sujeito é um domínio diferente".
      - Meta-avaliação: task_id fixo = um dos seus vídeos reais, para medir
        few-shot adaptação a esse domínio específico.
    """

    def __init__(
        self,
        windows: Sequence[Window],
        task_index: Dict[str, Dict[str, List[int]]],
        n_way: Optional[int] = None,
        k_shot: int = 5,
        q_query: int = 10,
        seed: Optional[int] = None,
    ):
        self.windows = windows
        if n_way is None:
            # infere do próprio dado (funciona tanto para o esquema de 3
            # classes quanto para o binário Alert/Not-Alert) em vez de
            # assumir CLASSES fixo
            all_classes = {c for classes in task_index.values() for c in classes}
            n_way = len(all_classes)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self._rng = random.Random(seed)

        needed = k_shot + q_query
        # só mantemos, por task, as classes com exemplos suficientes
        self.usable_tasks: Dict[str, Dict[str, List[int]]] = {}
        for task_id, class_map in task_index.items():
            usable_classes = {
                c: idxs for c, idxs in class_map.items() if len(idxs) >= needed
            }
            if len(usable_classes) >= n_way:
                self.usable_tasks[task_id] = usable_classes

        if not self.usable_tasks:
            raise ValueError(
                f"Nenhuma task tem >= {n_way} classes com >= {needed} "
                f"janelas cada (k_shot={k_shot} + q_query={q_query}). "
                "Reduza k_shot/q_query, aumente window overlap, ou revise "
                "min_valid_ratio no build_windows."
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
        chosen_classes = self._rng.sample(list(class_map.keys()), self.n_way)

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
        """Generator conveniente para usar direto no loop de treino/avaliação."""
        for _ in range(num_episodes):
            yield self.sample_episode(task_id=task_id)


# ---------------------------------------------------------------------------
# Demonstração com dados sintéticos — roda sem depender dos seus dados reais.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def make_fake_video(task_id: str, n_frames: int = 6000, fps: int = 30):
        # rótulo por frame: majoritariamente Alert, com blocos longos de
        # Fatigue/Distraction (blocos longos = mais janelas por classe,
        # necessário pra demo ter k_shot+q_query exemplos por classe)
        labels = ["Alert"] * n_frames
        for start, length, lbl in [
            (800, 1500, "Fatigue"),
            (2800, 1200, "Distraction"),
            (4300, 1400, "Fatigue"),
        ]:
            for i in range(start, min(start + length, n_frames)):
                labels[i] = lbl

        # detecção facial: simula gaps (operador ausente / falha de detecção)
        detected = rng.random(n_frames) > 0.15  # ~85% de detecção nesse exemplo

        # features fake [EAR, MAR, Pitch, Yaw, Roll], plausíveis por classe
        features = rng.normal(loc=0.0, scale=1.0, size=(n_frames, 5)).astype(
            np.float32
        )
        return labels, detected, features

    # janela de 1s a 30fps, com overlap (stride < window_len) para gerar mais
    # janelas por classe a partir de um mesmo bloco rotulado
    window_len = 30
    stride = 10

    # esquema binário: Alert vs Not-Alert (Fatigue + Distraction fundidas)
    all_windows: List[Window] = []
    for subj in range(6):  # simula 6 sujeitos de um dataset público (meta-treino)
        labels, detected, feats = make_fake_video(f"public_subj{subj:02d}")
        labels = merge_labels(labels, BINARY_LABEL_MAP)
        all_windows += build_windows(
            labels,
            detected,
            feats,
            task_id=f"public_subj{subj:02d}",
            window_len=window_len,
            stride=stride,
        )

    # simula 1 vídeo próprio (target do few-shot / meta-avaliação)
    labels, detected, feats = make_fake_video("video_01", n_frames=6000)
    labels = merge_labels(labels, BINARY_LABEL_MAP)
    own_windows = build_windows(
        labels, detected, feats, task_id="video_01", window_len=window_len, stride=stride
    )

    task_index = index_by_task_class(all_windows)
    # n_way=2 aqui (ou omita n_way e deixe o sampler inferir do próprio dado)
    sampler = EpisodeSampler(all_windows, task_index, n_way=2, k_shot=5, q_query=10, seed=42)

    print(f"Tasks utilizáveis para meta-treino: {sampler.task_ids}")
    ep = sampler.sample_episode()
    print(f"Episódio sorteado da task '{ep.task_id}', classes: {ep.class_order}")
    print(f"support_x: {tuple(ep.support_x.shape)}  support_y: {tuple(ep.support_y.shape)}")
    print(f"query_x:   {tuple(ep.query_x.shape)}  query_y:   {tuple(ep.query_y.shape)}")

    # meta-avaliação: episódios sempre da mesma task (seu vídeo real)
    own_index = index_by_task_class(own_windows)
    try:
        eval_sampler = EpisodeSampler(
            own_windows, own_index, n_way=2, k_shot=5, q_query=10, seed=0
        )
        eval_ep = eval_sampler.sample_episode(task_id="video_01")
        print(f"\nEpisódio de avaliação em '{eval_ep.task_id}', classes: {eval_ep.class_order}")
        print(f"support_x: {tuple(eval_ep.support_x.shape)}  query_x: {tuple(eval_ep.query_x.shape)}")
    except ValueError as e:
        print(f"\n[aviso] vídeo próprio sintético não tem janelas suficientes: {e}")