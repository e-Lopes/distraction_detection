"""
load_own_data.py

Lê o CSV unificado (classificacoes_frames_exatos.csv) contendo as colunas de
landmarks reais e estados binários (Alert vs Not-Alert), construindo as janelas temporais.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from episode_sampler import (
    BINARY_LABEL_MAP,
    Window,
    build_windows,
)

TASK_COLUMN = "video_id"
FRAME_INDEX_COLUMN = "frame_index"
TIMESTAMP_COLUMN = "timestamp_seconds"
STATE_COLUMN = "state"
DETECTED_COLUMN = "face_detected"
FEATURE_COLUMNS: List[str] = ["ear", "mar", "pitch", "yaw", "roll"]


def estimate_fps(df: pd.DataFrame) -> float:
    diffs = df[TIMESTAMP_COLUMN].diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        raise ValueError("Não foi possível estimar fps.")
    return 1.0 / diffs.median()


def load_unified_csv_to_windows(
    csv_path: str,
    window_len: int,
    stride: Optional[int] = None,
    min_valid_ratio: float = 0.5,
    verbose: bool = True,
) -> List[Window]:
    df = pd.read_csv(csv_path)
    all_windows: List[Window] = []
    
    # Capitaliza o estado para corresponder às chaves do BINARY_LABEL_MAP
    df[STATE_COLUMN] = df[STATE_COLUMN].astype(str).str.capitalize()

    for task_id, group_df in df.groupby(TASK_COLUMN):
        df_sorted = group_df.sort_values(FRAME_INDEX_COLUMN).reset_index(drop=True)
        
        states = df_sorted[STATE_COLUMN]
        is_valid_state = states.isin(list(BINARY_LABEL_MAP.keys()))
        face_ok = df_sorted[DETECTED_COLUMN].astype(bool)
        
        # Frame é considerado válido se a face foi detectada e o estado é válido
        frame_detected = (face_ok & is_valid_state).tolist()
        
        # Mapeia os rótulos para o esquema binário (Alert vs Not-Alert)
        raw_labels = df_sorted[STATE_COLUMN].tolist()
        labels = [BINARY_LABEL_MAP.get(lbl, lbl) for lbl in raw_labels]

        # Trata features ausentes preenchendo por propagação ou zeros
        feats_df = df_sorted[FEATURE_COLUMNS].ffill().bfill().fillna(0.0)
        features = feats_df.to_numpy(dtype=np.float32)

        windows = build_windows(
            frame_labels=labels,
            frame_detected=frame_detected,
            frame_features=features,
            task_id=str(task_id),
            window_len=window_len,
            stride=stride,
            min_valid_ratio=min_valid_ratio,
        )
        all_windows += windows
        if verbose:
            fps = estimate_fps(df_sorted) if len(df_sorted) > 1 else 0.0
            print(f"Vídeo '{task_id}': {len(df_sorted)} frames (~{fps:.1f} fps) -> {len(windows)} janelas")

    return all_windows


def print_class_window_counts(windows: List[Window]) -> None:
    from episode_sampler import index_by_task_class
    idx = index_by_task_class(windows)
    for task_id, class_map in idx.items():
        counts = {cls: len(ids) for cls, ids in class_map.items()}
        print(f"  Task {task_id}: {counts}")


def save_windows(windows: List[Window], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(windows, f)


def load_windows(path: str) -> List[Window]:
    with open(path, "rb") as f:
        return pickle.load(f)