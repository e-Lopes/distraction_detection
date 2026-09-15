"""
load_own_data.py

Converte os seus CSVs (um por vídeo, colunas:
video_id, frame_index, timestamp_seconds, ear, mar, pitch, yaw, roll,
face_detected, operational_state, legacy_heuristic_state)
para o formato Window usado por episode_sampler.py.

IMPORTANTE — assunção que estou fazendo e que precisa da sua confirmação:
  Usei `legacy_heuristic_state` como rótulo de classe (é a única coluna com
  alert/fatigue/distraction no seu CSV). O nome da coluna sugere que é uma
  heurística antiga, não necessariamente a anotação "de verdade" — se você
  tiver uma anotação mais confiável em outro lugar, troque LABEL_COLUMN.
  Reparei também que, nos frames com face_missing, legacy_heuristic_state
  aparece como "alert" (provavelmente o heurístico só mantém o último
  estado conhecido) — como essas janelas tendem a ser descartadas pelo
  filtro de min_valid_ratio, isso deve ter pouco impacto, mas vale
  confirmar.

Uso:
    from load_own_data import load_all_videos
    windows = load_all_videos(
        ["video_01.csv", "video_02.csv", "video_03.csv", "video_04.csv"],
        window_len=90,       # ex.: 5s a ~18fps (ver estimate_fps)
        min_valid_ratio=0.5,
        binary=True,
    )
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
    index_by_task_class,
    merge_labels,
)

# Colunas do seu CSV, na ordem que vira features da janela: EAR, MAR, Pitch, Yaw, Roll
CSV_FEATURE_COLUMNS: List[str] = ["ear", "mar", "pitch", "yaw", "roll"]
LABEL_COLUMN = "legacy_heuristic_state"
DETECTED_COLUMN = "face_detected"
TASK_COLUMN = "video_id"
FRAME_INDEX_COLUMN = "frame_index"
TIMESTAMP_COLUMN = "timestamp_seconds"

# rótulos do CSV (minúsculo) -> nomes usados no episode_sampler (capitalizado)
CSV_LABEL_MAP = {"alert": "Alert", "fatigue": "Fatigue", "distraction": "Distraction"}


def estimate_fps(df: pd.DataFrame) -> float:
    """Estima o fps a partir do timestamp, pra ajudar a escolher window_len."""
    diffs = df[TIMESTAMP_COLUMN].diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        raise ValueError("Não foi possível estimar fps (timestamps insuficientes).")
    return 1.0 / diffs.median()


def _reindex_frames_sem_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante uma linha por frame_index sem buracos. Se o CSV pular índices
    (frame descartado inteiramente, não só marcado como face_missing),
    preenche a lacuna como frame sem detecção, pra não bagunçar o
    janelamento posicional do build_windows.
    """
    full_range = pd.RangeIndex(
        df[FRAME_INDEX_COLUMN].min(), df[FRAME_INDEX_COLUMN].max() + 1
    )
    df = df.set_index(FRAME_INDEX_COLUMN).reindex(full_range)
    df.index.name = FRAME_INDEX_COLUMN

    n_missing = df[TASK_COLUMN].isna().sum()
    if n_missing:
        print(f"  [aviso] {n_missing} frame(s) ausente(s) no CSV — tratados como não detectados")

    df[TASK_COLUMN] = df[TASK_COLUMN].ffill().bfill()
    df[DETECTED_COLUMN] = df[DETECTED_COLUMN].fillna(0)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].ffill().bfill()
    return df.reset_index()


def load_video_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(FRAME_INDEX_COLUMN).reset_index(drop=True)
    df = _reindex_frames_sem_gaps(df)
    return df


def csv_to_windows(
    df: pd.DataFrame,
    window_len: int,
    stride: Optional[int] = None,
    min_valid_ratio: float = 0.5,
    binary: bool = True,
) -> List[Window]:
    task_id = str(df[TASK_COLUMN].iloc[0])

    detected = df[DETECTED_COLUMN].astype(bool).tolist()

    # frames sem detecção ficam com ear/mar/pitch/yaw/roll vazios (NaN);
    # preenche por propagação (ffill/bfill) em vez de zero, pra não injetar
    # um valor artificial na série — essas janelas tendem a ser descartadas
    # pelo min_valid_ratio de qualquer forma, mas o array precisa ser numérico
    feats_df = df[CSV_FEATURE_COLUMNS].ffill().bfill()
    if feats_df.isna().any().any():
        raise ValueError(
            f"Vídeo '{task_id}': ainda há NaN após ffill/bfill — provavelmente "
            "o vídeo inteiro não tem nenhuma detecção válida."
        )
    features = feats_df.to_numpy(dtype=np.float32)

    raw_labels = df[LABEL_COLUMN].str.lower().map(CSV_LABEL_MAP)
    if raw_labels.isna().any():
        bad = sorted(df[LABEL_COLUMN][raw_labels.isna()].unique())
        raise ValueError(f"Vídeo '{task_id}': rótulo(s) não reconhecido(s) em '{LABEL_COLUMN}': {bad}")
    labels = raw_labels.tolist()

    if binary:
        labels = merge_labels(labels, BINARY_LABEL_MAP)

    return build_windows(
        labels,
        detected,
        features,
        task_id=task_id,
        window_len=window_len,
        stride=stride,
        min_valid_ratio=min_valid_ratio,
    )


def load_all_videos(
    csv_paths: List[str],
    window_len: int,
    stride: Optional[int] = None,
    min_valid_ratio: float = 0.5,
    binary: bool = True,
    verbose: bool = True,
) -> List[Window]:
    all_windows: List[Window] = []
    for path in csv_paths:
        df = load_video_csv(path)
        fps = estimate_fps(df)
        windows = csv_to_windows(df, window_len, stride, min_valid_ratio, binary)
        all_windows += windows
        if verbose:
            print(
                f"{Path(path).name}: {len(df)} frames (~{fps:.1f} fps) "
                f"-> {len(windows)} janelas"
            )
    return all_windows


def print_class_window_counts(windows: List[Window]) -> None:
    """Diagnóstico: quantas janelas por classe existem em cada vídeo — útil
    pra saber se dá pra pedir k_shot+q_query no EpisodeSampler."""
    idx = index_by_task_class(windows)
    for task_id, class_map in idx.items():
        counts = {cls: len(ids) for cls, ids in class_map.items()}
        print(f"  {task_id}: {counts}")


def save_windows(windows: List[Window], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(windows, f)


def load_windows(path: str) -> List[Window]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Demonstração: procura CSVs em /mnt/user-data/uploads; se não achar, gera
# CSVs sintéticos no seu formato exato pra validar o conversor de ponta a
# ponta (incluindo o caso de frame_index com lacunas).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob

    real_csvs = sorted(glob.glob("/mnt/user-data/uploads/video_*.csv"))

    if real_csvs:
        print(f"Encontrados {len(real_csvs)} CSVs em /mnt/user-data/uploads:")
        csv_paths = real_csvs
    else:
        print("Nenhum CSV real encontrado — gerando CSVs sintéticos no seu formato para teste.\n")
        csv_paths = []
        rng = np.random.default_rng(0)
        tmp_dir = Path("/tmp/fake_own_data")
        tmp_dir.mkdir(exist_ok=True)

        for v in range(1, 3):
            n_frames = 2000
            fps = 18.0
            rows = []
            state = "alert"
            for i in range(n_frames):
                # bloco de face_missing simulado (operador ausente/falha de detecção)
                missing = 300 <= i < 340 or 900 <= i < 950

                if i == 500:
                    state = "fatigue"
                elif i == 900:
                    state = "distraction"
                elif i == 1300:
                    state = "alert"

                if missing:
                    ear = mar = pitch = yaw = roll = ""
                    face_detected = 0
                    op_state = "face_missing"
                else:
                    ear = round(float(rng.normal(0.25, 0.05)), 6)
                    mar = round(float(rng.normal(0.05, 0.02)), 6)
                    pitch = round(float(rng.normal(-20, 15)), 6)
                    yaw = round(float(rng.normal(0, 20)), 6)
                    roll = round(float(rng.normal(-150, 5)), 6)
                    face_detected = 1
                    op_state = "valid"

                rows.append(
                    {
                        "video_id": f"video_{v:02d}",
                        "frame_index": i,
                        "timestamp_seconds": i / fps,
                        "ear": ear,
                        "mar": mar,
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "face_detected": face_detected,
                        "operational_state": op_state,
                        "legacy_heuristic_state": state,
                    }
                )

            # simula uma lacuna real no frame_index (frame descartado do CSV)
            rows = [r for r in rows if not (1000 <= r["frame_index"] < 1005)]

            out_path = tmp_dir / f"video_{v:02d}.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            csv_paths.append(str(out_path))
        print(f"CSVs sintéticos gerados em {tmp_dir}\n")

    windows = load_all_videos(
        csv_paths, window_len=90, stride=45, min_valid_ratio=0.5, binary=True
    )
    print(f"\nTotal de janelas geradas: {len(windows)}")
    print("Janelas por vídeo/classe:")
    print_class_window_counts(windows)