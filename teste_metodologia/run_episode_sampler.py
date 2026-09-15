"""
run_episode_sampler.py

Script único pra rodar o pipeline completo: CSVs -> janelas -> episódio de
exemplo. Ajuste as variáveis na seção CONFIGURAÇÃO abaixo e rode:

    python run_episode_sampler.py

Requer episode_sampler.py e load_own_data.py na mesma pasta.
"""

from pathlib import Path

from load_own_data import load_all_videos, print_class_window_counts, save_windows
from episode_sampler import index_by_task_class, EpisodeSampler

# ============================================================================
# CONFIGURAÇÃO — ajuste aqui
# ============================================================================

# Caminho dos 4 CSVs (um por vídeo). Pode ser caminho relativo ou absoluto.
CSV_PATHS = [
    "fase_2/data/interim/legacy_extraction/video_01.csv",
    "fase_2/data/interim/legacy_extraction/video_02.csv",
    "fase_2/data/interim/legacy_extraction/video_03.csv",
    "fase_2/data/interim/legacy_extraction/video_04.csv",
]

# Janelamento
WINDOW_LEN = 90          # tamanho da janela em frames (ex.: 5s a ~18fps -> ~90)
STRIDE = 45              # passo entre janelas; None = sem overlap (stride = WINDOW_LEN)
MIN_VALID_RATIO = 0.5    # fração mínima de frames com detecção válida por janela

# Esquema de classes
BINARY = True            # True = Alert vs Not-Alert | False = Alert/Fatigue/Distraction

# Episódio few-shot
N_WAY = None              # None = infere automaticamente (2 se BINARY=True, 3 se False)
K_SHOT = 5                # exemplos de suporte por classe
Q_QUERY = 10              # exemplos de consulta por classe
SEED = 42

# Task específica para amostrar o episódio de exemplo no final.
# None = sorteia entre as tasks disponíveis.
DEMO_TASK_ID = None

# Onde salvar as janelas processadas (pra não precisar reprocessar os CSVs
# toda vez). None = não salvar.
SAVE_WINDOWS_PATH = "teste_metodologia/results/windows.pkl"

# ============================================================================
# EXECUÇÃO — normalmente não precisa mexer daqui pra baixo
# ============================================================================


def main() -> None:
    missing = [p for p in CSV_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Os seguintes CSVs não foram encontrados (confira CSV_PATHS no "
            f"topo do script): {missing}"
        )

    print("=== Carregando CSVs e construindo janelas ===")
    windows = load_all_videos(
        CSV_PATHS,
        window_len=WINDOW_LEN,
        stride=STRIDE,
        min_valid_ratio=MIN_VALID_RATIO,
        binary=BINARY,
    )
    print(f"\nTotal de janelas: {len(windows)}")

    print("\n=== Janelas por vídeo/classe ===")
    print_class_window_counts(windows)

    if SAVE_WINDOWS_PATH:
        save_windows(windows, SAVE_WINDOWS_PATH)
        print(f"\nJanelas salvas em: {SAVE_WINDOWS_PATH}")

    print("\n=== Montando o EpisodeSampler ===")
    task_index = index_by_task_class(windows)
    sampler = EpisodeSampler(
        windows,
        task_index,
        n_way=N_WAY,
        k_shot=K_SHOT,
        q_query=Q_QUERY,
        seed=SEED,
    )
    print(f"n_way usado: {sampler.n_way}")
    print(f"Tasks utilizáveis (têm janelas suficientes por classe): {sampler.task_ids}")

    if not sampler.task_ids:
        return  # EpisodeSampler já teria levantado erro antes de chegar aqui

    print("\n=== Episódio de exemplo ===")
    episode = sampler.sample_episode(task_id=DEMO_TASK_ID)
    print(f"task_id: {episode.task_id}")
    print(f"class_order: {episode.class_order}")
    print(f"support_x: {tuple(episode.support_x.shape)}  support_y: {tuple(episode.support_y.shape)}")
    print(f"query_x:   {tuple(episode.query_x.shape)}  query_y:   {tuple(episode.query_y.shape)}")


if __name__ == "__main__":
    main()