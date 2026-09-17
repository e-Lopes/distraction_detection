"""
run_episode_sampler.py

Processa o CSV unificado com as features reais e gera as janelas binárias.
"""

from pathlib import Path
from load_own_data import load_unified_csv_to_windows, print_class_window_counts, save_windows

TEST_DIR = Path(__file__).resolve().parent
UNIFIED_CSV_PATH = TEST_DIR / "results/classificacoes_com_mediapipe.csv"
WINDOW_LEN = 90
STRIDE = 15
MIN_VALID_RATIO = 0.5
N_WAY = 2
K_SHOT = 3
Q_QUERY = 5
SEED = 42
DEMO_TASK_ID = None
SAVE_WINDOWS_PATH = TEST_DIR / "results/windows_binary.pkl"

def main() -> None:
    csv_path = Path(UNIFIED_CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"O CSV unificado não foi encontrado: {csv_path.absolute()}")

    print("=== Processando CSV Unificado e Construindo Janelas ===")
    windows = load_unified_csv_to_windows(
        csv_path=str(csv_path),
        window_len=WINDOW_LEN,
        stride=STRIDE,
        min_valid_ratio=MIN_VALID_RATIO,
        verbose=True,
    )
    print(f"\nTotal geral de janelas válidas: {len(windows)}")
    print_class_window_counts(windows)

    if SAVE_WINDOWS_PATH:
        Path(SAVE_WINDOWS_PATH).parent.mkdir(parents=True, exist_ok=True)
        save_windows(windows, SAVE_WINDOWS_PATH)
        print(f"Janelas salvas em: {SAVE_WINDOWS_PATH}")

if __name__ == "__main__":
    main()
