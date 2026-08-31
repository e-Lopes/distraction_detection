from fase_2.src.training.temporal_multiseed import (
    plot_fold_seed_heatmap,
    plot_metric_variability,
)


def test_plots_accept_validation_only_generation(tmp_path):
    seed_rows = [
        {
            "model": "lstm",
            "subset": "validation",
            "seed": 42,
            "macro_f1_all_classes": 0.4,
        }
    ]
    summary_rows = [
        {
            "model": "lstm",
            "subset": "validation",
            "seed": 42,
            "fold": fold,
            "macro_f1_all_classes": value,
        }
        for fold, value in ((1, 0.35), (2, 0.45))
    ]
    variability = tmp_path / "variability.png"
    heatmap = tmp_path / "heatmap.png"
    plot_metric_variability(seed_rows, variability)
    plot_fold_seed_heatmap(summary_rows, heatmap)
    assert variability.exists() and variability.with_suffix(".svg").exists()
    assert heatmap.exists() and heatmap.with_suffix(".svg").exists()
