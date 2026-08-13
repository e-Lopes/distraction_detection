# Baselines clássicos — janela de 60 frames

Resultados médios nos quatro vídeos de teste. Macro F1 usa as três classes; classes ausentes continuam sem suporte nos CSVs por classe.

| Modelo | Ablação | Macro F1 | Balanced accuracy |
|---|---|---:|---:|
| svm | facial_only | 0.4517 | 0.5685 |
| svm | facial_plus_missingness | 0.4472 | 0.5684 |
| svm | missingness_only | 0.4635 | 0.5853 |
| random_forest | facial_only | 0.4690 | 0.5903 |
| random_forest | facial_plus_missingness | 0.4697 | 0.5918 |
| random_forest | missingness_only | 0.4611 | 0.5834 |
| xgboost | facial_only | 0.4737 | 0.6134 |
| xgboost | facial_plus_missingness | 0.4728 | 0.6096 |
| xgboost | missingness_only | 0.4657 | 0.5719 |

Checkpoints e previsões individuais permanecem fora do Git em `outputs/models/classical_60/` e `outputs/predictions/classical_60/`.
