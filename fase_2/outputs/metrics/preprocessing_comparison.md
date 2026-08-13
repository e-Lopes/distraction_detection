# Comparação de pré-processamento

Medianas são ajustadas somente no treino de cada fold. A interpolação ocorre dentro de cada bloco e nunca atravessa treino, validação, teste ou purge gaps.

| Modelo | Estratégia | Janela | Macro F1 | Balanced accuracy | Recall fadiga* |
|---|---|---:|---:|---:|---:|
| svm | zero_fill_baseline | 30 | 0.4502 | 0.5274 | 0.0378 |
| svm | zero_fill_baseline | 60 | 0.4472 | 0.5684 | 0.0299 |
| svm | zero_fill_baseline | 150 | 0.4521 | 0.5742 | 0.0000 |
| svm | short_gap_interpolation | 30 | 0.4538 | 0.5317 | 0.0280 |
| svm | short_gap_interpolation | 60 | 0.4487 | 0.5747 | 0.0075 |
| svm | short_gap_interpolation | 150 | 0.4547 | 0.5758 | 0.0000 |
| svm | short_gap_interpolation_with_flags | 30 | 0.4523 | 0.5254 | 0.0231 |
| svm | short_gap_interpolation_with_flags | 60 | 0.4625 | 0.5867 | 0.0462 |
| svm | short_gap_interpolation_with_flags | 150 | 0.4568 | 0.5760 | 0.0000 |
| random_forest | zero_fill_baseline | 30 | 0.4658 | 0.5399 | 0.0000 |
| random_forest | zero_fill_baseline | 60 | 0.4692 | 0.5893 | 0.0000 |
| random_forest | zero_fill_baseline | 150 | 0.4602 | 0.5727 | 0.0000 |
| random_forest | short_gap_interpolation | 30 | 0.4662 | 0.5405 | 0.0000 |
| random_forest | short_gap_interpolation | 60 | 0.4701 | 0.5934 | 0.0000 |
| random_forest | short_gap_interpolation | 150 | 0.4654 | 0.5780 | 0.0000 |
| random_forest | short_gap_interpolation_with_flags | 30 | 0.4372 | 0.4879 | 0.0000 |
| random_forest | short_gap_interpolation_with_flags | 60 | 0.4382 | 0.5409 | 0.0000 |
| random_forest | short_gap_interpolation_with_flags | 150 | 0.4289 | 0.5275 | 0.0000 |
| xgboost | zero_fill_baseline | 30 | 0.4615 | 0.5467 | 0.0000 |
| xgboost | zero_fill_baseline | 60 | 0.4740 | 0.6122 | 0.0000 |
| xgboost | zero_fill_baseline | 150 | 0.4720 | 0.5952 | 0.0000 |
| xgboost | short_gap_interpolation | 30 | 0.4592 | 0.5456 | 0.0000 |
| xgboost | short_gap_interpolation | 60 | 0.4580 | 0.5900 | 0.0000 |
| xgboost | short_gap_interpolation | 150 | 0.4778 | 0.6132 | 0.0000 |
| xgboost | short_gap_interpolation_with_flags | 30 | 0.4266 | 0.4836 | 0.0000 |
| xgboost | short_gap_interpolation_with_flags | 60 | 0.4347 | 0.5448 | 0.0000 |
| xgboost | short_gap_interpolation_with_flags | 150 | 0.4520 | 0.5770 | 0.0000 |

Observação: recall de fadiga usa apenas folds externos com suporte.

Estes resultados comparam representações usando hiperparâmetros fixos. Eles não selecionam a configuração final pelo conjunto de teste.
