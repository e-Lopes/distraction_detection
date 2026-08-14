# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.4326 ± 0.0000 | 0.4326 | 0.4326 | 0.4326 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.3170 ± 0.0000 | 0.3170 | 0.3170 | 0.3170 | [nan, nan] |
| lstm | test | best_epoch | 5.0000 ± 0.0000 | 5.0000 | 5.0000 | 5.0000 | [nan, nan] |
| lstm | test | stopping_epoch | 20.0000 ± 0.0000 | 20.0000 | 20.0000 | 20.0000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.3561 ± 0.0000 | 0.3561 | 0.3561 | 0.3561 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.3343 ± 0.0000 | 0.3343 | 0.3343 | 0.3343 | [nan, nan] |
| lstm | validation | best_epoch | 5.0000 ± 0.0000 | 5.0000 | 5.0000 | 5.0000 | [nan, nan] |
| lstm | validation | stopping_epoch | 20.0000 ± 0.0000 | 20.0000 | 20.0000 | 20.0000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.4684 ± 0.0000 | 0.4684 | 0.4684 | 0.4684 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.3512 ± 0.0000 | 0.3512 | 0.3512 | 0.3512 | [nan, nan] |
| tcn | test | best_epoch | 8.7500 ± 0.0000 | 8.7500 | 8.7500 | 8.7500 | [nan, nan] |
| tcn | test | stopping_epoch | 23.7500 ± 0.0000 | 23.7500 | 23.7500 | 23.7500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.3831 ± 0.0000 | 0.3831 | 0.3831 | 0.3831 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.3617 ± 0.0000 | 0.3617 | 0.3617 | 0.3617 | [nan, nan] |
| tcn | validation | best_epoch | 8.7500 ± 0.0000 | 8.7500 | 8.7500 | 8.7500 | [nan, nan] |
| tcn | validation | stopping_epoch | 23.7500 ± 0.0000 | 23.7500 | 23.7500 | 23.7500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
