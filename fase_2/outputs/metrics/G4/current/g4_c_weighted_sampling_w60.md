# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.5972 ± 0.0000 | 0.5972 | 0.5972 | 0.5972 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4590 ± 0.0000 | 0.4590 | 0.4590 | 0.4590 | [nan, nan] |
| lstm | test | best_epoch | 8.7500 ± 0.0000 | 8.7500 | 8.7500 | 8.7500 | [nan, nan] |
| lstm | test | stopping_epoch | 23.7500 ± 0.0000 | 23.7500 | 23.7500 | 23.7500 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.5216 ± 0.0000 | 0.5216 | 0.5216 | 0.5216 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.3906 ± 0.0000 | 0.3906 | 0.3906 | 0.3906 | [nan, nan] |
| lstm | validation | best_epoch | 8.7500 ± 0.0000 | 8.7500 | 8.7500 | 8.7500 | [nan, nan] |
| lstm | validation | stopping_epoch | 23.7500 ± 0.0000 | 23.7500 | 23.7500 | 23.7500 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5868 ± 0.0000 | 0.5868 | 0.5868 | 0.5868 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4578 ± 0.0000 | 0.4578 | 0.4578 | 0.4578 | [nan, nan] |
| tcn | test | best_epoch | 8.2500 ± 0.0000 | 8.2500 | 8.2500 | 8.2500 | [nan, nan] |
| tcn | test | stopping_epoch | 23.2500 ± 0.0000 | 23.2500 | 23.2500 | 23.2500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.5528 ± 0.0000 | 0.5528 | 0.5528 | 0.5528 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4035 ± 0.0000 | 0.4035 | 0.4035 | 0.4035 | [nan, nan] |
| tcn | validation | best_epoch | 8.2500 ± 0.0000 | 8.2500 | 8.2500 | 8.2500 | [nan, nan] |
| tcn | validation | stopping_epoch | 23.2500 ± 0.0000 | 23.2500 | 23.2500 | 23.2500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
