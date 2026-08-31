# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | validation | balanced_accuracy | 0.5259 ± 0.0000 | 0.5259 | 0.5259 | 0.5259 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.4044 ± 0.0000 | 0.4044 | 0.4044 | 0.4044 | [nan, nan] |
| lstm | validation | best_epoch | 8.7500 ± 0.0000 | 8.7500 | 8.7500 | 8.7500 | [nan, nan] |
| lstm | validation | stopping_epoch | 23.7500 ± 0.0000 | 23.7500 | 23.7500 | 23.7500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.5204 ± 0.0000 | 0.5204 | 0.5204 | 0.5204 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4035 ± 0.0000 | 0.4035 | 0.4035 | 0.4035 | [nan, nan] |
| tcn | validation | best_epoch | 5.7500 ± 0.0000 | 5.7500 | 5.7500 | 5.7500 | [nan, nan] |
| tcn | validation | stopping_epoch | 20.7500 ± 0.0000 | 20.7500 | 20.7500 | 20.7500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
