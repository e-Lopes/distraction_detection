# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.4094 ± 0.0000 | 0.4094 | 0.4094 | 0.4094 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.3106 ± 0.0000 | 0.3106 | 0.3106 | 0.3106 | [nan, nan] |
| lstm | test | best_epoch | 2.0000 ± 0.0000 | 2.0000 | 2.0000 | 2.0000 | [nan, nan] |
| lstm | test | stopping_epoch | 17.0000 ± 0.0000 | 17.0000 | 17.0000 | 17.0000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.3323 ± 0.0000 | 0.3323 | 0.3323 | 0.3323 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.3163 ± 0.0000 | 0.3163 | 0.3163 | 0.3163 | [nan, nan] |
| lstm | validation | best_epoch | 2.0000 ± 0.0000 | 2.0000 | 2.0000 | 2.0000 | [nan, nan] |
| lstm | validation | stopping_epoch | 17.0000 ± 0.0000 | 17.0000 | 17.0000 | 17.0000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.4154 ± 0.0000 | 0.4154 | 0.4154 | 0.4154 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.2929 ± 0.0000 | 0.2929 | 0.2929 | 0.2929 | [nan, nan] |
| tcn | test | best_epoch | 1.7500 ± 0.0000 | 1.7500 | 1.7500 | 1.7500 | [nan, nan] |
| tcn | test | stopping_epoch | 16.7500 ± 0.0000 | 16.7500 | 16.7500 | 16.7500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.3407 ± 0.0000 | 0.3407 | 0.3407 | 0.3407 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.3167 ± 0.0000 | 0.3167 | 0.3167 | 0.3167 | [nan, nan] |
| tcn | validation | best_epoch | 1.7500 ± 0.0000 | 1.7500 | 1.7500 | 1.7500 | [nan, nan] |
| tcn | validation | stopping_epoch | 16.7500 ± 0.0000 | 16.7500 | 16.7500 | 16.7500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
