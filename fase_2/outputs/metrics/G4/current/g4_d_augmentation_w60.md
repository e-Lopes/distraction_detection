# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.5894 ± 0.0000 | 0.5894 | 0.5894 | 0.5894 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4544 ± 0.0000 | 0.4544 | 0.4544 | 0.4544 | [nan, nan] |
| lstm | test | best_epoch | 12.2500 ± 0.0000 | 12.2500 | 12.2500 | 12.2500 | [nan, nan] |
| lstm | test | stopping_epoch | 27.2500 ± 0.0000 | 27.2500 | 27.2500 | 27.2500 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.4259 ± 0.0000 | 0.4259 | 0.4259 | 0.4259 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.4006 ± 0.0000 | 0.4006 | 0.4006 | 0.4006 | [nan, nan] |
| lstm | validation | best_epoch | 12.2500 ± 0.0000 | 12.2500 | 12.2500 | 12.2500 | [nan, nan] |
| lstm | validation | stopping_epoch | 27.2500 ± 0.0000 | 27.2500 | 27.2500 | 27.2500 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5996 ± 0.0000 | 0.5996 | 0.5996 | 0.5996 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4691 ± 0.0000 | 0.4691 | 0.4691 | 0.4691 | [nan, nan] |
| tcn | test | best_epoch | 3.5000 ± 0.0000 | 3.5000 | 3.5000 | 3.5000 | [nan, nan] |
| tcn | test | stopping_epoch | 18.5000 ± 0.0000 | 18.5000 | 18.5000 | 18.5000 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.4283 ± 0.0000 | 0.4283 | 0.4283 | 0.4283 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4113 ± 0.0000 | 0.4113 | 0.4113 | 0.4113 | [nan, nan] |
| tcn | validation | best_epoch | 3.5000 ± 0.0000 | 3.5000 | 3.5000 | 3.5000 | [nan, nan] |
| tcn | validation | stopping_epoch | 18.5000 ± 0.0000 | 18.5000 | 18.5000 | 18.5000 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
