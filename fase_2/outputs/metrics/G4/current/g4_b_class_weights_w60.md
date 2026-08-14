# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.6004 ± 0.0000 | 0.6004 | 0.6004 | 0.6004 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4614 ± 0.0000 | 0.4614 | 0.4614 | 0.4614 | [nan, nan] |
| lstm | test | best_epoch | 11.5000 ± 0.0000 | 11.5000 | 11.5000 | 11.5000 | [nan, nan] |
| lstm | test | stopping_epoch | 26.5000 ± 0.0000 | 26.5000 | 26.5000 | 26.5000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.5248 ± 0.0000 | 0.5248 | 0.5248 | 0.5248 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.4004 ± 0.0000 | 0.4004 | 0.4004 | 0.4004 | [nan, nan] |
| lstm | validation | best_epoch | 11.5000 ± 0.0000 | 11.5000 | 11.5000 | 11.5000 | [nan, nan] |
| lstm | validation | stopping_epoch | 26.5000 ± 0.0000 | 26.5000 | 26.5000 | 26.5000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5929 ± 0.0000 | 0.5929 | 0.5929 | 0.5929 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4674 ± 0.0000 | 0.4674 | 0.4674 | 0.4674 | [nan, nan] |
| tcn | test | best_epoch | 4.2500 ± 0.0000 | 4.2500 | 4.2500 | 4.2500 | [nan, nan] |
| tcn | test | stopping_epoch | 19.2500 ± 0.0000 | 19.2500 | 19.2500 | 19.2500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.5347 ± 0.0000 | 0.5347 | 0.5347 | 0.5347 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4029 ± 0.0000 | 0.4029 | 0.4029 | 0.4029 | [nan, nan] |
| tcn | validation | best_epoch | 4.2500 ± 0.0000 | 4.2500 | 4.2500 | 4.2500 | [nan, nan] |
| tcn | validation | stopping_epoch | 19.2500 ± 0.0000 | 19.2500 | 19.2500 | 19.2500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
