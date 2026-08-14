# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.5869 ± 0.0000 | 0.5869 | 0.5869 | 0.5869 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4535 ± 0.0000 | 0.4535 | 0.4535 | 0.4535 | [nan, nan] |
| lstm | test | best_epoch | 12.0000 ± 0.0000 | 12.0000 | 12.0000 | 12.0000 | [nan, nan] |
| lstm | test | stopping_epoch | 27.0000 ± 0.0000 | 27.0000 | 27.0000 | 27.0000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.4240 ± 0.0000 | 0.4240 | 0.4240 | 0.4240 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.4044 ± 0.0000 | 0.4044 | 0.4044 | 0.4044 | [nan, nan] |
| lstm | validation | best_epoch | 12.0000 ± 0.0000 | 12.0000 | 12.0000 | 12.0000 | [nan, nan] |
| lstm | validation | stopping_epoch | 27.0000 ± 0.0000 | 27.0000 | 27.0000 | 27.0000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5899 ± 0.0000 | 0.5899 | 0.5899 | 0.5899 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4716 ± 0.0000 | 0.4716 | 0.4716 | 0.4716 | [nan, nan] |
| tcn | test | best_epoch | 6.0000 ± 0.0000 | 6.0000 | 6.0000 | 6.0000 | [nan, nan] |
| tcn | test | stopping_epoch | 21.0000 ± 0.0000 | 21.0000 | 21.0000 | 21.0000 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.4255 ± 0.0000 | 0.4255 | 0.4255 | 0.4255 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4125 ± 0.0000 | 0.4125 | 0.4125 | 0.4125 | [nan, nan] |
| tcn | validation | best_epoch | 6.0000 ± 0.0000 | 6.0000 | 6.0000 | 6.0000 | [nan, nan] |
| tcn | validation | stopping_epoch | 21.0000 ± 0.0000 | 21.0000 | 21.0000 | 21.0000 | [nan, nan] |
| transformer | test | balanced_accuracy | 0.6002 ± 0.0000 | 0.6002 | 0.6002 | 0.6002 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4828 ± 0.0000 | 0.4828 | 0.4828 | 0.4828 | [nan, nan] |
| transformer | test | best_epoch | 7.5000 ± 0.0000 | 7.5000 | 7.5000 | 7.5000 | [nan, nan] |
| transformer | test | stopping_epoch | 22.5000 ± 0.0000 | 22.5000 | 22.5000 | 22.5000 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4225 ± 0.0000 | 0.4225 | 0.4225 | 0.4225 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.4065 ± 0.0000 | 0.4065 | 0.4065 | 0.4065 | [nan, nan] |
| transformer | validation | best_epoch | 7.5000 ± 0.0000 | 7.5000 | 7.5000 | 7.5000 | [nan, nan] |
| transformer | validation | stopping_epoch | 22.5000 ± 0.0000 | 22.5000 | 22.5000 | 22.5000 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
