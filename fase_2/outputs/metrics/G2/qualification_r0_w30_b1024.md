# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.5382 ± 0.0000 | 0.5382 | 0.5382 | 0.5382 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4513 ± 0.0000 | 0.4513 | 0.4513 | 0.4513 | [nan, nan] |
| lstm | test | best_epoch | 12.0000 ± 0.0000 | 12.0000 | 12.0000 | 12.0000 | [nan, nan] |
| lstm | test | stopping_epoch | 27.0000 ± 0.0000 | 27.0000 | 27.0000 | 27.0000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.4228 ± 0.0000 | 0.4228 | 0.4228 | 0.4228 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.4027 ± 0.0000 | 0.4027 | 0.4027 | 0.4027 | [nan, nan] |
| lstm | validation | best_epoch | 12.0000 ± 0.0000 | 12.0000 | 12.0000 | 12.0000 | [nan, nan] |
| lstm | validation | stopping_epoch | 27.0000 ± 0.0000 | 27.0000 | 27.0000 | 27.0000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5431 ± 0.0000 | 0.5431 | 0.5431 | 0.5431 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4697 ± 0.0000 | 0.4697 | 0.4697 | 0.4697 | [nan, nan] |
| tcn | test | best_epoch | 5.7500 ± 0.0000 | 5.7500 | 5.7500 | 5.7500 | [nan, nan] |
| tcn | test | stopping_epoch | 20.7500 ± 0.0000 | 20.7500 | 20.7500 | 20.7500 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.4210 ± 0.0000 | 0.4210 | 0.4210 | 0.4210 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4065 ± 0.0000 | 0.4065 | 0.4065 | 0.4065 | [nan, nan] |
| tcn | validation | best_epoch | 5.7500 ± 0.0000 | 5.7500 | 5.7500 | 5.7500 | [nan, nan] |
| tcn | validation | stopping_epoch | 20.7500 ± 0.0000 | 20.7500 | 20.7500 | 20.7500 | [nan, nan] |
| transformer | test | balanced_accuracy | 0.5331 ± 0.0000 | 0.5331 | 0.5331 | 0.5331 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4550 ± 0.0000 | 0.4550 | 0.4550 | 0.4550 | [nan, nan] |
| transformer | test | best_epoch | 12.7500 ± 0.0000 | 12.7500 | 12.7500 | 12.7500 | [nan, nan] |
| transformer | test | stopping_epoch | 27.7500 ± 0.0000 | 27.7500 | 27.7500 | 27.7500 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4218 ± 0.0000 | 0.4218 | 0.4218 | 0.4218 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.4062 ± 0.0000 | 0.4062 | 0.4062 | 0.4062 | [nan, nan] |
| transformer | validation | best_epoch | 12.7500 ± 0.0000 | 12.7500 | 12.7500 | 12.7500 | [nan, nan] |
| transformer | validation | stopping_epoch | 27.7500 ± 0.0000 | 27.7500 | 27.7500 | 27.7500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
