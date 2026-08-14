# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| transformer | test | balanced_accuracy | 0.5920 ± 0.0000 | 0.5920 | 0.5920 | 0.5920 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4503 ± 0.0000 | 0.4503 | 0.4503 | 0.4503 | [nan, nan] |
| transformer | test | best_epoch | 8.2500 ± 0.0000 | 8.2500 | 8.2500 | 8.2500 | [nan, nan] |
| transformer | test | stopping_epoch | 23.2500 ± 0.0000 | 23.2500 | 23.2500 | 23.2500 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4644 ± 0.0000 | 0.4644 | 0.4644 | 0.4644 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.3923 ± 0.0000 | 0.3923 | 0.3923 | 0.3923 | [nan, nan] |
| transformer | validation | best_epoch | 8.2500 ± 0.0000 | 8.2500 | 8.2500 | 8.2500 | [nan, nan] |
| transformer | validation | stopping_epoch | 23.2500 ± 0.0000 | 23.2500 | 23.2500 | 23.2500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
