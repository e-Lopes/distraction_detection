# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| transformer | test | balanced_accuracy | 0.5914 ± 0.0000 | 0.5914 | 0.5914 | 0.5914 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4633 ± 0.0000 | 0.4633 | 0.4633 | 0.4633 | [nan, nan] |
| transformer | test | best_epoch | 5.2500 ± 0.0000 | 5.2500 | 5.2500 | 5.2500 | [nan, nan] |
| transformer | test | stopping_epoch | 20.2500 ± 0.0000 | 20.2500 | 20.2500 | 20.2500 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4454 ± 0.0000 | 0.4454 | 0.4454 | 0.4454 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.4139 ± 0.0000 | 0.4139 | 0.4139 | 0.4139 | [nan, nan] |
| transformer | validation | best_epoch | 5.2500 ± 0.0000 | 5.2500 | 5.2500 | 5.2500 | [nan, nan] |
| transformer | validation | stopping_epoch | 20.2500 ± 0.0000 | 20.2500 | 20.2500 | 20.2500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
