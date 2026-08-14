# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| transformer | test | balanced_accuracy | 0.5940 ± 0.0000 | 0.5940 | 0.5940 | 0.5940 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4493 ± 0.0000 | 0.4493 | 0.4493 | 0.4493 | [nan, nan] |
| transformer | test | best_epoch | 10.2500 ± 0.0000 | 10.2500 | 10.2500 | 10.2500 | [nan, nan] |
| transformer | test | stopping_epoch | 25.2500 ± 0.0000 | 25.2500 | 25.2500 | 25.2500 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4531 ± 0.0000 | 0.4531 | 0.4531 | 0.4531 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.3878 ± 0.0000 | 0.3878 | 0.3878 | 0.3878 | [nan, nan] |
| transformer | validation | best_epoch | 10.2500 ± 0.0000 | 10.2500 | 10.2500 | 10.2500 | [nan, nan] |
| transformer | validation | stopping_epoch | 25.2500 ± 0.0000 | 25.2500 | 25.2500 | 25.2500 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
