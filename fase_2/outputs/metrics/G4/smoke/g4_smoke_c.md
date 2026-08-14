# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| tcn | test | balanced_accuracy | 0.3333 ± 0.0000 | 0.3333 | 0.3333 | 0.3333 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.1647 ± 0.0000 | 0.1647 | 0.1647 | 0.1647 | [nan, nan] |
| tcn | test | best_epoch | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | [nan, nan] |
| tcn | test | stopping_epoch | 2.0000 ± 0.0000 | 2.0000 | 2.0000 | 2.0000 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.3333 ± 0.0000 | 0.3333 | 0.3333 | 0.3333 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.1818 ± 0.0000 | 0.1818 | 0.1818 | 0.1818 | [nan, nan] |
| tcn | validation | best_epoch | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | [nan, nan] |
| tcn | validation | stopping_epoch | 2.0000 ± 0.0000 | 2.0000 | 2.0000 | 2.0000 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
