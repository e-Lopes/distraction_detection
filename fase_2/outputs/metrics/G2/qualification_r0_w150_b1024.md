# Modelos temporais — estabilidade entre seeds

Seeds independentes: `[42]`. Cada seed foi executada nos folds `[1, 2, 3, 4]`, com early stopping independente. A unidade da estatística abaixo é a média dos folds de cada seed (`n=1`), não as janelas individuais.

| Modelo | Subset | Métrica | Média ± DP | Mediana | Mínimo | Máximo | IC 95% |
|---|---|---|---:|---:|---:|---:|---:|
| lstm | test | balanced_accuracy | 0.5660 ± 0.0000 | 0.5660 | 0.5660 | 0.5660 | [nan, nan] |
| lstm | test | macro_f1_all_classes | 0.4422 ± 0.0000 | 0.4422 | 0.4422 | 0.4422 | [nan, nan] |
| lstm | test | best_epoch | 11.5000 ± 0.0000 | 11.5000 | 11.5000 | 11.5000 | [nan, nan] |
| lstm | test | stopping_epoch | 26.5000 ± 0.0000 | 26.5000 | 26.5000 | 26.5000 | [nan, nan] |
| lstm | validation | balanced_accuracy | 0.4081 ± 0.0000 | 0.4081 | 0.4081 | 0.4081 | [nan, nan] |
| lstm | validation | macro_f1_all_classes | 0.3946 ± 0.0000 | 0.3946 | 0.3946 | 0.3946 | [nan, nan] |
| lstm | validation | best_epoch | 11.5000 ± 0.0000 | 11.5000 | 11.5000 | 11.5000 | [nan, nan] |
| lstm | validation | stopping_epoch | 26.5000 ± 0.0000 | 26.5000 | 26.5000 | 26.5000 | [nan, nan] |
| tcn | test | balanced_accuracy | 0.5795 ± 0.0000 | 0.5795 | 0.5795 | 0.5795 | [nan, nan] |
| tcn | test | macro_f1_all_classes | 0.4696 ± 0.0000 | 0.4696 | 0.4696 | 0.4696 | [nan, nan] |
| tcn | test | best_epoch | 5.0000 ± 0.0000 | 5.0000 | 5.0000 | 5.0000 | [nan, nan] |
| tcn | test | stopping_epoch | 20.0000 ± 0.0000 | 20.0000 | 20.0000 | 20.0000 | [nan, nan] |
| tcn | validation | balanced_accuracy | 0.4077 ± 0.0000 | 0.4077 | 0.4077 | 0.4077 | [nan, nan] |
| tcn | validation | macro_f1_all_classes | 0.4032 ± 0.0000 | 0.4032 | 0.4032 | 0.4032 | [nan, nan] |
| tcn | validation | best_epoch | 5.0000 ± 0.0000 | 5.0000 | 5.0000 | 5.0000 | [nan, nan] |
| tcn | validation | stopping_epoch | 20.0000 ± 0.0000 | 20.0000 | 20.0000 | 20.0000 | [nan, nan] |
| transformer | test | balanced_accuracy | 0.5773 ± 0.0000 | 0.5773 | 0.5773 | 0.5773 | [nan, nan] |
| transformer | test | macro_f1_all_classes | 0.4639 ± 0.0000 | 0.4639 | 0.4639 | 0.4639 | [nan, nan] |
| transformer | test | best_epoch | 6.5000 ± 0.0000 | 6.5000 | 6.5000 | 6.5000 | [nan, nan] |
| transformer | test | stopping_epoch | 21.5000 ± 0.0000 | 21.5000 | 21.5000 | 21.5000 | [nan, nan] |
| transformer | validation | balanced_accuracy | 0.4232 ± 0.0000 | 0.4232 | 0.4232 | 0.4232 | [nan, nan] |
| transformer | validation | macro_f1_all_classes | 0.4081 ± 0.0000 | 0.4081 | 0.4081 | 0.4081 | [nan, nan] |
| transformer | validation | best_epoch | 6.5000 ± 0.0000 | 6.5000 | 6.5000 | 6.5000 | [nan, nan] |
| transformer | validation | stopping_epoch | 21.5000 ± 0.0000 | 21.5000 | 21.5000 | 21.5000 | [nan, nan] |

O desempenho final é sempre apresentado como média ± desvio padrão entre seeds. Nenhuma configuração ou conclusão usa apenas a melhor seed observada.
