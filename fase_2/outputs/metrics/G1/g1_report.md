# G1 - regras fixas e classicos sobre R0 achatado

Esta geracao usa distribuicao original (`balancing: none`). Os parametros e thresholds sao congelados antes da avaliacao; o teste externo nao seleciona modelos ou janelas.

| Modelo | Janela | Subset | Macro F1 medio | Balanced accuracy media |
|---|---:|---|---:|---:|
| fixed_rules | 30 | validation | 0.1597 | 0.2648 |
| fixed_rules | 30 | test | 0.1179 | 0.1992 |
| fixed_rules | 60 | validation | 0.0904 | 0.2364 |
| fixed_rules | 60 | test | 0.0784 | 0.2316 |
| fixed_rules | 150 | validation | 0.0715 | 0.2134 |
| fixed_rules | 150 | test | 0.0668 | 0.2361 |
| random_forest | 30 | validation | 0.3984 | 0.4136 |
| random_forest | 30 | test | 0.4631 | 0.5333 |
| random_forest | 60 | validation | 0.4022 | 0.4153 |
| random_forest | 60 | test | 0.4645 | 0.5814 |
| random_forest | 150 | validation | 0.3983 | 0.4070 |
| random_forest | 150 | test | 0.4558 | 0.5642 |
| svm | 30 | validation | 0.3782 | 0.4024 |
| svm | 30 | test | 0.3995 | 0.4883 |
| svm | 60 | validation | 0.4191 | 0.4544 |
| svm | 60 | test | 0.4547 | 0.5794 |
| svm | 150 | validation | 0.3576 | 0.3702 |
| svm | 150 | test | 0.4159 | 0.5451 |
| xgboost | 30 | validation | 0.4000 | 0.4154 |
| xgboost | 30 | test | 0.4620 | 0.5318 |
| xgboost | 60 | validation | 0.4024 | 0.4149 |
| xgboost | 60 | test | 0.4678 | 0.5852 |
| xgboost | 150 | validation | 0.4017 | 0.4096 |
| xgboost | 150 | test | 0.4642 | 0.5767 |

Resultados de teste sao descritivos. Qualquer escolha para G2 deve usar apenas treino e validacao.
