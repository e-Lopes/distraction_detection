# Dummy baseline

Baseline `most_frequent` ajustado separadamente em cada fold. Em todos os casos, a classe majoritária de treino foi `alert`.

| Janela | Acurácia média | Balanced accuracy média | Macro F1 (3 classes) |
|---:|---:|---:|---:|
| 30 | 0.8316 | 0.3750 | 0.3021 |
| 60 | 0.8343 | 0.4167 | 0.3026 |
| 150 | 0.8385 | 0.4167 | 0.3034 |

A acurácia elevada reflete o desbalanceamento e não desempenho útil nas classes minoritárias. Os folds 2 e 4 não possuem fadiga no teste para janelas de 60 e 150 frames; métricas por classe sem suporte ficam vazias no CSV correspondente.
