# Cenário binário: atenção e não atenção

> Atenção = `alert`. Não atenção = `fatigue` + `distraction`.

Esta é uma reavaliação das previsões dos modelos de três classes; não houve novo treinamento.

| Posição | Modelo | Janela | F1 geral | Melhor fold | Pior fold | Acurácia balanceada | Precisão: não atenção | Recall: não atenção | F1: não atenção |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **mantis_frozen_ridge** | 60 | 0.560 | 1 (0.607) | 2 (0.532) | 0.600 | 0.208 | 0.406 | 0.271 |
| 2 | **SVM** | 60 | 0.537 | 1 (0.587) | 2 (0.484) | 0.567 | 0.183 | 0.352 | 0.235 |
| 3 | **hydra_multirocket_ridge** | 60 | 0.507 | 2 (0.582) | 3 (0.407) | 0.575 | 0.171 | 0.480 | 0.251 |
| 4 | **multirocket_ridge** | 60 | 0.487 | 1 (0.590) | 3 (0.389) | 0.592 | 0.183 | 0.571 | 0.261 |

## Interpretação

A precisão baixa de não atenção indica muitos falsos alarmes. O recall mostra quanto da não atenção real foi encontrado. Um treinamento diretamente binário pode produzir resultados diferentes e deve ser validado antes de qualquer conclusão final.
