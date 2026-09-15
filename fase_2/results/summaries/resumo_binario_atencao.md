# Cenário binário: atenção e não atenção

> Atenção = `alert`. Não atenção = `fatigue` + `distraction`.

Esta é uma reavaliação das previsões dos modelos de três classes; não houve novo treinamento.

| Posição | Modelo | Janela | F1 geral | Melhor fold | Pior fold | Acurácia balanceada | Precisão: não atenção | Recall: não atenção | F1: não atenção |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Random Forest** | 60 | 0.601 | 4 (0.635) | 3 (0.573) | 0.592 | 0.335 | 0.248 | 0.281 |
| 2 | **XGBoost** | 60 | 0.600 | 1 (0.637) | 3 (0.575) | 0.608 | 0.291 | 0.319 | 0.297 |
| 3 | **MiniRocket + Ridge** | 60 | 0.588 | 1 (0.615) | 3 (0.571) | 0.612 | 0.251 | 0.367 | 0.294 |
| 4 | **Shapelet + Ridge** | 60 | 0.566 | 3 (0.589) | 1 (0.542) | 0.630 | 0.218 | 0.499 | 0.300 |
| 5 | **MiniRocket + Ridge** | 30 | 0.565 | 1 (0.586) | 2 (0.537) | 0.601 | 0.214 | 0.400 | 0.277 |
| 6 | **MiniRocket + Ridge** | 150 | 0.562 | 3 (0.574) | 4 (0.530) | 0.571 | 0.224 | 0.253 | 0.227 |
| 7 | **SVM** | 60 | 0.551 | 1 (0.582) | 3 (0.527) | 0.587 | 0.194 | 0.376 | 0.254 |
| 8 | **Regressão logística** | 60 | 0.540 | 1 (0.570) | 2 (0.494) | 0.582 | 0.184 | 0.396 | 0.249 |

## Interpretação

A precisão baixa de não atenção indica muitos falsos alarmes. O recall mostra quanto da não atenção real foi encontrado. Um treinamento diretamente binário pode produzir resultados diferentes e deve ser validado antes de qualquer conclusão final.
