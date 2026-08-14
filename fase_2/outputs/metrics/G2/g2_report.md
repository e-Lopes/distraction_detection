# G2 — modelos temporais em R0

Matriz oficial: 36 execucoes (3 modelos x 3 janelas x 4 folds), seed 42, distribuicao original e sem balanceamento.

| Modelo | Janela | Macro F1 de validacao (media +/- DP) | F1 Fatigue (media +/- DP) |
|---|---:|---:|---:|
| TCN | 60 | 0.4125 +/- 0.0211 | 0.0000 +/- 0.0000 |
| TRANSFORMER | 150 | 0.4081 +/- 0.0136 | 0.0000 +/- 0.0000 |
| TCN | 30 | 0.4065 +/- 0.0150 | 0.0000 +/- 0.0000 |
| TRANSFORMER | 60 | 0.4065 +/- 0.0146 | 0.0000 +/- 0.0000 |
| TRANSFORMER | 30 | 0.4062 +/- 0.0179 | 0.0000 +/- 0.0000 |
| LSTM | 60 | 0.4044 +/- 0.0169 | 0.0000 +/- 0.0000 |
| TCN | 150 | 0.4032 +/- 0.0260 | 0.0000 +/- 0.0000 |
| LSTM | 30 | 0.4027 +/- 0.0147 | 0.0000 +/- 0.0000 |
| LSTM | 150 | 0.3946 +/- 0.0106 | 0.0000 +/- 0.0000 |

Os desvios acima medem variacao entre folds/sessoes, nao estabilidade entre seeds. A G2 de qualificacao nao confirma H3 nem autoriza selecao final: todas as configuracoes ainda foram avaliadas com uma unica seed.

Matrizes de confusao, metricas por classe, historicos por epoca, distribuicoes de previsao, configuracoes resolvidas e hashes dos checkpoints acompanham este relatorio.
