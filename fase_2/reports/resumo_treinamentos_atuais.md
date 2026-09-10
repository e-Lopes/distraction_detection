# Resumo dos treinamentos realizados

> Atualizado em 10/09/2026 às 16:48. 
> **32 treinamentos de comparação concluídos e salvos.**

## Desempenho na validação

A tabela usa somente a validação interna, que é a parte adequada para comparar e escolher modelos.

| Posição | Modelo | Entrada | Janela | Folds | F1 geral | Variação | Acurácia balanceada | F1 de fadiga | Tempo total |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Random Forest** | temporal_behavior_v1 | 60 | 4/4 | 0.406 | ± 0.020 | 0.406 | 0.000 | 36.0 s |
| 2 | **XGBoost** | temporal_behavior_v1 | 60 | 4/4 | 0.405 | ± 0.017 | 0.419 | 0.000 | 40.1 s |
| 3 | **MiniRocket + Ridge** | R0 | 60 | 4/4 | 0.397 | ± 0.013 | 0.422 | 0.000 | 16.4 s |
| 4 | **MiniRocket + Ridge** | R0 | 30 | 4/4 | 0.384 | ± 0.019 | 0.417 | 0.014 | 24.2 s |
| 5 | **Shapelet + Ridge** | R0 | 60 | 4/4 | 0.383 | ± 0.019 | 0.474 | 0.008 | 76.3 min |
| 6 | **Regressão logística** | temporal_behavior_v1 | 60 | 4/4 | 0.373 | ± 0.033 | 0.436 | 0.025 | 1.8 min |
| 7 | **MiniRocket + Ridge** | R0 | 150 | 4/4 | 0.371 | ± 0.011 | 0.379 | 0.000 | 26.8 s |
| 8 | **SVM** | temporal_behavior_v1 | 60 | 4/4 | 0.370 | ± 0.017 | 0.397 | 0.000 | 43.9 s |

## Leitura rápida

- Melhor F1 geral: **Random Forest**, janela de 60 imagens (0.406).
- F1 geral considera igualmente alerta, fadiga e distração. Quanto mais perto de 1, melhor.
- Acurácia balanceada reduz o efeito da grande diferença de quantidade entre as classes.
- F1 de fadiga deve ser analisado separadamente; um valor próximo de zero indica que o comportamento quase não foi reconhecido.
- Os resultados do vídeo externo permanecem guardados, mas não foram usados para ordenar esta tabela.

## Situação das etapas

| Etapa | Quantidade | Situação | Próxima ação |
|---|---:|---|---|
| Comparação dos modelos | 32 | **Concluída e salva** | Revisar esta tabela e escolher candidatos |
| KNN-DTW | 4 | Bloqueado pelo custo | Executar somente após aceitar o custo elevado |
| Confirmação SVM/LSTM | 24 | Aguardando escolha | Criar `screening_promotion.yaml` |

## Observação

A posição representa apenas o resultado médio da validação atual. Ela não significa que o primeiro colocado já esteja pronto para uso real. A escolha final também precisa considerar fadiga, falsos alarmes, estabilidade entre vídeos e custo.
