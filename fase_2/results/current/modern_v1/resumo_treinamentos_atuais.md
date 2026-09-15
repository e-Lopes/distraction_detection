# Resumo dos treinamentos realizados

> Atualizado em 15/09/2026 às 14:48. 
> **32 treinamentos de comparação concluídos e salvos.**

## Desempenho na validação

A tabela usa somente a validação interna, que é a parte adequada para comparar e escolher modelos.

| Posição | Modelo | Entrada | Janela | Folds | F1 geral | Variação | Acurácia balanceada | F1 de fadiga | Tempo total |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **mantis_frozen_ridge** | R0 | 60 | 4/4 | 0.369 | ± 0.025 | 0.400 | 0.000 | 28.2 s |
| 2 | **SVM** | R0_flat | 60 | 4/4 | 0.359 | ± 0.033 | 0.382 | 0.000 | 12.8 s |
| 3 | **hydra_multirocket_ridge** | R0 | 60 | 4/4 | 0.336 | ± 0.046 | 0.378 | 0.003 | 15.9 s |
| 4 | **multirocket_ridge** | R0 | 60 | 4/4 | 0.332 | ± 0.059 | 0.412 | 0.022 | 27.8 s |

## Leitura rápida

- Melhor F1 geral: **mantis_frozen_ridge**, janela de 60 imagens (0.369).
- F1 geral considera igualmente alerta, fadiga e distração. Quanto mais perto de 1, melhor.
- Acurácia balanceada reduz o efeito da grande diferença de quantidade entre as classes.
- F1 de fadiga deve ser analisado separadamente; um valor próximo de zero indica que o comportamento quase não foi reconhecido.
- Os resultados do vídeo externo permanecem guardados, mas não foram usados para ordenar esta tabela.

## Situação das etapas

| Etapa | Quantidade | Situação | Próxima ação |
|---|---:|---|---|
| Comparação dos modelos | 32 | **Concluída e salva** | Revisar esta tabela e escolher candidatos |
| KNN-DTW | 0 | Bloqueado pelo custo | Executar somente após aceitar o custo elevado |
| Confirmação SVM/LSTM | 24 | Aguardando escolha | Criar `screening_promotion.yaml` |

## Observação

A posição representa apenas o resultado médio da validação atual. Ela não significa que o primeiro colocado já esteja pronto para uso real. A escolha final também precisa considerar fadiga, falsos alarmes, estabilidade entre vídeos e custo.
