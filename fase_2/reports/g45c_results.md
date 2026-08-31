# Resultados G4.5C — classificação hierárquica

**Data:** 31/08/2026  
**Escopo:** validação, R0, janela 60, quatro folds, seed 42  
**Teste externo:** não carregado

Foram concluídos os oito pipelines previstos (SVM e LSTM em quatro folds), com dois
estimadores independentes por pipeline. O nível 1 separou `Alert` de `Non-Alert`; o nível 2
foi ajustado exclusivamente com `Fatigue` e `Distraction` do treino. As probabilidades foram
compostas na ordem `Alert`, `Fatigue`, `Distraction`.

| Modelo | Macro F1 (média ± DP) | Balanced accuracy (média ± DP) | F1 Alert | F1 Fatigue | F1 Distraction |
|---|---:|---:|---:|---:|---:|
| SVM hierárquico | 0,3327 ± 0,0319 | 0,3490 ± 0,0320 | 0,9201 | 0,0000 | 0,0780 |
| LSTM hierárquica | 0,4186 ± 0,0649 | 0,4452 ± 0,0484 | 0,8642 | 0,0923 | 0,2994 |

## Comparação pareada da LSTM

| Fold | Hierárquica | LSTM/B | LSTM/Focal | Delta vs. B | Delta vs. Focal |
|---:|---:|---:|---:|---:|---:|
| 1 | 0,5156 | 0,4623 | 0,4495 | +0,0533 | +0,0660 |
| 2 | 0,3810 | 0,3382 | 0,3599 | +0,0429 | +0,0211 |
| 3 | 0,3938 | 0,3976 | 0,4030 | -0,0039 | -0,0092 |
| 4 | 0,3841 | 0,4037 | 0,4049 | -0,0196 | -0,0208 |

## Decisão

A LSTM hierárquica melhorou o Macro F1 médio, mas o ganho não foi consistente nos quatro folds
e não melhorou de forma convincente a classe `Fatigue`: seu F1 médio (0,0923) ficou abaixo de
LSTM/Focal (0,1039). O SVM hierárquico não recuperou nenhuma janela de `Fatigue`.

Conforme o critério congelado, nenhum modelo hierárquico é promovido à G5. A hierarquia é
preservada como análise qualificatória. G5 fica liberada com LSTM/B e SVM/B como candidatos
principais; TCN/C e SVM/C permanecem análise secundária.

Os CSVs agregados estão em `outputs/metrics/G45/g45c_*.csv`. Checkpoints, logs e predições por
janela são locais e não versionados por conterem artefatos sensíveis ou volumosos.
