# Diagnóstico de missingness facial

- Frames auditados: 122.337
- Frames sem face: 46.677 (38.15%)
- Missing é definido exclusivamente por `face_detected=0`; zeros não são usados como sentinela.
- Gap curto: até 15 frames, definido antes dos folds de teste.

| Vídeo | Frames | Detecção | Missing | Gaps | Curtos | Longos | Maior gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| video_01 | 32013 | 56.98% | 43.02% | 159 | 95 | 64 | 5658 |
| video_02 | 28200 | 83.70% | 16.30% | 53 | 24 | 29 | 1390 |
| video_03 | 30334 | 67.48% | 32.52% | 95 | 68 | 27 | 7163 |
| video_04 | 31790 | 41.99% | 58.01% | 90 | 51 | 39 | 15706 |

## Missingness por classe anotada

| Classe | Frames | Missing |
|---|---:|---:|
| alert | 75064 | 10.29% |
| distraction | 14081 | 61.91% |
| fatigue | 1488 | 40.19% |

## Missingness médio por janela

| Janela | Classe | Janelas | Missing médio |
|---:|---|---:|---:|
| 30 | alert | 4999 | 10.34% |
| 30 | distraction | 932 | 62.13% |
| 30 | fatigue | 97 | 40.69% |
| 60 | alert | 4983 | 10.36% |
| 60 | distraction | 922 | 62.65% |
| 60 | fatigue | 88 | 43.77% |
| 150 | alert | 4961 | 10.59% |
| 150 | distraction | 893 | 63.67% |
| 150 | fatigue | 84 | 46.29% |

Os CSVs complementares preservam os recortes por vídeo e por fold. Janelas `mixed` permanecem nos arquivos para auditoria, mas não integram o treinamento principal.
