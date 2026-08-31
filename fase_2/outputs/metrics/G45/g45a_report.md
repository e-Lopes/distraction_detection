# G4.5A — threshold de Fatigue com cross-fit por sessão

Somente predições de validação da G4 foram carregadas. Cada vídeo-alvo foi excluído da calibração de seu threshold; janelas repetidas foram deduplicadas por média das probabilidades e renormalização.

## Thresholds escolhidos

| Modelo | Vídeo-alvo | Threshold | Macro F1 de calibração | Macro F1 retido |
|---|---|---:|---:|---:|
| lstm_b | video_01 | 0.39 | 0.4391 | 0.2312 |
| lstm_b | video_02 | 0.38 | 0.3833 | 0.3928 |
| lstm_b | video_03 | 0.37 | 0.3473 | 0.4955 |
| lstm_b | video_04 | 0.39 | 0.3859 | 0.3907 |
| tcn_b | video_01 | 0.35 | 0.4316 | 0.2619 |
| tcn_b | video_02 | 0.35 | 0.3853 | 0.4008 |
| tcn_b | video_03 | 0.45 | 0.3669 | 0.4405 |
| tcn_b | video_04 | 0.36 | 0.3954 | 0.3755 |
| tcn_c | video_01 | 0.50 | 0.4317 | 0.2583 |
| tcn_c | video_02 | 0.50 | 0.3769 | 0.4227 |
| tcn_c | video_03 | 0.50 | 0.3539 | 0.4916 |
| tcn_c | video_04 | 0.50 | 0.3909 | 0.3808 |

## Reconstrução por fold

| Modelo | Método | Macro F1 médio | F1 Fatigue médio | Recall Fatigue médio |
|---|---|---:|---:|---:|
| lstm_b | argmax | 0.4005 | 0.0883 | 0.3920 |
| lstm_b | crossfit_threshold | 0.3390 | 0.0512 | 0.2574 |
| tcn_b | argmax | 0.4029 | 0.0528 | 0.3971 |
| tcn_b | crossfit_threshold | 0.3608 | 0.0298 | 0.3529 |
| tcn_c | argmax | 0.4035 | 0.0652 | 0.4785 |
| tcn_c | crossfit_threshold | 0.3811 | 0.0102 | 0.2500 |

Os resultados são qualificatórios, usam seed 42 e não incluem o teste externo. Melhora em validação não autoriza promover configuração antes das demais etapas congeladas da G4.5.
