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
