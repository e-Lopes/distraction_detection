# Importacao e teste GPU

Origem: `C:\Users\Eduardo\Desktop\Mestrado\distraction_detection\distraction_detection\fase_1\resultados_legados\extracao_ao_vivo\20260916_204539_018100`.
Frames: 122337; sem rotulo: 2 (mantidos como unknown).
Janelas estritas: 35, comprimento/stride 90.

```text
label      0  1
video_id       
video_01  12  0
video_02   9  0
video_03  10  1
video_04   3  0
```

GPU: NVIDIA GeForce RTX 2060 SUPER; forwards auditados: 0.
Encoder aleatorio em eval; normalizacao apenas no suporte. Nenhum treino ou checkpoint.
K/Q deste teste nao substituem a avaliacao K=1/5/10 planejada.
Janelas sem overlap, sem NaN e homogeneas na classe binaria. Sem imputacao.
Filtro estrito introduz vies de selecao; consultar frame_coverage e excluded_windows.
Angulos ainda nao validados. Nao interpretar este teste como resultado cientifico.

## Proximo passo
Auditar landmarks e erro de reprojecao antes de tratar lacunas. Fixar extrator,
preparar dados publicos para meta-treino e usar os videos proprios so na meta-avaliacao.
Os scripts numerados legados nao foram executados: usam outro CSV/preenchimento
e o terceiro treina nos videos proprios, em desacordo com o plano.
