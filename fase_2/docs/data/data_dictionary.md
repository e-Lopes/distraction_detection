# Dicionário de dados

| Campo | Nível | Descrição |
|---|---|---|
| `video_id` | vídeo | Identificador anônimo do vídeo |
| `frame_idx` | frame | Índice sequencial do frame |
| `timestamp_s` | frame | Tempo em segundos desde o início |
| `ear` | frame | Eye Aspect Ratio |
| `mar` | frame | Mouth Aspect Ratio |
| `pitch`, `yaw`, `roll` | frame | Orientação estimada da cabeça |
| `face_detected` | frame | Detecção facial válida |
| `was_interpolated` | frame | Valor preenchido por interpolação |
| `missing_duration_so_far` | frame | Duração acumulada do gap até o frame |
| `frame_label` | frame | Alert, Fatigue ou Distraction |
| `operational_state` | frame | Valid, FaceMissing, Occlusion ou OperatorAbsent |

## Anotações temporais da fase 2

As anotações manuais são intervalos em resolução de segundos. O arquivo local configurado
em `configs/data/base.yaml` não é versionado por conter informação derivada dos vídeos.

| Campo | Descrição |
|---|---|
| `video_id` | Identificador anônimo `video_01` a `video_04` |
| `start_time`, `end_time` | Limites inclusivos no formato `MM:SS` |
| `behavior_label` | `alert`, `fatigue`, `distraction` ou vazio quando não há alvo comportamental |
| `operational_state` | `valid` ou `operator_absent` nesta versão |
| `source_label` | Rótulo original em português, preservado para auditoria |
| `annotation_version` | Versão congelada da anotação |

`Ausente` é uma condição operacional e nunca é convertida automaticamente em fadiga ou
distração. As grafias `Distração` e `Distraido` da fonte são normalizadas como
`distraction`, mantendo-se o texto original em `source_label`.

## Semântica das flags de missingness

- `face_detected = 1` somente quando o frame possui detecção facial válida e os cinco sinais
  observados; permanece `0` mesmo quando os sinais são posteriormente preenchidos.
- `was_interpolated = 1` somente quando os cinco sinais ausentes foram produzidos pela política
  de interpolação curta; preenchimento por zero ou mediana do treino não ativa a flag.
- `missing_duration_so_far` conta causalmente os frames consecutivos sem detecção até o frame
  atual e volta a zero quando `face_detected = 1`; nunca contém a duração futura total do gap.

Na representação R2, a ordem é `[ear, mar, pitch, yaw, roll, face_detected, was_interpolated,
missing_duration_so_far]`. As duas flags binárias não são padronizadas. Os cinco sinais e a
duração usam somente parâmetros estimados no treino.
