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
