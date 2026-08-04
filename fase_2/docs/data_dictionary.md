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

