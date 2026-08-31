# G4.7 — MediaPipe Face Mesh × YOLO26-Pose facial

**Status:** infraestrutura implementada; anotação CVAT, WFLW e pesos customizados pendentes  
**Escopo inicial:** seed 42, quatro folds, YOLO26 n/s/m, seleção por validação

## Contrato

O baseline permanece `mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)`. Não são
usados MediaPipe Tasks nem MediaPipe Pose. MediaPipe e YOLO são convertidos para o schema
anatômico de 22 pontos em `configs/features/g47_face_landmarks.yaml`; EAR, MAR e os seis pontos
do `solvePnP` passam pelas mesmas funções.

Os CSVs G0–G4.6 não são alterados. Falhas e oclusões permanecem `NaN`. Resultados G4.7 ficam em
`outputs/metrics/G47`, figuras em `outputs/figures/G47` e pesos em `outputs/models/G47`.

## Gates obrigatórios

1. O ambiente `g47_env` deve passar o manifesto CUDA e carregar `yolo26n-pose.yaml`.
2. O YAML precisa declarar `kpt_shape: [22, 3]` e o `flip_idx` congelado.
3. Um par original/espelhado deve ser aprovado visualmente e registrado por hash do schema.
4. Os 10% reanotados precisam ter NME mediano ≤ 0,03 e p95 ≤ 0,08.
5. Busca de perdas somente no Nano e somente em treino/validação.
6. `pose_loss` não pode conter NaN/Inf nem terminar acima de 5× a mediana inicial.
7. Teste operacional não participa da seleção do extrator ou filtro.

## Fluxo operacional

```bash
# 0. Clone isolado; não altera yolo_env nem a .venv CPU.
bash fase_2/scripts/setup_g47_env.sh
conda activate g47_env

# 1. Contrato e seleção dos 800 frames locais.
python -m fase_2.src.data.g47_landmarks validate-schema \
  --schema fase_2/configs/features/g47_face_landmarks.yaml
python -m fase_2.src.data.g47_landmarks sample-local \
  --video video_01=fase_2/data/raw/videos/1.mp4 \
  --video video_02=fase_2/data/raw/videos/2.mp4 \
  --video video_03=fase_2/data/raw/videos/3.mp4 \
  --video video_04=fase_2/data/raw/videos/4.mp4 \
  --series-dir fase_2/data/interim/geometry_v2 \
  --intervals fase_2/data/manifests/annotation_frame_intervals.csv \
  --output-dir fase_2/data/external/G47/cvat_sample

# 2. Após preencher o mapeamento WFLW e concluir/revisar o CVAT:
python -m fase_2.src.data.g47_landmarks convert-wflw --help
python -m fase_2.src.data.g47_landmarks convert-cvat --help
python -m fase_2.src.data.g47_landmarks agreement --help
python -m fase_2.src.data.g47_landmarks audit-flip --help

# 3. O dry-run exige dataset e marcador de aprovação do flip válidos.
python -m fase_2.src.training.g47_yolo_face \
  --data 'fase_2/data/external/G47/fold_{fold}.yaml' --scale n --search-losses --dry-run

# 4. Benchmark curto antes dos vídeos completos.
python -m fase_2.src.evaluation.g47_benchmark \
  --video fase_2/data/raw/videos/1.mp4 --video-id video_01 \
  --model fase_2/outputs/models/G47/weights/yolo26n-face-custom__fold_1.pt \
  --device cuda:0 --max-frames 100

# 5. Avaliação downstream permanece validation-only por padrão.
python -m fase_2.src.evaluation.g47_downstream \
  --input-dir fase_2/outputs/metrics/G47/yolo26n_cuda
```

## Suavização e promoção

Os sinais brutos são obrigatórios. EMA e One Euro são avaliados somente se o jitter YOLO for
maior que 1,5× o MediaPipe ou se Bland–Altman exceder ±5° em yaw/roll ou ±7,5° em pitch. O
filtro é ajustado em treino/validação e reinicia após 0,5 s sem detecção.

Uma variante só é promovida com ≥20 FPS, p95 ≤55 ms, perda de cobertura ≤2 pontos
percentuais, perda de Macro F1 ≤0,01 e melhora em mais de um fold. Em empate de Macro F1 de
até 0,01, vence menor latência/memória.

## Pendências externas explícitas

- preencher os 22 índices WFLW após conferir a versão adquirida;
- concluir e revisar as 800 anotações no CVAT;
- criar os quatro YAMLs leave-one-video-out e o marcador de aprovação do flip;
- treinar os pesos customizados; `l/x` ficam para a GPU mais forte;
- executar benchmark completo três vezes por condição.
