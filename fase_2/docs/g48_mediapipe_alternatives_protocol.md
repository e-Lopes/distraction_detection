# G4.8 — Alternativas ao MediaPipe para indicadores faciais

**Estado:** G4.8-E1 em preparação  
**Plano de origem:** `../../plano_experimento_alternativas_mediapipe.md`  
**Geração de artefatos:** `G48`  
**Escopo principal:** MediaPipe, InsightFace, OpenFace 2.0 e MMPose/RTMW  
**Extensão de eficiência:** OpenSeeFace  
**Extensão condicionada:** YOLO26-Pose facial customizado

## Finalidade e separação experimental

A G4.8 compara extratores faciais sob um contrato anatômico e temporal comum. Ela não substitui
nem reclassifica resultados de G0–G4.7. A G4.7 permanece como exploração YOLO26; seus pesos COCO
de 17 pontos podem fornecer um proxy de pose da cabeça, mas não participam da comparação principal
de EAR/MAR e landmarks faciais da G4.8.

Nenhum resultado exploratório será gravado como oficial. Artefatos G4.8 usam exclusivamente:

- `outputs/metrics/G48/`;
- `outputs/figures/G48/`;
- `outputs/models/G48/`;
- `outputs/predictions/G48/`;
- `outputs/logs/G48/`;
- `outputs/cache/G48/`.

## Etapas e gates

### G4.8-E1 — Preparação e congelamento

**Objetivo:** congelar candidatos, licenças, versões, amostra, contrato anatômico e condições de
benchmark antes de comparar resultados.

Entregáveis:

- matriz de licença separando código e pesos;
- ambientes/dependências reproduzíveis por extrator;
- manifesto estratificado de 300–600 frames, sem copiar vídeos para o Git;
- mapeamento anatômico preliminar por extrator;
- protocolo de ROI, resolução, confiança, warm-up e hardware;
- hashes dos modelos/configurações.

Gate: nenhum extrator avança sem carregar localmente, declarar sua licença, produzir a saída
normalizada e passar por inspeção visual do mapeamento em frames frontal, lateral e ocluído.

### G4.8-E2 — Ground truth

**Objetivo:** obter a referência manual comum para landmarks e indicadores.

Entregáveis: anotações dos pontos mínimos comuns, protocolo de oclusão, segunda anotação cega da
subamostra e métricas intra/interavaliador. A diagonal da bounding box facial é a normalização
principal do NME; distância interocular é análise secundária.

Gate: ambiguidades revisadas e variabilidade da anotação reportada antes de ranquear extratores.

### G4.8-E3 — Integração dos extratores

**Objetivo:** implementar adaptadores com um contrato único:

```text
video_id, frame_index, timestamp_seconds, extractor,
face_bbox, landmarks_xy, landmark_confidence,
face_valid, failure_reason,
ear, mar, pitch, yaw, roll,
indicator_validity, inference_ms
```

EAR, MAR e pose principal usam as mesmas fórmulas, modelo 3D, câmera e convenção de eixos. Pose
nativa de frameworks é registrada somente como análise secundária. Falhas permanecem ausentes,
nunca zero.

Gate: testes geométricos e auditoria visual de cada mapeamento aprovados.

### G4.8-E4 — Avaliação isolada

**Objetivo:** medir qualidade geométrica, robustez e estabilidade sem classificação downstream.

Métricas principais: NME por diagonal da face, failure rate por motivo, CED/AUC, MAE/RMSE/p95 e
Bland–Altman dos indicadores, jitter em segmentos estáveis e estatísticas de gaps.

Gate: mesma amostra e regras congeladas para todos os extratores; frames consecutivos não são
tratados como observações independentes.

### G4.8-E5 — Benchmark computacional

**Objetivo:** comparar custo nos mesmos frames, ROI, resolução e hardware.

Cada condição terá cinco repetições, warm-up, execução sem visualização/gravação e sincronização
de GPU. A latência p95 é principal; também são registrados p50/p99, FPS, CPU/GPU, RAM/VRAM,
inicialização, tamanho dos pesos e tempo por vídeo. Modos com tracking e frame a frame são
identificados separadamente.

Gate: manifesto de hardware/software e hashes acompanham toda tabela oficial.

### G4.8-E6 — Avaliação downstream

**Objetivo:** medir o efeito do extrator em SVM, LSTM, TCN e Transformer.

Folds, janelas, seeds, hiperparâmetros, balanceamento, preprocessing e regras de episódios ficam
constantes na comparação principal. Usam-se splits temporais sem sobreposição, scaler train-only
e janelas causais de 30/60/90/150 frames; 300 é opcional. Macro F1 é a métrica principal, com
recall/F1 de Fadiga, PR-AUC, falsos episódios/hora e estabilidade entre folds/seeds.

Gate: seleção somente por treino/validação; teste operacional não ajusta extrator, thresholds ou
pós-processamento.

### G4.8-E7 — Decisão

**Objetivo:** selecionar soluções não dominadas por análise de Pareto, sem escore arbitrário.

Eixos mínimos: qualidade, cobertura/estabilidade, latência e desempenho downstream. Licença,
recall de Fadiga e falsos episódios/hora são restrições de promoção. A decisão final registra
manter/substituir MediaPipe e se os critérios para YOLO26 facial customizado foram satisfeitos.

## Ordem de execução

```text
E1 Preparação → E2 Ground truth → E3 Integração → E4 Qualidade/robustez
                                           └──→ E5 Custo
                         E4 + E5 → E6 Downstream → E7 Pareto/decisão
```

E2 e a preparação técnica de E3 podem avançar em paralelo, mas nenhum ranking oficial ocorre
antes dos respectivos gates. OpenSeeFace entra depois do smoke dos quatro extratores principais.
YOLO26 facial permanece condicionado à decisão E7.

## Estado inicial e próximos passos

1. Auditar licença de código e pesos dos quatro candidatos principais.
2. Congelar versões/modelos e decidir se cada framework roda em ambiente isolado.
3. Gerar o manifesto estratificado de 300–600 frames usando os quatro vídeos e intervalos atuais.
4. Criar os quatro mapeamentos anatômicos e aprová-los visualmente.
5. Somente então iniciar a extração comparativa curta e a anotação manual.

