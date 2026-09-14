# Qualidade dos indicadores com câmera superior/lateral — v1

Data: 14/09/2026. Estado: implementação e amostras de desenvolvimento; **sem reextração integral e sem treinamento oficial**. Complementa a comparação de famílias modernas, não a cruza automaticamente.

## 1. Auditoria do estado real

- Interface existente preservada: `python -m fase_2 status/prepare/train/report/all`, configuração herdada de `configs/final_experiment.yaml`, orquestração em `src/pipeline.py`, relatório em `src/reporting.py`. Novo perfil: `configs/measurement_experiment.yaml`; novo subcomando público de extração: `measurement-extract`.
- Alterações locais da rodada anterior foram preservadas. Originais, anotações e resultados históricos não foram sobrescritos. Novos outputs ficam em `outputs/measurement_v1`, amostras em `outputs/measurement_samples`, bruto completo futuro em `data/interim/measurement_raw_v1`.
- Extrator canônico: MediaPipe Face Mesh legado; ambiente da amostra 0.10.21, OpenCV registrado no JSON da execução. V1 produz EAR médio e pose; V2 já preserva EAR por olho e usa qualidade geométrica por largura relativa interocular. Esse score é heurística do projeto, não confiança oficial por landmark.
- EAR direito: `[33,160,158,133,153,144]`; esquerdo: `[362,385,387,263,373,380]`. Distância entre dois pares verticais dividida por duas vezes a distância horizontal. Implementação canônica multiplica x por largura e y por altura: não calcula distância misturando coordenadas normalizadas de escalas distintas.
- MAR: distância entre centróides superiores `[82,13,312]` e inferiores `[87,14,317]`, dividida pela distância `[78,308]`. Não equivale a anotação de bocejo e não recebe correção por cosseno do yaw.
- ROI histórica original: `[1145,521,538,491]`, em vídeos 1920×1080. O extrator escolhia o primeiro rosto; não havia identidade persistente. O manifesto aponta `data/raw/1.mp4`, etc., mas os arquivos locais estão em `data/raw/videos/`; o novo perfil explicita esses caminhos sem modificar o manifesto histórico.
- Ausência na extração canônica: campo vazio; `face_detected` é global. `preprocess_block` zera todos os indicadores quando a face falta. R0 preenche ausência com zero antes do scaler; R1 interpola até 15 frames e usa medianas de treino; R2 acrescenta três flags. O validador histórico rejeita uma linha detectada com apenas um indicador ausente, apesar de a extração poder perder somente pose: incompatibilidade concreta resolvida apenas para o novo schema.
- **Não há média móvel de 60 frames no caminho de treinamento ativo** (`extract_facial_series` → `temporal_data`). A plotagem tem mediana móvel de 5 segundos, apenas visual. A referência nova não insere um filtro de 60 frames por memória de uma versão antiga. O número 60 ativo é tamanho de janela; 60 frames corresponderiam a aproximadamente 3,33/3,82/3,53/3,37 segundos por vídeo, não uma duração única.
- Timestamps históricos são `frame_index/FPS`; o novo bruto preserva o timestamp entregue pelo decoder OpenCV e sua origem. Timestamps inexistentes ou não monotônicos geram erro, sem fallback silencioso para FPS. Essa origem não certifica sincronização absoluta de hardware. Na amostra, frame 180 tem 11,533s e não exatamente 180/FPS.
- LOVO, quatro folds, validação em blocos contíguos e purge 150 frames. Janelas históricas 30/60/150, stride 15, maioria mínima 0,60. A maioria considera frames anotados; janelas parcialmente sem rótulo não são automaticamente descartadas. Janelas mistas são excluídas igualmente; nenhuma variante descarta janelas difíceis por qualidade.
- Modelos-sonda: SVM linear sobre **trajetória achatada**, não estatísticas agregadas; LSTM de duas camadas/hidden 64. w60, seed42, class weights train-only, hiperparâmetros herdados/fixos. As demais famílias continuam no experimento próprio.

## 2. Convenção e limitações da pose

O modelo genérico 3D e os seis pontos `[1,152,33,263,61,291]` permanecem em `extract_facial_series.py`. Historicamente solvePnP usa foco igual à largura do **crop**, centro no crop e distorção zero. No novo cálculo, landmarks são reconvertidos para pixels originais; K aproximada usa fx=fy=largura original e centro do frame. Não confundir essa aproximação com câmera calibrada.

A matriz R transforma coordenadas do modelo para a câmera. No OpenCV: x para a direita, y para baixo, z para frente. A decomposição histórica corresponde a `Rz @ Ry @ Rx`, mas suas colunas são chamadas **pitch=ângulo Y, yaw=ângulo Z, roll=ângulo X**. Mantivemos os nomes para compatibilidade e documentamos a permutação; não interpretar os nomes como convenção anatômica padrão. Unidades: graus. [Documentação oficial solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

Preservam-se matriz, translação, K, erro RMS de reprojeção e ângulos originais; ângulos com K histórica ficam em `legacy_pitch/yaw/roll` para a referência pareada. V2 histórica colapsa yaw/roll módulo 180 em ±90; essa transformação não foi aplicada ao novo bruto. Tratamento Q desfaz saltos de 360° por unwrap apenas em segmentos contínuos; não resolve ambiguidade de modelo/PnP e não rejeita movimento rápido por si só.

Pose relativa disponível como utilitário `R_rel = R_ref.T @ R`, seguido de decomposição. D permanece **desabilitada**: início da gravação não é referência neutra comprovada. Habilitar exige matriz operacional revisada apenas no desenvolvimento; o código rejeita D sem matriz. Não usar rótulos Alert externos, mediana do vídeo externo inteiro ou subtração genérica de ângulos. Calibração por sessão/exclusão de trecho não foi implementada nesta v1.

## 3. Extração bruta e identidade

`measurement_raw.py` produz CSV `measurement_v1` + JSON de proveniência por vídeo. Inclui timestamp, ROI, escala x/y efetiva e padding do letterbox, bbox, landmarks essenciais em pixels originais, EAR por olho, MAR, pose/matriz, face detectada, candidatos, track_id, tracking_status, critérios geométricos e motivos. Ausências são NaN. Candidatos sem identidade confirmada ficam identificados como diagnósticos, nunca como medições válidas do operador para classificação.

Redimensionamento para 640×640 preserva proporção com arredondamento a pixels e registra as duas escalas efetivas. Não se usa super-resolução, frontalização ou realce de contraste. Ampliação não recupera detalhe ocular ausente.

O lock exige uma **âncora visual verificada** `[x,y,w,h]` do rosto do operador, em resolução original, no frame inicial. Uma detecção única e IoU≥0,25 mantém continuidade. Vários candidatos, desaparecimento ou salto insuficientemente sobreposto provocam perda; não há readquisição automática. Uma nova âncora verificada pode reiniciar o track naquele frame. ROI/IoU não são biometria e não garantem identidade quando alguém substitui outra pessoa na mesma posição: isso requer revisão humana e é limitação explícita, não uma propriedade certificada.

Por segurança `verified_anchors` está vazio nos quatro vídeos. Reextração completa é bloqueada sem âncora frame 0. Revisor deve inspecionar identidade inicial e perdas/entradas de outras pessoas antes de classificar. Configure, por exemplo, `verified_anchors: {0: [x, y, w, h], 1234: [x2, y2, w2, h2]}` com coordenadas **realmente revisadas**, não os símbolos do exemplo. Não extrapolar âncoras de um vídeo para outro.

## 4. Políticas e matriz

| Variante | Intervenção | Canais | Fits |
|---|---|---:|---:|
| QA | referência pareada: EAR médio, pose histórica crop, zero-fill, sem suavização | 5 | 8 |
| QB0 | ausência explícita/flags, sem rejeições geométricas adicionais | 15 | 8 |
| QB | QB0 + qualidade por indicador | 15 | 8 |
| QC | QB + olho predominante fixo | 15 | 8 |
| QD | QC + rotação relativa | 15 | 0, desabilitada |
| QE01 | QC + interpolação offline ≤0,1s | 15 | 8 |
| QE02 | QC + interpolação offline ≤0,2s | 15 | 8 |
| QES | QC + mediana causal trailing 0,1s, EAR/MAR apenas | 15 | 8 |
| Diagnóstico | LogisticRegression somente médias das 5 máscaras e 5 flags de QB | 10 features | 4 |

Teto: **60 fits**, 28 LSTM, 28 SVM, quatro diagnósticos. Sem combinação interpolação+suavização, sem múltiplas seeds ou janelas nesta triagem. QC/E ficam bloqueados enquanto `predominant_eye: null`; os dois trechos inspecionados sugerem o direito como hipótese geométrica, mas não cobrem orientação suficiente para impor uma decisão à instalação. Não escolher maior EAR nem melhor F1 externo.

QA reproduz **a política numérica** corrente sobre uma extração pareada, incluindo pose crop e zero-fill. Não é reprodução bit a bit dos CSVs históricos: letterbox, detector stateless e lock são diferentes. A referência histórica original permanece preservada e deve ser apresentada separadamente. QB é intervenção composta (representação, máscaras, pose consistente com frame original e qualidade); QB0 versus QB isola a contribuição das rejeições. Não atribuir todo delta QA→QB somente à qualidade.

Os 15 canais são cinco sinais padronizados + cinco máscaras **observado e aceito** + cinco flags **interpolado**. Média/desvio são ajustados apenas em valores observados dos frames de treino (sem duplicar frames por sobreposição). Máscaras não são normalizadas. Valores ainda ausentes recebem zero **após** padronização; esse zero não é observação física. Canal sem nenhuma observação de treino usa média0/escala1 fixas, sem consultar avaliação. Registrar cobertura antes de interpretar desempenho.

Exceção de referência: QA conserva o scaler R0 calculado sobre janelas de treino, incluindo zeros e repetição de frames nas sobreposições. Essa diferença faz parte da intervenção composta de representação e é explícita.

Qualidade v1: todos os landmarks relevantes dentro do crop, largura ocular≥8 pixels, boca≥12 pixels, pose com reprojeção RMS≤15 pixels e profundidade positiva. São hipóteses conservadoras de desenvolvimento, centralizadas e congeladas, não limiares validados. Largura horizontal degenerada é problema numérico; não há limiar inferior de EAR nem superior de MAR. Desfoque (variância do Laplaciano da ROI) é apenas diagnóstico, sem threshold; não há detector confiável de oclusão por indicador. Scores oficiais por landmark indisponíveis ficam explicitamente ausentes, nunca inventados. Não interpretar blendshapes/limiares de detecção como confiança de cada olho. [Saídas documentadas do Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python); a API legada usada aqui não é a mesma API Tasks.

Interpolação por indicador exige extremidades válidas, mesmo track e bloco. Duração conservadora = distância temporal **entre as âncoras**. Com ~15 FPS, uma única falta já demanda ~0,133s: candidato0,1s poderá não preencher nada. Isso será resultado, não motivo para alterar retrospectivamente a regra. Não extrapola, não atravessa perdas de identidade, sessões ou splits. Pose usa SLERP de rotações, não média linear de Euler. Flags preservam que um fechamento interpolado não foi observado.

Mediana de0,1s é causal trailing e não usa futuro, mas pode atrasar transições/apagar eventos; atraso é dependente do sinal, limitado ao histórico de0,1s, não um atraso constante de fase. Reinicia em lacunas/identidade/timestamps descontínuos. Não suaviza pose nesta v1. Primeiro inspecionar fechamentos curtos, fala e movimentos reais, mantendo grade pequena.

## 5. Amostra visual e limitações observadas

Dois trechos predefinidos sem consulta a rótulos: vídeo2 frames180–224 e600–644, ambos dentro de **train do fold1**. Total90 frames (~6s), não quatro vídeos. Artefatos locais:

- `outputs/measurement_samples/v1_video02_f180/video_02.csv`, `.json`, `_audit.png`;
- `outputs/measurement_samples/v1_video02_f600/video_02.csv`, `.json`, `_audit.png`.

Figura sincroniza ROI, landmarks e curvas/máscaras; versões regeneradas também mostram eixos da pose. Não há interpolação na amostra bruta nem rótulos de classificação sobrepostos. Todos os frames ficaram `unverified_operator`: a amostra não concede autorização de identidade para treino. O candidato inspecionado está sentado diante dos controles; outra pessoa aparece parcialmente na ROI.

No primeiro trecho, candidato único em45/45 frames. Olho esquerdo com largura3,65–8,16px (critério aceitou2/45), direito12,38–15,91px (45/45). EAR esquerdo0,22–0,82 versus direito0,15–0,29. Isso evidencia assimetria/instabilidade, **não** erro geométrico quantitativo sem ground truth. Pose passa em45/45 com reprojeção2,06–3,27px, apesar de saltos de roll; critério não detecta todas as ambiguidades. Segundo trecho também mostra instabilidades de pose e rejeições pontuais.

Não foram cobertos de modo verificado: rosto frontal, cabeça muito abaixada, fechamento completo de olhos, fala/bocejo, oclusão prolongada, desfoque forte e mudanças de iluminação. O procedimento é selecionar até90 frames por trecho dentro do treino do fold de desenvolvimento, registrar fenômeno visual e revisar sem F1. Externos só após congelamento, identificados como análise de erro, sem reajuste. Inspeção de vídeo2 influencia desenvolvimento de folds que o reservam: todos os resultados futuros deste estudo são exploratórios entre sessões, não teste intocado.

Tempo medido da extração dessas amostras:0,64s e0,57s (exclui geração da figura e hash completo do vídeo). Projeção puramente linear para122.337 frames: cerca de27min **nesta máquina**, não benchmark confiável para orientações difíceis, I/O, revisão humana ou RTX4060. Não foi medida a duração de treino; agendar até30min/fit como teto de planejamento (30h para60), não estimativa empírica nem limite imposto pelo código. MediaPipe continua CPU; RTX4060 será usada pela LSTM com PyTorch/CUDA.

## 6. Preparação, cobertura e relatório

Prepare valida schema completo, hashes, timestamps, intervalos e isolamento; samples não podem passar como extração completa. Preserva por tratamento/fold/bloco (somente desenvolvimento) `treated_frames/`, com sinais brutos/tratados, origem do olho, máscaras, rejeições, interpolação, duração de lacuna e necessidade de imputação. A extração bruta original nunca é sobrescrita.

Relatório existente recebe seção específica e mantém métricas de classificação (Macro F1, métricas/suporte por classe, balanced accuracy, confusões, sessão/fold/seed). Arquivos adicionais:

- `quality_coverage.csv`: válidos/interpolados/rejeitados/imputados por indicador, classe, vídeo, fold e subset;
- `quality_gaps.csv`: episódios de lacuna e duração, sinalizando extremidades sem âncoras;
- `quality_rejections.csv`: motivos e mudanças de origem ocular;
- `quality_windows.csv`: cobertura e janelas totalmente sem medições; conjuntos comuns são comparados e diferenças abortam preparação.

Ausência parcial nunca zera os demais indicadores na variante nova. Não há abstenção na avaliação principal. Comparar remoções em Fatigue/Distraction versus Alert; diagnóstico só de máscaras é associação, não causalidade. Janelas sobrepostas e seeds não são réplicas independentes.

Avaliação externa **não habilitada** nesta v1. Selecionar tratamento pela validação interna de cada fold, depois congelar protocolo externo versionado antes de pontuar. Não usar relatórios antigos para declarar teste intocado. Nenhum resultado antigo comprova cobertura por olho: CSV de cinco números não contém essa informação. Reutilizáveis: vídeos/annotations/splits/contagens e resultados históricos como referência descritiva, não checkpoints com dimensão15.

## 7. Verificações desta etapa

Suíte completa: **213 passaram, 1 opcional pulado**, sem treinamento oficial. Testes incluem crop/letterbox e EAR com aspecto conhecido; perda/ambiguidade de identidade sem readquisição; ausência parcial/total; EAR zero e MAR alto sem rejeição circular; limites por timestamp; gaps longos, bordas, identidade e splits; composição de rotação/SLERP em ±180°; mediana causal sem atravessar gaps; scaler observado-only sem influência da avaliação; máscaras binárias; janelas comuns; serialização; fingerprint; orçamento; SVM/diagnóstico pelo executor real com dados sintéticos e retomada; backward sintético LSTM15 canais; figuras/deltas com métricas sintéticas em diretório temporário. Shell passou em `bash -n`; `git diff --check` sem erros.

O relatório oficial foi gerado vazio de resultados novos e sem proxies históricos importados. Três figuras históricas produzidas na primeira verificação do gerador foram movidas para `outputs/measurement_samples/report_diagnostic_historical/`, fora dos resultados desta rodada. Nenhuma métrica sintética foi adicionada ao relatório científico.

As anotações continuam vinculadas aos índices do manifesto existente. Timestamps reais agora usados em tratamentos não corrigem automaticamente eventual imprecisão da conversão histórica segundos→frames; a sincronização deve ser revisada antes de inferências temporais finas.

## 8. Comandos futuros (raiz do repositório)

No ambiente da máquina RTX4060, instalar PyTorch com CUDA compatível; para extração, API MediaPipe legada:

```bash
fase_2/.venv/bin/python -m pip install -e './fase_2[temporal,extraction,dev]'
fase_2/.venv/bin/python -m pip check
fase_2/.venv/bin/python -u -m fase_2 train --plan --scope screening --config fase_2/configs/measurement_experiment.yaml
```

Amostra pequena (usar diretório novo para não sobrescrever bruto):

```bash
fase_2/.venv/bin/python -u -m fase_2 measurement-extract --config fase_2/configs/measurement_experiment.yaml --video video_02 --start-frame 1200 --max-frames 90 --output fase_2/outputs/measurement_samples/video02_f1200
```

Depois de confirmar ROI, âncoras/identidade e congelar critérios, **reextração integral futura**, sequencial, por vídeo (não executada nesta etapa):

```bash
for video in video_01 video_02 video_03 video_04; do
  fase_2/.venv/bin/python -u -m fase_2 measurement-extract --config fase_2/configs/measurement_experiment.yaml --video "$video" --full || break
done
```

Saída é atômica por vídeo, com manifesto atualizado; arquivo já existente não é sobrescrito. Se interromper, preservar `.tmp` para diagnóstico, mas retomada de extração no meio do vídeo não foi implementada. Retomar só vídeos ainda ausentes, após revisar perdas de tracking.

Preparar sem treinar / classificar depois / apenas relatar:

```bash
fase_2/.venv/bin/python -u -m fase_2 prepare --config fase_2/configs/measurement_experiment.yaml
bash fase_2/scripts/run_measurement_experiment.sh
fase_2/.venv/bin/python -u -m fase_2 report --config fase_2/configs/measurement_experiment.yaml
```

Wrapper exige CUDA, mostra progresso e salva log. Modelos clássicos em CPU; LSTM CUDA, best/last pelo runner existente. Retomada por fit compatível; fingerprints incluem configuração resolvida, rawCSV/JSON, extração/ROI/pose/qualidade, splits, código e dependências. Não reutilizar checkpoint antigo após mudança de canais/tratamento.
