# Linha de base acadêmica da qualificação

Este documento resume os compromissos científicos formalizados em `DocQualificacao.pdf` e
os reconcilia com os ativos encontrados. Ele não substitui o documento acadêmico original.

## Identificação

- Título: *Fatigue and Distraction Detection for Heavy Machinery Teleoperation*.
- Tipo: projeto de dissertação apresentado ao PPGIa/PUCPR.
- Extensão: 104 páginas, PDF/A-2b.
- SHA-256: `7ebbbd07aba33656f28aadb90a1c8697033c3c34a398c43362b5fb9b15fe1eda`.
- Dataset declarado: quatro vídeos, aproximadamente 119 minutos, 122.337 frames, cerca de
  17 FPS e resolução 1920×1080.

Os metadados declarados precisam ser confirmados pelo manifesto dos arquivos reais antes de
serem usados computacionalmente.

## Hipóteses e estado formal

| Hipótese | Conteúdo | Estado na qualificação |
|---|---|---|
| H1 | Indicadores comportamentais podem ser extraídos com MediaPipe e YOLO. | Confirmada; sistema integrado com 87% de acurácia nos 40.778 frames anotados. |
| H2 | Modelos genéricos não são suficientes; adaptação ao domínio é necessária. | Confirmada; detector de celular passou de F1 0,03 para aproximadamente 0,995. |
| H3 | Arquiteturas temporais superam regras fixas e classificadores sem modelagem temporal. | Em aberto; constitui o núcleo experimental restante. |

H1 e H2 são antecedentes acadêmicos. A fase 2 não precisa reexecutá-los para iniciar H3, mas
deve preservar seus dados, limitações e resultados como legado auditável.

## Contribuições concluídas

### Monitoramento corporal

- Postura por landmarks corporais e inclinação da cabeça.
- Mãos fora da região de controle.
- Ausência do operador.
- Hierarquia de alertas e suavização temporal por regras.
- Validação manual em 40.778 frames e cinco categorias.

Esse sistema fornece contexto corporal e operacional, mas não mede adequadamente fadiga.

### Adaptação do detector de celular

- Evidência do domain gap do YOLO COCO na cabine lateral e com baixa iluminação.
- Dataset específico anotado com LabelImg.
- Treino das variantes YOLO11n/s/m/l/x.
- Seleção acadêmica do YOLO11-Nano pelo compromisso entre desempenho e eficiência.

As métricas publicadas foram obtidas com validação cruzada aleatória de frames próximos. Elas
demonstram adaptação ao domínio, mas não comprovam generalização temporal ou entre sessões.

### Baseline facial por regras

O documento declara a extração de EAR, MAR, pitch, yaw e roll por MediaPipe Face Mesh e
OpenCV, com regras:

- EAR menor que 0,25 por pelo menos 20 frames: fadiga;
- MAR maior que 0,55: fadiga;
- pitch menor que −15°: distração;
- demais frames: alerta.

Uma média móvel de 60 frames foi usada para suavização. O baseline produziu excesso de
distração porque o ângulo lateral da câmera desloca sistematicamente o pitch.

## Evidências faciais relatadas

| Indicador | Vídeo 1 | Vídeo 2 | Vídeo 3 | Vídeo 4 |
|---|---:|---:|---:|---:|
| EAR médio | 0,286 | 0,285 | 0,269 | 0,264 |
| MAR médio | 0,018 | 0,016 | 0,032 | 0,022 |
| Pitch médio | −20,8° | −31,6° | −30,3° | −29,1° |
| Detecção facial | 57,0% | 83,7% | 67,5% | 42,0% |

Esses valores são resultados históricos. O dataset por frame que os originou não está nesta
máquina e pode permanecer na máquina da extração original. Ele não pode alimentar os modelos
até ser transferido, inventariado e validado contra esses valores e os vídeos reais.

## Ground truth temporal

A qualificação definiu anotação por segmentos (`start_time`, `end_time`, `state`) e as classes
Alert, Fatigue e Distraction. Ausência, câmera bloqueada ou estado não avaliável eram tratados
como excluídos. A anotação manual fornecida posteriormente concretiza esse trabalho em 71
intervalos e separa ausência como estado operacional.

O protocolo acadêmico também previa confiança alta/média e observação textual. Esses campos
não foram fornecidos na anotação atual e ficam registrados como metadados opcionais, não como
valores a serem inventados.

## Núcleo experimental restante — H3

Todos os modelos devem receber a mesma fonte de cinco indicadores e os mesmos folds:

- B1: baseline de regras já especificado;
- B2: SVM, Random Forest e XGBoost sobre janelas achatadas `N × 5`;
- P1: LSTM;
- P2: TCN;
- P3: Transformer experimental e não bloqueante.

A comparação deve separar três perguntas:

1. B1 versus B2: aprender fronteiras é melhor que usar thresholds fixos?
2. B2 versus P1–P3: preservar ordem temporal traz ganho adicional?
3. B1 versus P1–P3: qual é o ganho total da arquitetura proposta?

Macro F1 é a métrica principal. Precision, recall, AUC-ROC e matrizes de confusão são
complementares. Os splits devem ser session-disjoint/leave-one-video-out.

## Divergências e pendências de reconciliação

| Tema | Documento | Ativos auditados | Tratamento |
|---|---|---|---|
| Dataset de celular | 2.879 imagens | 2.906 imagens; 2.878 labels; 2.870 boxes | Não alterar silenciosamente; investigar origem da contagem acadêmica. |
| Cross-validation YOLO | 5 folds | Existem execuções de 5, 3 e 2 folds | Vincular cada tabela ao run/configuração exatos antes de republicar. |
| Indicadores faciais | Baseline executado e estatísticas publicadas | Dataset tabular pendente de transferência da outra máquina | Recuperar e auditar primeiro; regenerar somente se ausente ou inválido. |
| Missing facial | Vetores zerados | Decisão atual: zero-fill como baseline com flag de detecção | Preservar comparação; interpolação está adiada. |
| Anotações | Planejadas com confiança e observação | 71 intervalos sem esses campos | Não fabricar metadados; documentar a ausência. |

## Critério de preservação

Resultados da qualificação permanecem válidos como histórico acadêmico, mas nenhum número é
promovido automaticamente a resultado da fase 2. Reutilização exige vínculo com dados,
configuração, split, código e artefato de execução identificáveis.
