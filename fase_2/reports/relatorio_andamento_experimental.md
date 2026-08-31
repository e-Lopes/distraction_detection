# Relatório de andamento e resultados experimentais

**Projeto:** Detecção de fadiga e distração para teleoperação de máquinas pesadas

**Etapa:** classificação temporal dos estados do operador

**Data de consolidação:** 30/08/2026

**Status:** G1–G4 e G4.5A/B concluídas; G4.5C pendente; hipótese H3 ainda em avaliação

## 1. Resumo executivo

O trabalho implementou um pipeline reproduzível para extrair cinco indicadores faciais — EAR,
MAR, pitch, yaw e roll —, convertê-los em janelas temporais e comparar regras fixas,
classificadores clássicos e arquiteturas temporais na classificação de `Alert`, `Fatigue` e
`Distraction`. Os quatro vídeos reais somam 122.337 frames e aproximadamente 119 minutos.

Os resultados obtidos até G4 respondem parcialmente à hipótese da qualificação:

- modelos aprendidos superaram amplamente o baseline de regras fixas;
- a melhor arquitetura temporal sem balanceamento, TCN/60, atingiu Macro F1 de validação
  `0,4125 ± 0,0211`, contra `0,1597` da melhor configuração de regras;
- o melhor clássico da comparação principal, SVM/60, atingiu `0,4191 ± 0,0593` e permaneceu
  ligeiramente acima da TCN/60;
- a modelagem temporal demonstrou vantagem sobre regras, mas ainda não demonstrou vantagem
  geral sobre classificadores sem arquitetura temporal;
- `Fatigue` é o principal gargalo: existem somente 84 segundos anotados e apenas dois eventos
  que geram janelas avaliáveis de 60 frames;
- class weights e weighted sampling recuperaram parte do recall de fadiga, mas elevaram falsos
  alarmes e reduziram o Macro F1.

A hipótese H3 permanece aberta. A conclusão confirmatória depende de repetição multi-seed dos
contrastes selecionados e de uma leitura por sessão/evento, sem tratar janelas sobrepostas como
observações independentes.

## 2. Pergunta, objetivo e hipótese

### 2.1 Pergunta de pesquisa

> Uma arquitetura temporal pode aprender múltiplos indicadores de fadiga e distração e superar
> abordagens baseadas em regras no contexto da teleoperação de mineração?

### 2.2 Objetivo geral

Investigar e avaliar arquiteturas temporais para detecção de fadiga e distração em operadores de
máquinas pesadas teleoperadas.

### 2.3 Objetivos específicos

1. Extrair indicadores visuais em vídeos reais de teleoperação.
2. Avaliar a necessidade de adaptação de domínio.
3. Construir um baseline baseado em regras.
4. Comparar arquiteturas temporais e classificadores clássicos.
5. Avaliar o impacto do tamanho da janela temporal.

### 2.4 Hipótese em avaliação

Arquiteturas temporais capazes de processar janelas de múltiplos indicadores superam abordagens
baseadas em regras fixas e classificadores sem modelagem temporal.

Para evitar uma conclusão composta ambígua, a hipótese foi separada em dois contrastes:

- **H-temporal-regras:** temporais superam regras fixas;
- **H-temporal-clássicos:** temporais superam classificadores sem modelagem temporal.

O primeiro contraste possui evidência preliminar favorável; o segundo permanece não confirmado.
O escopo operacional prevê sempre o mesmo operador, permitindo personalização, mas a avaliação
científica principal continua multiclasses. A agregação `normal/non_normal` será somente análise
operacional complementar.

## 3. Dados e indicadores

### 3.1 Vídeos

| Vídeo | Frames | Duração aproximada | FPS | Resolução |
|---|---:|---:|---:|---:|
| video_01 | 32.013 | 29,60 min | 18,023 | 1920×1080 |
| video_02 | 28.200 | 29,91 min | 15,716 | 1920×1080 |
| video_03 | 30.334 | 29,73 min | 17,007 | 1920×1080 |
| video_04 | 31.790 | 29,74 min | 17,814 | 1920×1080 |
| **Total** | **122.337** | **aprox. 119 min** | — | — |

Os metadados completos e as limitações do backend estão em
[`videos.csv`](../data/manifests/videos.csv).

### 3.2 Distribuição temporal dos rótulos

| Estado | Frames | Duração estimada | Proporção do dataset |
|---|---:|---:|---:|
| Alert | 75.064 | 4.436,04 s | 61,36% |
| Distraction | 14.081 | 824,94 s | 11,51% |
| Fatigue | 1.488 | 83,90 s | 1,22% |
| OperatorAbsent | 31.704 | 1.793,95 s | 25,91% |

`OperatorAbsent` é estado operacional e não é convertido em fadiga ou distração. Janelas sem
predominância comportamental suficiente recebem `mixed` e ficam fora do treino principal.

### 3.3 Indicadores extraídos

- EAR: abertura relativa dos olhos;
- MAR: abertura relativa da boca;
- pitch, yaw e roll: orientação estimada da cabeça;
- `face_detected`: validade da observação facial;
- flags derivadas de interpolação e duração de ausência, quando a representação exige.

Foram detectados 75.660 dos 122.337 frames. A taxa global de ausência facial foi 38,15%, com
forte variação entre vídeos:

| Vídeo | Detecção facial | Missingness |
|---|---:|---:|
| video_01 | 56,98% | 43,02% |
| video_02 | 83,70% | 16,30% |
| video_03 | 67,48% | 32,52% |
| video_04 | 41,99% | 58,01% |

Essa variação motivou a comparação controlada das representações R0–R2. Os dados, anotações e
fontes estão descritos em [`data_sources_and_provenance.md`](../docs/data_sources_and_provenance.md).

## 4. Protocolo experimental

### 4.1 Unidade de entrada

- janelas de 30, 60 e 150 frames;
- stride de 15 frames;
- rótulo pela maioria dos frames comportamentais;
- proporção dominante mínima de 60%;
- janelas `mixed` excluídas do treino principal.

Nos vídeos atuais, essas janelas correspondem aproximadamente a 1,7–1,9 s, 3,3–3,8 s e
8,3–9,5 s. Assim, o estudo já compara três escalas curtas, embora contexto mais longo para
fadiga permaneça como trabalho posterior.

### 4.2 Divisão dos dados

- quatro folds leave-one-video/session-out;
- validação em blocos temporais contínuos dentro das sessões de treino;
- purge gap mínimo de 150 frames;
- proibição de divisão aleatória de janelas sobrepostas;
- normalização, imputação e pesos ajustados somente no treino;
- teste externo excluído da seleção de modelo, representação, janela e threshold.

### 4.3 Métricas

A métrica principal é Macro F1 nas três classes. Também são preservados balanced accuracy,
precision, recall e F1 por classe, matrizes de confusão, distribuição de previsões, tempo de
treino e métricas operacionais de falsos episódios de fadiga.

### 4.4 Famílias comparadas

- **B1:** regras fixas baseadas em EAR, MAR e pitch;
- **B2:** SVM, Random Forest e XGBoost sem arquitetura temporal;
- **P1:** LSTM;
- **P2:** TCN;
- **P3:** Transformer como controle experimental.

O protocolo detalhado está em
[`experimental_protocol.md`](../docs/methodology/experimental_protocol.md).

## 5. Trabalho implementado

### 5.1 Fundação e rastreabilidade

- manifesto dos quatro vídeos;
- normalização e validação dos 71 intervalos anotados;
- conversão consistente de segundos para frames;
- separação entre target comportamental e estado operacional;
- distribuição por frame, janela, fold e classe;
- diagnóstico de missingness por vídeo, classe, janela e fold;
- fingerprints, configurações resolvidas e manifestos SHA-256 dos experimentos;
- testes contra vazamento entre treino, validação e teste.

### 5.2 Representações

- **R0:** cinco sinais com zero-fill, referência principal;
- **R1:** cinco sinais com interpolação curta e mediana do treino;
- **R2:** R1 acrescida de `FaceDetected`, `WasInterpolated` e
  `MissingDurationSoFar`;
- **R3 histórico:** estatísticas agregadas por janela para baselines clássicos;
- **temporal_behavior_v1:** nova representação ainda não avaliada, com distribuição, deltas,
  PERCLOS, eventos oculares/orais, pose e missingness configurável.

### 5.3 Infraestrutura de treinamento

- execução em CPU/CUDA e mixed precision;
- early stopping por Macro F1 de validação;
- checkpoints best/last, retomada e estado de RNG;
- dry-run e smoke tests;
- geração de predições, métricas, tabelas e figuras;
- suporte a class weights, weighted sampling e augmentation isolados.

## 6. Resultados experimentais

Todos os valores desta seção são médias de validação. Resultados externos de teste existem nos
artefatos, mas não são usados para ordenar configurações.

### 6.1 G1 — regras fixas e classificadores clássicos

| Modelo | 30 frames | 60 frames | 150 frames | Melhor janela |
|---|---:|---:|---:|---:|
| Regras fixas | **0,1597** | 0,0904 | 0,0715 | 30 |
| SVM | 0,3782 | **0,4191** | 0,3576 | 60 |
| Random Forest | 0,3984 | **0,4022** | 0,3983 | 60 |
| XGBoost | 0,4000 | **0,4024** | 0,4017 | 60 |

Os modelos aprendidos superaram o baseline de regras em todas as janelas. O baseline fixo foi
prejudicado principalmente pelo pitch sistematicamente deslocado pela câmera lateral. SVM/60
foi o melhor resultado de validação da G1.

Na janela de 60 frames, o SVM obteve F1 `0,8818` em Alert, `0,2900` em Distraction e `0,0855`
em Fatigue. Random Forest e XGBoost não detectaram Fatigue nessa configuração.

Fonte: [`g1_report.md`](../outputs/metrics/G1/g1_report.md).

### 6.2 G2 — arquiteturas temporais em R0, sem balanceamento

| Arquitetura | 30 frames | 60 frames | 150 frames | Melhor configuração |
|---|---:|---:|---:|---:|
| LSTM | 0,4027 | **0,4044** | 0,3946 | LSTM/60 |
| TCN | 0,4065 | **0,4125** | 0,4032 | TCN/60 |
| Transformer | 0,4062 | 0,4065 | **0,4081** | Transformer/150 |

TCN/60 foi o melhor temporal e ficou `0,0066` abaixo do SVM/60 da G1. Nenhuma das nove
combinações temporais produziu F1 útil de Fatigue: precision, recall e F1 ficaram em zero. Em
Distraction, TCN/60 obteve F1 `0,3325`, acima do SVM/60, mas o ganho não compensou a ausência de
detecção de fadiga no Macro F1.

Fonte: [`g2_report.md`](../outputs/metrics/G2/g2_report.md).

### 6.3 G3 — interpolação e missingness

| Modelo/janela | R0 | R1 | R2 | Melhor |
|---|---:|---:|---:|---:|
| LSTM/60 | **0,4044** | 0,3163 | 0,3343 | R0 |
| TCN/60 | **0,4125** | 0,3167 | 0,3617 | R0 |
| Transformer/150 | 0,4081 | 0,3138 | **0,4112** | R2 |
| **Média agregada** | **0,4083** | 0,3156 | 0,3691 | **R0** |

R1 piorou os três modelos. R2 recuperou parte da perda e melhorou o Transformer em apenas
`+0,0032`, sem consistência nas demais arquiteturas. Nenhuma representação recuperou Fatigue.
R0 avançou por melhor desempenho agregado e menor complexidade.

Fonte: [`g3_report.md`](../outputs/metrics/G3/g3_report.md).

### 6.4 G4 — mitigação do desbalanceamento

Os cenários foram aplicados isoladamente:

- A: distribuição original;
- B: class weights;
- C: weighted sampling;
- D: augmentation leve das classes minoritárias.

| Modelo/cenário | Macro F1 | F1 Fatigue | Recall Fatigue | Falsos episódios de fadiga/h |
|---|---:|---:|---:|---:|
| SVM/A | 0,4191 ± 0,0593 | 0,0855 | 0,1544 | 53,10 |
| SVM/D | **0,4203 ± 0,0597** | 0,0869 | 0,1544 | 47,23 |
| LSTM/B | 0,4004 ± 0,0507 | **0,0885** | 0,3920 | 56,92 |
| TCN/C | 0,4035 ± 0,0156 | 0,0652 | **0,4785** | 84,79 |

SVM/D apresentou o maior Macro F1, mas o ganho de `+0,0012` sobre SVM/A é pequeno e não resolve
fadiga. LSTM/B apresentou o compromisso temporal mais equilibrado. TCN/C atingiu o maior recall
de fadiga, porém com baixa precisão (`0,0370`) e elevado número de falsos episódios.

Sob a mesma estratégia B, LSTM/B superou SVM/B em Macro F1 (`0,4004` contra `0,3898`) e F1 de
Fatigue (`0,0885` contra `0,0745`). Esse contraste é favorável à modelagem temporal, mas ainda é
qualificatório e baseado somente na seed 42.

Fonte: [`g4_imbalance_results.md`](g4_imbalance_results.md).

### 6.5 G4.5A — ajuste cross-fit do threshold de fadiga

Após a consolidação inicial deste relatório, foi executada a primeira intervenção pré-registrada
da G4.5. Foram usadas exclusivamente as probabilidades de validação de LSTM/B, TCN/B e TCN/C.
Janelas repetidas entre folds foram deduplicadas, as probabilidades foram renormalizadas e cada
threshold foi calibrado nos outros três vídeos, sem acesso ao vídeo-alvo ou ao teste externo.

| Modelo | Argmax: Macro F1 | Cross-fit: Macro F1 | Argmax: F1 Fatigue | Cross-fit: F1 Fatigue |
|---|---:|---:|---:|---:|
| LSTM/B | **0,4005** | 0,3390 | **0,0883** | 0,0512 |
| TCN/B | **0,4029** | 0,3608 | **0,0528** | 0,0298 |
| TCN/C | **0,4035** | 0,3811 | **0,0652** | 0,0102 |

O ajuste de threshold piorou Macro F1 e F1 de Fatigue nos três casos. Os thresholds escolhidos
variaram de 0,35 a 0,50, mostrando forte dependência da sessão; TCN/C selecionou o limite máximo
0,50 para todos os vídeos. A evidência indica que a dificuldade de fadiga não decorre apenas da
regra argmax: as probabilidades não separam a classe rara de maneira estável entre sessões.

Fonte: [`g45a_report.md`](../outputs/metrics/G45/g45a_report.md).

### 6.6 G4.5B — Focal Loss

Foram concluídos oito novos runs com LSTM/60 e TCN/60, R0, quatro folds e seed 42. A Focal Loss
usou `gamma=2,0` e alphas balanceados calculados somente no treino. A geração avaliou somente
validação e não produziu predições de teste externo.

| Modelo | Referência B: Macro F1 | Focal: Macro F1 | Referência B: F1 Fatigue | Focal: F1 Fatigue |
|---|---:|---:|---:|---:|
| LSTM | 0,4004 | **0,4044** | 0,0885 | **0,1039** |
| TCN | 0,4029 | **0,4035** | 0,0528 | **0,0566** |

A LSTM apresentou ganho pequeno: `+0,0039` em Macro F1 e `+0,0154` em F1 de Fatigue, com
precision `0,0730` e recall `0,4067`. Na TCN, o efeito foi praticamente nulo e o recall de
fadiga diminuiu de `0,3971` para `0,3436`. Focal Loss permanece candidata apenas para LSTM e
ainda não supera o melhor SVM/A (`0,4191`) em Macro F1.

Fonte: [`g45_results.md`](g45_results.md).

## 7. Impacto do tamanho da janela

Os resultados não indicam uma janela universalmente melhor:

- regras fixas: 30 frames;
- SVM, Random Forest e XGBoost: 60 frames;
- LSTM e TCN: 60 frames;
- Transformer: 150 frames.

Janelas de 60 frames produziram o melhor compromisso para a maioria dos modelos. O Transformer
se beneficiou modestamente de 150 frames. Como 150 frames ainda representam menos de dez
segundos, o estudo atual avalia contexto curto. Fadiga pode exigir uma segunda escala de dezenas
de segundos, que deverá ser introduzida em experimento separado com purge recalculado.

## 8. Avaliação dos objetivos específicos

| Objetivo | Estado | Evidência atual |
|---|---|---|
| Extrair indicadores visuais | Concluído | Cinco sinais por frame, manifesto e diagnóstico de validade |
| Avaliar adaptação de domínio | Parcial | Adaptação YOLO histórica; R0–R2 e variação entre sessões avaliadas; normalização pessoal pendente |
| Baseline baseado em regras | Concluído | G1, thresholds congelados e três janelas |
| Comparar temporais e clássicos | Parcialmente concluído | G1/G2/G4 completos com seed 42; confirmação multi-seed pendente |
| Avaliar tamanho da janela | Concluído na qualificação | 30/60/150 em regras, clássicos e temporais |

## 9. Avaliação da hipótese H3

### 9.1 Temporal versus regras fixas

O melhor temporal atingiu `0,4125`, contra `0,1597` das regras. A diferença de `+0,2528` em
Macro F1 é grande e consistente com a inadequação dos thresholds fixos ao posicionamento lateral
da câmera. O contraste possui evidência preliminar forte, embora o resultado final deva preservar
as repetições e unidades por sessão.

### 9.2 Temporal versus classificadores sem modelagem temporal

O melhor temporal sem balanceamento, TCN/60 (`0,4125`), não superou SVM/60 (`0,4191`). No
cenário pareado B, LSTM/B superou SVM/B, mas ambos apresentaram desempenho baixo em fadiga e a
comparação usa apenas uma seed. Portanto, esse contraste permanece inconclusivo.

### 9.3 Estado da hipótese

**H3 não está confirmada nem rejeitada integralmente.** A evidência atual sustenta que aprender
fronteiras é muito melhor que aplicar regras fixas, mas não sustenta ainda que a ordem temporal
produz ganho geral sobre o melhor classificador clássico.

## 10. Limitações

- apenas quatro sessões do mesmo operador;
- somente 84 segundos de fadiga anotada;
- dois eventos de fadiga geram janelas avaliáveis de 60 frames;
- vídeo 2 não contém fadiga e vídeo 4 não contém janela de fadiga em 60/150 frames;
- janelas sobrepostas não são observações estatisticamente independentes;
- resultados G1–G4 usam uma única seed;
- missingness varia de 16,30% a 58,01% entre sessões;
- contexto máximo atual inferior a dez segundos;
- o Transformer/CUDA não é determinístico bit a bit no kernel de atenção utilizado;
- alta sensibilidade à fadiga veio acompanhada de precisão baixa e muitos falsos alarmes.

Essas limitações impedem interpretar pequenas diferenças de Macro F1 como superioridade
definitiva.

## 11. Reprodutibilidade e verificação

Os experimentos preservam:

- configurações YAML e configurações resolvidas;
- IDs com geração, modelo, representação, janela, fold e seed;
- métricas por run, fold e classe;
- matrizes de confusão e distribuições de previsão;
- checkpoints e históricos locais;
- manifestos de artefatos com SHA-256;
- testes de janelamento, splits, preprocessing, modelos e retomada.

Na consolidação deste relatório, os arquivos oficiais continham:

- G1: 96 linhas de avaliação, correspondentes a 48 combinações avaliadas em validação e teste;
- G2: 72 linhas de avaliação, correspondentes a 36 runs;
- G3: 72 linhas consolidadas, com 12 runs R0 reutilizados e 24 novos R1/R2;
- G4: 128 linhas de avaliação, correspondentes a 64 comparações novas/reutilizadas.

Reexecutar G1–G4 sem mudança pré-registrada não verificaria “melhora”; produziria somente nova
realização estocástica ou duplicação dos resultados. A G4.5A foi executada por ser uma nova
intervenção congelada e mostrou piora com threshold cross-fit. As próximas verificações adequadas
são as demais intervenções G4.5, a G5 multi-seed e uma comparação separada para qualquer
representação nova.

## 12. Próximas etapas

1. Concluir G4.5C com classificação hierárquica; G4.5A piorou os scores e G4.5B trouxe ganho
   pequeno somente para LSTM.
2. Congelar finalistas sem consultar o teste externo.
3. Executar G5 com as seeds `42, 123, 456, 789, 2026`.
4. Reportar deltas pareados por sessão e estabilidade entre seeds.
5. Validar `temporal_behavior_v1` em comparação separada, sem misturá-la a G1–G5.
6. Executar ablação dos grupos ocular, oral, pose e missingness.
7. Avaliar normalização personalizada como adaptação de domínio em experimento isolado.
8. Avaliar contexto multiescala somente após recalcular janelas, splits e purge.
9. Gerar estudos de caso por evento, falsos positivos e falsos negativos.

## 13. Conclusão provisória

O pipeline desenvolvido consegue aprender padrões muito superiores às regras históricas e
identifica distração de forma útil, mas ainda apresenta dificuldade severa na detecção confiável
de fadiga. A melhor configuração global continua sendo clássica, enquanto estratégias temporais
mostram vantagens parciais sob tratamento de desbalanceamento. O resultado mais correto neste
momento é que a primeira parte da hipótese possui suporte preliminar e a segunda continua em
aberto. A continuação deve priorizar estabilidade multi-seed, avaliação por evento e ganho
temporal pareado contra o clássico, sem buscar apenas aumento isolado de score.

## Referências internas

- [`DocQualificacao.pdf`](../../DocQualificacao.pdf)
- [`qualification_baseline.md`](../docs/qualification_baseline.md)
- [`research_question_alignment.md`](../docs/research_question_alignment.md)
- [`experimental_protocol.md`](../docs/methodology/experimental_protocol.md)
- [`integrated_plan_gap_analysis.md`](../docs/integrated_plan_gap_analysis.md)
- [`g1_report.md`](../outputs/metrics/G1/g1_report.md)
- [`g2_report.md`](../outputs/metrics/G2/g2_report.md)
- [`g3_report.md`](../outputs/metrics/G3/g3_report.md)
- [`g4_imbalance_results.md`](g4_imbalance_results.md)
