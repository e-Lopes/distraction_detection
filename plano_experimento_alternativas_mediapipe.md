# Plano Experimental — Análise de Alternativas ao MediaPipe para Extração de Indicadores Faciais

## 1. Contexto

Este experimento integra o projeto de detecção de fadiga e distração em operadores de equipamentos pesados em ambiente de teleoperação. O sistema atual utiliza o **MediaPipe Face Mesh** para extrair landmarks faciais e, a partir deles, calcular cinco indicadores por frame:

- Eye Aspect Ratio (EAR);
- Mouth Aspect Ratio (MAR);
- Pitch;
- Yaw;
- Roll.

Esses indicadores formam séries temporais utilizadas por modelos de classificação, como SVM, LSTM, TCN e Transformer, para distinguir os estados **Alerta**, **Fadiga** e **Distração**.

O MediaPipe apresenta baixo custo computacional e facilidade de uso, mas sua taxa de detecção varia entre os vídeos e pode ser afetada pela posição lateral da câmera, oclusões, equipamentos de proteção individual, iluminação e baixa resolução do rosto. Por isso, serão avaliados outros extratores de landmarks faciais.

## 2. Objetivo geral

Comparar alternativas ao MediaPipe Face Mesh quanto à qualidade dos landmarks, robustez, estabilidade temporal, custo computacional e impacto no desempenho dos classificadores de fadiga e distração.

## 3. Objetivos específicos

1. Implementar uma interface comum para diferentes extratores faciais.
2. Mapear landmarks anatomicamente equivalentes entre os extratores.
3. Extrair EAR, MAR, Pitch, Yaw e Roll com procedimentos padronizados.
4. Avaliar a precisão geométrica dos landmarks em uma amostra anotada manualmente.
5. Medir cobertura, falhas consecutivas e estabilidade temporal.
6. Comparar latência, throughput e consumo de recursos computacionais.
7. Avaliar como cada extrator afeta SVM, LSTM, TCN e Transformer em diferentes janelas temporais.
8. Identificar alternativas no melhor compromisso entre qualidade, robustez, custo e desempenho final.

## 4. Perguntas de pesquisa

### PQ1 — Qualidade geométrica

Qual extrator produz landmarks faciais mais próximos das anotações manuais no ambiente real de teleoperação?

### PQ2 — Robustez

Qual extrator mantém a maior cobertura diante de posição lateral, iluminação variável, oclusões e uso de equipamentos de proteção?

### PQ3 — Estabilidade temporal

Qual extrator produz séries de EAR, MAR e Head Pose mais estáveis e com menos descontinuidades?

### PQ4 — Custo computacional

Qual extrator apresenta o melhor desempenho computacional em CPU e GPU?

### PQ5 — Impacto downstream

Qual extrator produz indicadores que resultam no melhor desempenho dos classificadores temporais?

### PQ6 — Compromisso operacional

Qual solução oferece o melhor equilíbrio entre Macro F1, sensibilidade à Fadiga, falsos episódios por hora, latência e uso de recursos?

## 5. Hipóteses experimentais

- **H1:** pelo menos uma alternativa apresenta menor taxa de falha que o MediaPipe.
- **H2:** extratores com menor erro geométrico nos olhos produzem estimativas de EAR mais confiáveis.
- **H3:** menor erro geométrico não implica necessariamente melhor desempenho na classificação temporal.
- **H4:** soluções com rastreamento temporal produzem menor jitter que soluções independentes por frame.
- **H5:** o MediaPipe permanece competitivo em custo computacional, mas pode ser superado em robustez ou desempenho downstream.
- **H6:** um extrator mais pesado somente será vantajoso se a melhoria de detecção compensar sua latência e consumo de recursos.

## 6. Dados disponíveis

- Quatro vídeos reais de teleoperação;
- aproximadamente 119 minutos no total;
- aproximadamente 122.337 frames;
- taxa média próxima de 17 FPS;
- câmera RGB monocular 1080p;
- ângulo predominantemente lateral;
- ROI do operador definida por vídeo;
- classes anotadas: Alerta, Fadiga, Distração e Excluído;
- ground truth temporal produzido manualmente em nível de frame.

Os vídeos não devem ser misturados aleatoriamente entre treino e validação. A proximidade temporal entre frames e janelas poderia causar vazamento de informação.

## 7. Extratores candidatos

### 7.1 MediaPipe Face Mesh

Baseline atual, com 478 landmarks faciais.

### 7.2 InsightFace

Alternativa baseada em análise facial 2D/3D, com modelos capazes de produzir 106 landmarks 2D ou 68 landmarks 3D. Deve-se verificar separadamente a licença do código e dos pesos pré-treinados antes de qualquer uso fora da pesquisa.

### 7.3 OpenFace 2.0

Ferramenta voltada à análise de comportamento facial. Produz landmarks, Head Pose, confiança, sucesso da detecção, gaze e Action Units. Será avaliado tanto o Head Pose nativo quanto o Head Pose padronizado.

### 7.4 MMPose/RTMW

Framework moderno de pose. Modelos whole-body podem produzir landmarks corporais, faciais e das mãos, permitindo futura integração de indicadores corporais e faciais.

### 7.5 OpenSeeFace

Alternativa leve com rastreamento facial temporal e foco em execução em CPU. Será incluída principalmente na avaliação de estabilidade e custo computacional.

### 7.6 YOLO26-Pose facial customizado — etapa exploratória

O YOLO26-Pose padrão utiliza os 17 keypoints corporais do COCO e não fornece pontos suficientes para EAR e MAR. Sua inclusão exige treinamento ou ajuste de um modelo facial com landmarks específicos.

Essa alternativa será tratada como extensão porque envolve anotação, treinamento e validação adicionais. Caso seja treinada usando pseudo-rótulos do MediaPipe, uma amostra deverá ser revisada manualmente para reduzir o risco de o modelo apenas reproduzir os erros do baseline.

## 8. Escopo sugerido

### Etapa principal

1. MediaPipe;
2. InsightFace;
3. OpenFace;
4. MMPose/RTMW.

### Etapa de eficiência

5. OpenSeeFace.

### Etapa exploratória condicionada aos resultados

6. YOLO26-Pose facial customizado.

O avanço para YOLO26 deverá ocorrer somente se as soluções prontas não atingirem cobertura ou estabilidade satisfatórias, ou se houver interesse explícito em adaptação ao domínio da cabine.

## 9. Padronização dos landmarks

Os frameworks utilizam topologias e índices diferentes. Portanto, os pontos não serão comparados por número de índice, mas por correspondência anatômica.

Será criado um arquivo de configuração por extrator contendo, no mínimo:

- seis pontos para cada olho;
- oito pontos para a boca;
- ponta do nariz;
- queixo;
- cantos externos dos olhos;
- cantos da boca;
- confiança ou validade de cada ponto, quando disponível.

Exemplo conceitual:

```yaml
extractor: mediapipe
left_eye: [p1, p2, p3, p4, p5, p6]
right_eye: [p1, p2, p3, p4, p5, p6]
mouth: [p1, p2, p3, p4, p5, p6, p7, p8]
head_pose:
  nose_tip: p1
  chin: p2
  left_eye_outer: p3
  right_eye_outer: p4
  mouth_left: p5
  mouth_right: p6
```

Uma inspeção visual deverá confirmar o mapeamento de cada extrator antes da extração oficial.

## 10. Padronização dos indicadores

### 10.1 EAR

O EAR será calculado pela mesma fórmula para todos os extratores:

$$
EAR = \frac{\lVert p_2-p_6\rVert + \lVert p_3-p_5\rVert}{2\lVert p_1-p_4\rVert}
$$

Será calculado separadamente para cada olho. Quando ambos forem válidos, será utilizada a média. A regra para uso de apenas um olho deverá ser definida antes da avaliação oficial.

### 10.2 MAR

O MAR será calculado por uma fórmula única baseada em três distâncias verticais e uma distância horizontal:

$$
MAR = \frac{\lVert p_2-p_8\rVert + \lVert p_3-p_7\rVert + \lVert p_4-p_6\rVert}{3\lVert p_1-p_5\rVert}
$$

### 10.3 Head Pose padronizado

Para isolar a qualidade dos landmarks, o Head Pose principal será calculado com o mesmo procedimento para todos os extratores:

1. mesmos seis pontos anatômicos 2D;
2. mesmo modelo facial 3D;
3. mesma matriz intrínseca ou mesma aproximação de câmera;
4. `cv2.solvePnP`;
5. mesma conversão para Pitch, Yaw e Roll;
6. mesma convenção de sinais e eixos.

Quando um framework oferecer Head Pose nativo, ele será registrado em uma análise secundária de solução completa.

### 10.4 Valores ausentes

Falhas de detecção não serão codificadas automaticamente como zero. Cada frame deverá armazenar:

- valores dos cinco indicadores, quando válidos;
- máscara de validade;
- confiança dos landmarks, quando disponível;
- motivo da invalidade;
- identificador do extrator.

As estratégias de missingness serão aplicadas posteriormente e ajustadas somente nos dados de treino.

## 11. Construção do ground truth de landmarks

### 11.1 Amostragem

Será criada uma amostra estratificada dos quatro vídeos contendo:

- diferentes vídeos;
- Alerta, Fadiga e Distração;
- rosto frontal, lateral e parcialmente ocluído;
- diferentes condições de iluminação;
- presença de equipamentos de proteção;
- frames com e sem detecção no baseline;
- diferentes níveis de qualidade aparente.

Sugestão inicial: entre 300 e 600 frames, balanceados por vídeo e condição. O tamanho final deverá considerar o tempo disponível para anotação.

### 11.2 Landmarks manuais

Serão anotados somente os pontos anatômicos comuns necessários para:

- EAR;
- MAR;
- Head Pose;
- avaliação por regiões.

Não é necessário anotar as topologias completas de 478, 106 ou 68 pontos para a comparação principal.

### 11.3 Confiabilidade da anotação

Uma subamostra deverá ser anotada duas vezes, preferencialmente por dois avaliadores. Quando houver apenas um avaliador, deverá ocorrer uma segunda anotação cega após intervalo de tempo.

Serão registrados:

- erro intra-avaliador;
- erro interavaliador, quando possível;
- protocolo para pontos ocluídos;
- critérios de exclusão.

O erro do modelo deverá ser interpretado em relação à variabilidade da própria anotação.

## 12. Métricas de qualidade geométrica

### 12.1 NME — Normalized Mean Error

Métrica principal para landmarks:

$$
NME = \frac{1}{N}\sum_{i=1}^{N}\frac{\lVert p_i-\hat{p}_i\rVert_2}{d}
$$

Em razão do ângulo lateral, a normalização principal será feita pela diagonal da bounding box facial:

$$
d = \sqrt{w^2+h^2}
$$

A distância interocular poderá ser apresentada como análise secundária, mas não como normalização principal, pois um dos olhos pode estar ocluído ou fortemente comprimido pela perspectiva.

Resultados:

- NME médio;
- mediana;
- desvio-padrão;
- intervalo de confiança;
- percentil 95;
- NME dos olhos;
- NME da boca;
- NME dos pontos usados no Head Pose.

### 12.2 Taxa de falha

$$
FailureRate = \frac{\text{frames sem landmarks válidos}}{\text{total de frames}}
$$

Serão separados:

- rosto não detectado;
- landmarks ausentes;
- confiança insuficiente;
- geometria impossível ou degenerada;
- falha no cálculo de EAR/MAR;
- falha do `solvePnP`.

### 12.3 Curva de erro acumulado

Quando houver amostra suficiente, será construída uma Cumulative Error Distribution (CED), mostrando a proporção de frames abaixo de diferentes limites de NME. Também poderão ser apresentados:

- AUC da CED dentro de um intervalo predefinido;
- percentual de frames com NME abaixo de limites congelados.

## 13. Métricas dos indicadores

EAR, MAR e Head Pose derivados dos landmarks manuais serão utilizados como referência.

Para EAR e MAR:

- MAE;
- RMSE;
- erro mediano;
- percentil 95;
- correlação de Spearman;
- análise de concordância de Bland–Altman.

A correlação não será interpretada isoladamente como precisão.

Para Head Pose, quando houver referência angular confiável:

- MAE de Pitch em graus;
- MAE de Yaw em graus;
- MAE de Roll em graus;
- erro mediano;
- percentil 95;
- percentual de frames com erro menor que 5°, 10° e 15°.

Sem ground truth angular confiável, serão anotadas categorias direcionais, como frontal, esquerda, direita, acima e abaixo. Nesse caso, serão usados Macro F1, recall por direção e matriz de confusão, sem afirmar precisão angular absoluta.

## 14. Estabilidade temporal

### 14.1 Jitter normalizado

$$
Jitter = \frac{1}{T-1}\sum_{t=2}^{T}\frac{\lVert p_t-p_{t-1}\rVert_2}{d_t}
$$

Como o movimento real também aumenta essa medida, o jitter deverá ser calculado:

- em segmentos manualmente identificados como estáveis; ou
- após compensar o movimento global do rosto.

### 14.2 Continuidade

Serão medidos:

- maior intervalo consecutivo sem detecção;
- duração média dos gaps;
- número de gaps por minuto;
- percentual de gaps passíveis de interpolação curta;
- quantidade de saltos não plausíveis em EAR, MAR, Pitch, Yaw e Roll.

### 14.3 Estabilidade dos indicadores

Nos segmentos estáveis:

- desvio-padrão de EAR e MAR;
- variação absoluta entre frames consecutivos;
- variação de Pitch, Yaw e Roll;
- número de picos acima de limites definidos no conjunto de treino.

## 15. Benchmark computacional

### 15.1 Condições controladas

Todos os extratores serão executados com:

- mesmos vídeos e frames;
- mesma ROI;
- mesma resolução de entrada, quando tecnicamente possível;
- mesmo hardware;
- mesmo sistema operacional e ambiente documentado;
- inferência sem visualização ou gravação de vídeo;
- período de warm-up;
- pelo menos cinco repetições;
- sincronização explícita da GPU antes das medições de tempo;
- CPU e GPU avaliadas separadamente quando suportadas.

Se um extrator usar tracking em modo de vídeo e outro processar cada frame independentemente, essa diferença será registrada. Serão apresentados, quando possível, dois regimes:

1. modo recomendado pelo framework para vídeo;
2. modo comparável frame a frame.

### 15.2 Métricas

- latência média em ms/frame;
- latência mediana;
- latência p95 e p99;
- FPS sustentado;
- utilização média e máxima de CPU;
- utilização média e máxima de GPU;
- pico de RAM;
- pico de VRAM;
- tamanho dos pesos;
- tempo de inicialização;
- tempo total por vídeo;
- energia consumida, se houver instrumento confiável.

A latência p95 será a métrica computacional principal.

## 16. Avaliação downstream em séries temporais

### 16.1 Representação

Cada extrator produzirá, por frame:

```text
[EAR, MAR, Pitch, Yaw, Roll, máscaras de validade e confianças disponíveis]
```

As versões comparadas deverão utilizar o mesmo conjunto de atributos de entrada. Confianças específicas de um framework poderão ser avaliadas apenas em uma análise secundária.

### 16.2 Janelas temporais

Considerando aproximadamente 17 FPS:

| Frames | Duração aproximada |
| ---: | ---: |
| 30 | 1,8 s |
| 60 | 3,5 s |
| 90 | 5,3 s |
| 150 | 8,8 s |
| 300 | 17,6 s |

A janela de 300 frames poderá ser descartada se sua latência temporal for incompatível com o uso pretendido.

As janelas oficiais para uso em tempo real serão causais, utilizando somente passado e presente. Janelas centralizadas, se testadas, serão identificadas como análise offline.

### 16.3 Modelos

- SVM com atributos estatísticos da janela;
- LSTM;
- TCN;
- Transformer.

### 16.4 Controle experimental

Entre extratores, deverão permanecer constantes:

- folds;
- janelas;
- seeds;
- arquitetura e hiperparâmetros;
- número máximo de épocas;
- early stopping;
- tratamento de desbalanceamento;
- regra de rotulagem das janelas;
- métricas;
- política de seleção de checkpoints;
- pós-processamento temporal.

Uma análise secundária poderá ajustar hiperparâmetros por extrator, mas não deverá substituir a comparação controlada principal.

## 17. Protocolo de validação temporal

Não será usada divisão aleatória de frames ou janelas.

Opções aceitas:

1. leave-one-video-out, quando a distribuição das classes permitir;
2. blocos temporais contínuos por vídeo;
3. protocolo já congelado nas gerações anteriores, desde que não haja sobreposição entre treino e validação.

Requisitos:

- nenhum frame compartilhado entre conjuntos;
- nenhum par de janelas sobrepostas atravessando treino e validação;
- gap mínimo de uma janela completa entre blocos, quando necessário;
- scaler ajustado somente no treino;
- interpolação ajustada somente no treino;
- threshold ajustado somente em dados internos de treino/validação;
- conjunto de teste preservado para avaliação final, se possível.

## 18. Métricas de classificação

### 18.1 Métrica principal

- Macro F1.

### 18.2 Métricas complementares

- Balanced Accuracy;
- acurácia convencional;
- precisão, recall e F1 por classe;
- matriz de confusão;
- PR-AUC por classe;
- suporte por classe;
- média e desvio-padrão entre folds e seeds;
- intervalos de confiança.

A acurácia convencional não será usada como critério principal devido ao desbalanceamento entre Alerta, Fadiga e Distração.

## 19. Métricas temporais e operacionais

- Event Precision;
- Event Recall;
- Event F1;
- cobertura temporal do episódio;
- tempo até a primeira detecção;
- latência mediana e p95 da detecção;
- falsos episódios por hora;
- duração média dos falsos alertas;
- fragmentação de episódios;
- trocas de classe por minuto;
- percentual de episódios de Fadiga detectados.

As regras de união de janelas em episódios, tolerância temporal e duração mínima de alerta deverão ser definidas usando somente os dados de treino e congeladas antes do teste final.

## 20. Análises estatísticas

Como os mesmos frames, folds e seeds serão usados entre os extratores, as comparações deverão ser pareadas.

Sugestões:

- intervalos de confiança por bootstrap respeitando vídeo ou episódio como unidade de reamostragem;
- teste de Wilcoxon pareado quando a normalidade não for justificável;
- comparação pareada dos deltas de Macro F1 entre folds/seeds;
- correção para múltiplas comparações quando vários extratores forem comparados simultaneamente;
- tamanho de efeito, não somente valor de significância.

Frames consecutivos não deverão ser tratados como observações estatisticamente independentes.

## 21. Tabelas principais

### 21.1 Qualidade dos landmarks

| Extrator | NME geral ↓ | NME olhos ↓ | NME boca ↓ | Falha ↓ | Jitter ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MediaPipe |  |  |  |  |  |
| InsightFace |  |  |  |  |  |
| OpenFace |  |  |  |  |  |
| MMPose/RTMW |  |  |  |  |  |
| OpenSeeFace |  |  |  |  |  |
| YOLO26 customizado |  |  |  |  |  |

### 21.2 Indicadores

| Extrator | EAR MAE ↓ | MAR MAE ↓ | Pitch MAE ↓ | Yaw MAE ↓ | Roll MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MediaPipe |  |  |  |  |  |
| InsightFace |  |  |  |  |  |
| OpenFace |  |  |  |  |  |
| MMPose/RTMW |  |  |  |  |  |
| OpenSeeFace |  |  |  |  |  |

### 21.3 Custo computacional

| Extrator | FPS ↑ | Latência p95 ↓ | CPU ↓ | GPU ↓ | RAM ↓ | VRAM ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MediaPipe |  |  |  |  |  |  |
| InsightFace |  |  |  |  |  |  |
| OpenFace |  |  |  |  |  |  |
| MMPose/RTMW |  |  |  |  |  |  |
| OpenSeeFace |  |  |  |  |  |  |

### 21.4 Classificação downstream

| Extrator | Modelo | Janela | Macro F1 ↑ | F1 Fadiga ↑ | Recall Fadiga ↑ | PR-AUC Fadiga ↑ | Falsos episódios/h ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |  |

## 22. Gráficos recomendados

1. boxplot do NME por extrator e vídeo;
2. NME por região facial;
3. CED dos erros de landmarks;
4. taxa de detecção por vídeo e classe;
5. duração dos gaps sem detecção;
6. exemplos qualitativos de sucesso e falha;
7. séries temporais sobrepostas de EAR, MAR e Head Pose;
8. Bland–Altman para EAR e MAR;
9. latência p50/p95/p99;
10. Macro F1 por extrator, modelo e janela;
11. F1 de Fadiga versus falsos episódios por hora;
12. gráfico de Pareto entre Macro F1 e latência;
13. gráfico de Pareto entre NME e custo computacional.

## 23. Critério de seleção

Não será criada uma pontuação arbitrária combinando todas as métricas. A seleção será baseada em análise de Pareto.

Um extrator será considerado candidato principal quando:

1. não for dominado simultaneamente em qualidade, robustez, custo e desempenho downstream;
2. alcançar inferência compatível com a taxa do vídeo ou requisito operacional;
3. não apresentar aumento inaceitável de falsos episódios por hora;
4. preservar ou melhorar o recall de Fadiga;
5. apresentar estabilidade entre vídeos, folds e seeds;
6. possuir licença compatível com pesquisa e possível transferência à empresa parceira.

O extrator de maior NME pode ainda ser selecionado se produzir séries mais estáveis e melhor classificação final. Da mesma forma, o maior Macro F1 não será suficiente se houver recall de Fadiga inadequado ou falsos alarmes excessivos.

## 24. Critérios para avançar ao YOLO26 customizado

O treinamento de um YOLO26-Pose facial será iniciado apenas se pelo menos uma das condições ocorrer:

- nenhuma solução pronta atingir cobertura aceitável;
- os erros se concentrarem em características específicas da cabine;
- houver evidência de que adaptação ao domínio pode melhorar olhos ou boca;
- a integração corpo-rosto em uma arquitetura YOLO tiver valor adicional;
- houver recursos para anotação e revisão independente dos landmarks.

Antes do treinamento, deverão ser congelados:

- topologia dos landmarks;
- dataset de treino, validação e teste;
- política para pseudo-rótulos;
- amostra revisada manualmente;
- arquitetura YOLO26 escolhida;
- resolução;
- seeds;
- early stopping;
- métricas e critérios de checkpoint.

## 25. Estrutura de implementação

```text
experimento_extratores/
├── configs/
│   ├── extractors/
│   ├── landmarks/
│   ├── benchmark/
│   └── temporal/
├── data/
│   ├── annotations/
│   ├── manifests/
│   └── splits/
├── src/
│   ├── extractors/
│   ├── indicators/
│   ├── head_pose/
│   ├── metrics/
│   ├── benchmark/
│   └── temporal/
├── outputs/
│   ├── landmarks/
│   ├── indicators/
│   ├── benchmarks/
│   ├── predictions/
│   ├── tables/
│   └── figures/
├── tests/
├── reports/
└── README.md
```

Cada extrator deverá implementar uma interface comum, por exemplo:

```python
class FaceLandmarkExtractor:
    def load(self): ...
    def infer(self, frame): ...
    def normalize_output(self, result): ...
    def close(self): ...
```

A saída normalizada deverá conter:

```text
video_id, frame_id, timestamp, extractor,
face_bbox, landmarks_xy, landmark_confidence,
face_valid, failure_reason,
ear, mar, pitch, yaw, roll,
indicator_validity, inference_time_ms
```

## 26. Reprodutibilidade

Deverão ser preservados:

- versões das bibliotecas;
- hashes dos modelos e arquivos de configuração;
- ambiente de hardware e software;
- seeds;
- manifests dos frames;
- splits oficiais;
- parâmetros da câmera;
- mapeamentos de landmarks;
- thresholds de confiança;
- logs de execução;
- predições por frame e por janela;
- checkpoints;
- tabelas e gráficos gerados automaticamente;
- hashes SHA-256 dos artefatos oficiais.

Os vídeos originais não deverão ser versionados no Git. Somente manifests, metadados, anotações autorizadas e resultados derivados deverão ser incluídos conforme as regras de privacidade do projeto.

## 27. Etapas de execução

### E1 — Preparação

- congelar extratores da etapa principal;
- verificar licenças;
- definir ambientes e dependências;
- congelar amostra de frames;
- definir landmarks anatômicos comuns.

### E2 — Ground truth

- anotar landmarks manuais;
- executar segunda anotação da subamostra;
- calcular variabilidade da anotação;
- revisar frames ambíguos.

### E3 — Integração

- implementar adaptadores;
- validar mapeamentos visualmente;
- padronizar EAR, MAR e Head Pose;
- criar testes automatizados.

### E4 — Avaliação isolada

- calcular NME;
- calcular taxa de falha;
- avaliar estabilidade temporal;
- analisar EAR, MAR e Head Pose;
- gerar casos qualitativos.

### E5 — Benchmark computacional

- executar CPU e GPU;
- repetir medições;
- registrar latência, FPS e memória;
- gerar tabelas e gráficos.

### E6 — Avaliação downstream

- gerar séries por extrator;
- aplicar os mesmos splits;
- treinar modelos e janelas congelados;
- avaliar métricas por janela e por episódio;
- executar análise pareada.

### E7 — Decisão

- construir fronteiras de Pareto;
- selecionar extratores não dominados;
- decidir sobre experimento YOLO26;
- documentar limitações e recomendação final.

## 28. Riscos e limitações

- pequeno número de vídeos e baixa diversidade de operadores;
- dependência de anotações produzidas por um único avaliador;
- ausência de ground truth angular instrumental para Head Pose;
- diferenças anatômicas entre topologias de landmarks;
- bibliotecas com tracking e inferência frame a frame não totalmente equivalentes;
- thresholds de EAR e MAR não diretamente transferíveis entre extratores;
- diferenças de resolução ou backend;
- licenças distintas para código e pesos;
- modelos whole-body podem dedicar menor resolução efetiva ao rosto;
- adaptação YOLO26 pode introduzir vantagem de domínio não disponível aos modelos prontos;
- resultados não devem ser generalizados para ambientes ou operadores não representados.

## 29. Resultado esperado

O experimento deverá produzir:

1. uma comparação reprodutível de alternativas ao MediaPipe;
2. uma base anotada de landmarks representativos do domínio;
3. métricas geométricas, temporais e computacionais;
4. avaliação do efeito de cada extrator nos modelos temporais;
5. identificação das soluções no melhor compromisso de Pareto;
6. decisão fundamentada sobre manter o MediaPipe, substituí-lo ou avançar para adaptação com YOLO26;
7. material diretamente utilizável na dissertação e em publicação científica.

## 30. Referências técnicas iniciais

- [MediaPipe Face Landmarker — Google AI Edge](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/)
- [InsightFace — projeto oficial](https://github.com/deepinsight/insightface)
- [OpenFace 2.0 — projeto oficial](https://github.com/TadasBaltrusaitis/OpenFace)
- [MMPose — projeto oficial](https://github.com/open-mmlab/mmpose)
- [MMPose — documentação](https://mmpose.readthedocs.io/)
- [OpenSeeFace — projeto oficial](https://github.com/emilianavt/OpenSeeFace)
- [Face Alignment Network](https://github.com/1adrianb/face-alignment)
- [Ultralytics YOLO26-Pose](https://docs.ultralytics.com/tasks/pose/)

