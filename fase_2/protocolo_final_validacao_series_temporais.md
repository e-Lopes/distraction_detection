# Protocolo Final de Validação e Comparação de Algoritmos para Séries Temporais

## Classificação de fadiga e distração por indicadores faciais

## 1. Objetivo

Este documento define um protocolo experimental para comparar algoritmos de classificação aplicados a séries temporais formadas por indicadores extraídos de landmarks faciais.

Os sinais básicos considerados são:

- **EAR (Eye Aspect Ratio):** abertura dos olhos;
- **MAR (Mouth Aspect Ratio):** abertura da boca;
- **Pitch:** inclinação vertical da cabeça;
- **Yaw:** rotação lateral da cabeça;
- **Roll:** inclinação lateral da cabeça;
- **validade dos landmarks:** frames válidos, ausentes ou interpolados.

As classes previstas são:

- **Alert**;
- **Fatigue**;
- **Distraction**.

O protocolo deve responder:

1. Qual algoritmo apresenta o melhor desempenho global?
2. Qual reconhece melhor a classe rara **Fatigue**?
3. Qual produz menos falsos alarmes e melhor delimita episódios?
4. Qual é mais estável entre vídeos, folds e seeds?
5. Qual apresenta o melhor compromisso entre desempenho e custo computacional?

O objetivo não é forçar um vencedor absoluto. Quando necessário, devem ser identificados separadamente o melhor modelo global, o mais sensível à fadiga e o melhor compromisso operacional.

---

## 2. O que está sendo comparado

É necessário separar três perguntas experimentais.

### 2.1 Comparação de algoritmos

O extrator facial, as séries de entrada, o pré-processamento, a representação e os folds permanecem fixos. Somente o algoritmo muda.

Exemplos:

- Regressão Logística;
- SVM;
- Random Forest ou gradient boosting;
- LSTM;
- TCN;
- Transformer;
- MiniROCKET.

### 2.2 Comparação de representações

O classificador e o protocolo de validação permanecem fixos, enquanto a representação muda:

- valores brutos de EAR, MAR, Pitch, Yaw e Roll;
- estatísticas por janela;
- indicadores comportamentais, como PERCLOS;
- features de `catch22`, `tsfresh` ou TSFEL;
- transformação ROCKET/MiniROCKET.

### 2.3 Comparação de pipelines completos

Cada família pode utilizar sua melhor representação, janela, balanceamento e hiperparâmetros. Nesse caso, o objeto comparado é:

> extrator + indicadores + representação + algoritmo + threshold + pós-processamento

Essa comparação indica a melhor solução final, mas não permite atribuir a diferença exclusivamente ao algoritmo.

### 2.4 Ordem recomendada

1. Comparar representações usando classificadores simples.
2. Comparar algoritmos sob uma representação comum.
3. Selecionar internamente a melhor configuração de cada família.
4. Comparar os pipelines completos nos mesmos folds externos.

---

## 3. Características das séries temporais

As features devem ser calculadas separadamente para cada canal e, quando aplicável, entre pares de canais.

### 3.1 Sequência bruta

Entrada de dimensão `N × C`, em que `N` é o número de frames da janela e `C` é o número de canais. Essa representação preserva a ordem temporal e é adequada para LSTM, TCN e Transformer.

### 3.2 Features no domínio do tempo

| Grupo | Features candidatas |
|---|---|
| Tendência central | média, mediana e quantis |
| Dispersão | desvio-padrão, variância, amplitude e intervalo interquartil |
| Extremos | mínimo e máximo |
| Forma | assimetria e curtose |
| Tendência | inclinação, diferença inicial-final e regressão linear |
| Velocidade | média, desvio-padrão e máximo de `Δx` |
| Aceleração | estatísticas de `Δ²x` |
| Persistência | autocorrelação em lags previamente definidos |
| Mudanças | cruzamentos da média, mudanças de direção e número de picos |
| Variação total | line length ou soma das diferenças absolutas |

Essas features formam um baseline interpretável para SVM, Regressão Logística, Random Forest e gradient boosting.

### 3.3 Complexidade e frequência

Features candidatas:

- Sample Entropy e Approximate Entropy;
- spectral entropy;
- energia espectral e frequência dominante;
- energia em bandas;
- FFT ou densidade espectral de potência;
- wavelets DWT/CWT;
- Hurst exponent, somente em janelas suficientemente longas.

Entropia, frequência e Hurst podem ser instáveis em janelas curtas. A taxa de amostragem, a duração da janela e o tratamento de dados ausentes precisam ser consistentes.

### 3.4 Relações entre canais

Como a série é multivariada, avaliar:

- correlação EAR–MAR;
- correlação EAR–Pitch;
- correlação MAR–Pitch;
- covariâncias entre canais;
- defasagem entre movimentos;
- sincronização entre fechamento dos olhos, bocejo e queda da cabeça.

### 3.5 Indicadores comportamentais

Features candidatas:

- PERCLOS;
- taxa de piscadas;
- duração média e máxima das piscadas;
- tempo acumulado com olhos fechados;
- número e duração de possíveis bocejos;
- variabilidade de EAR e MAR;
- média, amplitude, velocidade e variabilidade de Pitch, Yaw e Roll;
- frequência e duração do desvio de cabeça.

Features baseadas em thresholds, como PERCLOS, devem compor um baseline adicional. Elas não devem substituir o experimento principal de fronteiras aprendidas pelos dados.

### 3.6 Missingness e qualidade dos landmarks

Para cada janela, registrar:

- proporção de frames válidos;
- proporção de frames ausentes;
- proporção de frames interpolados;
- maior sequência consecutiva sem face;
- número de perdas e recuperações da face;
- confiança média dos landmarks, quando disponível.

Flags de validade podem ajudar, mas também podem criar um atalho, associando ausência facial diretamente a uma classe. Comparar:

1. modelo sem flags;
2. modelo com flags;
3. desempenho estratificado por missingness.

### 3.7 Extração automática

Podem ser usados como baselines:

- **catch22:** conjunto compacto de 22 características;
- **tsfresh:** extração ampla seguida de seleção;
- **TSFEL:** features organizadas por domínios;
- **ROCKET/MiniROCKET:** transformação por convoluções aleatórias com classificador linear.

Esses métodos devem ser tratados como baselines fortes, sem presumir superioridade. Toda seleção automática deve ser ajustada apenas no treinamento de cada fold.

---

## 4. Construção das janelas

Devem ser congelados:

- tamanhos de janela candidatos;
- stride e sobreposição;
- regra de rótulo;
- tratamento de transições;
- limite de missingness;
- tratamento de janelas sem face;
- relação entre frames e segundos.

### 4.1 Tamanhos de janela

Cada configuração deve ser identificada pelo par modelo/janela, por exemplo `SVM/60`, `LSTM/60`, `TCN/90` e `Transformer/150`.

Para cada tamanho, recalcular:

- quantidade de janelas;
- distribuição das classes;
- quantidade de transições;
- missing rate;
- duração em segundos;
- desempenho e custo computacional.

### 4.2 Rótulo da janela

Opções:

- classe do frame central;
- classe do último frame;
- classe majoritária;
- classe presente em proporção mínima;
- descarte de janelas que atravessam transições.

A regra deve ser definida antes da avaliação oficial e aplicada igualmente.

### 4.3 Sobreposição

A sobreposição aumenta o número de janelas, mas não o número de observações independentes. Janelas vizinhas não podem ser distribuídas aleatoriamente entre treino e teste.

---

## 5. Protocolo de validação

### 5.1 Validação externa recomendada

Com quatro vídeos, utilizar **leave-one-video-out**, também chamado leave-one-session-out quando cada vídeo representa uma sessão:

| Fold | Treinamento e validação interna | Teste externo |
|---|---|---|
| 1 | Vídeos 2, 3 e 4 | Vídeo 1 |
| 2 | Vídeos 1, 3 e 4 | Vídeo 2 |
| 3 | Vídeos 1, 2 e 4 | Vídeo 3 |
| 4 | Vídeos 1, 2 e 3 | Vídeo 4 |

Cada vídeo é avaliado uma vez sem participar do treinamento, da escolha de hiperparâmetros ou do threshold daquele fold.

Esse protocolo mede generalização para uma sessão não vista. Se os vídeos forem do mesmo operador, não demonstra generalização para novos operadores.

### 5.2 Quando utilizar LOSO por sujeito

Leave-One-Subject-Out só deve ser usado quando existirem vários operadores identificáveis. Um sujeito inteiro fica fora do treinamento em cada fold. Não se deve chamar leave-one-video-out de LOSO por sujeito.

### 5.3 Validação interna

Dentro dos três vídeos de desenvolvimento de cada fold externo, selecionar:

- representação e features;
- hiperparâmetros;
- tamanho da janela;
- balanceamento;
- épocas, early stopping e checkpoint;
- threshold e pós-processamento.

Pode-se utilizar leave-one-video-out interno ou blocos temporais contínuos quando os vídeos internos não contiverem classes suficientes. O mesmo orçamento de busca deve ser oferecido às famílias comparadas.

### 5.4 Prevenção de vazamento

É obrigatório:

- dividir vídeos ou blocos antes de criar janelas;
- impedir janelas que atravessem partições;
- impedir compartilhamento de frames entre partições;
- usar gap compatível com a maior janela nas divisões do mesmo vídeo;
- ajustar scaler, interpolação aprendida e seleção de features somente no treino;
- aplicar balanceamento e data augmentation somente no treino;
- ajustar calibração, threshold e pós-processamento somente na validação interna;
- impedir uso do teste externo para early stopping ou checkpoint.

### 5.5 Predições out-of-fold

Preservar as predições externas dos quatro vídeos. O conjunto out-of-fold será usado para:

- matriz de confusão consolidada;
- Macro F1 e métricas por classe;
- curvas Precision–Recall;
- formação de episódios;
- métricas temporais.

Os resultados individuais por vídeo também devem ser mantidos.

---

## 6. Balanceamento e treinamento

Estratégias candidatas:

- nenhuma correção;
- pesos de classe;
- weighted sampling;
- focal loss;
- augmentação das séries;
- classificação hierárquica;
- Alert versus Non-Alert como análise complementar.

Regras:

- treino pode ser balanceado; validação e teste preservam a distribuição original;
- o efeito do balanceamento deve ser separado do efeito do algoritmo;
- todos recebem orçamento comparável de ajuste;
- o checkpoint é escolhido na validação interna;
- modelos neurais usam as mesmas seeds e critérios de parada.

---

## 7. Métricas por janela

| Métrica | Função | Papel |
|---|---|---|
| Accuracy | Proporção total de acertos | Complementar |
| Balanced Accuracy | Média do recall das classes | Complementar importante |
| Precision por classe | Previsões positivas corretas | Mede falsos alarmes |
| Recall por classe | Casos reais detectados | Central para Fatigue |
| F1 por classe | Equilíbrio Precision–Recall | Obrigatória |
| Macro F1 | Mesmo peso para todas as classes | Métrica global principal |
| Weighted F1 | Peso pela frequência | Complementar; favorece Alert |
| PR-AUC por classe | Precision–Recall variando threshold | Importante para classe rara |
| ROC-AUC por classe | Separação entre classes | Complementar |
| Matriz de confusão | Tipos de erro | Obrigatória |

### 7.1 Métrica principal

Utilizar **Macro F1 out-of-fold**:

\[
Macro\ F1 = \frac{F1_{Alert} + F1_{Fatigue} + F1_{Distraction}}{3}
\]

Accuracy e Weighted F1 não devem determinar sozinhos o melhor modelo.

### 7.2 Resultados por vídeo

Apresentar:

- Macro F1 por vídeo;
- F1 por classe e vídeo;
- média, desvio-padrão e pior vídeo;
- distribuição de classes por vídeo.

Se uma classe estiver ausente em um vídeo, isso deve ser explicitado; não se deve substituir silenciosamente uma métrica indefinida.

### 7.3 Probabilidades e calibração

Quando houver probabilidades, avaliar opcionalmente:

- log loss;
- Brier score;
- Expected Calibration Error;
- curvas de calibração.

Isso é relevante quando thresholds ou fusão probabilística forem utilizados.

---

## 8. Avaliação temporal e por episódio

### 8.1 Formação dos episódios

Antes da avaliação oficial, congelar:

- threshold probabilístico;
- duração mínima;
- tolerância para gaps;
- suavização;
- cooldown;
- correspondência entre eventos reais e previstos.

### 8.2 Métricas

| Métrica | Interpretação |
|---|---|
| Event Precision | Episódios previstos que correspondem a eventos reais |
| Event Recall | Episódios reais detectados |
| Event F1 | Equilíbrio entre Event Precision e Event Recall |
| IoU temporal | Sobreposição entre intervalo real e previsto |
| FP/h | Falsos episódios por hora |
| Latência | Tempo do início real à primeira detecção |
| Fragmentação | Alertas gerados para o mesmo evento |
| Erro de duração | Diferença entre duração prevista e real |
| Taxa de transições | Oscilações entre estados por tempo |

### 8.3 Correspondência de episódios

Uma regra possível exige classe correta e sobreposição mínima. Se for usado um limite como `IoU >= 0,30`, ele deve ser definido previamente ou selecionado apenas na validação interna. Cada episódio previsto deve corresponder a no máximo um episódio real.

### 8.4 Curva operacional

Enquanto a empresa não definir limite de falsos alarmes, apresentar:

- Event Recall versus FP/h;
- F1 de Fatigue versus FP/h;
- latência versus FP/h;
- desempenho em thresholds definidos na validação interna.

Isso é preferível a inventar um único limite operacional.

---

## 9. Regra para selecionar o melhor modelo

Não somar arbitrariamente métricas com unidades diferentes. Utilizar decisão hierárquica.

### 9.1 Hierarquia

1. **Macro F1 out-of-fold:** desempenho global.
2. **F1 e PR-AUC de Fatigue:** qualidade da classe rara.
3. **Recall de Fatigue:** sensibilidade.
4. **Event F1 e FP/h:** utilidade temporal.
5. **Pior vídeo e variabilidade:** robustez.
6. **Latência, FPS e memória:** desempate prático.

### 9.2 Regra formal

> O melhor modelo global será aquele com maior Macro F1 out-of-fold. Um modelo que não reconheça Fatigue não será considerado solução satisfatória para o objetivo central. Quando os modelos apresentarem desempenho global próximo ou incerteza sobreposta, a escolha será feita, nesta ordem, por F1/PR-AUC de Fatigue, Event F1, menor FP/h, maior estabilidade e menor custo computacional.

### 9.3 Relevância prática

Antes dos resultados, definir uma margem mínima `δ` para diferença relevante de Macro F1, por exemplo `0,01` ou `0,02`.

Se a diferença for inferior a `δ`, tratar os modelos como praticamente equivalentes no desempenho global e aplicar os critérios seguintes. O valor de `δ` deve ser justificado e congelado antes da comparação oficial.

### 9.4 Fronteira de Pareto

Na ausência de requisitos operacionais, apresentar a fronteira de Pareto considerando:

- Macro F1;
- F1 ou PR-AUC de Fatigue;
- FP/h;
- latência.

Um modelo domina outro quando é igual ou melhor em todos os critérios e estritamente melhor em pelo menos um.

---

## 10. Thresholds e pós-processamento

Em cada fold externo:

1. treinar nos dados internos;
2. gerar probabilidades na validação interna;
3. selecionar threshold e pós-processamento apenas nela;
4. congelar as decisões;
5. aplicar uma vez ao vídeo externo;
6. armazenar as predições sem reajuste.

Possíveis objetivos internos:

- maximizar Macro F1;
- maximizar F1 de Fatigue;
- maximizar Recall sob limite de FP/h;
- minimizar FP/h sob Recall mínimo;
- minimizar um custo definido pela aplicação.

Todos os algoritmos devem utilizar o mesmo objetivo de seleção.

---

## 11. Estabilidade e seeds

Para modelos estocásticos:

- utilizar no mínimo 5 seeds;
- preferencialmente 10, se viável;
- usar as mesmas seeds entre algoritmos;
- reportar média, desvio-padrão e pior resultado;
- preservar checkpoints e predições por seed.

Modelos determinísticos podem ser executados uma vez se o determinismo for verificado. Seeds medem variação de otimização, mas não criam novos vídeos nem novas unidades independentes.

---

## 12. Comparação estatística

### 12.1 Unidade independente

Frames e janelas sobrepostas não são observações independentes. A análise deve respeitar:

- vídeos ou sessões;
- episódios;
- blocos temporais contínuos.

### 12.2 Procedimento recomendado com quatro vídeos

Para cada par de modelos:

1. calcular a diferença por vídeo;
2. informar em quantos vídeos cada modelo venceu;
3. calcular efeito absoluto e relativo;
4. estimar intervalo de confiança por bootstrap hierárquico ou block bootstrap;
5. repetir a análise por episódio;
6. interpretar conjuntamente magnitude, consistência e incerteza.

O bootstrap deve reamostrar vídeos, episódios ou blocos contínuos, nunca janelas como se fossem independentes.

### 12.3 Testes que não devem ser superinterpretados

Com quatro vídeos independentes:

- teste t pareado tem pressupostos difíceis de verificar;
- Wilcoxon possui poder muito baixo;
- Friedman e Nemenyi não sustentam comparação forte;
- um diagrama de diferença crítica não acrescenta evidência robusta.

Esses métodos ficam mais apropriados com muitos sujeitos, sessões independentes ou datasets. Para o conjunto atual, são preferíveis resultados completos por vídeo, deltas pareados, intervalos que preservem a dependência temporal e análise de sensibilidade.

### 12.4 Linguagem de conclusão

Usar:

- “apresentou o maior resultado médio”;
- “venceu em três dos quatro vídeos”;
- “obteve menor variabilidade”;
- “a diferença foi inferior à margem prática”;
- “não foi possível estabelecer superioridade conclusiva”.

Evitar afirmar superioridade estatística sem evidência suficiente.

---

## 13. Robustez e ablação

Avaliar por:

- vídeo e classe;
- janela e stride;
- threshold;
- iluminação;
- orientação facial;
- PPE e oclusões;
- taxa de detecção facial;
- missingness e interpolação;
- duração dos episódios;
- transições entre classes.

Estratificar missingness, quando possível, em `0–10%`, `10–30%`, `30–50%` e `>50%`.

Realizar ablação:

1. EAR e MAR;
2. Head Pose;
3. EAR + MAR + Head Pose;
4. conjunto completo com features derivadas;
5. conjunto completo com flags de validade.

---

## 14. Custo computacional

Registrar:

- tempo de treinamento e por época;
- latência mediana e percentil 95 por janela;
- throughput e FPS;
- CPU, RAM, GPU e VRAM máximos;
- parâmetros e tamanho do modelo;
- hardware, precisão numérica e batch size.

As medições devem utilizar o mesmo hardware e aquecimento.

Separar:

1. custo do extrator de landmarks;
2. custo da geração dos indicadores;
3. custo do classificador;
4. custo fim a fim.

---

## 15. Desenho experimental final

### Etapa A — Auditoria

- verificar vídeos, FPS e timestamps;
- confirmar ground truth;
- calcular distribuição por frame;
- identificar episódios e transições;
- calcular missingness por vídeo e classe;
- confirmar quantos operadores existem.

### Etapa B — Congelamento das janelas

- definir tamanhos, stride e rótulo;
- impedir vazamento nas fronteiras;
- recalcular distribuição para cada janela.

### Etapa C — Baselines de representação

- sequência bruta;
- estatísticas temporais;
- indicadores comportamentais;
- catch22 ou equivalente reduzido;
- MiniROCKET.

### Etapa D — Comparação controlada

- mesma representação e folds;
- mesmo orçamento de hiperparâmetros;
- comparação de modelos clássicos e temporais em condições compatíveis.

### Etapa E — Melhor pipeline por família

- selecionar internamente janela, representação, hiperparâmetros e balanceamento;
- selecionar threshold e pós-processamento;
- avaliar uma vez no fold externo.

### Etapa F — Avaliação final

- consolidar predições out-of-fold;
- calcular métricas por janela e episódio;
- analisar robustez e incerteza;
- medir custo computacional;
- gerar fronteira de Pareto;
- declarar vencedor somente quando sustentado pelos critérios congelados.

---

## 16. Tabelas finais

### 16.1 Desempenho por janela

| Pipeline | Janela | Macro F1 OOF | Balanced Acc. | F1 Alert | F1 Fatigue | F1 Distraction | PR-AUC Fatigue |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM | — | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — | — |
| TCN | — | — | — | — | — | — | — |
| Transformer | — | — | — | — | — | — | — |
| MiniROCKET | — | — | — | — | — | — | — |

### 16.2 Avaliação temporal

| Pipeline | Event Precision | Event Recall | Event F1 | FP/h | Latência mediana | Latência P90 | Fragmentação |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM | — | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — | — |
| TCN | — | — | — | — | — | — | — |
| Transformer | — | — | — | — | — | — | — |
| MiniROCKET | — | — | — | — | — | — | — |

### 16.3 Estabilidade e custo

| Pipeline | Pior vídeo | Vitórias/4 | Desvio entre folds | Latência P50/P95 | FPS | RAM | VRAM | Tamanho |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | — | — | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — | — | — |
| TCN | — | — | — | — | — | — | — | — |
| Transformer | — | — | — | — | — | — | — | — |
| MiniROCKET | — | — | — | — | — | — | — | — |

---

## 17. Gráficos finais

1. Matriz de confusão out-of-fold.
2. Curvas Precision–Recall por classe.
3. Macro F1 por pipeline e janela.
4. F1/PR-AUC de Fatigue por pipeline.
5. Resultado por vídeo.
6. Distribuição entre seeds.
7. Linha temporal real versus prevista.
8. Event Recall versus FP/h.
9. Macro F1 versus FP/h.
10. Latência versus desempenho.
11. Desempenho por missingness.
12. Ablação dos indicadores.
13. Fronteira de Pareto.

---

## 18. Checklist de validade

- [ ] A pergunta está identificada como comparação de algoritmo, representação ou pipeline.
- [ ] Todos os modelos usam os mesmos folds externos.
- [ ] Nenhum frame aparece em treino e teste no mesmo fold.
- [ ] Nenhuma janela atravessa uma fronteira.
- [ ] Scalers e seletores são ajustados somente no treino.
- [ ] Interpolação não utiliza informação futura do teste.
- [ ] Balanceamento e augmentação ocorrem somente no treino.
- [ ] O teste externo não seleciona hiperparâmetros, épocas ou checkpoints.
- [ ] Threshold e pós-processamento são definidos na validação interna.
- [ ] A mesma regra de seleção é usada para todos.
- [ ] Seeds são compartilhadas e registradas.
- [ ] Predições out-of-fold são preservadas.
- [ ] São reportadas métricas por classe, não apenas Accuracy.
- [ ] Há avaliação por episódio e FP/h.
- [ ] A incerteza respeita vídeos e dependência temporal.
- [ ] O custo é medido no mesmo hardware.
- [ ] Falhas e resultados ausentes são documentados.

---

## 19. Critério congelado recomendado

> **O Macro F1 out-of-fold será a métrica principal de desempenho global. O modelo também deverá demonstrar capacidade de reconhecer Fatigue, avaliada por F1, PR-AUC e Recall da classe. Se a diferença de Macro F1 entre os melhores modelos for inferior à margem prática `δ` ou apresentar incerteza amplamente sobreposta, serão usados Event F1, falsos episódios por hora, desempenho no pior vídeo e custo computacional, nesta ordem. Thresholds, hiperparâmetros e pós-processamento serão selecionados exclusivamente na validação interna de cada fold.**

Enquanto a empresa não definir limites operacionais, os resultados também serão apresentados por fronteira de Pareto e curvas Event Recall versus FP/h.

---

## 20. Forma recomendada de conclusão

Exemplo quando modelos vencem em critérios diferentes:

> A SVM apresentou o maior desempenho global e o menor custo computacional, mas capacidade limitada de reconhecer Fatigue. A LSTM apresentou o melhor compromisso entre Macro F1, F1 de Fatigue e falsos episódios por hora. A TCN alcançou o maior Recall de Fatigue, porém produziu uma taxa elevada de alarmes falsos. Não foi observada superioridade absoluta de um único algoritmo.

Exemplo quando os resultados são próximos:

> Embora o modelo A tenha obtido o maior Macro F1 médio, sua diferença para o modelo B foi inferior à margem prática e apresentou ampla incerteza. Os modelos foram considerados equivalentes em desempenho global, sendo o modelo B selecionado por apresentar menos falsos episódios e menor custo computacional.

---

## 21. Referências metodológicas úteis

- [Scikit-learn — Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Bake Off Redux: avaliação de algoritmos de classificação de séries temporais](https://arxiv.org/abs/2304.13029)
- [UEA Multivariate Time Series Classification Archive](https://arxiv.org/abs/1811.00075)

