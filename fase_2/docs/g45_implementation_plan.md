# Protocolo congelado de implementação da G4.5

**Status:** aprovado e congelado antes da geração de resultados

**Data do congelamento:** 14/08/2026

**Escopo:** diagnóstico, ajuste de decisão e intervenções dirigidas à classe `Fatigue`

**Restrição:** este documento não autoriza o início da G5 nem o uso do teste externo

Este documento preserva o plano aprovado para a G4.5. Ele deve ser tratado como protocolo
pré-resultados: objetivos, dados, modelos, métricas, regras de seleção e critérios de sucesso não
podem ser alterados depois da inspeção dos resultados sem uma emenda explícita, datada e
justificada. A G4.5 usa somente a seed `42`; as cinco seeds dos finalistas pertencem à G5.

## 1. Motivação e objetivo

A G4 mostrou que as estratégias de desbalanceamento recuperaram parcialmente `Fatigue`, mas
não produziram uma solução satisfatória. `LSTM/B` apresentou o compromisso temporal mais
equilibrado, `TCN/C` obteve maior sensibilidade com custo operacional elevado e o controle
clássico continuou competitivo. A G4.5 investiga, sem misturar efeitos:

1. se o limiar de decisão suprime sistematicamente a classe rara;
2. se Focal Loss melhora o aprendizado dos exemplos difíceis;
3. se decompor a tarefa em classificação hierárquica melhora a separação entre `Fatigue` e
   `Distraction`;
4. se eventuais ganhos são consistentes entre folds, eventos e sessões, sem selecionar pelo
   melhor resultado isolado.

## 2. Auditoria congelada antes dos resultados

### 2.1 Integridade dos artefatos da G4

- O manifesto da G4 contém 477 entradas e teve todos os hashes verificados.
- O relatório oficial está em `fase_2/reports/g4_imbalance_results.md`.
- `LSTM/B`, `TCN/B` e `TCN/C` possuem, cada um, 4.629 predições de validação e as colunas
  `prob_alert`, `prob_fatigue` e `prob_distraction`.
- A ordem canônica das classes é `alert`, `fatigue`, `distraction`.
- As somas das probabilidades apresentam pequenas diferenças em torno de 1 decorrentes de AMP;
  por isso, serão renormalizadas antes da análise.
- As predições SVM da G4 não armazenam probabilidades. Os checkpoints usam
  `SVC(probability=False)` e oferecem apenas margens de `decision_function`, que não serão
  reinterpretadas como probabilidades.
- Consequentemente, o ajuste de threshold da seção 4 é limitado a `LSTM/B`, `TCN/B` e `TCN/C`.
  O SVM permanece no experimento hierárquico, treinado com probabilidades apropriadas.

### 2.2 Disponibilidade da classe Fatigue

Para R0 com janela de 60 frames, a auditoria encontrou:

| Fold | Treino (`Alert/Fatigue/Distraction`) | Eventos Fatigue no treino | Validação (`Alert/Fatigue/Distraction`) | Eventos Fatigue avaliáveis na validação |
|---|---:|---:|---:|---:|
| 1 | 2.832 / 14 / 381 | 2 (8 + 6 janelas) | 1.014 / 4 / 162 | 1 (4 janelas) |
| 2 | 2.282 / 68 / 553 | 4 (36 + 18 + 8 + 6) | 1.080 / 17 / 76 | 2 (13 + 4 janelas) |
| 3 | 2.589 / 54 / 658 | 2 (36 + 18 janelas) | 986 / 13 / 152 | 1 (13 janelas) |
| 4 | 2.965 / 68 / 688 | 4 (36 + 18 + 8 + 6) | 1.012 / 17 / 96 | 2 (13 + 4 janelas) |

Existem somente dois eventos únicos de `Fatigue` que geram janelas avaliáveis: um evento de
`video_01`, com 13 janelas, e um de `video_03`, com 4 janelas. Um evento anotado de 34 frames em
`video_03` não gera janela-alvo e será tratado como estruturalmente não avaliável. Resultados por
janela não podem ser apresentados como se fossem evidência de muitos eventos independentes.

### 2.3 Duplicação entre folds e decisão de cross-fit

Os blocos de validação se repetem entre folds. Foram encontradas as seguintes quantidades de
janelas exatamente duplicadas:

| Pares de folds | Janelas duplicadas |
|---|---:|
| 1–2 | 810 |
| 1–3 | 788 |
| 1–4 | 762 |
| 2–3 | 781 |
| 2–4 | 755 |
| 3–4 | 733 |

Portanto, calibrar o threshold nos outros três folds e aplicá-lo ao fold retido causaria
vazamento das mesmas janelas. A unidade de cross-fit congelada é a **sessão/vídeo**, não o fold.
O teste externo permanece totalmente excluído de calibração, seleção e análise exploratória.

## 3. Dados, identidade e normalização das predições

- Usar R0 e as janelas já selecionadas na G4: 60 frames para LSTM, TCN e SVM.
- Usar os mesmos quatro folds, seed `42`, splits, orçamento, hiperparâmetros e early stopping da
  G4, salvo a intervenção explicitamente testada.
- Criar um inventário normalizado de validação com `run_id`, `model`, `strategy`, `fold`, `seed`,
  `video_id`, `start_frame`, `end_frame`, classe real, classe prevista, probabilidades, configuração
  resolvida, fingerprint e origem do artefato.
- Definir `window_id` como `video_id:start_frame:end_frame`.
- Quando a mesma janela aparecer em mais de um fold, calcular a média das probabilidades e
  renormalizá-las para soma 1 antes do cross-fit por vídeo.
- Preservar a rastreabilidade até os arquivos e runs originais; não sobrescrever artefatos da G4.

## 4. G4.5A — ajuste de threshold com cross-fit por sessão

### 4.1 Modelos

- `LSTM/B`, janela 60;
- `TCN/B`, janela 60;
- `TCN/C`, janela 60.

O SVM não participa desta etapa, pois suas margens não são probabilidades calibradas. O threshold
não será combinado com Focal Loss ou classificação hierárquica nesta geração, para manter cada
efeito isolado.

### 4.2 Procedimento

Para cada modelo e cada vídeo-alvo:

1. deduplicar as janelas repetidas conforme a seção 3;
2. usar os outros três vídeos como conjunto de calibração;
3. testar `threshold_fatigue` de `0,05` a `0,50`, inclusive, em passos de `0,01`;
4. prever `Fatigue` quando `prob_fatigue >= threshold_fatigue`; nos demais casos, escolher a
   maior probabilidade entre `Alert` e `Distraction`;
5. escolher o threshold que maximiza a média de Macro F1 dando peso igual a cada um dos três
   vídeos de calibração, e não a cada janela;
6. aplicar o threshold escolhido uma única vez ao vídeo retido;
7. repetir até que cada vídeo tenha sido retido uma vez;
8. reconstruir os resultados por fold usando o threshold do vídeo correspondente a cada janela.

### 4.3 Desempate determinístico

Em caso de empate, usar sucessivamente:

1. maior F1 de `Fatigue`;
2. maior precision de `Fatigue`;
3. threshold mais próximo de `1/3`;
4. maior threshold.

Registrar para cada vídeo a curva completa de calibração, o threshold escolhido e as métricas no
vídeo retido. Nenhuma informação do vídeo-alvo pode participar da escolha de seu threshold.

## 5. Diagnóstico dirigido à Fatigue

Executar o diagnóstico sobre as predições de validação, preservando resultados por vídeo, fold e
evento:

- comparar distribuições de probabilidades e sinais com mediana, intervalo interquartil e
  Cliff's delta;
- usar os thresholds históricos `EAR < 0,25` e `MAR > 0,55` apenas de forma descritiva, nunca
  como regra ajustada ou prova confirmatória;
- normalizar cada evento em 10 bins temporais para comparar sua evolução;
- calcular inclinação e velocidade de pitch;
- calcular correlações de Spearman, deixando explícita a dependência entre janelas sobrepostas;
- exibir distribuições de `prob_fatigue` por classe real, vídeo, fold e modelo;
- relacionar falsos positivos e falsos negativos aos sinais, missingness e posição temporal.

Para avaliação por evento, mesclar predições consecutivas ou sobrepostas de `Fatigue` em episódios
e realizar pareamento um-para-um por sobreposição com eventos reais. Reportar precision, recall e
F1 de evento. O F1 principal de evento considera somente eventos avaliáveis; uma análise de
sensibilidade adicional inclui o evento curto estruturalmente não avaliável e o identifica como
tal.

## 6. G4.5B — Focal Loss

### 6.1 Matriz

- `LSTM/60`, R0, quatro folds, seed `42`;
- `TCN/60`, R0, quatro folds, seed `42`.

Total: **8 runs**. Não usar weighted sampling, augmentation, threshold ajustado ou outra técnica
de desbalanceamento em conjunto com Focal Loss.

### 6.2 Definição congelada

- `gamma = 2,0`;
- `alpha_c = N / (3 N_c)`, calculado somente no treino de cada fold;
- ordem de classes: `alert`, `fatigue`, `distraction`;
- não aplicar normalização adicional aos alphas;
- calcular `log_softmax` em `float32`, mesmo sob AMP;
- redução: soma da focal loss ponderada dividida pela soma de `alpha_y` do batch.

Os hiperparâmetros, orçamento de épocas, scheduler, early stopping e critérios de checkpoint da G4
permanecem congelados. Cada run recebe fingerprint novo, mantendo registrados os fingerprints dos
dados e splits de origem.

## 7. G4.5C — classificação hierárquica

### 7.1 Estrutura

- **Nível 1:** `Alert` versus `Non-Alert`;
- **Nível 2:** `Fatigue` versus `Distraction`, treinado apenas nas amostras `Non-Alert` do treino.

Calcular pesos balanceados independentemente em cada nível usando `N / (2 N_c)`, somente com o
treino do fold. A composição final deve produzir probabilidades na ordem canônica:

- `P(Alert) = P_1(Alert)`;
- `P(Fatigue) = P_1(Non-Alert) × P_2(Fatigue)`;
- `P(Distraction) = P_1(Non-Alert) × P_2(Distraction)`.

Renormalizar apenas para corrigir erro numérico. Não ajustar threshold adicional.

### 7.2 Modelos

**SVM/60:** dois classificadores lineares independentes, `C = 1`, `probability=True` e
`random_state=42`. A calibração interna necessária ao `SVC` deve usar somente dados de treino.

**LSTM/60:** dois modelos binários independentes, com os mesmos hiperparâmetros, orçamento e
early stopping da LSTM da G4. Cada nível terá histórico, configuração, checkpoint e identificação
próprios.

Total: **8 pipelines hierárquicos** — dois modelos por quatro folds — e **16 estimadores** nesses
pipelines.

## 8. Matriz oficial e identificação dos runs

A G4.5 contém **16 pipelines oficiais**:

| Intervenção | Modelo | Janela | Folds | Pipelines | Estimadores treinados |
|---|---|---:|---:|---:|---:|
| Focal Loss | LSTM | 60 | 4 | 4 | 4 |
| Focal Loss | TCN | 60 | 4 | 4 | 4 |
| Hierárquica | SVM | 60 | 4 | 4 | 8 |
| Hierárquica | LSTM | 60 | 4 | 4 | 8 |
| **Total** |  |  |  | **16** | **24** |

O ajuste de threshold é uma reavaliação cross-fit de artefatos existentes e não adiciona treino à
matriz. Usar os IDs:

```text
G45__focal_r0_w60__{model}__r0__w60__fold_{n}__seed_42
G45__hierarchical_b_w60__{model}__r0__w60__fold_{n}__seed_42
```

Os artefatos hierárquicos de cada nível recebem os sufixos `__level_1` e `__level_2`.

Antes da matriz completa, executar quatro smoke tests isolados: LSTM focal, TCN focal, SVM
hierárquico e LSTM hierárquica. Smokes não entram nos resultados oficiais.

## 9. Testes obrigatórios antes da execução oficial

### 9.1 Threshold e auditoria

- detectar e deduplicar `window_id` repetido;
- renormalizar probabilidades e preservar a ordem canônica;
- provar que o vídeo-alvo não participa da calibração;
- testar grade inclusiva de `0,05` a `0,50` e todos os desempates;
- validar reconstrução por fold a partir de thresholds por vídeo;
- falhar se qualquer artefato de teste externo for carregado;
- testar pareamento um-para-um de episódios e evento curto não avaliável.

### 9.2 Focal Loss

- equivalência com cross-entropy ponderada quando `gamma = 0`;
- menor contribuição relativa de exemplo fácil e maior de exemplo difícil;
- cálculo train-only de alpha e ordem correta das classes;
- estabilidade numérica em AMP/`float32` e saídas finitas;
- redução ponderada conforme a definição congelada;
- serialização, reload e resume com loss e configuração resolvida.

### 9.3 Hierarquia

- mapeamento correto dos rótulos do nível 1;
- filtro train-only de `Non-Alert` no nível 2;
- pesos binários calculados separadamente por nível;
- probabilidades finais finitas, não negativas, somando 1 e na ordem canônica;
- comportamento explícito quando faltar classe em um nível;
- salvamento e recarga reproduzindo predições;
- checkpoints, logs e fingerprints independentes por nível.

## 10. Métricas, gráficos e artefatos

Para cada configuração e fold, preservar:

- matriz de confusão;
- precision, recall e F1 por classe;
- Macro F1 e balanced accuracy;
- distribuição das previsões e probabilidades;
- falsos episódios de `Fatigue` por hora;
- métricas por vídeo e por evento;
- melhor época, época de parada, training/validation loss, tempo e checkpoints;
- configuração resolvida, seed, fingerprints e origem dos dados/splits.

Gerar, no mínimo:

- curvas de loss e Macro F1 por época;
- curvas de calibração do threshold por vídeo;
- box/violin e ECDF de `prob_fatigue` por classe;
- matrizes de confusão por fold e agregadas;
- comparação de métricas por fold com os baselines G4;
- linha temporal dos eventos e episódios previstos;
- gráfico de precisão versus recall de `Fatigue` e falsos episódios por hora;
- visualização dos 10 bins temporais e das tendências de pitch.

O relatório consolidado será salvo em
`fase_2/outputs/metrics/G4_5/g45_report.md`. Checkpoints, logs, predições, figuras, tabelas,
configurações resolvidas e manifesto com hashes devem permanecer sob a árvore G4.5, sem
sobrescrever gerações anteriores.

## 11. Comparações e critérios de sucesso

Cada intervenção deve ser comparada ao seu controle G4 compatível, usando somente validação. Uma
configuração somente será promovida como finalista se cumprir simultaneamente:

1. ganho absoluto de F1 de `Fatigue` de pelo menos `0,05`;
2. delta positivo de F1 de `Fatigue` em pelo menos 3 dos 4 folds;
3. queda de Macro F1 não superior a `0,02`;
4. detecção dos dois eventos únicos avaliáveis de `Fatigue`;
5. ausência de explosão de falsos positivos/episódios em relação ao controle.

O item 5 deve ser julgado com a distribuição por vídeo, falsos episódios por hora e curva de
precision/recall, e não por um único número agregado. Se nenhuma configuração cumprir todos os
critérios, não forçar um vencedor: apresentar a fronteira de Pareto entre Macro F1, F1/recall de
`Fatigue` e custo de falsos episódios.

Nenhuma decisão será tomada pelo melhor fold, melhor evento ou melhor execução isolada. Os
resultados desta geração são qualificatórios porque usam apenas uma seed. Confirmação de H3 exige
a repetição dos finalistas nas seeds `42`, `123`, `456`, `789` e `2026`, com early stopping
independente, na G5.

## 12. Limites e proibições

- Não acessar nem avaliar o teste externo durante a G4.5.
- Não iniciar a G5 nem executar múltiplas seeds nesta etapa.
- Não combinar threshold, Focal Loss e hierarquia no experimento principal.
- Não modificar dados, splits, folds ou artefatos das gerações anteriores.
- Não interpretar margens SVM como probabilidades.
- Não confirmar H3 com a seed 42.
- Não ocultar a baixa quantidade de eventos independentes ou a não avaliabilidade do evento
  curto.
- Qualquer desvio deste protocolo requer emenda prévia, explícita, datada e versionada.

## 13. Ordem futura de implementação

1. implementar e testar o inventário/deduplicação das predições;
2. implementar e testar o cross-fit por sessão e o diagnóstico por evento;
3. implementar e testar Focal Loss;
4. implementar e testar a classificação hierárquica;
5. executar os quatro smoke tests;
6. congelar as configurações resolvidas e imprimir a matriz oficial;
7. executar os 16 pipelines oficiais da seed 42;
8. consolidar métricas, gráficos, manifesto e relatório G4.5;
9. aplicar os critérios de sucesso e registrar a decisão antes de qualquer G5.

Este registro apenas preserva o plano para implementação futura. Nenhum item desta ordem foi
iniciado pela criação deste documento.
