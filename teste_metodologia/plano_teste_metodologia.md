# Plano de Validação da Metodologia: Landmarks+Séries Temporais vs. Vídeo Bruto

## 1. Objetivo

Validar formalmente se a metodologia adotada no protocolo principal da dissertação —
extrair landmarks faciais e convertê-los em 5 séries temporais (EAR, MAR, Pitch, Yaw,
Roll) para depois classificar o estado do operador — generaliza melhor do que treinar
um modelo de deep learning diretamente sobre os frames/vídeo bruto (CNN, 3D-CNN), dado
o volume reduzido de dados próprios (4 vídeos de operação real).

A motivação original da escolha por séries temporais foi justamente a escassez de
dados. Este experimento testa essa premissa de forma controlada.

## 2. Hipótese

Representações compactas e já normalizadas (razões e ângulos, como EAR/MAR/Pitch/Yaw/
Roll) generalizam melhor entre domínios diferentes (câmera frontal de datasets
públicos vs. câmera não frontal de operação real) do que representações em pixels
brutos, que são muito mais sensíveis a pose, iluminação e identidade do sujeito.

## 3. Desenho experimental: few-shot learning via meta-learning

Diferente do few-shot clássico de classificação de imagens (onde cada classe é uma
categoria nova nunca vista), aqui:

- **As classes são fixas e compartilhadas entre todas as tasks**: Alert / Not-Alert
  (esquema binário — ver seção 4).
- **Cada task é um domínio diferente**: um sujeito/vídeo de um dataset público durante
  o meta-treino, ou um dos vídeos reais de teleoperação durante a meta-avaliação.
- **O objetivo é generalizar para um domínio novo** (câmera/sujeito nunca visto) com
  poucos exemplos rotulados por classe — "few-shot domain adaptation", usando a
  técnica de **prototypical networks**.

### Fluxo geral

1. **Meta-treino**: em datasets públicos (NTHU-DDD, UTA-RLDD — câmera frontal, muitos
   sujeitos). Cada sujeito é uma task; a cada episódio, sorteia-se uma task e amostra-se
   um support set (K exemplos por classe) e um query set. O encoder aprende a produzir
   embeddings tais que exemplos da mesma classe fiquem próximos entre tasks diferentes.
2. **Meta-avaliação**: nos 4 vídeos reais de teleoperação (câmera não frontal, o
   domínio-alvo real). Cada vídeo é uma task nova nunca vista durante o meta-treino;
   mede-se a capacidade de classificar corretamente com apenas K exemplos rotulados
   por classe daquele vídeo específico.

### As duas pipelines comparadas

| | Pipeline A (landmarks) | Pipeline B (vídeo bruto) |
|---|---|---|
| Entrada | janela das 5 séries temporais (EAR, MAR, Pitch, Yaw, Roll) | janela de frames/clipe de vídeo |
| Encoder | 1D-CNN ou LSTM/TCN pequeno → embedding | CNN 2D + pooling temporal ou 3D-CNN (provavelmente com backbone pré-treinado, já que dificilmente treina do zero com tão pouco dado) |
| Classificação | protótipo por classe (média dos embeddings do support set) + distância ao protótipo mais próximo | mesma lógica de protótipos |

Um ponto relevante da comparação: a Pipeline B praticamente depende de transfer
learning (backbone pré-treinado) só para ser viável com tão pouco dado — o que já é,
em si, uma evidência indireta a favor da hipótese: se a abordagem "crua" só funciona
emprestando conhecimento externo, e mesmo assim generaliza pior para o domínio-alvo,
isso reforça o argumento de que a abstração em séries temporais é mais robusta à
mudança de domínio (frontal → não frontal).

### Métrica de comparação

Para cada valor de K (ex.: 1-shot, 5-shot, 10-shot por classe), rodar múltiplos
episódios reamostrados (bootstrap) sobre os vídeos próprios como alvo. Como há apenas
4 vídeos, a variância entre episódios tende a ser alta — múltiplas amostragens com
intervalo de confiança são necessárias para qualquer conclusão ser defensável na
dissertação. Métricas: acurácia e F1 por classe, comparadas entre Pipeline A e B sob o
mesmo orçamento de K-shot.

## 4. Esquema de classes

Decidido trabalhar com **esquema binário**: `Alert` vs `Not-Alert` (fundindo Fatigue +
Distraction), em vez das 3 classes originais do protocolo principal (Alert/Fatigue/
Distraction). Isso também suaviza o desbalanceamento entre tasks, já que Alert domina
a maior parte do tempo de operação.

Existe ainda uma terceira categoria nos dados brutos — `Absent` (operador fora de
quadro) — que **não é tratada como classe de classificação**. Frames/janelas rotulados
como `Absent` são excluídos do treino/avaliação (marcados como não-detectados antes do
voto majoritário da janela), já que não representam um estado de alerta do operador,
apenas ausência física.

## 5. Dados próprios: problemas encontrados e como foram resolvidos

### 5.1 Câmera não frontal / baixa taxa de detecção

Teste inicial com MediaPipe no video_01 mostrou taxa de detecção facial de apenas
57,1%, com indicadores muito ruidosos (EAR chegando a 1,0–1,5, valores fisicamente
anômalos; Roll com baseline constante em torno de -130°, sugerindo câmera fisicamente
rotacionada em relação ao rosto; Pitch/Yaw oscilando violentamente).

Ficou definido que nem toda "falha de detecção" é um problema do framework: em dados
de operação real, o operador de fato levanta da cadeira ou sai de quadro. Por isso a
métrica de detecção bruta não é a mais informativa — o que importa é a taxa de
detecção durante presença confirmada do operador (daí a necessidade da anotação manual
de presença/ausência, ver 5.2).

### 5.2 Rótulos: de `legacy_heuristic_state` para `classificacoes_frames_exatos.csv`

A coluna `legacy_heuristic_state` dos CSVs extraídos originalmente (única coluna com
Alert/Fatigue/Distraction disponível de início) se mostrou pouco confiável: ao rodar o
pipeline inicial, ela classificava a maioria das janelas como `Not-Alert`, o oposto do
que a inspeção manual dos vídeos mostra — majoritariamente `Alerta`, com trechos
pontuais de Fadiga/Distração/Ausência.

**Decisão final**: os rótulos agora vêm de `classificacoes_frames_exatos.csv` — um CSV
com o rótulo validado manualmente por frame exato (coluna `state`: alert/fatigue/
distraction/absent), já combinado com os indicadores calculados pelo MediaPipe (ear,
mar, pitch, yaw, roll, face_detected) no mesmo arquivo. Isso eliminou a necessidade de
um parser de timestamps por trecho (a ideia inicial de um módulo
`manual_annotations.py` foi abandonada, já que o rótulo por frame já vem pronto).

### 5.3 Janelas por vídeo/classe (checagem de viabilidade do K-shot)

Rodagem inicial (ainda usando `legacy_heuristic_state`, portanto desatualizada) só
teve 1 dos 4 vídeos com janelas suficientes nas duas classes para `k_shot=5,
q_query=10`. Com a troca para os rótulos de `classificacoes_frames_exatos.csv`, essa
contagem precisa ser refeita com os 4 vídeos reais — é o próximo passo prático antes de
seguir para o encoder.

## 6. Pipeline de dados implementado (código)

- **`episode_sampler.py`** — `Window`, `build_windows`, `index_by_task_class`,
  `EpisodeSampler` (amostrador de episódios N-way K-shot), `merge_labels` +
  `BINARY_LABEL_MAP` (esquema Alert/Not-Alert, com `Absent` mapeado como identidade
  para nunca virar rótulo de janela).
- **`load_own_data.py`** — conversão dos dados próprios para o formato `Window`,
  suportando dois layouts: um único CSV com todos os vídeos juntos
  (`classificacoes_frames_exatos.csv`, colunas `video_id, frame_index,
  timestamp_seconds, state, ear, mar, pitch, yaw, roll, face_detected`) via
  `load_from_single_csv`, ou um CSV por vídeo via `load_all_videos`. O rótulo vem
  diretamente da coluna `state` (já validada manualmente), com `absent` tratado como
  não-detectado (combinado com `face_detected`) antes do voto majoritário da janela.
  Inclui tratamento de frames ausentes no `frame_index` (reindexação) e preenchimento
  de indicadores faltantes por propagação (ffill/bfill).
- **`run_episode_sampler.py`** — script único com variáveis de configuração no topo
  (`SINGLE_CSV_PATH` ou `CSV_PATHS`, `window_len`, `stride`, `min_valid_ratio`, esquema
  binário, `n_way`/`k_shot`/`q_query`) que executa o pipeline completo e imprime o
  diagnóstico de janelas por vídeo/classe.
- **`test_episode_sampler.py`** — suite de testes automatizados (pytest) cobrindo
  shapes dos episódios, não sobreposição entre support/query, reprodutibilidade via
  seed, filtro de `min_valid_ratio`, e o caso de erro por dado insuficiente.

## 7. Próximos passos

**Andamento (2026-09-16):** resultados completos GPU da fase_1 copiados com
verificacao SHA-256 para `results/imported_fase_1/20260916_204539_018100/`.
`run_imported_gpu_smoke.py` prepara dados separados, sem imputacao/overlap e sem
usar features do CSV de rotulos. Com janelas estritas de 90 frames, nenhum video
forma episodio K=1/Q=1; com 15 frames, somente para teste funcional, videos 1/3/4
executaram inferencia CUDA. Nao houve meta-treino ou mudanca do protocolo final.
Ver `VALIDACAO_EXTRACAO.md` para comandos, contagens e relatorios. A auditoria
geometrica dos angulos permanece anterior ao tratamento das lacunas.

0. **Validar a medicao antes de novas janelas/treino.** Executar
   `python teste_metodologia/compare_extraction_backends.py`: mesmos modelos faciais
   e pesos em CPU/CUDA, nos mesmos frames, com referencias MediaPipe com/sem
   tracking. Saidas em `results/backend_comparison/<execucao>/`. O teste nao aplica
   filtro corporal, nao imputa valores e nao substitui os CSVs anteriores.
   Cobertura, paridade numerica e acuracia sao criterios diferentes; concordancia
   com o legado nao demonstra acuracia. Esta amostra por estado e diagnostica,
   nao uma estimativa representativa dos videos inteiros.
   Depois, inspecionar landmarks em trechos independentes e fixar o extrator.
   Preservar NaN/mascaras individuais; definir tratamento de lacunas e calibracao
   sem consultar query. A normalizacao dos indicadores nao elimina automaticamente
   diferencas entre cameras/modelos: isso permanece uma hipotese experimental.

1. Rodar `run_episode_sampler.py` com o CSV real (`classificacoes_frames_exatos.csv`)
   e validar a contagem de janelas por vídeo/classe com os rótulos corretos.
2. Implementar o encoder da Pipeline A (1D-CNN ou LSTM/TCN pequeno) e a prototypical
   network (cálculo de protótipos + classificação por distância).
3. Definir e implementar a Pipeline B (encoder sobre frames/vídeo bruto, com backbone
   pré-treinado) para comparação justa sob o mesmo protocolo de episódios.
4. Rodar o loop de meta-treino nos datasets públicos (NTHU-DDD, UTA-RLDD) para as duas
   pipelines.
5. Meta-avaliação nos 4 vídeos próprios, variando K (1/5/10-shot), com bootstrap para
   intervalo de confiança, comparando Pipeline A vs. Pipeline B.

**Restricoes de validade:** os quatro videos proprios permanecem reservados para
meta-avaliacao, nao meta-treino. Janelas de support/query nao podem compartilhar
frames, mesmo quando seus indices de janela diferem. Usar os mesmos episodios e
criterios de elegibilidade em A/B e relatar a exclusao causada por falhas faciais
por classe; excluir essas falhas silenciosamente pode favorecer a Pipeline A.
Estatisticas de normalizacao e limiares nao podem ser estimados sobre query.

## 8. Limitações conhecidas

- Apenas 4 vídeos próprios — poucas tasks de meta-avaliação, alta variância entre
  episódios; conclusões precisam vir acompanhadas de intervalo de confiança, não de
  números pontuais.
- Overlap de janelas (`stride < window_len`) aumenta o volume de exemplos mas também a
  correlação entre janelas vizinhas — especialmente relevante para as classes raras
  (Fatigue/Distraction dentro de Not-Alert).
- A Pipeline B, mesmo com backbone pré-treinado, enfrenta um domain gap maior
  (aparência de pixels muda muito mais com pose/câmera do que os indicadores já
  normalizados da Pipeline A) — isso é parte do que o experimento busca quantificar,
  não apenas um problema a "resolver".
