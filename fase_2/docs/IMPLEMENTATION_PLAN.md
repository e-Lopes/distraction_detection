# Plano incremental de implementação do pipeline de ML

Este documento compara o plano de pesquisa proposto em agosto de 2026 com o estado real da
Fase 2. A implementação seguirá por marcos verificáveis; não será criada uma matriz completa de
experimentos antes de estabilizar dados, splits e baselines.

## Estado atual

Já estão implementados e executados com os quatro vídeos reais:

- manifesto, normalização e validação das anotações;
- extração canônica de EAR, MAR, pitch, yaw, roll e `face_detected` por frame;
- diagnóstico de classes, falhas de detecção e sequências ausentes;
- janelas configuráveis de 30, 60 e 150 frames com regra de maioria mínima;
- quatro folds externos leave-one-video-out;
- validação por blocos contínuos nos vídeos de desenvolvimento, com purge gap bilateral;
- testes automatizados contra sobreposição de janelas e frames entre subconjuntos;
- Dummy, SVM, Random Forest e XGBoost com métricas por fold e classe;
- ablações entre indicadores faciais e atributos de missingness;
- checkpoints retomáveis, predições locais, tabelas e figuras dos baselines clássicos;
- grid search clássico selecionado exclusivamente pela validação.

## Decisões metodológicas

1. O problema principal permanece multiclasse: `alert`, `fatigue` e `distraction`.
2. Macro F1 é a métrica primária; balanced accuracy e métricas por classe são obrigatórias.
3. O vídeo externo de teste de cada fold não participa de scaler, escolha de atributos,
   hiperparâmetros, imputação, balanceamento ou checkpoint.
4. A busca atual é uma avaliação externa aninhada: cada fold escolhe seus parâmetros usando
   apenas os respectivos dados de treino e validação. Uma configuração final única será definida
   somente após consolidar esse procedimento, sem escolher pelo melhor resultado de teste.
5. Ausência natural de uma classe no teste será registrada como métrica não estimável. Também
   será mantida uma Macro F1 com três classes para comparabilidade, claramente identificada.
6. Missingness é um possível atalho de aprendizado. Resultados com e sem seus indicadores devem
   sempre aparecer lado a lado.

## Arquitetura proposta

A estrutura atual em `fase_2/` será preservada e ampliada, evitando uma migração destrutiva:

```text
fase_2/
├── configs/                 dados, splits, pré-processamento e experimentos
├── data/manifests/          metadados e splits versionáveis
├── src/data/                validação, janelas e splits
├── src/preprocessing/       imputação, flags e scaling
├── src/training/            baselines, tuning e motor de treino
├── src/models/              LSTM, TCN e modelos adicionais
├── src/evaluation/          métricas, erros e comparação estatística
├── outputs/                 métricas e figuras versionáveis; caches locais ignorados
├── docs/                    protocolo, decisões e roteiro
└── tests/                   integridade, vazamento, modelos e checkpoints
```

## Próximos marcos

### Marco 1 — Consolidar os baselines clássicos

- concluir e revisar o grid search de SVM, Random Forest e XGBoost;
- salvar ranking completo, vencedores por fold e gráficos comparativos;
- adicionar Regressão Logística como baseline linear interpretável;
- medir tempo de inferência, tamanho dos checkpoints e throughput;
- exportar erros por janela e timelines de predição por vídeo.

### Marco 2 — Pré-processamento como variável experimental

- [x] implementar zero-fill, interpolação apenas de gaps curtos e flags de validade;
- [x] ajustar qualquer transformação somente no treino de cada fold;
- [x] comparar imputação, atributos faciais e atributos de missingness;
- testar atributos derivados (deltas e estatísticas) em grupos configuráveis;
- [x] avaliar janelas de 30, 60 e 150 frames antes de ampliar a lista.

### Marco 3 — Motor temporal em PyTorch

- criar Dataset/DataLoader e interface comum de modelos;
- implementar primeiro LSTM e TCN; manter Transformer como comparação posterior;
- suportar CUDA, AMP, gradient clipping, early stopping e scheduler;
- salvar `best`, `last` e checkpoints periódicos com estado completo;
- retomar treinamento interrompido e registrar curvas por época.

### Marco 4 — Otimização e robustez

- usar grade controlada para fatores científicos discretos;
- usar Optuna ou random search para hiperparâmetros contínuos dos modelos temporais;
- repetir finalistas em múltiplas seeds e reportar média, desvio e intervalos;
- comparar balanceamento, augmentation e ablações sem modificar validação/teste;
- produzir análise de erros, importância, timelines e comparações estatísticas pareadas.

### Marco 5 — Seleção e teste final

- congelar uma configuração e um critério de seleção antes da avaliação final;
- manter um conjunto externo adicional intocado, caso novos vídeos sejam disponibilizados;
- executar a avaliação final uma única vez e marcar seus artefatos explicitamente;
- gerar tabelas e figuras da dissertação somente a partir de resultados rastreáveis.

## Riscos e limites atuais

- Quatro vídeos fornecem poucos grupos independentes e limitam a potência dos testes
  estatísticos. Intervalos e conclusões devem refletir essa incerteza.
- Leave-one-video-out mede generalização para vídeos/sessões não vistos. Só mede generalização
  para operadores não vistos se os vídeos realmente representarem operadores independentes.
- Fadiga está ausente ou quase ausente em alguns vídeos externos; accuracy isolada seria
  enganosa e algumas métricas por classe não são estimáveis nesses folds.
- A falha de detecção facial é correlacionada com o alvo e pode virar um atalho. Ela é relevante
  operacionalmente, mas não deve ser confundida com evidência fisiológica de fadiga.
- Transformer provavelmente tem variância alta neste volume de dados. LSTM e TCN são os modelos
  temporais prioritários; o Transformer só avança se houver controle de capacidade e evidência.
- Uma matriz irrestrita de modelos × janelas × imputações × seeds é cara e favorece seleção por
  acaso. Os experimentos serão filtrados por estágios e hipóteses registradas.

## Critérios de aceite

Cada marco só é concluído quando código, configuração, testes, documentação, métricas e figuras
podem ser regenerados. Resultados não podem depender de caminhos absolutos, dados sintéticos não
identificados ou decisões tomadas após observar o teste.
