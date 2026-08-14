# Plano incremental de implementacao da fase 2

Este roteiro operacionaliza o plano integrado sobre a implementacao existente. Cada etapa e
pequena, verificavel e preserva os artefatos anteriores. A matriz completa de aderencia esta em
[`integrated_plan_gap_analysis.md`](integrated_plan_gap_analysis.md).

## Regras de execucao

- O teste externo de cada fold nunca participa de selecao ou ajuste.
- Configuracoes e runs pertencem a uma geracao explicita: G0 a G6.
- Resultados exploratorios nao sao misturados aos resultados oficiais.
- Nenhum treinamento longo ocorre antes de testes, dry-run e smoke test.
- Checkpoints, predicoes e logs permanecem locais e ignorados pelo Git.
- Configuracoes finalistas usam as seeds `42, 123, 456, 789, 2026` e sao reportadas como
  media mais ou menos desvio-padrao, com estatisticas complementares.

## Etapa 0 - auditoria e preservacao

- **Objetivo:** congelar o entendimento do estado atual e impedir perda/mistura de resultados.
- **Arquivos:** `docs/integrated_plan_gap_analysis.md`, este plano e, depois, `README.md`.
- **Reuso:** toda a arvore atual, manifestos, outputs e testes.
- **Mudancas:** documentar aderencia, componentes, conflitos e classificacao dos outputs.
- **Testes:** `git status`, verificacao de ignorados, busca por extensoes sensiveis e arquivos
  rastreados grandes.
- **Conclusao:** matriz cita evidencia real e alteracoes do usuario continuam intactas.
- **Dependencias:** leitura integral do PDF e da documentacao existente.
- **Risco:** classificar como valido um artefato sem proveniencia suficiente.
- **Rollback:** somente documentos novos/atualizados; reversao independente do codigo.

## Etapa 1 - contrato R0-R3 e integridade numerica

- **Objetivo:** tornar a representacao uma variavel explicita e impedir NaN/Inf no treino.
- **Arquivos:** `src/training/temporal_data.py`, configs de preprocessing e testes temporais.
- **Reuso:** `preprocess_block`, `fit_training_medians`, `build_windows` e splits congelados.
- **Mudancas:** mapear `R0 -> cinco sinais/zero`, `R1 -> cinco sinais/interpolacao curta`,
  `R2 -> cinco sinais mais tres flags`, e manter R3 nos agregadores existentes.
- **Testes:** dimensoes 5/5/8, scaler train-only, nenhum cruzamento de blocos, valores finitos.
- **Conclusao:** cada representacao tem nome, numero de features e preprocessing resolvidos no run.
- **Dependencias:** Etapa 0.
- **Risco:** alterar silenciosamente os resultados antigos que assumiam oito colunas.
- **Rollback:** manter default de compatibilidade somente para runs antigos; novos IDs incluem R0-R3.

## Etapa 2 - estabilizacao do motor temporal

- **Objetivo:** completar CUDA/AMP, early stopping e resume antes de qualquer experimento real.
- **Arquivos:** `src/training/temporal_engine.py`, `configs/experiment/temporal_multiseed.yaml`,
  `tests/test_temporal_engine.py`.
- **Reuso:** loop PyTorch, LSTM/TCN, scheduler e formato de checkpoint existentes.
- **Mudancas:** defaults `auto/150/15/0.002`, monitor `val_macro_f1`,
  `best_macro_f1.pt`, `last.pt`, balanceamento configuravel, RNG e generator no checkpoint,
  verificacoes de finitude e memoria CUDA.
- **Testes:** forward/backward, checkpoint best/last/periodico, reload, resume real e predicoes.
- **Conclusao:** smoke CPU e CUDA de 2-3 epocas terminam sem NaN/Inf e retomam da epoca correta.
- **Dependencias:** Etapa 1 e PyTorch com CUDA para o smoke de GPU.
- **Risco:** checkpoints exploratorios antigos serem incompativeis.
- **Rollback:** usar novos diretorios por geracao/ID; jamais sobrescrever os antigos.

## Etapa 3 - tracking, dry-run e G0

- **Objetivo:** conhecer custo e matriz antes de executar.
- **Arquivos:** `src/training/temporal_multiseed.py`, configuracoes G0/G2 e `README.md`.
- **Reuso:** fingerprint, JSON por run, agregadores e geradores de graficos existentes.
- **Mudancas:** ID com geracao/modelo/representacao/janela/fold/seed/config; `--dry-run`, filtros
  de modelo/fold/seed/janela, config resolvida, status e estimativa do numero de runs.
- **Testes:** dry-run nao cria checkpoint; IDs sao unicos; resume rejeita fingerprint divergente.
- **Conclusao:** G0 lista e executa um fold, uma janela, amostra reduzida e 2-3 epocas.
- **Dependencias:** Etapa 2.
- **Risco:** explosao combinatoria por uma configuracao mal formada.
- **Rollback:** dry-run e filtros sao aditivos e podem ser removidos sem tocar dados/resultados.

## Etapa 4 - G1, baselines de qualificacao

- **Objetivo:** estabelecer B1 e B2a sobre a mesma entrada R0.
- **Arquivos:** novo avaliador de regras sob `src/training/`, adaptacao minima dos classicos,
  configs G1, testes e outputs G1.
- **Reuso:** heuristica historica, metricas, janelas, folds e leitores existentes.
- **Mudancas:** regras fixas congeladas e SVM/RF/XGBoost sobre R0 achatado; sem usar teste para
  ajustar thresholds ou parametros.
- **Testes:** casos de fronteira das regras, flatten deterministico e isolamento dos folds.
- **Conclusao:** relatorio G1 rastreavel compara B1 e B2a nos quatro folds.
- **Dependencias:** Etapas 1 e 3.
- **Risco:** thresholds historicos nao generalizarem para a camera lateral.
- **Rollback:** manter o baseline historico identificado, sem substituir os resultados R3.

## Etapa 5 - G2, comparacao temporal controlada

**Estado: concluida em 13/08/2026.** Foram executados os 36 runs oficiais. TCN/60 liderou a
validacao (`0.4125 +/- 0.0211`), mas nao superou SVM/60 da G1 e Fatigue permaneceu com F1 zero.
H3 continua em aberto.

- **Objetivo:** comparar LSTM, TCN e Transformer pequeno sobre R0 e janelas 30/60/150.
- **Arquivos:** `models/temporal.py`, configs G2, runner e testes.
- **Reuso:** motor estabilizado, splits e tracking.
- **Mudancas:** Transformer com positional encoding e capacidade controlada; busca pequena feita
  somente em treino/validacao. Distribuicao original (`balancing:none`).
- **Testes:** shape/gradientes dos tres modelos, mascara/posicao, smoke de cada arquitetura.
- **Conclusao:** uma configuracao por modelo/janela e escolhida por validacao, sem olhar teste.
- **Dependencias:** G0 concluida; G1 recomendado para contexto.
- **Risco:** variancia alta do Transformer e custo de 36 runs de qualificacao (3x3x4).
- **Rollback:** Transformer e comparacao isolados; LSTM/TCN continuam funcionais.

## Etapa 6 - G3 e G4, missingness e desbalanceamento

**Estado: G3 e G4 concluídas.** G3 comparou 12 runs R0 reutilizados com 24
novos runs R1/R2. R0 foi selecionada para G4 (`0.4083` agregado), enquanto R2 obteve `0.3691`
e R1 `0.3156`. A G4 comparou 16 condições A reutilizadas e 48 runs B/C/D. LSTM/B foi
recomendado como compromisso temporal, pareado com SVM/B; H3 permanece aberta.

- **Objetivo:** medir separadamente representacao e estrategias de classes raras.
- **Arquivos:** configs G3/G4, sampler/augmentation no motor e testes.
- **Reuso:** R0-R2, `imbalance.yaml`, diagnosticos de missingness.
- **Mudancas:** G3 compara R1/R2 ao R0 selecionado; G4 compara none, class weights, weighted
  sampling e augmentation leve somente no treino. Nao combinar weights e sampler no principal.
- **Testes:** sampler, reproducibilidade, limites fisicos e ausencia de transformacao em val/test.
- **Conclusao:** decisao registrada por validacao e impacto em Macro F1/falsos alertas.
- **Dependencias:** G2.
- **Risco:** missingness virar atalho e augmentation produzir valores nao plausiveis.
- **Rollback:** estrategias selecionadas por config, sem alterar dados persistidos.

## Etapa 7 - G5, repeticoes finais e estabilidade

- **Objetivo:** estimar desempenho e estabilidade dos finalistas.
- **Arquivos:** configs congeladas G5, runner, `stability.py`, metricas e figuras.
- **Reuso:** cinco seeds, early stopping independente e agregadores existentes.
- **Mudancas:** executar cada modelo/config/fold nas cinco seeds; salvar seed, melhores/parada,
  losses, metricas, tempo, probabilidades e eficiencia.
- **Testes:** agregacao sintetica, IC95, ausencia de selecao pela melhor seed e completude da matriz.
- **Conclusao:** media, DP, mediana, min, max e IC95; curvas/epocas/variancia por seed.
- **Dependencias:** configuracoes congeladas apos G4.
- **Risco:** apenas quatro sessoes limitam inferencia e ICs entre folds.
- **Rollback:** cada run e independente e retomavel; falhas nao invalidam runs completos.

## Etapa 8 - explicabilidade, ablacao e casos

- **Objetivo:** explicar ganhos e falhas dos finalistas.
- **Arquivos:** `src/explainability/`, configs de ablacao e geradores de relatorio.
- **Reuso:** predicoes, probabilities e metadados de janela.
- **Mudancas:** retraining por grupos, permutation/SHAP onde apropriado, TP/FP/FN e alto missing.
- **Testes:** grupos disjuntos, retraining real e exemplos escolhidos por regra predefinida.
- **Conclusao:** importancia estavel por fold e casos auditaveis sem dados sensiveis versionados.
- **Dependencias:** G5.
- **Risco:** cherry-picking de casos ou interpretacao causal indevida.
- **Rollback:** relatatorios derivados, sem mudar finalistas.

## Etapa 9 - G6, fusao opcional

- **Objetivo:** avaliar late fusion apenas se o nucleo estiver fechado.
- **Arquivos:** `src/fusion/`, configs G6 e testes.
- **Reuso:** probabilidades OOF e fontes legadas validadas.
- **Mudancas:** meta-classificador simples com probabilidades OOF, celular/corpo/ausencia.
- **Testes:** nenhuma previsao in-sample alimenta a fusao; alinhamento temporal e disponibilidade.
- **Conclusao:** comparacao isolada com o melhor facial ou registro de trabalho futuro.
- **Dependencias:** G5 completo e fontes multimodais validas.
- **Risco:** vazamento OOF e desalinhamento de fontes.
- **Rollback:** extensao totalmente opcional e separada.

## Matriz de geracoes prevista

| Geracao | Escopo | Criterio para avancar |
|---|---|---|
| G0 | smoke pequeno, um fold/janela, 2-3 epocas | **Concluido:** LSTM/TCN/Transformer em CUDA; checkpoint/reload/resume/predicao finitos |
| G1 | regras e classicos R0 achatado | **Concluido:** 48 runs, tres janelas e quatro folds; qualificacao seed 42 |
| G2 | LSTM/TCN/Transformer, R0, 30/60/150 | selecao somente por validacao |
| G3 | R1 e R2 nos finalistas | efeito de missingness documentado |
| G4 | none/weights/sampling/augmentation | estrategia congelada sem teste externo |
| G5 | cinco seeds por fold/config finalista | estabilidade e media mais ou menos DP |
| G6 | fusao opcional | somente apos fechamento facial-temporal |

O runner deve imprimir essa matriz em modo seco antes de criar qualquer artefato de treino.
