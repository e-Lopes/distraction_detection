# Relatorio final do experimento temporal

> Estado: primeira consolidacao. Resultados historicos validos foram reutilizados; nenhum novo treinamento oficial foi executado. O teste externo nunca foi usado para selecao.

## 1. Resumo executivo

O conjunto possui 122,337 frames (119.0 min), forte desbalanceamento e apenas 84 s anotados como Fatigue. SVM e redes temporais superam regras fixas, mas nao ha evidencia de superioridade geral dos temporais sobre o melhor classico. O resultado operacional de Fatigue ainda e insuficiente: ganhos de recall vieram acompanhados de muitos falsos episodios. A decisao atual e confirmar LSTM/B contra SVM/B em cinco seeds e completar avaliacao por episodio.

## 2. Pergunta, objetivo e hipotese

Pergunta: quanto contexto temporal melhora a classificacao de Alert, Fatigue e Distraction a partir de EAR, MAR, Pitch, Yaw e Roll? A hipotese e que modelos temporais ajudam, sobretudo em Fatigue, sem custo operacional desproporcional. A margem de equivalencia pratica e Macro F1 = 0.01.

## 3. Dados, videos, duracao, FPS e classes

Sao quatro sessoes leave-one-video-out, com FPS entre 15.72 e 18.02. Duracoes anotadas: Alert 4440 s, Distraction 825 s, Fatigue 84 s e ausencia do operador 1794 s.

## 4. Quantidade e duracao dos episodios

Ha 9 intervalos anotados como Fatigue no manifesto; depois de janelamento e restricoes de suporte, algumas configuracoes ficam com aproximadamente dois eventos uteis. Logo, janelas sobrepostas nao sao tratadas como observacoes estatisticamente independentes e inferencias fortes nao sao justificadas.

## 5. Qualidade dos landmarks e missingness

O missingness varia de 16.3% a 58.0% entre videos (38.15% global). Na comparacao preliminar do video 04, OpenFace detectou 88.32% dos frames com operador presente e MediaPipe 79.81%; MediaPipe foi mantido pelo menor custo/latencia e integracao, nao por maior cobertura. Os indicadores dos extratores nao sao intercambiaveis sem nova validacao.

## 6. Tratamentos de dados

R0 usa zero-fill; R1 interpola somente gaps curtos e usa mediana do treino; R2 acrescenta flags de deteccao, interpolacao e duracao da ausencia. Ajustes sao isolados por video/particao e scalers usam apenas treino. Gaps longos e ausencia do operador sao preservados.

## 7. Representacoes e features

A base usa EAR, MAR, Pitch, Yaw e Roll. `temporal_behavior_v1` inclui distribuicao e quantis robustos, IQR, inclinacao, diferenca inicio-fim, autocorrelacao, line length, picos, cruzamentos, primeira e segunda diferencas, velocidade, aceleracao, variacao total, PERCLOS, piscadas, possiveis bocejos, desvio de cabeca, correlacoes/covariancias e missingness/interpolacao. Correlacoes ou variacoes indefinidas em sinais constantes viram zero finito. Foram descartadas correlacoes defasadas, autorregressao de alta ordem e bandas espectrais por suporte instavel em janelas curtas e FPS heterogeneo. Features por threshold sao baselines comportamentais complementares.

## 8. Protocolo de janelas e splits

Janelas de 30, 60 e 150 frames, stride 15, quatro folds leave-one-video-out e purge bilateral de 150 frames. Como o FPS varia, a mesma janela representa duracoes diferentes; o baseline historico em frames foi preservado e resampling nao foi imposto silenciosamente.

## 9. Inventario de experimentos existentes

Importados: Dummy, regras fixas, SVM, Random Forest, XGBoost, LSTM, TCN, Transformer, comparacao de janelas, missingness, balanceamento, Focal Loss, threshold cross-fit, classificacao hierarquica, robustez de pose e comparacao de extratores. A taxonomia canonica e `rule`, `feature`, `distance`, `shapelet`, `transform`, `deep` e `ensemble`; nomes historicos nao foram convertidos em novos resultados. G47/YOLO e exploratorio; smoke/pilots e artefatos incompativeis nao entram na conclusao cientifica.

## 10. Resultados dos baselines

O Dummy majoritario fica perto de Macro F1 0.31 na validacao. Regras fixas atingem no melhor caso aproximadamente 0.160. Ambos servem como piso, nao como solucao operacional.

## 11. Resultados dos modelos classicos

SVM permanece competitivo: melhor media historica em w=60 de 0.4191. SVM, Random Forest e XGBoost historicos continuam intactos. O novo screening compara Regressao Logistica, SVM, Random Forest e XGBoost com exatamente a mesma tabela `temporal_behavior_v1`, w=60, seed 42 e quatro folds (16 runs).

## 12. Resultados dos modelos temporais

Em seed 42: LSTM/w60 0.4044, TCN/w60 0.4125 e Transformer/w150 0.4081. Eles vencem regras fixas, mas nao estabelecem superioridade geral sobre SVM. Esses resultados serao reutilizados; os 16 novos runs LSTM/B estao bloqueados ate promocao formal apos o screening.

## 13. Resultados por janela

As melhores configuracoes historicas concentram-se em 60 frames para SVM, LSTM e TCN; Transformer teve melhor resultado em 150 frames. A figura `window_duration_by_video.png` explicita a diferenca temporal causada pelo FPS.

## 14. Resultados por classe

Alert domina o conjunto. Distraction e separavel parcialmente. Fatigue tem suporte muito pequeno e F1 instavel; medias globais sem a leitura por classe seriam enganosas.

## 15. Avaliacao de Fatigue

LSTM/B obteve F1 medio de Fatigue 0.0885 e recall 0.392, com 56.92 falsos episodios/h. TCN/C elevou recall para 0.4785, mas chegou a 84.79 falsos episodios/h. Focal Loss levou LSTM a F1 0.1039, ganho pequeno. PR-AUC ainda nao pode ser reconstruida para todos os historicos porque probabilidades OOF completas nao estao versionadas.

## 16. Avaliacao por episodio

`event_metrics.csv` registra claramente a estatistica G4 como proxy preliminar baseada em janelas. Event Precision, Recall, F1, latencia, fragmentacao e erro de duracao aguardam predicoes OOF completas e matching temporal pre-registrado.

## 17. Estabilidade entre videos e seeds

Ha variacao material entre folds/videos e somente seed 42 para os finalistas historicos. A confirmacao multi-seed e pendente; a unidade de comparacao sera video e seed, nunca janela individual.

## 18. Custo computacional

Tempos de treino, tamanho, parametros e pico de GPU existem para parte dos modelos e foram importados no registro. DTW possui custo quadratico e limite de pares; shapelets usam 2.000 candidatos e no maximo 200 selecionados; MiniROCKET usa 5.040 kernels, um multiplo valido de 84. Latencia end-to-end, RAM e throughput serao medidos nos runs reais; nao se inventam valores ausentes.

## 19. Analise de robustez e ablacao

R0 foi selecionada sobre R1/R2 (0.4083 contra 0.3156/0.3691). Threshold cross-fit foi negativo. Hierarquico/LSTM chegou a 0.4186, mas foi inconsistente e pior em Fatigue; nao promovido. Correcao de pose reduziu dependencia geometrica, mas ganhou apenas ~0.0012 de Macro F1; nao promovida.

## 20. Comparacao pareada e incerteza

Com quatro videos, nao se usa teste t, Friedman, Nemenyi ou Wilcoxon como prova forte. A confirmacao reportara deltas por video, videos vencidos, magnitude e intervalos que preservem dependencia temporal.

## 21. Fronteira de Pareto

Ainda nao existe vencedor absoluto. SVM oferece Macro F1 competitivo e baixo custo; LSTM/B oferece algum sinal de Fatigue a custo de falsos alarmes; TCN/C amplia recall com mais alarmes. O screening adiciona quatro pipelines de features, 1-NN+DTW, shapelets+Ridge e MiniROCKET+Ridge. A comparacao e entre pipelines/representacoes, nao uma atribuicao causal isolada ao algoritmo.

## 22. Limitacoes

Quatro videos, poucos eventos de Fatigue, missingness heterogeneo, FPS diferentes, dados de um dominio restrito e probabilidades historicas incompletas. OpenFace foi comparado em um video e o benchmark YOLO e exploratorio.

## 23. Resposta as hipoteses

Contexto temporal supera regras manuais, mas a hipotese de superioridade global sobre o melhor classico nao foi confirmada. A hipotese de melhora robusta de Fatigue permanece inconclusiva devido a raridade e falsos alarmes.

## 24. Conclusao

O resultado defensavel hoje e um empate pratico global com trade-offs: SVM e referencia eficiente; LSTM/B e candidata temporal para confirmacao. Nenhum modelo deve ser apresentado como detector completo de Fatigue.

## 25. Proximos passos realmente pendentes

Executar primeiro o screening de 36 runs: features 16, DTW 4, shapelets 4 e MiniROCKET 12. Shapelets/MiniROCKET requerem o extra opcional `tsc`; DTW esta bloqueado porque a estimativa excede o limite configurado. Depois, promover no maximo o melhor pipeline de features, o melhor temporal nao neural, LSTM/B e SVM/B; TCN/C fica como sensibilidade. A confirmacao nao inicia sem `screening_promotion.yaml` e autorizacao.

### Matriz de screening reservada

| Paradigma | Pipeline | Representacao | Janelas | Runs | Estado inicial |
|---|---|---|---|---:|---|
| feature | Logistica/SVM/RF/XGBoost | temporal_behavior_v1 | 60 | 16 | pendente |
| distance | 1-NN + DTW dependente | R0 bruta | 60 | 4 | bloqueado por custo |
| shapelet | Random Shapelet Transform + Ridge | R0 bruta | 60 | 4 | dependencia opcional |
| transform | MiniROCKET + RidgeCV | R0 bruta | 30/60/150 | 12 | dependencia opcional |

### Secoes reservadas para resultados novos

- **DTW:** Sakoe-Chiba de 6 frames, matriz reutilizavel por fold e autorizacao explicita acima de 5 milhoes de pares.
- **Shapelets:** registrar canal, comprimento, importancia, exemplo de treino e presenca nos videos externos; verificar memorizacao de uma unica sessao.
- **MiniROCKET:** comparar as tres janelas diretamente com SVM, LSTM, TCN e Transformer.
- **Ensembles:** DrCIF, Arsenal e HIVE-COTE 2.0 permanecem desativados; TS-CHIEF e apenas documentado. So considerar apos resultado insuficiente dos metodos simples, estimativa de custo e autorizacao.

### Criterio de promocao

Usar apenas validacao interna: Macro F1 medio, F1/PR-AUC e recall de Fatigue, estabilidade entre videos e custo, com margem pratica de 0.01. Resultado negativo e preservado e nao recebe novas seeds. Metricas de episodio ausentes no screening ficam explicitamente pendentes para confirmation.

## 26. Apendice: nomes historicos e novo fluxo

| Historico | Conteudo | Etapa publica atual |
|---|---|---|
| G0/G1 | dados, Dummy, regras e classicos | prepare / train / report |
| G2 | temporais e janelas | train / report |
| G3 | missingness R0-R2 | prepare / report |
| G4 | balanceamento | train / report |
| G4.5 | threshold, focal e hierarquico | train / report |
| G4.6 | robustez de pose | prepare / report |
| G47 | benchmark exploratorio YOLO | report (exploratorio) |
| G48/G48A | comparacao de extratores | prepare / report |

### Auditoria interna consolidada

| Componente | Estado verdadeiro | Reutilizavel | Acao minima |
|---|---|---|---|
| dados e anotacoes | concluido | sim | validar manifesto |
| series faciais | concluido | sim | validar cache; bruto ausente |
| missingness | concluido | sim | importar |
| janelas e splits | concluido | sim | validar purge |
| features temporais | parcial | sim | ativar so se justificado |
| modelos classicos | parcial | sim | importar + Logistica |
| modelos temporais | parcial | sim | importar + multi-seed LSTM |
| avaliacao por episodio | parcial | sim | matching OOF |
| estabilidade multi-seed | parcial | sim | 16 runs pendentes |
| custo computacional | parcial | sim | consolidar novos runs |
| relatorio consolidado | concluido nesta versao | sim | regenerar apos treinos |

### Inconsistencias resolvidas

`split_validation.md` descreve um split antigo sem Fatigue na validacao; o manifesto atual e a ADR 004 mostram blocos corrigidos. `g45_results.md` e um relatorio de andamento chamam o hierarquico de pendente, mas `g45c_runs.csv` e seus artefatos provam conclusao posterior. `data_sources_and_provenance.md` registra pendencias antigas, enquanto manifestos e quatro series completas comprovam que os derivados estao disponiveis; somente os videos brutos faltam localmente.

### Figuras geradas

- `macro_f1_by_model.png`
- `fatigue_f1_by_model.png`
- `svm_w60_oof_confusion.png`
- `svm_w60_by_video.png`
- `window_duration_by_video.png`

Figuras que exigem OOF/multi-seed (confusao OOF final, resultado por video, distribuicao entre seeds, timeline, desempenho-custo e Pareto final) nao foram fabricadas nesta primeira consolidacao.
