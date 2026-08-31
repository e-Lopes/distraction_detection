# Próximas etapas executáveis

Este roteiro é histórico e foi reconciliado com o plano integrado em
`integrated_plan_gap_analysis.md`. As séries faciais já foram recuperadas/regeneradas. R0 usa
zero-fill; R1 e R2 permitem avaliar, de forma separada, interpolação curta e flags.

## Etapa A — Fundação de dados

- [x] Gerar o manifesto real dos quatro vídeos sem alterá-los.
- [x] Registrar e normalizar os 71 intervalos manuais.
- [x] Validar anotações contra classes, estados, cobertura e duração real.
- [x] Converter os limites em segundos para frames após confirmar o FPS.
- [x] Gerar distribuições frame-level e window-level.

## Etapa B — Janelas e splits

- [x] Construir janelas de 30, 60 e 150 frames, stride 15.
- [x] Aplicar maioria mínima de 60%; empate ou insuficiência vira `mixed`.
- [x] Gerar quatro folds leave-one-video-out.
- [x] Reservar blocos temporais contínuos para validação com purge gap de 150 frames.
- [x] Provar por testes que nenhuma janela ou frame atravessa subconjuntos.
- [x] Revisar a estratégia de validação contínua: blocos internos preservam as três classes
  em treino e validação, com purge gap bilateral de 150 frames.

### Diagnóstico dos splits atuais

Os arquivos gerados estão congelados para o primeiro protocolo de treino:

- Fold 2: teste (vídeo 2) não possui fadiga em nenhum tamanho.
- Fold 4: teste possui apenas quatro janelas de fadiga em 30 frames e nenhuma em 60/150.
- Todos os subconjuntos de treino e validação possuem as três classes em 30, 60 e 150 frames.

A ausência natural de fadiga no vídeo de teste deve ser relatada, não corrigida artificialmente.
Nos testes sem fadiga, a métrica dessa classe não é estimável e será reportada explicitamente.

## Etapa C — Integração das séries recuperadas

- [x] Recuperar e preservar o protótipo histórico que calculava as séries em memória.
- [x] Implementar extrator reproduzível com origem configurável, schema canônico e hashes.
- [x] Receber os arquivos em `data/interim/legacy_extraction/`.
- [x] Registrar hashes e schemas sem versionar o conteúdo.
- [x] Comparar contagens, timestamps e taxas de detecção com a qualificação.
- [x] Normalizar para a tabela canônica por frame.

## Etapa D — Baselines de aprendizado

- [x] Implementar zero-fill apenas na agregação de entrada e adicionar `face_detected`.
- [x] Ajustar transformações somente no treino.
- [x] Criar primeiras features agregadas por janela (média/desvio e missingness).
- [x] Treinar SVM, Random Forest e XGBoost nos mesmos folds.
- [x] Treinar Dummy `most_frequent` como piso de desempenho.
- [x] Registrar Macro F1, métricas por classe, balanced accuracy e matrizes de confusão.
- [x] Adicionar checkpoints retomáveis e figuras comparativas.
- [x] Consolidar o grid search clássico selecionado exclusivamente pela validação.

## Etapa E — Pré-processamento e contexto temporal

- [x] Comparar zero-fill, interpolação curta e flags de validade.
- [x] Comparar janelas de 30, 60 e 150 frames nos mesmos folds.
- [x] Adicionar deltas e grupos de atributos configuráveis em
  `src/features/temporal_window_features.py`, preservando o R3 histórico. A configuração
  `temporal_behavior_v1.yaml` define distribuição, dinâmica, ocular, oral, pose e missingness;
  sua comparação experimental ainda não foi executada.
- [x] Implementar LSTM e TCN com CUDA, AMP, early stopping e checkpoints completos.
- [x] Implementar a infraestrutura para repetições, estatísticas e gráficos entre seeds.
- [x] Executar G2 e congelar os candidatos por validação.
- [x] Executar G3 separadamente, com R0 reutilizado e 24 novos runs R1/R2.
- [x] Executar G4 sobre R0 com 48 runs novos e 16 reutilizados.
- [x] Executar G4.5A com threshold cross-fit por sessão; não promover porque houve piora nos
  três modelos.
- [x] Executar G4.5B com Focal Loss; preservar somente LSTM/Focal como candidata qualificatória.
- [ ] Implementar e executar G4.5C hierárquica com SVM/60 e LSTM/60, somente validação.
- [ ] Congelar LSTM/B + SVM/B para cinco seeds; manter TCN/C + SVM/C como análise secundária.

## Etapa F — G1 e comparação temporal

- [x] Congelar regras EAR/MAR/pitch sem ajuste no teste.
- [x] Avaliar regras, SVM, Random Forest e XGBoost sobre R0 achatado.
- [x] Executar 30/60/150 nos quatro folds, distribuição original e seed de qualificação 42.
- [x] Gerar métricas por classe, matrizes de confusão e gráficos comparativos.
- [x] Executar G2 com LSTM, TCN e Transformer sobre R0 nas três janelas.

Na validação G1, SVM linear com janela 60 obteve o maior Macro F1 médio (`0,4191`). Isso não
seleciona um resultado final e não elimina as três janelas de G2. O desempenho de fadiga ainda
é baixo e será tratado apenas nas gerações posteriores de desbalanceamento.

Na G2, o melhor resultado foi TCN/60 (`0,4125 ± 0,0211` de Macro F1 entre folds), seguido pelo
Transformer/150 (`0,4081 ± 0,0136`). O melhor LSTM foi LSTM/60 (`0,4044 ± 0,0169`). Nenhuma
configuração temporal detectou Fatigue de forma útil (`F1 = 0`). O SVM/60 da G1 permaneceu
ligeiramente superior, portanto H3 continua em aberto. O próximo experimento é G3, comparando
R1 e R2 contra R0 nos candidatos TCN/60, LSTM/60 e Transformer/150, sem introduzir ainda
balanceamento; G4 avaliará as estratégias de desbalanceamento separadamente.

Na G3, R0 foi selecionada para G4 (`0,4083` de Macro F1 agregado entre os três modelos), à
frente de R2 (`0,3691`) e R1 (`0,3156`). R2 melhorou ligeiramente apenas o Transformer/150,
mas não foi consistente entre arquiteturas e não produziu previsões de Fatigue. A próxima etapa
é G4, comparando as estratégias de desbalanceamento sobre R0 sem alterar simultaneamente a
representação. G4 foi concluída; LSTM/B foi o compromisso temporal recomendado e H3 permanece aberta.

### Diagnóstico do pré-processamento

Foram concluídos 108 treinamentos com hiperparâmetros fixos. A interpolação curta produziu
ganhos pequenos e dependentes do modelo/janela. Pela validação, Random Forest com interpolação
curta e janela de 60 frames obteve o maior Macro F1 médio (`0,4094`), praticamente empatado com
zero-fill (`0,4082`). As flags adicionais não apresentaram benefício consistente.

Na avaliação externa descritiva, o melhor Macro F1 foi `0,4778` para XGBoost com interpolação
curta e janela de 150 frames, mas esse resultado de teste não será usado para escolher a próxima
configuração. O recall de fadiga permaneceu próximo de zero, confirmando a necessidade de avaliar
as sequências temporais brutas com LSTM e TCN.

## Bloqueios explícitos

- Nenhum bloqueio de dados/GPU impede a continuação; G0–G4 e G4.5A/B foram concluídas.
- G5 permanece bloqueada até concluir G4.5C e congelar formalmente os candidatos.
- Nenhuma métrica sintética será apresentada como resultado científico.
