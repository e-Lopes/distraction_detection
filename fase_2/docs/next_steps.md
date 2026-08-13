# Próximas etapas executáveis

Este roteiro segue `PlanoPósBanca.pdf`. A recuperação das séries faciais da outra máquina
continua prioritária, mas não bloqueia a fundação abaixo. Interpolação está adiada; zero-fill
com indicação explícita de detecção será o primeiro baseline.

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

- [ ] Comparar zero-fill, interpolação curta e flags de validade.
- [ ] Comparar janelas de 30, 60 e 150 frames nos mesmos folds.
- [ ] Adicionar deltas e grupos de atributos configuráveis.
- [ ] Implementar LSTM e TCN com CUDA, AMP, early stopping e checkpoints completos.
- [ ] Repetir configurações finalistas em múltiplas seeds.

## Bloqueios explícitos

- Treino real depende das séries EAR/MAR/pitch/yaw/roll.
- Manifesto real depende de OpenCV ou `ffprobe` disponível no ambiente.
- Nenhuma métrica sintética será apresentada como resultado científico.
