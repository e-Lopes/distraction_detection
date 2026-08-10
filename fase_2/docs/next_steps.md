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
- [ ] Revisar a estratégia de validação contínua: os 20% finais deixam alguns subconjuntos
  sem janelas de fadiga, especialmente para 60 e 150 frames.

### Diagnóstico dos splits atuais

Os arquivos gerados são seguros contra sobreposição, mas ainda não devem ser congelados como
protocolo final de treino:

- Fold 1: treino não possui fadiga em janelas de 60/150 frames.
- Fold 2: teste (vídeo 2) não possui fadiga em nenhum tamanho.
- Fold 3: validação não possui fadiga em nenhum tamanho.
- Fold 4: teste possui apenas quatro janelas de fadiga em 30 frames e nenhuma em 60/150.

A ausência natural de fadiga no vídeo de teste deve ser relatada, não corrigida artificialmente.
Para treino/validação, será necessário escolher blocos contínuos sem vazamento que permitam
ajuste das três classes, ou declarar folds nos quais determinada métrica não é estimável.

## Etapa C — Integração das séries recuperadas

- [ ] Receber os arquivos em `data/interim/legacy_extraction/`.
- [ ] Registrar hashes e schemas sem versionar o conteúdo.
- [ ] Comparar contagens, timestamps e taxas de detecção com a qualificação.
- [ ] Normalizar para a tabela canônica por frame.

## Etapa D — Baselines de aprendizado

- [ ] Implementar zero-fill apenas na entrada do modelo e adicionar `face_detected`.
- [ ] Ajustar transformações somente no treino.
- [ ] Criar features agregadas por janela.
- [ ] Treinar Dummy, SVM, Random Forest e XGBoost nos mesmos folds.
- [ ] Registrar Macro F1, métricas por classe, balanced accuracy e matrizes de confusão.

## Bloqueios explícitos

- Treino real depende das séries EAR/MAR/pitch/yaw/roll.
- Manifesto real depende de OpenCV ou `ffprobe` disponível no ambiente.
- Nenhuma métrica sintética será apresentada como resultado científico.
