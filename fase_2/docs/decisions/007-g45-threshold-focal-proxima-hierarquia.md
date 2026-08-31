# 007 — G4.5A/B concluídas e próxima etapa hierárquica

**Status:** aceita em 31/08/2026

## Contexto

A G4 recuperou parcialmente `Fatigue` com class weights e weighted sampling, mas manteve baixa
precisão e muitos falsos episódios. O protocolo G4.5 foi congelado para testar separadamente
threshold cross-fit, Focal Loss e classificação hierárquica.

## Evidências produzidas

### G4.5A — threshold cross-fit

O threshold foi calibrado por sessão, excluindo sempre o vídeo-alvo. Nenhum teste externo foi
carregado. A intervenção piorou os três modelos:

| Modelo | Macro F1 argmax | Macro F1 cross-fit |
|---|---:|---:|
| LSTM/B | 0,4005 | 0,3390 |
| TCN/B | 0,4029 | 0,3608 |
| TCN/C | 0,4035 | 0,3811 |

Decisão: não promover ajuste de threshold. A dificuldade de fadiga não é explicada apenas pela
regra argmax.

### G4.5B — Focal Loss

Foram executados oito runs validation-only, com LSTM/TCN, quatro folds e seed 42. A LSTM/Focal
obteve Macro F1 `0,4044 ± 0,0366` e F1 de Fatigue `0,1039 ± 0,0208`, ganhos pequenos sobre
LSTM/B (`0,4004` e `0,0885`). Na TCN, o efeito foi praticamente nulo.

Decisão: preservar LSTM/Focal como candidata qualificatória; não promover TCN/Focal.

## Expectativa de desempenho

Os resultados atuais não sustentam expectativa de Macro F1 multiclasses próximo de `0,9` sob
leave-one-session-out. Alert isoladamente já se aproxima desse nível, e uma tarefa binária
personalizada pode futuramente alcançar valores altos, mas isso não responde sozinho à hipótese
multiclasses da qualificação. A próxima etapa não usará `0,9` como critério de sucesso nem fará
busca adaptativa de configurações até alcançar um score desejado.

## Próxima etapa autorizada — G4.5C

Implementar a classificação hierárquica já congelada:

1. nível 1: `Alert` versus `Non-Alert`;
2. nível 2: `Fatigue` versus `Distraction`, treinado somente com exemplos `Non-Alert` do treino;
3. SVM/60 e LSTM/60 em R0, quatro folds e seed 42;
4. pesos `N/(2N_c)` calculados separadamente e somente no treino de cada nível;
5. composição probabilística na ordem `Alert`, `Fatigue`, `Distraction`;
6. avaliação somente em validação até a decisão ser congelada;
7. checkpoints, logs, fingerprints e histórico independentes por nível.

## Critérios antes da execução oficial

- teste do mapeamento dos dois níveis;
- prova de que o nível 2 usa somente `Non-Alert` do treino;
- falha explícita quando faltar uma classe necessária;
- probabilidades finitas, não negativas e somando 1;
- recarga de checkpoints reproduzindo predições;
- smoke isolado de SVM e LSTM hierárquicos;
- dry-run com exatamente oito pipelines e dezesseis estimadores.

## Critério de decisão

A hierarquia será comparada com SVM/B, LSTM/B e LSTM/Focal usando Macro F1, métricas de Fatigue
e Distraction, balanced accuracy e falsos alarmes. Ganho em um fold isolado não será suficiente.
G5 permanece bloqueada até a consolidação da G4.5C.
