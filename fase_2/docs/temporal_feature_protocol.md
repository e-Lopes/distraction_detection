# Protocolo pré-experimento de atributos temporais explícitos

**Status:** implementação e testes concluídos; treinamento não iniciado

**Data:** 30/08/2026

## Objetivo

Avaliar se atributos temporais explícitos melhoram a separação científica primária entre
`Alert`, `Fatigue` e `Distraction` em relação às representações históricas, sem modificar
anotações, splits ou resultados G0–G4. Essa tarefa multiclasses responde à pergunta e à hipótese
da qualificação. A agregação personalizada `normal` versus `non_normal`, definida em
`configs/task/personalized_binary.yaml`, é somente uma análise operacional suplementar e não
pode selecionar a arquitetura principal. Esta etapa é uma qualificação de representação e não
substitui a G4.5 ou autoriza a G5.

Ausência do operador, oclusão, face ausente e janelas `mixed` não são evidência automática de
comportamento não normal. Esses casos são tratados pelo gate operacional e ficam fora do target
comportamental.

## Representação `temporal_behavior_v1`

O extrator em `src/features/temporal_window_features.py` produz grupos configuráveis:

- `signal_distribution`: média, dispersão, quantis, amplitude e inclinação de cada sinal;
- `signal_dynamics`: deltas somente entre pares de frames adjacentes válidos;
- `ocular`: PERCLOS descritivo, quantidade/taxa e maior duração relativa de fechamento;
- `oral`: proporção, quantidade/taxa e maior duração relativa de abertura da boca;
- `head_pose`: proporção, quantidade/taxa e maior duração relativa fora da pose central;
- `missingness`: taxa de detecção, taxa de ausência, quantidade e maior gap relativo.

Frames sem face não entram nas estatísticas faciais, interrompem eventos e permanecem
representados no grupo de missingness. O extrator não recebe nem consulta o rótulo da janela.

Os limiares iniciais estão congelados em `configs/features/temporal_behavior_v1.yaml`. EAR 0,25
e MAR 0,55 vêm das regras históricas; os limiares de pose são hipóteses de engenharia e não
evidência clínica ou operacional. Alterá-los após examinar resultados exige uma nova versão da
configuração e justificativa explícita.

## Comparação proposta

Antes do treino completo, implementar um runner com dry-run e smoke para comparar, nos mesmos
quatro folds/sessões do operador e sem acesso ao teste durante seleção:

1. agregação histórica R3;
2. apenas distribuição e dinâmica dos sinais;
3. apenas ocular, oral e pose;
4. todos os grupos sem missingness;
5. todos os grupos.

As representações serão comparadas sobre as três famílias exigidas pela qualificação: regras
fixas, classificadores clássicos sem modelagem temporal e arquiteturas temporais. Todos usam as
mesmas anotações, folds e janelas. Macro F1 e precision/recall/F1 de `Fatigue` e `Distraction`
formam a avaliação científica; contagens de janelas não serão tratadas como eventos
independentes. A análise binária poderá acrescentar F1 de `non_normal`, falsos alarmes por hora
e latência, mas não substituirá os resultados multiclasses.

## Critério para avançar

Avançar para o desenho multiescala somente se a nova representação:

- for finita e reproduzível em todos os folds;
- não usar frames fora da janela ou estatísticas do teste;
- melhorar a validação multiclasses em mais de um fold e em eventos independentes;
- não elevar falsos positivos de fadiga de forma operacionalmente desproporcional.

Resultados de teste serão exclusivamente descritivos e não escolherão grupos, limiares ou
janela.
