# Alinhamento com a pergunta de pesquisa e os objetivos da qualificação

## Pergunta de pesquisa

> Uma arquitetura temporal pode aprender múltiplos indicadores de fadiga e distração e superar
> abordagens baseadas em regras no contexto da teleoperação de mineração?

## Hipótese

Arquiteturas temporais capazes de processar janelas de múltiplos indicadores superam abordagens
baseadas em regras fixas e classificadores sem modelagem temporal.

O operador fixo define o escopo de aplicação e permite personalização, mas não altera o target
científico principal: `Alert`, `Fatigue` e `Distraction`. A análise `normal/non_normal` é
operacional e suplementar.

## Matriz de evidência

| Objetivo | Evidência necessária | Estado |
|---|---|---|
| Extrair indicadores visuais | EAR, MAR, pitch, yaw e roll por frame, com validade e proveniência | Implementado |
| Avaliar adaptação de domínio | Diagnóstico entre sessões e comparação controlada de representação/normalização | Parcial: missingness R0–R2 concluído; normalização pessoal ainda deve ser isolada |
| Construir baseline por regras | EAR/MAR/pitch históricos, sem ajuste pelo teste | Implementado em G1 |
| Comparar temporais e clássicos | Mesmos targets, folds, janelas, política de dados e métricas | Qualificação G1/G2 concluída; confirmação multi-seed pendente |
| Avaliar tamanho da janela | 30, 60 e 150 frames para cada família relevante | Implementado na qualificação; repetição final pendente |

## Comparação confirmatória

As famílias obrigatórias são:

1. regras fixas;
2. SVM e/ou outro classificador clássico sem arquitetura temporal;
3. LSTM, TCN e, como controle, Transformer.

Cada comparação deve preservar:

- target multiclasses `Alert/Fatigue/Distraction`;
- indicadores de entrada e representação declarados;
- janelas de 30, 60 e 150 frames quando a pergunta for efeito da janela;
- folds leave-one-video/session-out e purge congelados;
- seleção somente por validação;
- teste externo apenas descritivo depois do congelamento;
- múltiplas seeds para modelos estocásticos;
- Macro F1, balanced accuracy e métricas por classe, com atenção a `Fatigue` e `Distraction`.

Para evitar uma conclusão composta ambígua, a hipótese será examinada em dois contrastes:

- **H-temporal-regras:** arquitetura temporal supera regras fixas;
- **H-temporal-clássicos:** arquitetura temporal supera classificadores sem modelagem temporal.

Uma conclusão favorável no primeiro contraste não confirma automaticamente o segundo. Deltas
devem ser apresentados por sessão/fold e seed, juntamente com média, dispersão e consistência
entre sessões. As janelas sobrepostas não serão tratadas como unidades independentes para testes
de significância.

## Leitura dos resultados existentes

Na qualificação com seed 42, TCN/60 obteve Macro F1 de validação `0,4125`, contra até `0,1597`
das regras fixas e `0,4191` do SVM/60. Portanto, há evidência preliminar de superioridade sobre
regras, mas ainda não de superioridade sobre o melhor clássico. A hipótese completa permanece
em aberto até a comparação final controlada e multi-seed.
