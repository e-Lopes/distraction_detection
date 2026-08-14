# G4 — Mitigação isolada do desbalanceamento

## Resultado executivo

Foram consolidados **48 runs novos** e **16 reutilizados**, totalizando **64 comparações**. O maior Macro F1 médio de validação foi **0.4203 ± 0.0597**, em **svm / cenário D (augmentation)**. O F1 médio de Fatigue nessa configuração foi **0.0869 ± 0.1214**.

Esse máximo não resolve a classe rara: augmentation manteve Fatigue em zero nos temporais. O compromisso temporal mais equilibrado foi **LSTM/B**, com Macro F1 **0.4004 ± 0.0507**, F1 de Fatigue **0.0885 ± 0.0616**, recall **0.3920** e **56.92** falsos episódios/hora.

Os resultados permanecem qualificatórios: há somente a seed 42. H3 não é confirmada antes das cinco seeds dos finalistas.

## Desenho e controles

- R0, folds, janelas, arquiteturas, hiperparâmetros e orçamento foram congelados.
- A foi reutilizado após verificação de hashes; B, C e D foram aplicados isoladamente e apenas no treino.
- SVM/60 atuou como controle clássico nas quatro condições.
- Validação e teste não receberam sampling ou augmentation.

## Ranking de validação (média ± DP entre quatro folds)

- svm / D: 0.4203 ± 0.0597
- svm / A: 0.4191 ± 0.0593
- transformer / D: 0.4139 ± 0.0258
- tcn / A: 0.4125 ± 0.0211
- tcn / D: 0.4113 ± 0.0210
- transformer / A: 0.4081 ± 0.0136
- lstm / A: 0.4044 ± 0.0169
- tcn / C: 0.4035 ± 0.0156
- tcn / B: 0.4029 ± 0.0168
- lstm / D: 0.4006 ± 0.0144
- lstm / B: 0.4004 ± 0.0507
- transformer / C: 0.3923 ± 0.0269
- lstm / C: 0.3906 ± 0.0181
- svm / B: 0.3898 ± 0.0709
- transformer / B: 0.3878 ± 0.0222
- svm / C: 0.3580 ± 0.0340

## Interpretação científica

B e C recuperaram Fatigue nos temporais, mas reduziram o Macro F1. TCN/C obteve o maior recall médio de Fatigue (**0.4785**), ao custo de Macro F1 **0.4035** e **84.79** falsos episódios/hora. D preservou o Macro F1, porém não recuperou Fatigue nos temporais. A comparação principal usa médias entre folds e deltas pareados contra A; nenhum melhor fold isolado determina a seleção.

Para a comparação arquitetural justa, LSTM/B superou o controle SVM/B tanto em Macro F1 quanto em F1 de Fatigue, mas não superou o melhor SVM sem tratamento/augmentation. Portanto, há evidência preliminar favorável à modelagem temporal sob a mesma estratégia B, mas H3 continua aberta.

## Reprodutibilidade e limitação

Cada temporal possui checkpoints `last.pt` e `best_macro_f1.pt`, histórico por epoch, configuração resolvida, fingerprint e predições. O PyTorch advertiu que a atenção memory-efficient do Transformer em CUDA não é bit a bit determinística; esta limitação está registrada e deve ser considerada ao interpretar/repetir Transformer.

## Decisão

A seleção Pareto recomendada para a futura etapa de cinco seeds é **LSTM/B** como candidato primário, acompanhado de **SVM/B** como controle clássico pareado. **TCN/C** e **SVM/C** ficam como análise secundária de alta sensibilidade a Fatigue, devido ao custo operacional elevado. SVM/D é preservado como melhor Macro F1 observado, mas não é tratado como solução do desbalanceamento porque praticamente reproduziu A. A G5 não foi iniciada.
