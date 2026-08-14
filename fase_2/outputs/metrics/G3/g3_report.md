# G3 — missingness e representacao temporal

## 1. Objetivo e hipotese

Avaliar se interpolacao de gaps curtos (R1) e flags explicitas de missingness (R2) melhoram R0, mantendo modelos, folds, seed, distribuicao, orcamento e early stopping constantes.

## 2. Relacao com G1 e G2

G1 manteve SVM/60 como melhor qualificacao (0.4191). G3 reutiliza sem retreino os 12 runs R0 finalistas da G2 e adiciona 24 runs R1/R2.

## 3. Configuracoes congeladas

TCN/60, LSTM/60 e Transformer/150; seed 42; quatro folds; batch 1024; ate 150 epocas; patience 15; minimum delta 0.002; sem balanceamento. R1/R2 usam interpolacao linear offline de gaps internos de ate 15 frames e mediana do treino para gaps restantes.

## 4–7. Resultados reutilizados, novos, por fold e agregados

| Modelo | Representacao | Janela | Macro F1 validacao (media +/- DP) |
|---|---|---:|---:|
| LSTM | R0 | 60 | 0.4044 +/- 0.0169 |
| LSTM | R1 | 60 | 0.3163 +/- 0.0082 |
| LSTM | R2 | 60 | 0.3343 +/- 0.0454 |
| TCN | R0 | 60 | 0.4125 +/- 0.0211 |
| TCN | R1 | 60 | 0.3167 +/- 0.0066 |
| TCN | R2 | 60 | 0.3617 +/- 0.0564 |
| TRANSFORMER | R0 | 150 | 0.4081 +/- 0.0136 |
| TRANSFORMER | R1 | 150 | 0.3138 +/- 0.0059 |
| TRANSFORMER | R2 | 150 | 0.4112 +/- 0.0232 |

Os resultados completos por fold estao em `g3_runs.csv`; os desvios representam folds com seed 42, nao variabilidade entre seeds.

## 8. Deltas pareados

| Modelo | Comparacao | Delta medio de Macro F1 |
|---|---|---:|
| LSTM | R1-R0 | -0.0881 |
| LSTM | R2-R0 | -0.0702 |
| LSTM | R2-R1 | +0.0179 |
| TCN | R1-R0 | -0.0957 |
| TCN | R2-R0 | -0.0508 |
| TCN | R2-R1 | +0.0450 |
| TRANSFORMER | R1-R0 | -0.0943 |
| TRANSFORMER | R2-R0 | +0.0032 |
| TRANSFORMER | R2-R1 | +0.0974 |

## 9–10. Metricas por classe e Fatigue

Nenhuma representacao produziu deteccao util de Fatigue: F1 e recall permaneceram zero. As probabilidades, falsos positivos, falsos negativos e contagens reais estao em `g3_fatigue_analysis.csv`.

| Modelo | Representacao | Predicoes Fatigue (4 folds) | Prob. media nas janelas reais |
|---|---|---:|---:|
| LSTM | R0 | 0 | 0.0892 |
| LSTM | R1 | 0 | 0.2832 |
| LSTM | R2 | 0 | 0.2493 |
| TCN | R0 | 0 | 0.0073 |
| TCN | R1 | 0 | 0.0811 |
| TCN | R2 | 0 | 0.1051 |
| TRANSFORMER | R0 | 0 | 0.0165 |
| TRANSFORMER | R1 | 0 | 0.0634 |
| TRANSFORMER | R2 | 0 | 0.0191 |

Havia 51 janelas reais de Fatigue na validacao de w60 e 36 em w150. Todas foram falsos negativos e nao houve falsos positivos. A maior probabilidade de Fatigue observada em uma janela real foi 0.3284, ainda abaixo da referencia de 1/3 e sem alteracao de threshold.

As janelas de Fatigue dos blocos de validacao tinham missing ratio baixo (em geral zero ou proximo de zero). Assim, as flags de R2 nao forneceram um sinal forte para recuperar Fatigue nesses blocos; seus eventuais ganhos podem estar ligados sobretudo a outras classes e a correlacoes de missingness.

## 11–14. Confusao, previsoes, probabilidades e missingness

As matrizes, distribuicoes, probabilidades de Fatigue e relacao entre missing rate e erro foram preservadas em CSV e SVG. Como missingness e classe sao associados, qualquer ganho de R2 deve ser interpretado como correlacao potencial, nao como evidencia causal de comportamento facial.

## 15. Tempo e recursos

Os 24 novos runs consumiram 2672.2 segundos de treino registrados e usaram CUDA/AMP na RTX 2060 SUPER. Tempos de inferencia, throughput, tamanho e pico de memoria constam em `g3_execution_table.csv`.

## 16. Falhas e retomadas

Todos os 24 novos runs terminaram; 0 foram marcados como retomados. Os dois smokes permanecem separados e nenhuma falha foi incorporada.

## 17. Limitacoes

A G3 usa uma unica seed. R1/R2 com interpolacao linear sao offline. O kernel eficiente de atencao CUDA nao e deterministico bit a bit. Ha apenas quatro sessoes e poucos exemplos de Fatigue.

## 18. Decisao para G4

A representacao recomendada e **R0**, selecionada somente pela media de Macro F1 de validacao e favorecida tambem pela menor complexidade. Ranking agregado: R0=0.4083, R2=0.3691, R1=0.3156.

R1 piorou de forma consistente. R2 recuperou parte da perda e melhorou ligeiramente o Transformer, mas nao superou R0 no conjunto dos modelos nem resolveu Fatigue. G4 ainda nao foi iniciada. H3 continua em aberto; estes resultados de uma seed nao sao finais.
