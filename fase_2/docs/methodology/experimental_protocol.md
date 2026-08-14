# Protocolo experimental

## Questão central

Avaliar como diferentes representações e modelos temporais classificam Alerta, Fadiga e Distração a partir de indicadores faciais, considerando falhas de detecção, duração das janelas e desbalanceamento.

## Unidade de avaliação

- Indicadores por frame: EAR, MAR, pitch, yaw e roll.
- Entrada temporal: janelas de 30, 60 e 150 frames.
- Classe primária: maioria dos frames, condicionada à proporção mínima configurada.
- Janela de transição: janela sem predominância suficiente; excluída do treino principal e analisada separadamente.

## Divisão dos dados

- Avaliação externa leave-one-video-out.
- Validação em blocos contínuos dentro dos vídeos de treino.
- Purge gap igual à maior janela entre subconjuntos adjacentes.
- Proibido separar aleatoriamente janelas sobrepostas.

## Ordem dos experimentos

1. Diagnóstico do dataset e missingness.
2. Zero-fill versus interpolação curta com flags.
3. Seleção controlada de janela e representação.
4. Comparação entre regras fixas, XGBoost, LSTM e TCN.
5. Mitigação do desbalanceamento e augmentation.
6. Ablação, permutation importance e estudos de caso.
7. Late fusion opcional.

## Regra do conjunto de teste

O fold de teste não participa de normalização, escolha de janela, escolha do limiar de gap, seleção de features, ajuste de hiperparâmetros ou decisão sobre augmentation.

## Registro da validação interna — G1 concluída

A geração G1 foi concluída com 48 combinações: quatro modelos (regras fixas, SVM, Random
Forest e XGBoost), três janelas (30, 60 e 150 frames) e quatro folds leave-one-video-out. Foi
utilizada a seed de qualificação 42, representação R0 achatada e distribuição original, sem
class weights, weighted sampling ou augmentation.

Pela validação interna, os principais resultados foram:

| Configuração | Macro F1 médio | F1 de Fatigue médio |
|---|---:|---:|
| SVM / 60 frames | 0,4191 | 0,0855 |
| XGBoost / 60 frames | 0,4024 | 0,0000 |
| Random Forest / 60 frames | 0,4022 | 0,0000 |
| XGBoost / 150 frames | 0,4017 | 0,0000 |
| Regras fixas / 30–150 frames | 0,0715–0,1597 | insuficiente |

Esses resultados constituem evidência preliminar de que fronteiras aprendidas superam os
thresholds fixos históricos no domínio estudado. Eles não confirmam H3, pois G1 não compara
os classificadores com os modelos temporais e ainda usa apenas uma seed. A classe Fatigue
continua com desempenho insuficiente e não há justificativa para promover qualquer modelo a
resultado final.

Os resultados externos são apenas descritivos. A decisão de executar G2 nas três janelas já
estava definida pelo protocolo e não foi tomada olhando os folds externos.

### Evidências preservadas

- `outputs/metrics/G1/g1_execution_table.csv`: uma linha para cada uma das 48 execuções;
- `outputs/metrics/G1/g1_runs.csv`: resultados de validação e teste por fold;
- `outputs/metrics/G1/g1_per_class.csv`: precision, recall e F1 por classe/fold;
- `outputs/metrics/G1/g1_per_class_statistics.csv`: média e desvio-padrão por classe;
- `outputs/metrics/G1/g1_confusion.csv`: matrizes de confusão em formato longo;
- `outputs/metrics/G1/g1_fold_statistics.csv`: média, desvio-padrão, mínimo e máximo;
- `outputs/metrics/G1/g1_prediction_distribution.csv`: distribuição das classes preditas;
- `outputs/metrics/G1/g1_resolved_config.json`: configuração e fingerprint resolvidos;
- `outputs/metrics/G1/g1_artifact_manifest.csv`: identificação, tamanho e SHA-256 dos
  checkpoints ou configuração fixa associados aos 48 runs;
- `outputs/figures/G1/`: comparações, distribuição entre folds e matrizes de confusão.

Checkpoints e predições individuais permanecem locais, ignorados pelo Git, mas são
identificados pelo manifesto versionável.

## Configuração congelada da G2

A G2 compara LSTM, TCN e Transformer em R0, distribuição original, seed 42, quatro folds e
janelas 30/60/150. Não usa class weights, weighted sampling ou augmentation. O batch size foi
congelado em 1024 após o smoke e antes da matriz oficial, para viabilizar a execução na RTX
2060 SUPER. Uma tentativa operacional parcial com batch 256 permanece preservada sob IDs
anteriores e não será agregada à G2 oficial.

## Registro da validação interna — G2 concluída

A matriz oficial G2 foi concluída com 36 execuções: LSTM, TCN e Transformer, janelas de 30,
60 e 150 frames, quatro folds, representação R0, seed 42 e distribuição original. Cada run
aplicou early stopping de forma independente. Não foram usados class weights, weighted
sampling ou augmentation.

Os resultados de validação que lideraram cada arquitetura foram:

| Configuração | Macro F1 médio ± DP | F1 de Fatigue médio ± DP |
|---|---:|---:|
| TCN / 60 frames | 0,4125 ± 0,0211 | 0,0000 ± 0,0000 |
| Transformer / 150 frames | 0,4081 ± 0,0136 | 0,0000 ± 0,0000 |
| LSTM / 60 frames | 0,4044 ± 0,0169 | 0,0000 ± 0,0000 |

O melhor resultado temporal, TCN/60, ficou abaixo do SVM/60 da G1 (`0,4191`) nesta etapa de
qualificação. Assim, G2 não oferece evidência para confirmar H3 sob R0 e distribuição original.
Esse resultado não refuta de forma geral o valor da modelagem temporal: ainda faltam as
comparações controladas de representação e desbalanceamento e as repetições multi-seed. A
classe Fatigue permaneceu sem detecções úteis em todas as nove combinações temporais.

Os desvios acima representam variação entre os quatro folds/sessões, e não estabilidade entre
seeds. Nenhum resultado de teste externo foi usado para ordenar ou congelar as configurações.
Para G3 ficam congelados, com base apenas na validação, TCN/60 como candidato principal e os
melhores representantes das outras arquiteturas, LSTM/60 e Transformer/150, como controles.

### Evidências preservadas da G2

- `outputs/metrics/G2/g2_execution_table.csv`: uma linha para cada uma das 36 execuções;
- `outputs/metrics/G2/g2_runs.csv`: validação e teste por run/fold;
- `outputs/metrics/G2/g2_history.csv`: loss e Macro F1 por época;
- `outputs/metrics/G2/g2_per_class.csv` e `g2_per_class_statistics.csv`: métricas por classe;
- `outputs/metrics/G2/g2_confusion.csv`: matrizes de confusão em formato longo;
- `outputs/metrics/G2/g2_fold_statistics.csv`: média, DP, mínimo e máximo entre folds;
- `outputs/metrics/G2/g2_prediction_distribution.csv`: distribuição das previsões;
- `outputs/metrics/G2/g2_resolved_config.json`: três configurações oficiais resolvidas;
- `outputs/metrics/G2/g2_artifact_manifest.csv`: caminhos, tamanhos e SHA-256 de 180 artefatos
  (melhor checkpoint, último checkpoint, log e predições de validação/teste de cada run);
- `outputs/figures/G2/`: curvas por época, matrizes, comparações e variabilidade entre folds.

Checkpoints, logs e predições individuais permanecem locais e ignorados pelo Git. Uma limitação
de reprodutibilidade deve ser registrada: o PyTorch avisou que o kernel de atenção eficiente do
Transformer em CUDA não oferece determinismo bit a bit, embora a seed e os demais controles
determinísticos tenham sido configurados.

## Registro da validação interna — G3 concluída

A G3 investigou exclusivamente missingness e representação. Foram reutilizados, após validação
dos fingerprints e equivalência numérica das entradas, os 12 runs R0 de LSTM/60, TCN/60 e
Transformer/150 da G2. Foram executados 24 novos treinamentos R1/R2, totalizando 36 comparações
em quatro folds, seed 42, distribuição original e sem qualquer balanceamento.

R1 manteve os cinco sinais e aplicou interpolação linear offline apenas em gaps internos de até
15 frames; gaps nas bordas, acima do limite ou sem dois vizinhos válidos receberam a mediana do
treino. R2 acrescentou `FaceDetected`, `WasInterpolated` e `MissingDurationSoFar`. As flags
binárias permaneceram em 0/1 e a duração foi calculada causalmente, sem duração futura do gap.
A interpolação foi isolada por vídeo, split e janela, e as medianas/scalers foram ajustados
somente no treino.

| Modelo | R0 Macro F1 | R1 Macro F1 | R2 Macro F1 |
|---|---:|---:|---:|
| LSTM / 60 | 0,4044 ± 0,0169 | 0,3163 ± 0,0082 | 0,3343 ± 0,0454 |
| TCN / 60 | 0,4125 ± 0,0211 | 0,3167 ± 0,0066 | 0,3617 ± 0,0564 |
| Transformer / 150 | 0,4081 ± 0,0136 | 0,3138 ± 0,0059 | 0,4112 ± 0,0232 |

R1 piorou o Macro F1 nos três modelos. R2 recuperou parte dessa perda e apresentou ganho pequeno
no Transformer (`+0,0032`), mas piorou LSTM (`−0,0702`) e TCN (`−0,0508`) contra R0. Nenhuma
representação produziu previsões úteis de Fatigue: F1 e recall permaneceram zero. Embora R1/R2
tenham elevado algumas probabilidades de Fatigue, elas não alteraram a decisão argmax.

Pelo critério pré-registrado, R0 avança para G4: média agregada de Macro F1 `0,4083`, contra
`0,3691` de R2 e `0,3156` de R1, além da menor complexidade. A seleção usou exclusivamente a
validação. Naquele ponto G4 ainda não havia sido iniciada e H3 permanecia em aberto. Os desvios continuam representando
variação entre folds com seed 42, não estabilidade entre seeds.

### Evidências preservadas da G3

- `outputs/metrics/G3/g3_audit.json` e `g3_dry_run.csv`: auditoria e matriz pré-treino;
- `outputs/metrics/G3/g3_execution_table.csv`: 36 comparações, distinguindo R0 reutilizado;
- `outputs/metrics/G3/g3_runs.csv`, `g3_history.csv` e `g3_per_class.csv`;
- `outputs/metrics/G3/g3_confusion.csv` e `g3_prediction_distribution.csv`;
- `outputs/metrics/G3/g3_paired_deltas.csv` e estatísticas pareadas entre folds;
- `outputs/metrics/G3/g3_fatigue_analysis.csv`: probabilidades, TP, FP, FN e contagens;
- `outputs/metrics/G3/g3_missingness_error.csv`: missing rate e taxa de erro;
- `outputs/metrics/G3/g3_resolved_config.json`: ordem das features, fingerprints e decisão;
- `outputs/metrics/G3/g3_artifact_manifest.csv`: 180 artefatos identificados por SHA-256;
- `outputs/metrics/G3/g3_report.md` e `outputs/figures/G3/`: relatório e gráficos.

Os dois smokes R1/R2 permanecem isolados sob `G3_smoke` e não foram incorporados. Os 24 runs
oficiais terminaram sem falhas ou retomadas e registraram 2.672,2 segundos de treinamento.

## G4 — mitigação isolada do desbalanceamento

A G4 foi concluída com 48 runs novos e 16 reutilizados. Foram comparados SVM/60, LSTM/60,
TCN/60 e Transformer/150 em R0, seed 42 e quatro folds. B aplicou pesos `N/(K*N_c)`; C usou
amostragem ponderada com reposição e `N_train` exemplos; D aplicou uma cópia leve por janela de
Fatigue/Distraction antes da padronização. Nenhuma estratégia foi combinada e validação/teste
permaneceram intactos.

O maior Macro F1 de validação foi SVM/D (`0,4203 ± 0,0597`), praticamente igual a SVM/A
(`0,4191 ± 0,0593`) e sem resolver Fatigue. LSTM/B apresentou o compromisso temporal mais
equilibrado: Macro F1 `0,4004 ± 0,0507`, F1 de Fatigue `0,0885 ± 0,0616` e recall `0,3920`.
TCN/C obteve o maior recall de Fatigue (`0,4785`), mas com mais falsos episódios. D não recuperou
Fatigue nos temporais.

LSTM/B e SVM/B são o par primário recomendado para as cinco seeds; TCN/C e SVM/C ficam como
análise secundária. H3 continua aberta. A advertência de não determinismo bit a bit da atenção
memory-efficient do Transformer/CUDA está registrada em `g4_resolved_config.json`.

### Evidências preservadas da G4

- `outputs/metrics/G4/g4_audit.json` e `g4_dry_run.csv`;
- `g4_runs.csv`, `g4_history.csv`, `g4_per_class.csv` e `g4_confusion.csv`;
- `g4_paired_deltas.csv` e análise operacional de Fatigue;
- `g4_treatment_audit.csv` e `augmentation/`;
- `g4_resolved_config.json` e `g4_artifact_manifest.csv`;
- `reports/g4_imbalance_results.md` e `outputs/figures/G4/`.

