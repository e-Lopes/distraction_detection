# Auditoria de aderencia ao plano integrado

Auditoria realizada em 13 de agosto de 2026, depois da leitura integral de
`Plano_de_Acoes_Pos_Banca_Integrado.pdf`. O PDF integrado e o pedido que originou esta
auditoria prevalecem sobre os roteiros anteriores. A inspecao abrangeu codigo, configuracoes,
manifestos, testes, outputs versionados e artefatos locais ignorados pelo Git.

## Resumo executivo

A fundacao de dados e avaliacao e aproveitavel: os quatro videos possuem manifesto anonimo,
as anotacoes foram normalizadas, as janelas e os quatro folds foram congelados e ha testes
contra vazamento. Os baselines classicos e a comparacao de preprocessing tambem sao ativos
uteis, mas foram executados com uma unica seed e, em geral, sobre R3; nao constituem ainda a
comparacao final exigida pelo plano.

O motor temporal em desenvolvimento possui LSTM, TCN, CUDA/AMP, early stopping e checkpoints,
mas ainda nao e um resultado oficial. A execucao local interrompida em
`outputs/logs/temporal_multiseed/` usou R2, apenas janela 60, class weights obrigatorios e
criterios antigos (`50/8/0.001`). Seus 15 runs completos e um run parcial foram preservados e
classificados como **exploratorios e incompativeis com G2**. Eles nao serao misturados com
novas geracoes.

Nao foram encontrados videos, checkpoints, modelos ou tabelas por frame rastreados dentro de
`fase_2`. Os artefatos pesados e sensiveis estao cobertos por `.gitignore`. As alteracoes ja
existentes nos SVGs de `outputs/figures/classical_60/` e em
`outputs/metrics/classical_baselines_summary.csv` pertencem ao usuario e devem ser preservadas.

## Matriz de aderencia

| Area | Requisito do plano | Estado atual | Evidencia | Acao recomendada | Prioridade |
|---|---|---|---|---|---|
| Schema | Schema canonico e validacao de tipos/faixas/NaN/Inf | Partial | `src/features/extract_facial_series.py`, `src/preprocessing/missingness.py`, `docs/data_dictionary.md` | Completar contrato de tipos, unidades, faixas plausiveis e validacao antes do treino | Alta |
| Manifesto | Dataset versionado, anonimo e rastreavel | Implemented | `configs/data/base.yaml`, `data/manifests/videos.csv`, `data/manifests/data_sources.csv` | Acrescentar data de congelamento/consenso quando disponivel | Media |
| Proveniencia | Hashes e origem das series faciais | Implemented | `extract_facial_series.py`, manifesto local ignorado, `docs/data_sources_and_provenance.md` | Manter o manifesto local associado a cada run | Alta |
| Targets | Separar classes comportamentais e estados operacionais | Implemented | `configs/data/base.yaml`, `annotations.py`, ADR 001, testes de anotacao | Manter essa separacao em todos os consumidores | Alta |
| Janelas | Janelas de 30, 60 e 150 frames | Implemented | `base.yaml`, `windowing.py`, `window_distribution.csv` | Reusar sem alterar limites congelados | Alta |
| Stride | Stride explicito, inicialmente 15 | Implemented | `base.yaml: stride_frames`, testes de janelas | Registrar no run resolvido | Alta |
| Rotulo | Maioria de 60%; empate/insuficiencia vira mixed | Implemented | `windowing.py`, `test_window_labels.py` | Manter mixed fora do treino primario e nos diagnosticos | Alta |
| Distribuicoes | Diagnosticos frame-level e window-level | Implemented | `frame_distribution.csv`, `window_distribution.csv`, `split_window_distribution.csv` | Gerar novamente apenas se dataset/protocolo mudar | Media |
| Missingness | Taxa e duracao de gaps por video/classe/janela/fold | Implemented | `missingness.py`, `facial_missingness*.csv`, testes | Vincular versao dos diagnosticos ao dataset | Media |
| Split externo | Leave-one-video-out | Implemented | `splits.py`, `temporal_splits.csv`, `test_temporal_splits.py` | Reusar os quatro folds congelados | Alta |
| Validacao | Blocos temporais continuos internos | Implemented | `leave_one_video_out.yaml`, ADR 004 | Nao escolher blocos olhando o teste | Alta |
| Purge | Gap bilateral minimo de 150 frames | Implemented | `splits.py`, `split_validation.md`, testes | Manter 150 como minimo para todas as janelas | Alta |
| Vazamento | Fit de imputacao/scaler/pesos apenas no treino | Implemented | `strategies.py`, `temporal_data.py`, testes de preprocessing | Ampliar testes de caracterizacao ao runner temporal | Alta |
| R0 | N x 5 com zero-fill | Implemented | `baseline_zero_fill.yaml`, `representation_features`, testes de cinco colunas e smoke G0 | Usar como entrada inicial de G1/G2 | Alta |
| R1 | N x 5 com gaps curtos tratados | Implemented | G3 validou interpolacao por janela/split e 24 runs R1/R2 | Nao promover: Macro F1 agregado `0.3156` | Alta |
| R2 | N x 8 com tres flags causais | Implemented | ordem, flags binarias e duracao causal testadas; G3 `0.3691` | Preservar para analise, mas R0 avanca para G4 | Alta |
| R3 | Features agregadas por janela | Implemented | `dummy_baseline.py`, `strategies.py`, baselines classicos | Reusar nos classicos; documentar que nao equivale ao B2a achatado | Alta |
| Regras fixas | Baseline B1 reproduzivel | Implemented | `g1_baselines.py`, `g1_baselines.yaml`, 48 runs G1 e testes de fronteira | Manter thresholds congelados; nao selecionar pela avaliacao externa | Alta |
| Classico achatado | B2a sobre R0 achatado | Implemented | `flatten_r0` e SVM/RF/XGBoost G1 sobre 30/60/150 nos quatro folds | Repetir apenas finalistas em G5; G1 ainda e qualificacao de uma seed | Alta |
| SVM | Comparacao cientifica | Implemented | `classical_baselines.py`, `classical_grid_search.py` | Reexecutar finalista no protocolo/seeds definidos | Media |
| Random Forest | Comparacao cientifica | Implemented | mesmos modulos e outputs `classical_*` | Reexecutar finalista; preservar importance existente como exploratoria | Media |
| XGBoost | Classico central | Implemented | mesmos modulos, CUDA configuravel | Consolidar busca menor e repeticoes; medir eficiencia | Alta |
| LSTM | Modelo temporal | Implemented | G2 completa em R0; melhor janela 60 (`0.4044 +/- 0.0169`) | Comparar R1/R2 em G3 | Alta |
| TCN | Modelo temporal causal | Implemented | G2 completa; melhor temporal TCN/60 (`0.4125 +/- 0.0211`) | Candidato principal de G3 | Alta |
| Transformer | Comparacao controlada | Implemented | G2 completa; melhor janela 150 (`0.4081 +/- 0.0136`) | Manter como controle em G3 e registrar ressalva de determinismo CUDA | Media |
| GPU | `device:auto` e CUDA | Implemented | smoke G0 na RTX 2060 SUPER, PyTorch 2.5.1+cu121 | Medir pico de memoria por run | Alta |
| Mixed precision | AMP configuravel | Implemented | autocast/GradScaler exercitados no smoke CUDA | Manter configuravel | Alta |
| Early stopping | val Macro F1, max, 150/15/0.002 configuraveis | Implemented | config G5 e validacao em `temporal_engine.py`; G0 usa override curto | Manter valores G5 congelados | Alta |
| Best checkpoint | `best_macro_f1.pt` | Implemented | motor e testes temporais | Preservar `best.pt` apenas nos runs exploratorios antigos | Alta |
| Last/resume | `last.pt` com estado completo e RNG | Implemented | `temporal_engine.py` salva/restaura modelo, optimizer, scheduler, scaler, RNG e generator; teste retoma epoca | Confirmar novamente no smoke das tres arquiteturas | Alta |
| Seeds | 42, 123, 456, 789 e 2026 para finalistas | Partial | runner/config em desenvolvimento contem as cinco seeds | Separar qualification de G5 e nunca escolher a melhor seed | Alta |
| Tracking | ID, geracao, config resolvida, status e fingerprint | Partial | runner agora usa ID com geracao/config/modelo/R/janela/fold/seed, fingerprint e dry-run; falta manifesto de status consolidado | Acrescentar status e config resolvida por geracao | Alta |
| Metricas | Macro F1, balanced accuracy e por classe | Implemented | `evaluate_predictions`, CSVs classicos e runner temporal | Acrescentar probabilidades/calibracao quando aplicavel | Alta |
| Operacional | Falsos alertas por hora e latencia de deteccao | Missing | apenas tempo de treino e metricas por janela | Implementar depois de estabilizar predicoes temporais | Media |
| Repeticoes | Todos folds x seeds; media/DP/mediana/min/max/IC95 | Partial | `stability.py` e runner em desenvolvimento | Validar agregacao e usar seed/fold como unidades, nao janelas | Alta |
| Class weights | Cenario G4 isolado | Conflict | classicos e motor temporal aplicam pesos obrigatoriamente | Parametrizar `balancing`; G2 deve usar `none` | Alta |
| Weighted sampling | Cenario G4 | Missing | apenas declaracao em `imbalance.yaml` | Implementar depois de G2 | Media |
| Augmentation | Jitter/scale/masking apenas no treino | Missing | apenas declaracao em `imbalance.yaml` | Implementar com limites fisicos e testes | Media |
| Ablacao | Retreino removendo grupos | Partial | classicos possuem grupos; temporal nao retreina grupos | Criar matriz G4/G5 sem zerar apenas no teste | Media |
| Explicabilidade | SHAP/permutation e estabilidade entre folds | Partial | importance interna de RF/XGB; pacote SHAP opcional | Acrescentar permutation/SHAP aos finalistas | Baixa |
| Estudos de caso | TP/FP/FN e alto missing rate | Missing | predicoes locais existem, sem gerador de casos | Implementar apos finalistas | Media |
| Graficos | Curvas por epoca e variabilidade entre seeds | Implemented | runner gerou curvas de loss/Macro F1, early stopping, heatmap e variabilidade em G0 local | Regenerar com todos os folds/seeds em G2/G5 | Alta |
| Testes | Dados, splits, preprocessing, modelos/checkpoints | Implemented | suite em `tests/`; 51 testes anteriores e 7 temporais passaram isoladamente | Ampliar resume/NaN/representacoes/dry-run e rodar suite completa | Alta |
| Reprodutibilidade | Comandos a partir da raiz e configs resolvidas | Partial | `README.md` cobre etapas antigas, nao o novo runner | Atualizar README depois do smoke | Alta |
| Fusao | Late fusion com probabilidades OOF | Not applicable yet | diretorio `src/fusion/` vazio | Nao iniciar antes do nucleo facial-temporal | Baixa |

## Classificacao dos resultados existentes

| Artefato | Classificacao | Justificativa |
|---|---|---|
| Manifestos, distribuicoes e validacao de splits | valido | Configuracoes, codigo, dados de origem e testes sao identificaveis. |
| Dummy baseline | potencialmente valido | Protocolo rastreavel, mas uma seed/configuracao e sem repeticoes finais. |
| Baselines classicos fixos | potencialmente valido | Folds corretos; usam R3, uma seed e balanceamento embutido. |
| Grid search classico | exploratorio | Seleciona por validacao, mas a grade e ampla e antecede a estrategia geracional. |
| Comparacao de preprocessing (108 runs) | exploratorio | Util para formular hipoteses; resultados externos nao podem selecionar configuracoes. |
| Temporal local interrompido | incompativel com o protocolo atual | R2/w60/pesos e early stopping antigos; 15 runs completos e um parcial foram preservados localmente. |
| G1 regras/classicos R0 achatado | potencialmente valido | Matriz completa, splits/fingerprint/config/checkpoints rastreaveis; ainda e qualificacao com uma seed. |
| G2 temporais R0 | potencialmente valido | 36 runs, quatro folds e artefatos rastreaveis; qualificacao com uma seed, sem ganho sobre SVM/60 e sem F1 de Fatigue. |
| G3 missingness/representacao | potencialmente valido | 12 runs R0 reutilizados e 24 novos R1/R2; fingerprints, deltas e 180 artefatos; ainda uma seed. |

## Componentes que devem ser reutilizados

| Caminho | Responsabilidade e qualidade | Dependencias/consumidores | Encaixe, adaptacao e risco de substituicao |
|---|---|---|---|
| `src/data/annotations.py` | Normaliza e valida intervalos; boa cobertura sintetica | CLI de dados, manifestos e testes | Reusar sem mudanca estrutural. Substituir arriscaria alterar targets congelados. |
| `src/data/windowing.py` | Constroi janelas e regra majoritaria | diagnosticos, baselines e temporal | Reusar diretamente. E a fonte unica desejavel para 30/60/150 e mixed. |
| `src/data/splits.py` | Gera/valida LOSO, blocos e purge | todos os treinadores | Reusar. Mudar invalidaria a comparabilidade dos resultados existentes. |
| `src/preprocessing/strategies.py` | Interpola por bloco, fill e flags sem cruzar splits | comparacao classica e temporal | Adaptar por mapeamento R0-R2, preservando a interface atual. |
| `src/preprocessing/missingness.py` | Diagnosticos de gaps e expansao de labels | CLI e temporal | Reusar sem refatoracao ampla. |
| `src/training/dummy_baseline.py` | Carrega series e define classes/agregacao basica | treinadores classicos/temporais | Reusar carregamento e taxonomia; evitar duplicar leitores. |
| `src/training/classical_baselines.py` | Modelos, metricas, fingerprints, figuras/checkpoints | grid, preprocessing e temporal | Reusar metricas/fingerprint; adaptar balanceamento somente via config. |
| `src/training/classical_grid_search.py` | Busca e checkpoint por candidato | experimento classico | Preservar como exploratorio; criar matriz geracional menor em vez de apagar. |
| `src/models/temporal.py` | Interface comum LSTM/TCN | motor e testes temporais | Adaptar incrementalmente; substituir perderia o trabalho e testes recentes. |
| `src/training/temporal_engine.py` | CUDA/AMP/early stop/checkpoints | runner temporal | Adaptar nomes, RNG, balanceamento e checks finitos; nao criar outro loop. |
| `src/training/temporal_data.py` | Sequencias e scaler train-only | runner temporal | Adaptar para representacao configuravel; manter retorno e metadados. |
| `src/training/stability.py` | Estatistica descritiva e IC t | runner multi-seed | Reusar; validar unidades independentes e pequenos n. |
| `data/manifests/*.csv` | Dataset/splits congelados e anonimos | todos os experimentos | Reusar como entradas imutaveis; regenerar apenas com nova versao. |

## Problemas criticos e proxima decisao

As correcoes sem ambiguidade sao: tornar R0-R2 explicitos, remover balanceamento implicito de
G2, adotar os parametros de early stopping do plano, tornar resume mais fiel, criar dry-run e
executar smoke. A escolha de hiperparametros finais, o limiar operacional de falsos alertas e
qualquer mudanca nos thresholds das regras fixas alteram resultados cientificos e devem ser
registrados antes de observar os testes externos.

## Evidencia de G0

Em 13 de agosto de 2026, o dry-run listou tres runs e o smoke controlado executou LSTM, TCN e
Transformer em CUDA, no fold 1, R0, janela 30, seed 42, tres epocas e no maximo 384 janelas por
subconjunto. Foram exercitados forward/backward, AMP, Macro F1 por epoca, verificacao de
NaN/Inf, checkpoint best/last/periodico, recarga do melhor estado, predicoes, probabilidades e
graficos. A retomada de `last.pt` da epoca seguinte e coberta por teste automatizado. Os
artefatos estao em `outputs/models/G0_smoke/` e `outputs/logs/G0_smoke/`, ambos ignorados pelo
Git. Os scores de G0 nao sao resultados cientificos e nao serao usados para selecionar modelos.

## Evidencia de G1

A matriz G1 concluiu 48 combinacoes (`4 modelos x 3 janelas x 4 folds`) com seed 42 e
distribuicao original, gerando 96 linhas de avaliacao. Pela validacao interna, o maior Macro F1
medio foi do SVM linear com janela 60 (`0.4191`), seguido por XGBoost 60 (`0.4024`), Random
Forest 60 (`0.4022`) e XGBoost 150 (`0.4017`). O baseline fixo ficou muito abaixo dos modelos
aprendidos (`0.1597`, `0.0904` e `0.0715` para 30/60/150), coerente com o vies de pitch da
camera lateral registrado na qualificacao. O F1 de fadiga permaneceu zero para quase todas as
combinacoes; SVM 60 foi a excecao mais alta, ainda baixa (`0.0855` na validacao).

Os valores externos em `outputs/metrics/G1/g1_report.md` sao somente descritivos e nao foram
usados nessa ordenacao. G1 nao e resultado final multi-seed: RF e XGBoost possuem aleatoriedade,
portanto qualquer configuracao promovida precisa das cinco seeds em G5.

## Evidencia de G2

A matriz G2 concluiu 36 runs (`3 modelos x 3 janelas x 4 folds`) em R0, seed 42 e sem
balanceamento. Pela validacao, TCN/60 obteve `0.4125 +/- 0.0211`, Transformer/150 obteve
`0.4081 +/- 0.0136` e LSTM/60 obteve `0.4044 +/- 0.0169`. O F1 de Fatigue foi zero em todas as
configuracoes. Como o SVM/60 da G1 atingiu `0.4191`, H3 nao e confirmada nesta qualificacao.
Os 36 runs, historicos, metricas por classe, matrizes, distribuicoes, configuracoes e hashes de
180 artefatos foram consolidados em `outputs/metrics/G2/`.

## Evidencia de G3

A G3 reutilizou 12 runs R0 da G2 após verificar fingerprints, hashes dos 180 artefatos e
equivalência numérica das entradas. Executou 24 novos runs R1/R2 na GPU, sem balanceamento.
O Macro F1 agregado por representação foi R0 `0.4083`, R2 `0.3691` e R1 `0.3156`. R2 superou
R0 apenas no Transformer/150 por `+0.0032`, sem consistência nos demais modelos. Nenhuma
representação produziu predições úteis de Fatigue. R0 foi congelada para a G4, posteriormente
concluída. A evidência completa e os hashes de 180 artefatos estão em
`outputs/metrics/G3/`.

## Evidência de G4

A G4 consolidou 48 runs novos e 16 reutilizados. SVM/D obteve o maior Macro F1 (`0,4203`), mas
não resolveu a classe rara. LSTM/B recuperou Fatigue com F1 `0,0885`, recall `0,3920` e Macro F1
`0,4004`; TCN/C elevou o recall para `0,4785`, com mais falsos episódios por hora. LSTM/B e
SVM/B são o par recomendado para a futura G5. H3 permanece aberta e nenhuma múltipla seed foi
executada.
