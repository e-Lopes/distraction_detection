# Analise da extracao dos quatro videos

Execucao: `20260916_183656_888825`.

## Fontes e configuracao

- Series: `video_01.csv` a `video_04.csv`, nesta pasta.
- Estatisticas: `video_01_estatisticas.csv` a `video_04_estatisticas.csv`.
- Parametros: `config.json`.
- Rotulos para o cruzamento: `teste_metodologia/results/classificacoes_frames_exatos.csv`, relativo a raiz do repositorio.
- Cruzamento por `video_id` e `frame_index`, com correspondencia unica.
- EAR calculado somente no olho com menor coordenada horizontal media na imagem.
- Nenhuma referencia angular aplicada (`angle_reference: null`).
- Convencao historica: `pitch=Y`, `yaw=Z`, `roll=X`. Nao comparar diretamente com limiares de implementacoes que usam outra convencao.
- Modas calculadas por intervalos de 1 grau para angulos e 0,01 para EAR/MAR. Estatisticas ponderadas por frames, somente sobre valores finitos em frames com deteccao aceita.

As conclusoes por estado dependem da qualidade e do alinhamento das anotacoes existentes. Os dois frames finais do video 4 nao possuem rotulo nessa base.

## Integridade e cobertura

Os quatro videos estao marcados como completos. Foram extraidos 122.337 frames, sem indices duplicados por video e com timestamps crescentes. Os cinco indicadores estao disponiveis em 75.619 frames (61,8%).

| Video | Frames | Frames com cinco indicadores | Cobertura geral | Cobertura em alerta | Em distracao | Em fadiga |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32.013 | 18.200 | 56,9% | 95,2% | 14,7% | 42,8% |
| 2 | 28.200 | 23.602 | 83,7% | 97,9% | 10,5% | Sem amostras anotadas |
| 3 | 30.334 | 20.468 | 67,5% | 97,8% | 36,8% | 53,3% |
| 4 | 31.790 | 13.349 | 42,0% | 91,5% | 28,6% | 70,0% |

O resultado de fadiga no video 4 envolve somente 30 frames anotados, dos quais 21 foram detectados.

| Estado anotado | Frames | Frames com indicadores | Cobertura |
|---|---:|---:|---:|
| Alerta | 72.869 | 70.008 | 96,1% |
| Distracao | 13.304 | 2.452 | 18,4% |
| Fadiga | 1.360 | 637 | 46,8% |
| Ausencia | 33.420 | 1.445 | 4,3% |
| Desconhecido | 1.382 | 1.077 | 77,9% |

O principal achado e a perda de deteccao associada ao estado anotado: a cobertura em distracao e muito inferior a cobertura em alerta. Selecionar somente janelas detectadas favorece situacoes de alerta e pode introduzir vies no experimento. A cobertura geral baixa do video 4 tambem reflete seu longo periodo de ausencia.

## Postura predominante

| Video | Moda EAR | Moda MAR | Moda pitch | Moda yaw | Moda roll |
|---|---:|---:|---:|---:|---:|
| 1 | 0,15 | 0,01 | -34 graus | -21 graus | -148 graus |
| 2 | 0,13 | 0,01 | -32 graus | -22 graus | -147 graus |
| 3 | 0,13 | 0,01 | -32 graus | -21 graus | -146 graus |
| 4 | 0,13 | 0,01 | -30 graus | -24 graus | -146 graus |

As modas indicam uma orientacao frequente relativamente consistente. Sao candidatas a referencia, mas nao comprovam postura neutra: representam apenas frames detectados, predominantemente anotados como alerta. Cada moda marginal tambem nao garante que a combinacao dos tres angulos tenha ocorrido simultaneamente.

Medianas somente nos frames detectados e anotados como alerta:

| Video | EAR | Pitch | Yaw | Roll |
|---|---:|---:|---:|---:|
| 1 | 0,156 | -32,767 | -19,011 | -147,480 |
| 2 | 0,134 | -32,029 | -21,226 | -146,892 |
| 3 | 0,134 | -31,945 | -20,586 | -145,839 |
| 4 | 0,136 | -30,336 | -22,903 | -146,253 |

Qualquer referencia para uso em modelos deve ser definida dentro do protocolo de treino/calibracao, evitando usar dados de avaliacao indiscriminadamente.

## EAR, lacunas e angulos

- 98,8% dos frames detectados e anotados como alerta apresentam EAR abaixo de 0,25. Esse limiar seria inadequado para este olho e enquadramento; nao interpretar esses valores automaticamente como olhos fechados.
- O olho selecionado foi `RIGHT_EYE` em todos os frames validos, exceto tres do video 3. A selecao e espacial, nao pelo nome anatomico.
- Somente dois valores de EAR excederam 0,6, ambos no video 3. Nenhum excedeu 1. Isso nao substitui validacao visual dos landmarks.
- No video 2 existe uma lacuna de aproximadamente 92,6 segundos, entre 797,266 e 889,866 segundos, dentro de um trecho anotado como distracao. Nao preencher lacunas desse tamanho por interpolacao.
- Foram aceitas 1.445 deteccoes em frames anotados como ausencia: 1.307 no video 1, 124 no video 3 e 14 no video 4. Inspecionar para distinguir outra pessoa na ROI, falso positivo ou divergencia de anotacao.
- Foram observados saltos absolutos superiores a 180 graus entre frames consecutivos validos: yaw com 77/48/43/20 ocorrencias e roll com 69/65/50/26, respectivamente nos videos 1/2/3/4. Podem envolver passagem entre -180 e +180 ou instabilidade. Medias, desvios e interpolacao linear de angulos exigem cuidado.

## Diagnostico dos detectores

O campo atual `face_detected` corresponde a aceitacao conjunta de Pose e FaceMesh, e nao ao resultado bruto independente do FaceMesh.

| Pose detectado | Face aceita | Frames | Interpretacao possivel |
|---|---|---:|---|
| Sim | Sim | 75.619 | Ambos detectaram; cinco indicadores disponiveis nesta execucao |
| Sim | Nao | 17.483 | Pose detectou, mas FaceMesh nao forneceu face |
| Nao | Nao | 29.235 | FaceMesh independente desconhecido pelo CSV atual |

Nos frames anotados como distracao, Pose detectou a pessoa em 99,6%. Isso aponta para a deteccao facial como principal dificuldade nesse subconjunto. Nao e possivel recuperar do CSV atual o resultado independente do FaceMesh quando Pose falhou.

## Desempenho

Execucao em modo terminal (`headless: true`). O FPS de processamento registrado foi aproximadamente 24,25 / 22,71 / 24,00 / 25,47 nos videos 1 / 2 / 3 / 4. Esse indicador usa o tempo medido no loop e nao representa o tempo total incluindo inicializacao e gravacao dos arquivos.

## Proxima etapa

Antes de tratar ou preencher os dados, registrar independentemente o resultado bruto de Pose e FaceMesh, mantendo separada a regra de aceitacao dos indicadores. Isso permitira distinguir as causas das falhas hoje agrupadas.

Depois desse diagnostico, avaliar validacao por indicador, lacunas curtas versus longas e referencias angulares. Preservar as series brutas e identificar explicitamente qualquer valor imputado. Nao usar a ausencia de face, isoladamente, como rotulo de distracao ou ausencia do operador.

Este documento registra a analise; nenhum tratamento de dados ou alteracao da logica de deteccao foi realizado nesta etapa.
