# Validacao facial CPU/GPU

> CORRECAO DA AUDITORIA: esta execucao usou seek por numero de frame. Nos videos
> de taxa variavel, isso nao garantiu correspondencia ao frame sequencial/rotulo.
> As comparacoes CPU/GPU continuam pareadas na mesma imagem, mas cobertura por
> estado e identificacao de frames precisam ser refeitas. O caso nominal
> video_03/28673 era, por timestamp, video_03/28545 (1686.299 s). Consultar
> `results/angle_audit` para a investigacao sequencial. Nao usar as taxas por
> classe abaixo como resultados validados.

Amostra diagnostica: 795 frames em 14 trechos. Nao e avaliacao final de classificacao.

## Interpretacao desta execucao

- O novo modelo produziu 161 frames com cinco indicadores em CPU e os mesmos
  161 em GPU; nao houve discordancia de presenca facial nos 795 frames.
  Assim, nesta amostra, a perda de cobertura nao foi causada pelo dispositivo.
- EAR e MAR tiveram MAE CPU/GPU de aproximadamente 0.000060 e 0.000049.
  Isso e muito menor que a diferenca observada entre o novo modelo e o legado.
- **A pose angular ainda nao esta validada.** No video_03, frame 28673 (fatigue),
  pitch foi 36.810 graus na CPU e 2.589 na GPU; roll foi -174.091 e 149.388,
  respectivamente. Diferencas circulares: 34.221 e 36.521 graus.
  A mediana da diferenca de pitch foi apenas 0.003054 grau e o percentil 95,
  0.013178. A media sozinha esconderia essa divergencia pontual importante.
  A rotina legada usa solvePnP iterativo, sem avaliacao de erro de reprojecao.
  Instabilidade geometrica e uma hipotese a investigar, nao uma causa comprovada:
  este teste nao salvou os landmarks completos para localizar a origem do salto.
- MediaPipe com tracking entregou 384 frames validos; sem tracking, 237;
  o novo modelo, 161. Ha evidencia nesta amostra de beneficio do tracking e de
  diferencas adicionais entre os pipelines. Nao e possivel atribuir toda a perda
  exclusivamente ao modelo ou garantir que toda deteccao adicional esteja correta.
- Tempos faciais medios: PyTorch CPU 35.05 ms/frame, CUDA 20.92 ms/frame
  (aproximadamente 1.68x). E um benchmark eager, sem CUDA graphs ou YOLO;
  nao equivale ao desempenho do visualizador completo.
- Nao houve face aceita nos 180 frames rotulados absent desta amostra.
  Isso nao elimina os falsos positivos encontrados nas execucoes completas.

**Decisao para o experimento:** manter os CSVs anteriores intactos e nao usar a
extracao nova como substituta validada. O proximo passo e uma auditoria visual e
geometrica, salvando landmarks e erro de reprojecao, incluindo o frame divergente
e sua vizinhanca. So depois fixar o extrator e iniciar o tratamento das lacunas.
Os trechos utilizados aqui passam a ser amostras de desenvolvimento diagnostico;
se orientarem ajustes, nao devem sustentar uma alegacao de teste intocado.

## Protocolo
- Maior intervalo continuo por video/estado, recorte central; selecao independente das deteccoes.
- CPU/CUDA: mesmos pesos, preprocessing, FP32 eager, limiares 0.5; sem tracking e sem filtro corporal.
- MediaPipe original: referencias com/sem tracking, reiniciadas por trecho. Nao sao ground truth.
- EAR usa apenas o olho mais a esquerda da imagem; angulos usam a convencao legada, sem recalibracao.
- Tempos excluem leitura, inicializacao e calculo dos indicadores; nao estimam a execucao completa.

## Cobertura por estado

```text
           backend       state  frames  faces  valid   mean_ms  valid_fraction
  mediapipe_static      absent     180      0      0  4.943519        0.000000
  mediapipe_static       alert     240    136    136 10.930579        0.566667
  mediapipe_static distraction     240     32     32  9.873633        0.133333
  mediapipe_static     fatigue     135     69     69 10.399609        0.511111
mediapipe_tracking      absent     180      0      0  5.139633        0.000000
mediapipe_tracking       alert     240    230    230  7.317136        0.958333
mediapipe_tracking distraction     240     79     79  8.876960        0.329167
mediapipe_tracking     fatigue     135     75     75  8.338689        0.555556
         torch_cpu      absent     180      0      0 11.467203        0.000000
         torch_cpu       alert     240    111    111 45.579731        0.462500
         torch_cpu distraction     240     21     21 39.025688        0.087500
         torch_cpu     fatigue     135     29     29 40.687883        0.214815
        torch_cuda      absent     180      0      0  7.612059        0.000000
        torch_cuda       alert     240    111    111 26.687734        0.462500
        torch_cuda distraction     240     21     21 22.569887        0.087500
        torch_cuda     fatigue     135     29     29 25.463696        0.214815
```

## Diferencas pareadas

```json
[
  {
    "reference": "torch_cpu",
    "compared": "torch_cuda",
    "frames": 795,
    "face_disagreements": 0,
    "common_valid": 161,
    "ear_paired_n": 161,
    "ear_mae": 6.030386299259922e-05,
    "ear_max": 0.0008313025731996604,
    "mar_paired_n": 161,
    "mar_mae": 4.886357350191393e-05,
    "mar_max": 0.0004997817395697157,
    "pitch_paired_n": 161,
    "pitch_mae": 0.22492244235401582,
    "pitch_max": 34.22100887009046,
    "yaw_paired_n": 161,
    "yaw_mae": 0.013783061396446111,
    "yaw_max": 0.7050757206524452,
    "roll_paired_n": 161,
    "roll_mae": 0.23520187332786355,
    "roll_max": 36.520839450957624
  },
  {
    "reference": "torch_cpu",
    "compared": "mediapipe_tracking",
    "frames": 795,
    "face_disagreements": 245,
    "common_valid": 150,
    "ear_paired_n": 150,
    "ear_mae": 0.12506690934116554,
    "ear_max": 0.2870455368747536,
    "mar_paired_n": 150,
    "mar_mae": 0.0385502196748983,
    "mar_max": 0.3028767356554811,
    "pitch_paired_n": 150,
    "pitch_mae": 7.159455965566655,
    "pitch_max": 85.39737979481924,
    "yaw_paired_n": 150,
    "yaw_mae": 9.23461711007388,
    "yaw_max": 165.4699582212346,
    "roll_paired_n": 150,
    "roll_mae": 11.86427349416675,
    "roll_max": 77.23779733792996
  },
  {
    "reference": "torch_cpu",
    "compared": "mediapipe_static",
    "frames": 795,
    "face_disagreements": 136,
    "common_valid": 131,
    "ear_paired_n": 131,
    "ear_mae": 0.10394938258124364,
    "ear_max": 0.24979675513946956,
    "mar_paired_n": 131,
    "mar_mae": 0.032236248746559754,
    "mar_max": 0.23588513688114884,
    "pitch_paired_n": 131,
    "pitch_mae": 6.025416818473802,
    "pitch_max": 89.98399793273995,
    "yaw_paired_n": 131,
    "yaw_mae": 10.945265121695343,
    "yaw_max": 167.33964106825772,
    "roll_paired_n": 131,
    "roll_mae": 6.4945449957370895,
    "roll_max": 61.13277569320134
  }
]
```

MAE angular usa distancia circular em graus; EAR/MAR sao adimensionais. Pares finitos por indicador.
Concordancia CPU/GPU nao demonstra acuracia anatomica. Cobertura tambem nao e acuracia.

## Continuidade do experimento
1. Inspecionar visualmente landmarks em trechos independentes, sobretudo falhas em distraction/fatigue.
2. Fixar extrator e convencoes; manter NaN e mascaras individuais. Nao interpolar lacunas longas.
3. Ajustar normalizacao/tratamento apenas no treino ou suporte permitido; nao usar query para calibrar.
4. Recontar janelas Alert/Not-Alert, excluir absent, separar support/query sem frames compartilhados.
5. Meta-treinar nos dados publicos; reservar os quatro videos para meta-avaliacao das pipelines A/B.

Esta auditoria nao modifica os dados de entrada, gera janelas ou executa treinamento.
