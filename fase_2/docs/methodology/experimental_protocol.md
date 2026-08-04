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

