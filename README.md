# Fatigue and Distraction Detection for Heavy Machinery Teleoperation

Projeto de mestrado voltado à detecção de fadiga e distração de operadores durante a teleoperação de máquinas pesadas.

A pesquisa utiliza visão computacional e aprendizado de máquina para extrair e analisar sinais corporais e faciais do operador ao longo do tempo. O objetivo é desenvolver uma abordagem capaz de apoiar o monitoramento contínuo em ambientes industriais e de mineração, nos quais falhas de atenção podem representar riscos significativos à segurança.

## Organização do repositório

O desenvolvimento foi dividido em duas fases:

```text
distraction_detection/
├── fase_1/   estudos iniciais de monitoramento corporal e celular
└── fase_2/   modelagem temporal de fadiga e distração
```

### Fase 1 — Detecção de comportamentos e objetos

A primeira fase reúne os estudos iniciais realizados sobre o cenário de teleoperação:

- monitoramento da postura do operador;
- identificação das mãos fora da região dos controles;
- detecção da presença ou ausência do operador;
- detecção de uso de telefone celular;
- adaptação de modelos YOLO11 ao domínio da cabine de teleoperação.

Esses estudos demonstraram a importância de adaptar modelos de visão computacional ao ambiente real de operação.

Mais informações: [`fase_1/README.md`](fase_1/README.md).

### Fase 2 — Indicadores faciais e aprendizado temporal

A segunda fase concentra o núcleo atual da dissertação. A proposta utiliza MediaPipe Face Mesh e OpenCV para extrair cinco indicadores faciais por frame:

- EAR — Eye Aspect Ratio;
- MAR — Mouth Aspect Ratio;
- pitch;
- yaw;
- roll.

Esses indicadores são organizados em janelas temporais e utilizados para comparar regras fixas, classificadores clássicos e arquiteturas temporais, como LSTM e TCN.

Mais informações: [`fase_2/README.md`](fase_2/README.md).

Acesse o protocolo, as pendências e a execução pela interface desktop,
a partir da raiz do repositório: `python3 -m fase_2 interface`.

## Classes estudadas

- **Alert:** operador em estado normal de atenção.
- **Fatigue:** sinais compatíveis com fadiga ou sonolência.
- **Distraction:** atenção desviada da atividade de teleoperação.

Estados como falha de detecção facial, oclusão e ausência do operador são registrados separadamente como condições operacionais da observação.

## Tecnologias principais

- Python;
- PyTorch;
- OpenCV;
- MediaPipe Face Mesh;
- YOLO11;
- scikit-learn;
- XGBoost;
- LSTM e TCN.

## Privacidade e dados

Os vídeos foram coletados em um cenário operacional real e podem conter informações sensíveis da empresa e dos operadores. Por esse motivo, vídeos, frames, anotações completas, modelos treinados e outros dados internos não devem ser publicados neste repositório.

O Git deve conter apenas código-fonte, configurações experimentais, documentação, testes e resultados anonimizados cuja publicação seja autorizada.

## Autor

**Eduardo Lamy Lopes**  
Mestrado em Informática — Pontifícia Universidade Católica do Paraná (PUCPR)

Orientador: Prof. Dr. Marcelo Eduardo Pellenz  
Coorientador: Prof. Dr. Marco Antonio Simões Teixeira
