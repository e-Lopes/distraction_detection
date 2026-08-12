# Fase 1 — Monitoramento corporal e detecção de celular

Esta pasta reúne os estudos iniciais do projeto de detecção de comportamentos de risco durante a teleoperação de máquinas pesadas.

O foco desta etapa foi investigar sinais corporais e objetos associados à atenção do operador, além de avaliar a capacidade de modelos de visão computacional pré-treinados em um cenário operacional real.

## Objetivos

- detectar a presença ou ausência do operador;
- identificar desvios de postura;
- verificar se as mãos permanecem próximas à região dos controles;
- detectar o uso de telefone celular;
- avaliar a necessidade de adaptação de domínio para a cabine de teleoperação.

## Monitoramento corporal

A abordagem de monitoramento corporal combinou estimativa de pose e detecção de pessoas para produzir sinais relacionados à atividade do operador.

Os principais estados analisados foram:

- **Posture deviation:** desvio da postura esperado durante a operação;
- **Hands off controls:** punhos fora da região definida para os controles;
- **Operator absence:** operador não detectado na cabine;
- **Normal:** nenhuma condição de risco identificada.

O protótipo utilizou MediaPipe Pose e YOLO11, com regras temporais para reduzir oscilações entre estados consecutivos.

## Detecção de telefone celular

Também foi avaliada a detecção de celulares no ambiente da cabine. O desempenho inicial de um modelo YOLO11 pré-treinado no COCO foi insuficiente para o domínio estudado, apresentando grande quantidade de objetos não detectados.

Para reduzir essa diferença de domínio, foi construído um conjunto de imagens do cenário real e realizado o treinamento de variantes YOLO11. Os resultados mostraram que a adaptação ao ambiente operacional é essencial para obter uma detecção confiável.

## Principais conclusões

- modelos genéricos podem não representar adequadamente o domínio da cabine;
- iluminação, posição da câmera, oclusões e EPIs afetam a detecção;
- modelos compactos podem alcançar bom desempenho após treinamento específico;
- postura, mãos, presença e celular fornecem evidências complementares sobre o estado do operador;
- indicadores corporais isolados não são suficientes para caracterizar fadiga.

## Relação com a Fase 2

Os resultados desta fase motivaram a análise de sinais faciais e de sua evolução temporal. A [`fase_2`](../fase_2/README.md) utiliza EAR, MAR e pose da cabeça para comparar classificadores clássicos e modelos temporais na detecção de alerta, fadiga e distração.

No futuro, as evidências corporais e de celular poderão ser integradas às probabilidades do modelo facial por meio de uma estratégia de late fusion.

## Extração temporal recuperada

O script [`analise_temporal_fadiga_distracao.py`](analise_temporal_fadiga_distracao.py), recuperado do ambiente legado TELEOP, extrai por frame:

- EAR e MAR;
- pitch, yaw e roll da cabeça;
- indicador explícito de detecção facial;
- estado heurístico de alerta, fadiga ou distração.

Os vídeos devem permanecer localmente em `fase_1/videos/`, com os nomes `1.mp4` a `4.mp4`. A região de interesse usada no estudo está registrada em `roi_config_temporal.example.json`. Execute a partir de qualquer diretório com:

```bash
python fase_1/analise_temporal_fadiga_distracao.py
```

Além dos gráficos e do relatório histórico, o script agora grava `series_temporais_Video_N.csv` em `fase_1/presentation_outputs/`. Essa pasta é ignorada pelo Git porque contém resultados individuais derivados dos vídeos.

Os gráficos agregados e o relatório da execução histórica foram preservados separadamente em [`resultados_legados/`](resultados_legados/README.md). Frames com imagens dos operadores não foram importados.

## Dados e modelos

Arquivos de vídeo, frames, datasets completos, pesos treinados e informações internas da empresa não devem ser enviados ao GitHub. Consulte o `.gitignore` antes de adicionar novos arquivos.

## Referências acadêmicas do projeto

Esta fase está associada aos estudos sobre:

1. monitoramento corporal do operador em ambientes de teleoperação;
2. adaptação de domínio para detecção de telefone celular com YOLO11.

Os dados bibliográficos completos podem ser adicionados aqui após a publicação definitiva dos artigos.

