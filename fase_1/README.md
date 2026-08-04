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

## Dados e modelos

Arquivos de vídeo, frames, datasets completos, pesos treinados e informações internas da empresa não devem ser enviados ao GitHub. Consulte o `.gitignore` antes de adicionar novos arquivos.

## Referências acadêmicas do projeto

Esta fase está associada aos estudos sobre:

1. monitoramento corporal do operador em ambientes de teleoperação;
2. adaptação de domínio para detecção de telefone celular com YOLO11.

Os dados bibliográficos completos podem ser adicionados aqui após a publicação definitiva dos artigos.

