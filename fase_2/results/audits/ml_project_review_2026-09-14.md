# Avaliação técnica de machine learning e IA — 14/09/2026

## Parecer

O projeto constitui uma base de pesquisa experimental bem instrumentada, mas a evidência
atual não sustenta um detector operacional de fadiga nem superioridade geral de redes
temporais sobre modelos clássicos. A principal prioridade é aumentar a confiabilidade
da medição e do alvo anotado, seguida da avaliação por evento. Aumentar a quantidade
de arquiteturas, isoladamente, não resolve essas limitações.

Escopo: leitura de código, configurações, manifestos e resultados das duas fases,
com inspeção mais profunda do pipeline ativo da fase 2. Não houve reanotação dos
vídeos, novo treinamento ou auditoria linha a linha de todos os módulos. Números de
desempenho abaixo são os resultados históricos reportados, não medições novas.

## Diagnóstico por componente

| Componente | Avaliação | Evidência e consequência |
|---|---|---|
| Problema de pesquisa | Coerente | Comparar o valor do contexto temporal frente a baselines é uma pergunta verificável. Separar Alert/Fatigue/Distraction de ausência e falha facial é importante. |
| Fase 1 | Exploratória e complementar | Postura, mãos e celular podem fornecer sinais adicionais; não equivalem ao alvo de fadiga nem justificam fusão automática. A avaliação presente não revalida seus pesos ou resultados. |
| Dados | Principal restrição | Quatro sessões, 122.337 frames e cerca de 119 minutos; o relatório registra só 84 segundos e nove intervalos de fadiga. Janelas sobrepostas não aumentam o número de eventos independentes. |
| Generalização | Restrita ao domínio observado | Leave-one-video-out mede transferência entre essas sessões. Não demonstra desempenho em novos operadores, câmeras ou ambientes; a auditoria registra um único operador. |
| Anotação | Precisa de evidência adicional | A validade do rótulo Fatigue merece revisão humana independente e concordância entre anotadores. EAR/MAR e movimentos não bastam para certificar o estado anotado. Não encontrei, nesta inspeção, um estudo de concordância concluído. |
| Extração | Gargalo concreto | Missingness histórico de 16,3% a 58,0% por vídeo. A auditoria recente registra OpenFace suspeito/incompleto, seletores de face diferentes e mapeamento anatômico a validar. Cobertura não é acurácia geométrica. |
| Divisões | Boa base | `src/data/splits.py` exige vídeos externos separados, blocos sem sobreposição e purge; `temporal_data.py` calcula normalização com treino. É preciso manter essa separação em cada extensão. |
| Representações | Comparação útil, interpretação limitada | R0/R1/R2 permitem estudar ausência e interpolação. Zero-fill pode codificar contexto de falha facial; é necessário confrontar controles de missingness. Interpolação retrospectiva deve ter seu atraso explicitado numa aplicação online. |
| Modelos | Cobertura já suficiente para a pergunta inicial | SVM, LSTM, TCN e Transformer fornecem uma comparação defensável. Famílias modernas são uma extensão controlada, ainda dependente de dados válidos e orçamento. |
| Métricas | Bom desenho, implementação/evidência parcial | Macro F1, métricas por classe e pior vídeo são adequados ao diagnóstico. Parâmetros de matching de eventos seguem nulos; faltam probabilidades OOF completas em parte do histórico. |
| Reprodutibilidade | Forte para protótipo acadêmico | Configurações, seeds, registros, checkpoints e testes são pontos fortes. Há relatórios históricos divergentes do estado atual e ambientes/dependências que precisam ser congelados. |
| Interface | Informação demais na operação diária | Cinco abas, protocolos completos e textos redundantes dificultam localizar ações. Consulta do plano era repetida no ciclo de atualização. A interface deve mostrar estados e ações, sem substituir o protocolo científico. |

## O que os resultados permitem afirmar

O [relatório final histórico](../current/final/final_experiment_report.md) apresenta
Macro F1 médio de aproximadamente 0,4191 para SVM/w60, 0,4044 para LSTM/w60,
0,4125 para TCN/w60 e 0,4081 para Transformer/w150. São configurações distintas,
e a diferença isolada não estabelece equivalência estatística nem superioridade causal.

O mesmo relatório registra LSTM/B com F1 de fadiga 0,0885 e recall 0,392, acompanhados
de aproximadamente 56,92 falsos episódios/hora. A estatística de episódios é um proxy
histórico baseado em janelas. Não deve ser apresentada como desempenho operacional
validado de um sistema de alarmes.

A formulação defensável é: há evidência exploratória de utilidade dos sinais e do
contexto temporal, com trade-offs importantes; a melhora robusta de fadiga e a
generalização fora dessas sessões permanecem em aberto. Mais seeds avaliam variabilidade
do treinamento, mas não substituem mais operadores ou eventos independentes.

## Riscos técnicos prioritários

1. **Extratores e identidade do operador.** Sanear OpenFace e revisar amostras pareadas,
   anatomia e continuidade do operador antes de comparar extratores ou promover novas entradas.
2. **Alvo raro.** Revisar os poucos episódios de fadiga, registrar critérios e concordância,
   e ampliar sessões/operadores quando possível. Não otimizar repetidamente sobre os mesmos eventos.
3. **Avaliação externa.** O protocolo declara seleção por validação. A presença de resultados
   de teste em relatórios anteriores não permite provar, por inspeção, que nunca influenciaram
   decisões. Congelar decisões futuras e reservar uma avaliação adicional não consultada.
4. **Eventos e causalidade.** Congelar duração mínima, tolerância de gaps e matching no
   desenvolvimento; avaliar falsos alarmes/hora, atraso e fragmentação em predições contínuas.
5. **Estado de execução versus relatório.** Um CSV ou relatório existente não prova que
   o modelo/checkpoint e as predições atuais estejam completos. A UI deve distinguir
   concluído, reutilizado, pendente, bloqueado e dependência ausente.
6. **Fingerprint excessivamente abrangente.** Os perfis modernos incluem todos os `.py`
   de `src/` no hash. Assim, uma mudança puramente visual pode invalidar compatibilidade.
   Recomenda-se separar identidade científica da identidade da aplicação, com migração
   explícita e testes; não alterar hashes antigos para forçar reaproveitamento.
7. **Pose COCO G4.7.** A verificação anterior encontrou duas falhas de pose sintética em
   código não alterado pela reorganização. Corrigir e validar separadamente antes de usar
   esse proxy como evidência; ele não é intercambiável com a pose facial de seis pontos.

## Sequência de maior valor

1. Validar extração, operador e anotações; congelar manifesto e versões.
2. Preparar entradas e conferir suporte por classe/vídeo após janelamento.
3. Comparar SVM e uma referência temporal sob as mesmas condições; manter controles de ausência.
4. Selecionar pela validação e executar apenas os finalistas em múltiplas seeds.
5. Completar avaliação por evento, incerteza por vídeo e custo fim a fim.
6. Ampliar avaliação independente antes de reivindicar generalização ou uso operacional.

As extensões modernas já planejadas podem ser executadas sob seus protocolos;
a recomendação de priorização acima não altera configurações nem promove candidatos.

## Redesenho da interface decorrente da avaliação

- Três abas: Experimentos, Resultados e Logs.
- Ações principais: Verificar dados, Iniciar/Continuar e Parar.
- Contadores e tabela de execuções; motivo completo acessível ao selecionar uma linha.
- Configuração, filtros, preparação e extração em opções, com documentos sob demanda.
- Resultados compatíveis separados do histórico; acesso ao relatório e à galeria.
- Retomada e condições científicas continuam no pipeline; abrir a interface não treina.

### Verificação da implementação

Os nove testes da interface passaram, incluindo widgets Tk em janela oculta,
troca entre os três perfis, abertura das opções, consulta assíncrona do plano,
execução de processo curto, interrupção e filtragem de resultados incompatíveis.
A suíte completa terminou com 213 testes aprovados, seis ignorados e as duas
falhas de pose COCO G4.7 já identificadas. Os links locais deste parecer foram
verificados. Nenhum experimento de treinamento foi iniciado nesta avaliação.

## Fontes inspecionadas

- [Manifesto de vídeos](../../data/manifests/videos.csv),
  [anotações](../../data/manifests/annotation_frame_intervals.csv) e
  [missingness histórico](../../outputs/metrics/facial_missingness.csv).
- [Auditoria dos extratores](experimental_readiness_2026-09-14.md) e
  [relatório final](../current/final/final_experiment_report.md).
- [Configuração principal](../../configs/final_experiment.yaml),
  [qualidade das medidas](../../configs/measurement_experiment.yaml) e
  [famílias modernas](../../configs/modern_experiment.yaml).
- [Splits](../../src/data/splits.py), [dados temporais](../../src/training/temporal_data.py),
  [pipeline](../../src/pipeline.py) e [visão geral da fase 1](../../../fase_1/README.md).

Como referência metodológica externa, ajustar transformações somente no treino e
isolar dados externos reduz vazamento de informação; ver a documentação oficial de
[boas práticas do scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
Divisões por grupo devem refletir a unidade de generalização desejada; ver
[validação cruzada por grupos](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
