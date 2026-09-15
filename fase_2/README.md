# Deteccao temporal de distracao e fadiga

Pipeline experimental para classificar `Alert`, `Fatigue` e `Distraction` a partir de EAR, MAR, Pitch, Yaw e Roll. A central desktop reúne o protocolo e as três etapas: `prepare`, `train` e `report`. Os nomes G1-G48 identificam artefatos históricos preservados.

## Comece por aqui

| Ordem | Pasta | O que consultar ou fazer |
|---|---|---|
| 1 | [docs/](docs/README.md) | Planejamento, protocolos, descrição dos dados e decisões. |
| 2 | [data/](data/README.md) | Disponibilizar vídeos, anotações e manifestos. |
| 3 | [configs/](configs/README.md) | Escolher o experimento e seus parâmetros. |
| 4 | [src/](src/README.md) e [scripts/](scripts/README.md) | Executar o pipeline pela interface ou CLI. |
| 5 | [results/](results/README.md) | Consultar relatórios e gráficos por experimento. |
| 6 | [outputs/](outputs/README.md) | Conferir métricas, registros, predições e checkpoints. |
| 7 | `tests/` | Verificar o funcionamento do código. |
| Auxiliar | [tools/](tools/README.md) | Containers e benchmark facial independente. |

**Gráficos:** abra [a galeria local](results/index.html), com filtro de experimento,
busca por nome e acesso às versões PNG/SVG. Atualize-a com `python -m fase_2 results`.

```text
fase_2/
├── docs/       planning/ · protocols/ · data/ · decisions/
├── data/       dados locais e manifestos
├── configs/    perfis atuais e configurações dos estudos históricos
├── src/        código do pipeline
├── scripts/    execução e auditorias auxiliares
├── results/    current/ · historical/ · audits/ · summaries/ · index.html
├── outputs/    artefatos técnicos de execução
├── tests/      testes automatizados
└── tools/      docker/ · facial-landmarks-benchmark/
```

Pastas de entrada ou saída surgem quando são necessárias. Não é preciso criar
pastas vazias para funcionalidades ainda não implementadas. A orientação para
notebooks exploratórios está em [docs/exploration.md](docs/exploration.md).

## Interface desktop

Na raiz do repositório, execute:

```bash
python3 -m fase_2 interface
```

A tela principal segue três passos:

1. **Experimento:** selecione o perfil no topo da janela.
2. **Verificar dados:** confira as medidas já extraídas e as divisões de avaliação.
3. **Iniciar / continuar:** extraia os vídeos que faltam, prepare os dados e treine em sequência.

As três abas são **Experimentos**, **Resultados** e **Logs**. A primeira mostra
contadores e execuções; selecione uma linha para consultar seu motivo de bloqueio.
**Resultados** reúne as avaliações compatíveis, o relatório e os gráficos.
Em **Opções**, ficam configuração YAML, filtros, caminhos, preparação, protocolo e
extração de medidas. O padrão é comparar modelos; a confirmação
dos escolhidos continua dependendo da promoção documentada pela validação interna.

Os processos executam em segundo plano e os logs aparecem na janela. Apenas uma
ação executa por vez. O botão **Parar** encerra o processo atual;
fechar a janela durante um treino também interrompe a execução antes de sair.
Após interrupções, consulte o plano e o registro antes de retomar. A aba de logs
mostra até 10.000 linhas recentes; os artefatos continuam nos caminhos canônicos.

A interface usa Tkinter e requer uma sessão gráfica. No Debian/Ubuntu, o suporte
Tk do Python é fornecido pelo pacote `python3-tk`. Não há servidor web ou biblioteca
gráfica instalada via pip. Os documentos técnicos opcionais são exibidos como texto Markdown.

## Extração, conferência e retomada

As medidas do rosto ficam em `data/interim/legacy_extraction`, um CSV por vídeo.
A extração usa o MediaPipe do projeto na imagem inteira, com uma face por imagem.
A adequação desse enquadramento ao operador deve ser conferida nos vídeos reais.
Cada vídeo concluído recebe um registro de extração e um hash do arquivo salvo.
Ao retomar, o vídeo interrompido é refeito; os vídeos concluídos e compatíveis são reutilizados.
Após cada vídeo, são gerados um painel com os cinco indicadores e cinco gráficos individuais,
em PNG e SVG, dentro de `results/current/final/figures/facial_indicators`.

O comando de verificação checa identificação, sequência e tempos das imagens,
valores faciais, limites das anotações e isolamento de treino, validação e teste.
A preparação recalcula a ausência de detecção por classe e a quantidade de trechos.
Isso verifica integridade dos arquivos, sem certificar a qualidade das anotações humanas.

Cada etapa de treino corresponde a um modelo, tamanho de trecho, divisão de avaliação
e repetição. O modelo ajustado é salvo antes da avaliação. Predições, métricas por classe
e matriz de confusão são salvas por conjunto avaliado. Redes neurais salvam `last.pt`
e `progress.json` a cada época, além do melhor checkpoint e das cópias periódicas.
Modelos clássicos retomam do ajuste salvo; se a interrupção ocorrer dentro de `fit`,
aquele ajuste será refeito. Nenhum algoritmo clássico é apresentado como incremental
quando sua biblioteca não oferece esse recurso.

```bash
python3 -m fase_2 extract
python3 -m fase_2 check-data
python3 -m fase_2 chain
```

A fila salva seu andamento em `outputs/final/chain_state.json`, pausa em caso de falha
e preserva as etapas anteriores. Componentes ausentes e bloqueios ficam pendentes;
eles não são registrados como treinos concluídos. A auditoria fica em
`outputs/final/data_audit.json`. O código de saída 2 indica pendências ou falha.
Novas extrações não herdam automaticamente o estado concluído dos modelos históricos.

A extração requer o extra `extraction`; as redes e representações sequenciais usam
o extra `temporal`. O ambiente deve ter as dependências do `pyproject.toml` instaladas.

Para consultar apenas o resumo, sem interação nem alteração de artefatos:

```bash
python3 -m fase_2 interface --summary
```

O resumo verifica disponibilidade de arquivos, compatibilidade do cache e estados
do plano. Ele não certifica completude das predições OOF ou validade científica.
O arquivo de promoção continua sendo uma decisão documentada com base na validação
interna. Parâmetros de episódios ainda indefinidos aparecem como pendências.
Os comandos abaixo continuam disponíveis para automação. A interface respeita os
bloqueios de promoção e custo do pipeline; abrir a janela não inicia treinamentos.

## Estrutura

- `configs/final_experiment.yaml`: referência histórica e comparação final;
- `configs/measurement_experiment.yaml`: qualidade das medidas faciais;
- `configs/modern_experiment.yaml`: comparação de famílias modernas;
- `src/`: componentes reutilizaveis e orquestracao;
- `outputs/final/`: registro, métricas, predições e checkpoints;
- `results/current/final/figures/`: gráficos da comparação final;
- `results/current/final/final_experiment_report.md`: relatorio cientifico canonico;
- `docs/` e `outputs/metrics/G*/`: metodologia e resultados historicos.

## Pre-requisitos

Python 3.10 ou superior. Instale o projeto a partir desta pasta com as dependencias adequadas ao uso. PyTorch e necessario somente para treino temporal; MediaPipe, OpenFace e videos brutos nao sao carregados por `status`, `--plan` ou `report`.

## Fluxo recomendado

```bash
python -m fase_2 status
python -m fase_2 prepare --config fase_2/configs/final_experiment.yaml
python -m fase_2 train --scope screening --plan --config fase_2/configs/final_experiment.yaml
python -m fase_2 train --scope screening --config fase_2/configs/final_experiment.yaml
python -m fase_2 train --scope confirmation --plan --config fase_2/configs/final_experiment.yaml
python -m fase_2 report --config fase_2/configs/final_experiment.yaml
```

Para executar as tres etapas em ordem:

```bash
python -m fase_2 train --family all --plan --config fase_2/configs/final_experiment.yaml
python -m fase_2 all --config fase_2/configs/final_experiment.yaml
```

`--plan` apenas expande a matriz, verifica o registro e mostra reutilizacoes e pendencias; nao carrega videos, GPU ou modelos. Retomada e cache sao o comportamento padrao (`--resume` e implicito). `--force` invalida essa economia e deve ser usado somente de forma consciente.

O `screening` usa apenas a seed 42 e validacao interna. Ele compara os paradigmas `feature`, `distance`, `shapelet` e `transform`; use `--paradigm` para filtrar. A `confirmation` usa multiplas seeds somente para finalistas promovidos e permanece bloqueada ate existir `outputs/final/screening_promotion.yaml`.

Shapelets, MiniROCKET e ensembles futuros usam a dependencia opcional TSC, que nao e necessaria para `status`, `prepare`, `report` ou `--plan`:

```bash
python -m pip install -e "fase_2[tsc]"
```

O DTW possui uma protecao adicional de custo. Runs acima do limite ficam bloqueados no plano; `--allow-expensive` existe apenas para uma autorizacao consciente apos revisar a estimativa.

## Progresso e resultados

O terminal mostra etapa/run, modelo, janela, fold, seed, tempo e motivo de reutilizacao ou falha. Use `python -m fase_2 status` para conferir o que existe e o que falta. O registro incremental fica em `outputs/final/run_registry.csv`.

O resultado consolidado fica no [relatório final](results/current/final/final_experiment_report.md). Gere-o novamente depois de novos treinos com o comando `report`; esse comando nunca treina modelos.

## Reprodutibilidade historica

Os runners, configuracoes e relatorios G1-G48 permanecem no repositorio para auditoria. Eles nao fazem parte do fluxo normal e seus resultados validos sao importados automaticamente, sem retreinamento. Consulte [docs/README.md](docs/README.md) e o apendice do relatorio final.
## Experimentos planejados em 14/09/2026 na interface

Na raiz do repositório, execute `python -m fase_2 interface` no ambiente Python
com as dependências do experimento. No seletor superior, escolha **Qualidade das
medidas · 14/09** ou **Famílias modernas · 14/09**. Também é possível abrir uma
configuração YAML própria, mantendo os caminhos relativos à raiz do repositório.

1. Consulte **Opções → Protocolo** quando precisar dos detalhes científicos.
2. Use **Verificar dados** e, se necessário, **Opções → Preparar entradas**.
   Confira os bloqueios selecionando uma linha em **Experimentos**.
3. Use **Iniciar / continuar** para treinar e **Resultados → Gerar relatório** para consolidar.
   **Opções** permite filtrar a família e a etapa de avaliação.

No experimento de medidas, abra **Opções** e selecione vídeo, frame inicial e tamanho da amostra
(até 90 frames, dentro de um bloco de treino do fold de desenvolvimento).
**Extrair** usa a extração bruta do protocolo. Para repetir uma amostra,
informe uma pasta de saída nova. A opção de vídeo completo exige âncora do operador
verificada no frame zero, registrada no YAML. Amostras não liberam treinamento;
o pipeline valida os dados completos e as condições de cada tratamento.

Os novos perfis mantêm resultados em suas pastas próprias e não executam a
extração legada automaticamente. A referência histórica continua disponível no
seletor. A próxima rodada temporal com novos extratores ainda depende do saneamento
descrito em [auditoria de prontidão](results/audits/experimental_readiness_2026-09-14.md).
