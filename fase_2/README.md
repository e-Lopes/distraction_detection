# Deteccao temporal de distracao e fadiga

Pipeline experimental enxuto para classificar `Alert`, `Fatigue` e `Distraction` a partir de EAR, MAR, Pitch, Yaw e Roll. A interface publica possui somente tres etapas: `prepare`, `train` e `report`. Os nomes G1-G48 identificam apenas artefatos historicos preservados.

## Estrutura

- `configs/final_experiment.yaml`: unica configuracao para novas execucoes;
- `src/`: componentes reutilizaveis e orquestracao;
- `outputs/final/`: registro, metricas, predicoes, checkpoints e figuras finais;
- `reports/final_experiment_report.md`: relatorio cientifico canonico;
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

O resultado consolidado fica em [reports/final_experiment_report.md](reports/final_experiment_report.md). Gere-o novamente depois de novos treinos com o comando `report`; esse comando nunca treina modelos.

## Reprodutibilidade historica

Os runners, configuracoes e relatorios G1-G48 permanecem no repositorio para auditoria. Eles nao fazem parte do fluxo normal e seus resultados validos sao importados automaticamente, sem retreinamento. Consulte [docs/README.md](docs/README.md) e o apendice do relatorio final.
