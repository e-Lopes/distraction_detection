# Reorganização da fase 2 — 14/09/2026

## Estrutura adotada

- Documentação agrupada em `docs/planning`, `docs/protocols`, `docs/data` e `docs/decisions`.
- Benchmark independente e Dockerfiles auxiliares reunidos em `tools/`.
- Relatórios e gráficos agrupados em `results/current`, `results/historical`,
  `results/audits` e `results/summaries`; cada experimento mantém suas figuras juntas.
- Pastas vazias e módulos sem implementação removidos; orientação dos notebooks
  incorporada à documentação.
- Caminhos de checkpoints, registros, métricas, predições e dados preservados em `outputs/` e `data/`.

## Compatibilidade e verificação

Os comandos, a interface e os perfis de configuração apontam para os novos destinos
de relatórios e figuras. Os parâmetros científicos não foram alterados. Configurações
resolvidas e manifestos históricos preservam suas referências originais; o
[mapa de realocação](../relocations.csv) relaciona os caminhos antigos aos novos.
Fingerprints não foram adulterados: caches podem exigir revalidação após mudança de configuração.

- 312 arquivos de figuras movidos e verificados por SHA-256, sem alteração de conteúdo.
- 694 links da primeira versão da galeria verificados, sem destinos ausentes.
- Plano de execução antes/depois: 20 bloqueados, 16 pendentes, 16 com dependência ausente
  e 8 reutilizados, considerando todas as etapas.
- Consulta `python -m fase_2 interface --summary` executada com sucesso.
- Suíte final: 210 testes passaram, 7 ignorados e 2 falhas de pose COCO em
  `test_g47_schema.py`. O teste e o módulo `g47_schema.py` correspondem ao HEAD
  anterior à reorganização; as falhas foram preservadas para correção específica.
- Não foram iniciadas extrações nem rodadas de treinamento dos experimentos.

O teste gráfico da interface permanece dependente de execução explícita em sessão gráfica.
