# Artefatos técnicos de execução

Para leitura de resultados, comece em [results/](../results/README.md).

| Pasta | Conteúdo |
|---|---|
| `final/`, `measurement_v1/`, `modern_v1/` | Registros e artefatos dos perfis atuais; criados conforme a execução. |
| `metrics/<experimento>/` | Métricas históricas e diagnósticos tabulares. |
| `models/<experimento>/` | Modelos e checkpoints históricos. |
| `predictions/<experimento>/` | Predições históricas. |
| `logs/<experimento>/` | Logs históricos. |
| `cache/` | Entradas intermediárias reutilizáveis. |
| `G48A_all_frames/` | Extrações faciais e seus registros. |

Os caminhos técnicos foram mantidos: registros, checkpoints e configurações
resolvidas contêm referências a eles. Relatórios e figuras foram agrupados em
`results/`, e os comandos/configurações foram atualizados para gravar lá.
Logs e configurações resolvidas de execuções passadas preservam os caminhos da época.

Uma alteração na configuração pode exigir nova validação de cache. A reorganização
não promove execuções antigas nem modifica seus fingerprints para simular compatibilidade.
