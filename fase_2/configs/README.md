# Configurações

Os comandos são executados a partir da raiz do repositório; os caminhos de dados e
saídas são relativos a essa raiz. `extends` é relativo ao arquivo YAML que o declara.

| Perfil atual | Finalidade |
|---|---|
| `final_experiment.yaml` | Referência histórica, comparação e confirmação dos finalistas. |
| `measurement_experiment.yaml` | Qualidade das medidas; extração bruta e tratamentos QA–QES. |
| `modern_experiment.yaml` | Famílias modernas, com orçamento limitado. |

Os dois perfis novos herdam o primeiro. Selecione-os na interface ou passe `--config`.

As subpastas separam componentes: `data/` (fontes), `task/` (alvos), `features/`
(indicadores e mapeamentos), `preprocessing/` (tratamentos), `splits/` (divisões) e
`experiment/` (matrizes específicas, incluindo G1–G48). Estes caminhos permanecem
estáveis para reproduzir os comandos e identificar configurações antigas.
