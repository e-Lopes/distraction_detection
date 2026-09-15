# Resultados e gráficos

Abra [index.html](index.html) no navegador para consultar relatórios e figuras por
experimento. Há busca por nome, filtro por grupo e links PNG/SVG. Os dois formatos
ficam juntos; a galeria mostra uma única miniatura por par.

Atualize depois de gerar novos resultados:

```bash
python -m fase_2 results
```

| Grupo | Conteúdo |
|---|---|
| `current/final/` | Relatório e figuras da comparação final. |
| `current/measurement_v1/` | Qualidade das medidas faciais. |
| `current/modern_v1/` | Famílias modernas; criado conforme novos resultados. |
| `historical/<experimento>/` | Estudos G1–G48 e comparações anteriores; figuras em `figures/`. |
| `audits/` | Auditorias de integridade e prontidão. |
| `summaries/` | Sínteses acadêmicas e executivas. |

Os relatórios devem ser gerados a partir das métricas. A classificação `current`
identifica o perfil ativo, não confirma que o treinamento foi concluído ou validado.
O histórico mantém seus nomes G1–G48 para identificar a origem de cada gráfico.

Métricas, predições, logs e checkpoints continuam em [outputs/](../outputs/README.md).
Os links da galeria levam aos artefatos disponíveis. Não publique imagens que
contenham operadores ou dados internos; os arquivos locais continuam sujeitos às
regras de privacidade do projeto.

O [mapa de realocação](relocations.csv) registra o caminho antigo, o novo e o hash
após a migração. Use-o para localizar figuras citadas em manifestos históricos;
esses manifestos preservam os caminhos e hashes registrados na época.
