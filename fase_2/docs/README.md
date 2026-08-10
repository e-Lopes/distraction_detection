# Documentação da fase 2

## Hierarquia das fontes

Em caso de divergência, usar a seguinte ordem:

1. **Decisões explícitas mais recentes do pesquisador**, registradas em `docs/decisions/`.
2. **`PlanoPósBanca.pdf`**, plano de execução vigente para os próximos meses.
3. **`DocQualificacao.pdf`**, linha de base acadêmica entregue à banca, usada para preservar
   hipóteses, contribuições, resultados preliminares e limitações formalmente declaradas.
4. **`Plano_de_Acoes_Pos_Banca_Atualizado.pdf` e `docs/plano_pos_banca.md`**, referência
   complementar revisada, sem prevalência sobre o plano vigente.
5. Scripts e resultados exploratórios da fase 1, que exigem reconciliação antes de reutilização.

## Mapa da documentação

| Documento | Finalidade |
|---|---|
| `qualification_baseline.md` | Relacionar a qualificação aos ativos encontrados e ao trabalho restante. |
| `data_sources_and_provenance.md` | Inventário detalhado, proveniência, contagens e limitações dos dados. |
| `data_dictionary.md` | Definir campos e taxonomias da fase 2. |
| `methodology/experimental_protocol.md` | Protocolo experimental reproduzível. |
| `months/mes_01_dados_protocolo.md` | Checklist operacional do mês atual. |
| `decisions/` | Decisões metodológicas que prevalecem sobre propostas anteriores. |

## Política para a fase 1

A fase 1 é histórica e permanece congelada. Nenhum arquivo entra automaticamente na fase 2.
Código relevante deve ser reimplementado com configuração, testes e caminhos relativos. Dados
legados só podem ser usados com proveniência, semântica, split e finalidade documentados.
