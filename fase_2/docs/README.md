# Documentação da fase 2

## Hierarquia das fontes

Em caso de divergência, usar a seguinte ordem:

1. **Instruções explícitas mais recentes do pesquisador**.
2. **`Plano_de_Acoes_Pos_Banca_Integrado.pdf`**, referência científica, metodológica e de
   engenharia vigente; sua auditoria está em `integrated_plan_gap_analysis.md`.
3. **Decisões metodológicas registradas em `docs/decisions/`**, desde que não conflitem com o
   plano integrado ou com instrução posterior.
4. **`DocQualificacao.pdf`**, linha de base acadêmica entregue à banca, usada para preservar
   hipóteses, contribuições, resultados preliminares e limitações formalmente declaradas.
5. **Planos pós-banca anteriores e `docs/plano_pos_banca.md`**, referência
   complementar revisada, sem prevalência sobre o plano vigente.
6. Scripts e resultados exploratórios da fase 1, que exigem reconciliação antes de reutilização.

## Mapa da documentação

| Documento | Finalidade |
|---|---|
| `qualification_baseline.md` | Relacionar a qualificação aos ativos encontrados e ao trabalho restante. |
| `research_question_alignment.md` | Vincular pergunta, objetivos e hipótese às comparações e evidências exigidas. |
| `temporal_feature_protocol.md` | Congelar a avaliação dos novos atributos temporais antes dos resultados. |
| `data_sources_and_provenance.md` | Inventário detalhado, proveniência, contagens e limitações dos dados. |
| `data_dictionary.md` | Definir campos e taxonomias da fase 2. |
| `methodology/experimental_protocol.md` | Protocolo experimental reproduzível. |
| `integrated_plan_gap_analysis.md` | Aderência do código/artefatos ao plano integrado e componentes preservados. |
| `implementation_plan.md` | Sequência incremental G0-G6, testes, riscos e rollback. |
| `g45_implementation_plan.md` | Protocolo pré-resultados congelado para diagnóstico, Focal Loss e classificação hierárquica da G4.5. |
| `g48_mediapipe_alternatives_protocol.md` | Nova G4.8: comparação controlada de alternativas ao MediaPipe, do ground truth à decisão de Pareto. |
| `g48_e1_license_environment_audit.md` | Auditoria E1 separando licença de código, pesos, datasets e elegibilidade de transferência. |
| `g48_cvat_annotation_guide.md` | Exportação local, topologia e regras de anotação dos 22 landmarks da amostra G4.8. |
| `decisions/007-g45-threshold-focal-proxima-hierarquia.md` | Decisão após G4.5A/B e preparação da classificação hierárquica G4.5C. |
| `../reports/g4_imbalance_results.md` | Resultados, deltas, estabilidade e decisão da G4. |
| `months/mes_01_dados_protocolo.md` | Checklist operacional do mês atual. |
| `decisions/` | Decisões metodológicas que prevalecem sobre propostas anteriores. |

## Política para a fase 1

A fase 1 é histórica e permanece congelada. Nenhum arquivo entra automaticamente na fase 2.
Código relevante deve ser reimplementado com configuração, testes e caminhos relativos. Dados
legados só podem ser usados com proveniência, semântica, split e finalidade documentados.
