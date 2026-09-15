# Documentação da fase 2

## Hierarquia das fontes

Em caso de divergência, usar a seguinte ordem:

1. **Instruções explícitas mais recentes do pesquisador**.
2. **`Plano_de_Acoes_Pos_Banca_Integrado.pdf`**, referência científica, metodológica e de
   engenharia vigente; sua auditoria está em `planning/integrated_plan_gap_analysis.md`.
3. **Decisões metodológicas registradas em `docs/decisions/`**, desde que não conflitem com o
   plano integrado ou com instrução posterior.
4. **`DocQualificacao.pdf`**, linha de base acadêmica entregue à banca, usada para preservar
   hipóteses, contribuições, resultados preliminares e limitações formalmente declaradas.
5. **Planos pós-banca anteriores e `docs/planning/plano_pos_banca.md`**, referência
   complementar revisada, sem prevalência sobre o plano vigente.
6. Scripts e resultados exploratórios da fase 1, que exigem reconciliação antes de reutilização.

## Mapa da documentação

Avaliação do projeto: [parecer de ML e IA](../results/audits/ml_project_review_2026-09-14.md).

Auditoria atual: [prontidão em 14/09/2026](../results/audits/experimental_readiness_2026-09-14.md).
Próxima etapa condicionada: [rodada temporal](protocols/temporal_next_round_protocol.md).

| Documento | Finalidade |
|---|---|
| `planning/qualification_baseline.md` | Relacionar a qualificação aos ativos encontrados e ao trabalho restante. |
| `planning/research_question_alignment.md` | Vincular pergunta, objetivos e hipótese às comparações e evidências exigidas. |
| `protocols/temporal_feature_protocol.md` | Congelar a avaliação dos novos atributos temporais antes dos resultados. |
| `data/data_sources_and_provenance.md` | Inventário detalhado, proveniência, contagens e limitações dos dados. |
| `data/data_dictionary.md` | Definir campos e taxonomias da fase 2. |
| `protocols/experimental_protocol.md` | Protocolo experimental reproduzível. |
| `planning/integrated_plan_gap_analysis.md` | Aderência do código/artefatos ao plano integrado e componentes preservados. |
| `planning/implementation_plan.md` | Sequência incremental G0-G6, testes, riscos e rollback. |
| `protocols/g45_implementation_plan.md` | Protocolo pré-resultados congelado para diagnóstico, Focal Loss e classificação hierárquica da G4.5. |
| `protocols/g48_mediapipe_alternatives_protocol.md` | Nova G4.8: comparação controlada de alternativas ao MediaPipe, do ground truth à decisão de Pareto. |
| `protocols/g48_e1_license_environment_audit.md` | Auditoria E1 separando licença de código, pesos, datasets e elegibilidade de transferência. |
| `protocols/g48_cvat_annotation_guide.md` | Exportação local, topologia e regras de anotação dos 22 landmarks da amostra G4.8. |
| `decisions/007-g45-threshold-focal-proxima-hierarquia.md` | Decisão após G4.5A/B e preparação da classificação hierárquica G4.5C. |
| `../results/historical/G4/g4_imbalance_results.md` | Resultados, deltas, estabilidade e decisão da G4. |
| `planning/mes_01_dados_protocolo.md` | Checklist operacional do mês atual. |
| `decisions/` | Decisões metodológicas que prevalecem sobre propostas anteriores. |

## Ampliação moderna (2026)

O [protocolo de famílias modernas](protocols/modern_families_protocol.md) descreve auditoria,
matriz de 32 fits de desenvolvimento, limites de interpretação e execução posterior
na RTX 4060. Configuração: `configs/modern_experiment.yaml`; execução sequencial
com progresso e log: `bash fase_2/scripts/run_modern_experiment.sh`, na raiz.
Nenhum treinamento oficial foi iniciado durante a implementação.

## Qualidade das medições faciais (2026)

[Protocolo controlado de qualidade](protocols/measurement_quality_protocol.md): auditoria de ROI,
EAR por olho, pose, máscaras e tratamentos temporais; amostras pequenas de desenvolvimento,
matriz limitada SVM/LSTM e comandos futuros. Perfil `configs/measurement_experiment.yaml`.

## Histórico preservado

A fase 1 é histórica e permanece congelada. Nenhum arquivo entra automaticamente na fase 2.
Código relevante deve ser reimplementado com configuração, testes e caminhos relativos. Dados
legados só podem ser usados com proveniência, semântica, split e finalidade documentados.
