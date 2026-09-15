# Tratamento controlado dos indicadores — v1

Pergunta: tratar a qualidade das medições melhora Macro F1 e identificação de Fatigue, mantendo as mesmas janelas?
Relatório gerado pelo pipeline existente. O relatório histórico e suas seções/referências permanecem preservados; não importar conclusões históricas como evidência desta rodada.

Desenvolvimento somente. Quatro sessões de um operador; testes históricos já examinados.
SVM linear sobre trajetória achatada e LSTM, w60/stride15/seed42/class weights. Nenhuma janela excluída por qualidade.
QA: referência pareada (5 canais); demais: 5 valores + 5 máscaras de observação + 5 flags de interpolação = 15 canais.
QA usa medidas do mesmo extrator auditável; não é reprodução bit a bit dos arquivos antigos.
Diagnóstico missingness_logistic: somente médias das máscaras/flags (10 features); associação não implica causalidade.

| Variante | Habilitada | Política |
|---|---|---|
| QA | True | `{'reference_r0': True, 'reject_quality': False, 'eye': 'mean', 'interpolation_seconds': 0, 'smoothing_seconds': 0}` |
| QB0 | True | `{'reject_quality': False, 'eye': 'mean', 'interpolation_seconds': 0, 'smoothing_seconds': 0}` |
| QB | True | `{'reject_quality': True, 'eye': 'mean', 'interpolation_seconds': 0, 'smoothing_seconds': 0}` |
| QC | True | `{'reject_quality': True, 'eye': 'predominant', 'interpolation_seconds': 0, 'smoothing_seconds': 0}` |
| QD | False | `{'reject_quality': True, 'eye': 'predominant', 'reference_rotation': None}` |
| QE01 | True | `{'reject_quality': True, 'eye': 'predominant', 'interpolation_seconds': 0.1, 'smoothing_seconds': 0}` |
| QE02 | True | `{'reject_quality': True, 'eye': 'predominant', 'interpolation_seconds': 0.2, 'smoothing_seconds': 0}` |
| QES | True | `{'reject_quality': True, 'eye': 'predominant', 'interpolation_seconds': 0, 'smoothing_seconds': 0.1}` |

### Cobertura e resultados

Extração completa/preparação pendentes; nenhuma cobertura científica inferida da amostra.

### Classificação por tratamento, fold e seed

Nenhum treinamento oficial desta rodada. Sem ranking, ganho ou resultado de classificação.

### Comparações pareadas e cobertura

Arquivos de deltas pareiam modelo/fold/seed/métrica/classe com QA. QB0 versus QB deve ser examinado para separar qualidade de representação. Não usar janelas como réplicas.
Confusões e predições estão nos artefatos por run, sem limiar de abstenção. Cobertura de Fatigue/Distraction deve acompanhar qualquer ganho de F1.

### Falhas e custo

- final__screening__deep__lstm__qa__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qa__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qa__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qa__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb0__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb0__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb0__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb0__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qb__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__deep__lstm__qc__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qc__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qc__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qc__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe01__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe01__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe01__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe01__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe02__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe02__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe02__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qe02__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qes__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qes__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qes__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__deep__lstm__qes__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__missingness_logistic__qb__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__missingness_logistic__qb__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__missingness_logistic__qb__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__missingness_logistic__qb__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qa__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qa__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qa__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qa__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb0__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb0__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb0__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb0__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb__w60__fold_1__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb__w60__fold_2__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb__w60__fold_3__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qb__w60__fold_4__seed_42: blocked; Complete verified measurement_v1 raw extraction required (sample is not training data)
- final__screening__feature__svm__qc__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qc__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qc__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qc__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe01__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe01__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe01__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe01__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe02__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe02__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe02__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qe02__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qes__w60__fold_1__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qes__w60__fold_2__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qes__w60__fold_3__seed_42: blocked; Predominant eye requires installation/development verification
- final__screening__feature__svm__qes__w60__fold_4__seed_42: blocked; Predominant eye requires installation/development verification
Custo de classificação ainda não medido quando não há runs. A projeção da amostra não inclui revisão humana nem treinos.

### Limitações

ROI e continuidade espacial não certificam identidade. Após perda/ambiguidade, exige âncora humana para retomar.
Olho predominante não definido; C/E bloqueados até verificação. D desabilitada: sem referência operacional defensável.
Limiar geométrico é heurística de desenvolvimento; EAR baixo e MAR alto não são rejeitados pelo seu valor.
Pose com intrínsecos aproximados; erro baixo de reprojeção não comprova acurácia.
Interpolação usa futuro, não representa fechamento ocular observado. Mediana trailing 0,1s pode apagar eventos curtos e precisa de inspeção.
Protocolos e comandos: docs/protocols/measurement_quality_protocol.md.
