# Código do pipeline

Sequência de processamento:

1. `data/`: manifestos, anotações, alvos e divisões de avaliação.
2. `features/`: extração de indicadores e atributos por janela.
3. `preprocessing/`: ausência de detecção, qualidade e tratamentos dos sinais.
4. `models/` e `training/`: modelos, treinamento e retomada.
5. `evaluation/`: avaliação dos extratores e comparação de desempenho.

`pipeline.py` orquestra os experimentos; `workflow.py` executa o fluxo dos vídeos.
`desktop.py` e `terminal.py` são as interfaces. `reporting.py`, `modern_reporting.py`
e `measurement_reporting.py` geram os resultados para consulta.

Módulos G1–G48 são preservados para reprodução histórica. As pastas reservadas
`fusion/`, `explainability/` e `utils/`, que continham apenas descrições e não tinham
implementação nem consumidores, foram retiradas. Novos módulos devem ser criados
quando a funcionalidade existir.
