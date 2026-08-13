# 003 — Missingness estratificada

## Contexto

A extração canônica encontrou 46.677 frames sem face em 122.337 frames (38,15%). A taxa não é uniforme entre as classes manuais: 10,29% em alerta, 40,19% em fadiga e 61,91% em distração. Nas janelas de 150 frames, as taxas médias são respectivamente 10,59%, 46,29% e 63,67%.

Essa associação cria risco de o modelo aprender o padrão de falha do Face Mesh como atalho para a classe comportamental, especialmente para distração.

## Decisão

- Preservar `face_detected` como feature explícita e métricas ausentes como missing até a entrada do modelo.
- Manter zero-fill somente como baseline documentado, sem reinterpretar zero observado como ausência.
- Reportar missingness por vídeo, classe, tamanho de janela, fold e subconjunto.
- Avaliar separadamente o desempenho em janelas com baixa e alta missingness.
- Não escolher limiar de exclusão ou interpolação observando o fold de teste.

## Consequências

- Uma melhora de desempenho não será atribuída automaticamente aos indicadores faciais; será necessário verificar dependência da flag de detecção.
- Ablations com e sem `face_detected` passam a ser obrigatórias para interpretar os baselines.
- O desempenho em distração deve ser discutido junto à elevada taxa de falha facial dessa classe.
- Janelas `mixed` continuam disponíveis para auditoria, mas não entram no treinamento principal.
