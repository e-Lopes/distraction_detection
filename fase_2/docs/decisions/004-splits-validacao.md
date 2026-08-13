# 004 — Blocos de validação com cobertura de fadiga

## Contexto

Os blocos finais de 20% deixavam treino ou validação sem janelas de fadiga em alguns folds. Isso impedia ajuste e seleção de modelos com Macro F1 nas três classes, apesar de existirem episódios de fadiga nos vídeos de desenvolvimento.

## Decisão

- Manter leave-one-video-out, com um vídeo inteiro e intocado como teste.
- Manter validação com aproximadamente 20% de cada vídeo de desenvolvimento.
- Posicionar os blocos de validação usando somente anotações dos vídeos de desenvolvimento.
- Permitir treino antes e depois de um bloco interno de validação.
- Aplicar purge gap de 150 frames em ambos os limites do bloco interno.
- Exigir `alert`, `fatigue` e `distraction` em treino e validação para janelas de 30, 60 e 150 frames.
- Não alterar testes sem fadiga nos folds 2 e 4.

## Consequências

Os quatro folds permitem ajuste nas três classes sem inserir exemplos artificiais. Métricas de fadiga permanecem não estimáveis nos testes dos folds 2 e 4 e devem ser reportadas como ausentes, não como zero. A posição dos blocos fica congelada em `configs/splits/leave_one_video_out.yaml`.
