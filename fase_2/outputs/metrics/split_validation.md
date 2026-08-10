# Diagnóstico dos splits temporais

Os quatro folds leave-one-video-out foram gerados com blocos contínuos, validação nos 20%
finais de cada vídeo de desenvolvimento e purge gap de 150 frames.

## Integridade temporal

- Sobreposição entre treino, validação e teste: nenhuma.
- Vídeo de teste presente em desenvolvimento: não.
- Purge gap menor que 150 frames: não.
- Janelas que atravessam blocos: não são atribuídas.

## Cobertura de fadiga

| Janela | Fold | Treino | Validação | Teste |
|---:|---:|---:|---:|---:|
| 30 | 1 | 4 | 25 | 68 |
| 30 | 2 | 72 | 25 | 0 |
| 30 | 3 | 72 | 0 | 25 |
| 30 | 4 | 68 | 25 | 4 |
| 60 | 1 | 0 | 21 | 67 |
| 60 | 2 | 67 | 21 | 0 |
| 60 | 3 | 67 | 0 | 21 |
| 60 | 4 | 67 | 21 | 0 |
| 150 | 1 | 0 | 22 | 62 |
| 150 | 2 | 62 | 22 | 0 |
| 150 | 3 | 62 | 0 | 22 |
| 150 | 4 | 62 | 22 | 0 |

## Conclusão

Os splits passam nas regras contra vazamento, mas a política de reservar sempre os 20% finais
não suporta seleção de um classificador de três classes em todos os folds. O vídeo reservado
para teste permanece intocado mesmo quando não contém fadiga. A posição dos blocos de
validação nos vídeos de desenvolvimento deve ser definida antes de congelar o protocolo.
