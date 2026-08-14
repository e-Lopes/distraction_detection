# ADR 006 — Estratégias isoladas de desbalanceamento na G4

## Estado

Aceita em 2026-08-14. G4 concluída; G5 não iniciada.

## Contexto

G2/G3 suprimiram Fatigue nos modelos temporais. Comparar temporais balanceados com um SVM não
balanceado confundiria arquitetura e tratamento da classe rara. Por isso a G4 incluiu SVM/60,
LSTM/60, TCN/60 e Transformer/150 nos mesmos quatro cenários e folds.

## Decisão

- A reutiliza a distribuição original; B usa `N/(K*N_c)`; C usa amostragem com reposição e
  `N_train` itens; D cria uma cópia leve por janela minoritária.
- B, C e D não são combinados e atuam somente no treino.
- LSTM/B e SVM/B formam o par primário recomendado para futura repetição em cinco seeds.
- TCN/C e SVM/C são preservados como análise secundária de alta sensibilidade.
- SVM/D teve o maior Macro F1, mas não representa solução da classe rara.

## Consequências

H3 permanece aberta. Os resultados usam uma única seed e a atenção memory-efficient do
Transformer em CUDA emitiu advertência de não determinismo bit a bit. Nenhuma G5, fusão ou
avaliação externa adicional foi iniciada.
