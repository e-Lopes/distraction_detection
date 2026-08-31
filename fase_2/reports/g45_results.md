# G4.5 — resultados parciais

**Status:** G4.5A e G4.5B concluídas; G4.5C hierárquica pendente

**Seed:** 42

**Avaliação:** somente validação; nenhum arquivo de teste foi produzido pela G4.5B

## G4.5A — threshold cross-fit por sessão

O ajuste de threshold foi aplicado a LSTM/B, TCN/B e TCN/C. As janelas repetidas foram
deduplicadas por média das probabilidades, e cada vídeo-alvo foi excluído da calibração de seu
próprio threshold. A grade congelada foi de 0,05 a 0,50, em passos de 0,01.

| Modelo | Argmax: Macro F1 | Cross-fit: Macro F1 | Argmax: F1 Fatigue | Cross-fit: F1 Fatigue |
|---|---:|---:|---:|---:|
| LSTM/B | **0,4005** | 0,3390 | **0,0883** | 0,0512 |
| TCN/B | **0,4029** | 0,3608 | **0,0528** | 0,0298 |
| TCN/C | **0,4035** | 0,3811 | **0,0652** | 0,0102 |

O cross-fit piorou os três modelos. A hipótese de que o argmax era o principal responsável pela
supressão de `Fatigue` não foi sustentada. As probabilidades da classe rara não são separáveis de
forma estável entre sessões apenas por um threshold escalar.

## G4.5B — Focal Loss

A Focal Loss foi implementada com `gamma=2,0` e
`alpha_c=N/(3N_c)`, calculado exclusivamente no treino de cada fold. O `log_softmax` é executado
em float32 mesmo sob AMP, e a redução divide a soma das perdas ponderadas pela soma de
`alpha_y`. Não foram combinados class weights externos, weighted sampling, augmentation ou
threshold ajustado.

Foram concluídos oito runs: LSTM e TCN, janela de 60 frames, R0, quatro folds e seed 42.

| Modelo | Estratégia | Macro F1 | F1 Fatigue | Recall Fatigue | Precision Fatigue |
|---|---|---:|---:|---:|---:|
| LSTM | Class weights B | 0,4004 ± 0,0507 | 0,0885 ± 0,0616 | 0,3920 | 0,0501 |
| LSTM | **Focal** | **0,4044 ± 0,0366** | **0,1039 ± 0,0208** | **0,4067** | **0,0730** |
| TCN | Class weights B | 0,4029 ± 0,0168 | 0,0528 ± 0,0397 | **0,3971** | 0,0299 |
| TCN | **Focal** | **0,4035 ± 0,0145** | **0,0566 ± 0,0185** | 0,3436 | **0,0323** |

### Deltas contra class weights B

| Modelo | Δ Macro F1 | Δ F1 Fatigue | Interpretação |
|---|---:|---:|---|
| LSTM | +0,0039 | +0,0154 | ganho pequeno nas duas métricas e menor dispersão entre folds |
| TCN | +0,0006 | +0,0038 | efeito praticamente nulo; recall de fadiga diminuiu |

Focal Loss é promissora apenas para LSTM nesta qualificação. O ganho é pequeno e não autoriza
promoção antes da G4.5C e das repetições multi-seed. A LSTM/Focal ainda permanece abaixo do
melhor SVM/A em Macro F1 (`0,4191`).

## Reprodutibilidade

- configuração: `configs/experiment/g45_focal_w60.yaml`;
- resultados: `outputs/metrics/G45/focal_r0_w60__*.csv`;
- checkpoints: `outputs/models/G45/`;
- logs resolvidos: `outputs/logs/G45/`;
- predições: somente arquivos `__validation.csv` em `outputs/predictions/G45/`;
- curvas e estabilidade: `outputs/figures/G45/focal/`;
- testes automatizados cobrem equivalência com cross-entropy ponderada em `gamma=0`, exemplos
  fáceis/difíceis, alpha train-only, finitude e checkpoint/reload.

## Decisão provisória

- rejeitar ajuste de threshold cross-fit como melhoria;
- manter LSTM/Focal para a comparação final da G4.5;
- não promover TCN/Focal sobre TCN/B ou TCN/C;
- concluir a classificação hierárquica G4.5C antes de congelar candidatos para G5.
