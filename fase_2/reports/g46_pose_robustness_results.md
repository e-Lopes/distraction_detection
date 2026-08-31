# Resultados G4.6 — robustez à câmera lateral

**Data:** 31/08/2026  
**Escopo:** seed 42, quatro folds, treino/validação, sem seleção pelo teste externo

A extração v2 processou integralmente os quatro vídeos e preservou a extração histórica. Foram
avaliadas 12 condições: três representações nos quatro folds, com o mesmo Random Forest de
controle e janelas de 60 frames.

| Representação | Macro F1 | Balanced accuracy | F1 Fatigue | F1 Distraction | Falsos episódios de Fatigue/h |
|---|---:|---:|---:|---:|---:|
| Baseline v2 | 0,3993 ± 0,0085 | 0,4241 ± 0,0257 | 0,0000 | 0,3234 | 51,19 |
| Geometria normalizada | 0,3989 ± 0,0175 | 0,4238 ± 0,0276 | 0,0122 | 0,3058 | 37,62 |
| Correção condicionada à pose | 0,4005 ± 0,0100 | 0,4230 ± 0,0275 | 0,0111 | 0,3112 | 44,43 |

## Dependência de pose

A correção Huber quadrática, ajustada apenas em frames `Alert` válidos do treino, reduziu a
correlação de Spearman EAR–pitch em todos os folds:

| Fold | Antes | Depois |
|---:|---:|---:|
| 1 | -0,593 | -0,056 |
| 2 | -0,726 | -0,208 |
| 3 | -0,760 | -0,263 |
| 4 | -0,733 | -0,283 |

A extrapolação do primeiro protótipo gerava valores extremos. O protocolo final limita o EAR
aberto esperado ao intervalo `q10–q90` observado exclusivamente no treino, evitando NaN, Inf e
razões explosivas. Yaw e roll são canonicalizados em `[-90°, 90°]` para retirar saltos
equivalentes de 180 graus da instalação lateral.

## Decisão

A correção resolveu parte importante do artefato geométrico, mas não satisfez o critério de
promoção classificatória: a melhoria média de Macro F1 foi pequena (`+0,0012`) e não ocorreu em
mais de um fold contra o baseline. A recuperação de Fatigue apareceu somente no fold 4.

Portanto, a extração v2 e os diagnósticos são preservados, mas a representação corrigida não é
promovida para cinco seeds nem substitui os finalistas da G5. O próximo estudo permitido é o
pré-treino externo dos classificadores auxiliares, condicionado à aquisição licenciada dos
datasets e ao manifesto subject-independent.

Métricas: `outputs/metrics/G46/`. Figuras temporais e dispersões: `outputs/figures/G46/`.
