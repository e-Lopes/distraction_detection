# 005 — Política de interpolação e representações da G3

## Contexto

A G3 precisa comparar R0, R1 e R2 sem selecionar uma política de missingness olhando os folds
de teste. O código já possuía uma configuração anterior à G3 com interpolação linear, limite de
15 frames e preenchimento dos gaps longos pela mediana do treino. Não havia implementação de
forward fill ou mediana móvel pronta para uma comparação controlada.

## Decisão

- Congelar, antes dos resultados da G3, interpolação linear para gaps internos de no máximo 15
  frames.
- Aplicar a interpolação separadamente dentro de cada janela e de cada split; nunca usar pontos
  de outro vídeo, split ou fora da janela.
- Não interpolar gaps nas bordas nem gaps acima de 15 frames. Esses valores recebem a mediana
  calculada exclusivamente nos blocos de treino do fold.
- R1 contém apenas `[EAR, MAR, Pitch, Yaw, Roll]`.
- R2 contém, nessa ordem, os cinco sinais mais `FaceDetected`, `WasInterpolated` e
  `MissingDurationSoFar`.
- Manter as flags binárias em 0/1. Padronizar os cinco sinais e a duração acumulada somente com
  estatísticas do treino.
- Calcular `MissingDurationSoFar` causalmente desde o início do bloco do split, usando somente
  frames presentes e passados.

## Limitação

A interpolação linear usa o primeiro valor válido posterior ao gap. R1 e os sinais interpolados
de R2 são, portanto, adequados à análise offline. Esta G3 não fundamenta uma alegação de
inferência causal em tempo real. Uma variante online exigiria política própria em geração futura.

## Consequências

O limiar e o método não serão ajustados com base nos resultados intermediários da G3. Forward
fill e mediana móvel permanecem alternativas futuras e não serão misturadas nesta matriz. A G4
receberá apenas a representação escolhida pela validação da G3.
