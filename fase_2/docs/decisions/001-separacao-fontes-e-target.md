# 001 — Separação entre fontes legadas e target temporal

## Contexto

A fase 1 produziu frames classificados por estados corporais, validação manual de imagens,
bounding boxes de celular e modelos YOLO. A fase 2 possui anotações temporais independentes
para alerta, fadiga, distração e ausência.

## Decisão

As anotações temporais da fase 2 são a única fonte do target comportamental principal.
`operator_absent` permanece estado operacional. Dados da fase 1 são fontes auxiliares e só
podem entrar como contexto ou late fusion após validação e split próprios.

Não haverá conversão automática entre as taxonomias. Imagens, labels, ZIPs e pesos não serão
copiados para o Git; somente inventários anonimizados, schemas e resultados consolidados
autorizados serão versionados.

## Justificativa

As classes da fase 1 não representam fadiga e distração temporal. Além disso, os frames são
subamostrados, têm contador global sem vínculo persistido com o vídeo e podem conter overlays.
Usá-los diretamente como target introduziria erro semântico e risco de vazamento.

## Consequências

- O pipeline facial-temporal pode ser concluído sem depender do detector de celular.
- A reutilização do legado exige reconstrução de proveniência e splits por vídeo/bloco.
- Late fusion permanece complementar e não altera a taxonomia principal.
