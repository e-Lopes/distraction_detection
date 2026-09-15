# Revisão do tratamento dos dados e da retomada

## O que foi verificado

- Janelas que cruzam divisões são descartadas antes de alimentar os modelos.
- Médias, escalas e medianas aprendidas são ajustadas somente no treino.
- A configuração principal R0 preenche ausências com zero, sem interpolação.
- Valores residuais em imagens sem detecção válida agora são descartados antes do preenchimento.
- A preparação confere identificação, sequência, tempos e valores das séries, limites das
  anotações, sobreposição e separação dos conjuntos. Também recalcula os diagnósticos atuais.

## Correções na persistência

- A fila executa uma combinação modelo/trecho/divisão/repetição por vez.
- O wrapper temporal filtra explicitamente divisão e seed e registra o checkpoint real,
  em vez de marcar um grupo inteiro como concluído ao retornar do runner.
- Os caminhos de dados e parâmetros de janelas da configuração são repassados ao runner temporal.
- Modelos clássicos são salvos atomicamente antes de avaliar e são carregados ao retomar.
- Scalers das sequências usadas por classificadores clássicos são preservados separadamente.
- Resultados e predições são persistidos por conjunto; predições temporais são gravadas antes
  do marcador de conclusão. O plano volta a indicar pendência se checkpoint ou predições sumirem.
- Redes salvam checkpoint e histórico a cada época. A extração salva o progresso por vídeo completo.

## Limites que permanecem explícitos

- A regra histórica da maioria considera apenas frames anotados no denominador. Sua adequação
  a janelas parcialmente sem rótulo precisa de uma decisão experimental, sem alteração silenciosa.
- A interpolação linear das representações opcionais é retrospectiva dentro da janela;
  ela não representa uma implementação causal por frame para uso em tempo real.
- Medir zero-fill e flags de ausência continua exigindo ablação para investigar atalhos do modelo.
- Uma verificação automática não valida semanticamente as anotações nem identifica por si só
  qual pessoa é o operador. A extração padrão usa uma face na imagem inteira.
- Promoção, parâmetros dos episódios, avaliação OOF completa, incerteza e limites operacionais
  continuam sujeitos ao protocolo científico. Resultados históricos permanecem identificados.

## Verificação local

Testes usam dados sintéticos para verificar uma fila real de classificadores, treino temporal
curto, retomada após falha, persistência por época e reinício de extração por vídeo.
Eles não substituem a execução com os vídeos originais. Na revisão inicial deste ambiente,
os quatro CSVs faciais e os vídeos nos caminhos configurados estavam ausentes.
