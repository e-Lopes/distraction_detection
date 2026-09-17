# Auditoria inicial: identificacao de frames invalida

Esta execucao usou CAP_PROP_POS_FRAMES. A checagem do indice retornado nao foi
suficiente: no video 3, o pedido 28673 retornou timestamp 1686.299, que corresponde
ao frame sequencial 28545. O frame 28673 da extracao completa tem timestamp
1692.499. Portanto, NAO associar imagens/poses desta pasta aos indices nominais.

O teste CPU/GPU sobre a mesma imagem e a perturbacao geometrica ainda descrevem
essa imagem, mas os rotulos e a correspondencia aos saltos do CSV nao sao validos.
Usar a auditoria posterior com decodificacao sequencial e verificacao de timestamp.
