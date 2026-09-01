# G48A — resultado da G1 Smoke Test

Execução concluída em CPU nos 30 frames congelados. As contagens abaixo descrevem
disponibilidade do contrato, não precisão e não constituem ranking.

| Extrator | Faces/30 | EAR válido | MAR válido | pose válida |
|---|---:|---:|---:|---:|
| MediaPipe Face Mesh | 18 | 18 | 18 | 18 |
| InsightFace 2d106 | 28 | 28 | 28 | 28 |
| OpenFace 68 | 24 | 24 | 24 | 24 |

Por dificuldade, as detecções foram respectivamente `7/8/3` para Face Mesh,
`10/10/8` para InsightFace e `9/9/6` para OpenFace na ordem
fácil/intermediário/difícil.

## Validação do mapeamento

Os painéis dos 22 pontos foram revisados nos 30 alvos. Nos casos detectados, olhos,
boca, nariz e queixo ficaram em posições anatomicamente plausíveis nos três
frameworks. Dois falsos positivos iniciais do InsightFace correspondiam ao trabalhador
ao fundo quando o operador estava de costas; uma guarda espacial os converteu em
`operator_face_missing`. O OpenFace falhou em seis imagens e o Face Mesh em doze; as
medidas desses casos são NaN, nunca zero.

O tempo mediano observado foi 3,85 ms para a inferência do frame-alvo do Face Mesh e
aproximadamente 69 ms para o InsightFace. O lote do OpenFace não expôs temporização
por frame comparável, portanto esse campo permanece NaN e não deve ser usado em uma
comparação de desempenho.

## Decisão da G1

Os três mapeamentos são tecnicamente viáveis para cálculo dos indicadores nos frames
em que há face válida. InsightFace 2d106 e OpenFace podem avançar para uma eventual
G2 junto da baseline, mantendo as ressalvas de licença acadêmica, a guarda de identidade
do operador e a necessidade de avaliar precisão contra referência. Esta decisão não
declara um framework vencedor e não inicia a G2.

Relatório navegável e painéis locais: `fase_2/outputs/G48A_framework_smoke/report/index.html`.
