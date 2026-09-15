# Diagnóstico de Macro F1 e plano de melhoria

Data: 15/09/2026. Base: rodada `modern_v1`, 32 execuções concluídas na RTX 4060,
quatro folds, seed 42, janela de 60 frames. Estes resultados são de validação interna.

## Avaliação

O maior potencial de melhora está na qualidade dos sinais e das anotações, na
diversidade dos eventos e na representação temporal. Os resultados sugerem que
estes fatores limitam diferentes arquiteturas. Não há evidência para prometer
um Macro F1 específico após as intervenções.

| Modelo | Macro F1 médio | F1 fadiga | F1 distração |
|---|---:|---:|---:|
| Transformer | 0,4418 | 0,2116 | 0,2927 |
| LSTM | 0,4332 | 0,1114 | 0,3323 |
| Inception individual | 0,4200 | 0,0644 | 0,3358 |
| TCN | 0,4096 | 0,1014 | 0,2981 |

O Transformer obteve 0,6220 em um fold e 0,3774/0,3882/0,3796 nos demais.
Sua liderança depende fortemente de uma divisão. Não demonstra generalização
para nova sessão ou novos operadores. O conjunto contém quatro sessões do mesmo operador.

Evidência: `outputs/modern_v1/metrics_long.csv`, `window_counts.csv`,
`missingness_by_class.csv`, históricos e `checkpoints/*/progress.json`.

## 1. Revisar erros, vídeos e rótulos

Há nove intervalos de fadiga, somando aproximadamente 84 segundos: 56,98 s no
vídeo 1, 22,99 s no vídeo 3 e 3,93 s no vídeo 4. Isso oferece pouca diversidade.

Revisar visualmente todos os eventos de fadiga, falsos positivos das classes
minoritárias, cabeça virada, olhos ocluídos, perda da face e limites temporais das
anotações. Conferir sincronização entre vídeos, rótulos e indicadores.

Pergunta central: o comportamento anotado está visível em EAR, MAR, pitch, yaw e
roll? Uma rotação de cabeça pode significar distração ou uma ação necessária na
cabine. Definir consistentemente se fadiga significa bocejo, fechamento ocular ou
estado sustentado de sonolência. Obter uma segunda anotação independente ajuda a
identificar ambiguidades; preservar versões e discordâncias, sem adaptar rótulos
para favorecer previsões.

Referência: [In-the-wild Drowsiness Detection from Facial Expressions](https://arxiv.org/abs/2010.11162),
que apresenta protocolo de coleta e diretrizes explícitas de anotação.

## 2. Corrigir a medição facial

| Classe original | Frames sem medidas faciais |
|---|---:|
| Alert | 10,3% |
| Distraction | 62,1% |
| Fatigue | 41,0% |

No fold 3, as janelas de fadiga do treino têm aproximadamente 66% de medidas
ausentes; as da validação têm 0%. Isso sugere diferença relevante entre aprender e
avaliar. Zero-fill pode codificar falhas do extrator como se fossem comportamento.

Comparar os extratores nos mesmos trechos e verificar anatomia dos pontos,
seleção do operador, pose lateral, oclusão, EAR por olho e confiança por medida.
Separar ausência do operador de falha de extração. Maior cobertura só ajuda se a
geometria estiver correta. Revalidar OpenFace antes de usar os arquivos suspeitos.
A interpolação não recupera informação de gaps longos; flags por si só não garantem ganho.

## 3. Ampliar os episódios independentes

No fold 1 há apenas 14 janelas de fadiga no treino e quatro na validação em w60,
ainda sobrepostas. Mais janelas dos mesmos episódios não produzem diversidade.

Coletar novas sessões com eventos verificáveis, variações de postura, iluminação
e condições de operação; incluir negativos difíceis (falar, olhar instrumentos,
movimentar a cabeça). Adicionar operadores caso esse seja o objetivo de generalização.
Pesos de classe e oversampling aumentam a influência dos exemplos existentes,
mas não acrescentam novos comportamentos.

## 4. Investigar contexto e sinais mais informativos

A rodada moderna usou 60 frames, aproximadamente 3–4 segundos. Hipótese:
combinar contexto curto (piscadas, fechamento ocular, movimentos) com contexto
mais longo (frequência e persistência). Começar com LSTM e Transformer e poucas
durações físicas predefinidas, sem busca ampla guiada pelo teste.

Janelas longas exigem rever rótulos, sobreposição e separação temporal; podem diluir
eventos curtos. Resampling e janelas em segundos devem constituir uma nova versão
explícita. Sinais de olhos, boca, orientação em relação à tarefa e ações corporais
podem acrescentar informação; a fase 1 contém componentes relevantes para distração.
Avaliar cada grupo separadamente por ablação.

## 5. Ajustar modelos após o diagnóstico

Há sinais de sobreajuste: no Inception, Macro F1 médio da última época foi
aproximadamente 0,85 no treino e 0,37 na validação. Testar busca pequena por redes
menores, dropout/regularização, taxa de aprendizado e pesos menos extremos.

Não aumentar épocas como primeira medida: os 16 treinos neurais usaram early
stopping, limite 150, paciência 15, melhora mínima >0,002 no Macro F1 da validação.
Foram 343 épocas no total; os pesos da melhor época foram restaurados.

Focal loss, interpolação, thresholds e hierarquia já tiveram ganhos pequenos ou
resultados negativos no histórico. Repetir requer hipótese nova.

## Próxima rodada recomendada

1. Auditoria visual dos erros, rótulos e qualidade dos sinais.
2. Comparação controlada de extração/representação com dois modelos.
3. Em paralelo, ampliar os episódios independentes.
4. Ajustes pequenos de modelos, seleção pela validação e confirmação de estabilidade.

Manter divisão por sessão, transformações ajustadas somente no treino e avaliação
final reservada em novos dados. Consultar repetidamente os mesmos resultados pode
favorecer escolhas sem generalização. Referência:
[scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## Decisão posterior: novo target principal binário

O pesquisador autorizou Atenção = `alert` e Distração = `fatigue` + `distraction`.
Neste target, “Distração” é o nome operacional do agrupamento e inclui fadiga;
não implica equivalência clínica ou comportamental entre os rótulos originais.

O agrupamento deve ocorrer antes da votação das janelas. Treinar novamente com
duas saídas, duas classes na perda e métricas próprias, preservando anotações e
resultados ternários. Ausência do operador, oclusão e falha de extração não viram
Distração automaticamente. Um Macro F1 binário não é diretamente comparável ao
Macro F1 ternário: são tarefas diferentes e a classe rara foi incorporada a outra.
Manter análise por rótulo original para verificar se o agrupamento esconde falhas
nos trechos de fadiga. Consulte o protocolo binário para execução na interface.
