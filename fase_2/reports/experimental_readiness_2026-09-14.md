# Auditoria de prontidão experimental — 14/09/2026

## Decisão

A G2 histórica está completa para 30/60/150 frames. A comparação integral dos três
extratores **não está pronta**. A próxima rodada temporal permanece planejada, sem
promover resultados nem iniciar treinamento longo enquanto esses problemas persistirem.

Auditoria reproduzível (executar na raiz, com progresso):

```bash
fase_2/.venv/bin/python -u -m fase_2.scripts.audit_experiment_readiness
```

Artefatos: `outputs/metrics/experimental_audit/{extractor_integrity,window_coverage,g2_matrix}.csv`
e `readiness.json`. O último arquivo é diagnóstico, não autorização automática de treino.

## Janelas: geração não significa treinamento concluído

| Experimento | Janelas | Evidência e situação |
|---|---|---|
| G2 / R0 / sem balanceamento | 30, 60, 150 | 36 runs: LSTM, TCN e Transformer × quatro folds × três janelas; seed 42; métricas de validação presentes |
| G1 clássico e regras | 30, 60, 150 | Quatro folds reutilizados por modelo no registro final |
| G3/G4 | Subconjunto de 60/150 | Ablacões dirigidas aos candidatos; não equivalem ao produto cartesiano de todos os modelos/janelas/representações |
| Screening MiniRocket | 30, 60, 150 | 12 runs concluídos no registro final |
| Screening temporal_behavior_v1 | 60 | 16 runs clássicos concluídos; não avaliou essa representação em 30/150 |
| Screening shapelet | 60 | Quatro runs concluídos |
| Screening DTW | 60 | Quatro runs bloqueados |
| Confirmação | 60 | LSTM: 20 runs bloqueados; SVM: quatro bloqueados; não confundir com resultados |
| 90/300 frames, segundos fixos, multiescala | Ausentes desta matriz | Extensões propostas não são testes realizados; 300 era opcional no G48 |

A configuração vigente `configs/final_experiment.yaml` exige 30/60/150 frames,
stride 15 e maioria mínima de 60%. A G2 atende essa matriz histórica. FPS variam entre
vídeos: as janelas não têm duração física idêntica (30: cerca de 1,66–1,91 s;
60: 3,33–3,82 s; 150: 8,32–9,54 s). Uma análise em segundos exige versão própria,
resampling declarado e novo controle de sobreposição/purge; não substituir silenciosamente
a referência em frames. Aumentar para 300 exige revisar também `max_length` e purge.

## Comparação facial: estado dos arquivos atuais

| Extrator | Vídeo 1 | Vídeo 2 | Vídeo 3 | Vídeo 4 |
|---|---:|---:|---:|---:|
| MediaPipe: detecções | 18.264/32.013 | 23.591/28.200 | 20.462/30.334 | 13.302/31.790 |
| InsightFace: detecções | 22.294/32.013 | 26.419/28.200 | 21.439/30.334 | 15.125/31.790 |
| OpenFace | 0/32.013 (suspeito) | 0/28.200 (suspeito) | 0/30.334 (suspeito) | CSV final ausente; existe `.tmp` |

MediaPipe e InsightFace têm IDs, sequência e timestamps alinhados ao manifesto.
Isso confirma integridade estrutural, não precisão anatômica. Os três CSVs OpenFace
têm tanto `openface_tracking_failed` quanto `openface_output_missing`.
O parser atual já aceita CSV de imagem sem `success`; os arquivos são compatíveis com
o erro do parser anterior, mas a proveniência completa ainda precisa ser confirmada.
Não interpretar zero detecções como desempenho científico do OpenFace.

O relatório `g48a_all_frames_results.md` descreve uma extração anterior do vídeo 4,
com OpenFace positivo. Ele não representa o estado atual e não pode justificar promoção.
O smoke de 30 frames anteriormente descrito como validado também não demonstrou validade
de detecção apenas por terminar sem erro.

Há diferenças metodológicas adicionais: InsightFace seleciona a face pela região plausível
do operador; OpenFace escolhe a maior confiança; MediaPipe rastreia uma única face.
Antes de declarar superioridade, inspecionar falsos alvos na mesma amostra, padronizar o
critério de seleção ou declarar a comparação como sistemas completos com seletores distintos.
Não reutilizar thresholds EAR/MAR de outro extrator sem calibração em treino.
Latência do OpenFace está ausente; não comparar CPU com GPU como velocidade intrínseca dos modelos.
IC por frame não resolve autocorrelação: preferir diferenças pareadas com bootstrap em blocos,
relatório por vídeo e ressalva de um único operador. Cobertura não mede erro geométrico.

## Ordem de saneamento

1. Validar OpenFace nos 30 alvos congelados, incluindo casos anteriormente positivos;
   conferir arquivos esperados, schema, anatomia e seleção do operador.
2. Reextrair em uma raiz nova para preservar os CSVs suspeitos e a execução parcial.
   Não usar `--reuse-detections` nos CSVs zerados: essa opção preservaria o erro.
3. Validar todos os IDs, timestamps, máscaras e NaNs; calcular cobertura condicionada
   à presença do operador, falsos positivos na ausência e gaps por vídeo.
4. Gerar comparação pareada dos três extratores nos mesmos frames e relatórios separados
   por vídeo, com FPS do manifesto. O comando estatístico atual usa vídeo 4/FPS fixos por
   padrão e permite extratores ausentes: passar parâmetros e verificar completude explicitamente.
5. Congelar versão, pesos, ROI, seletor, modo temporal, hashes dos CSVs e hardware.
   Só então aprovar a entrada da próxima rodada temporal.

Comando de reextração após validação do smoke, executado pelo pesquisador no terminal:

```bash
fase_2/.venv/bin/python -u -m fase_2.scripts.g48a_run_openface_all_frames \
  --output fase_2/outputs/G48A_openface_revalidated --batch-size 250
```

A raiz nova contém somente OpenFace; é necessário compor um conjunto validado com os
outros dois braços antes de usar o comparador. Não trocar os caminhos do treinamento
apenas porque uma reextração terminou. Logs são persistidos em `logs/openface/` na raiz
de saída; a afirmação anterior de que seriam removidos estava incorreta.

## Quarto framework

Recomenda-se **RTMPose facial via MMPose**, como candidato opcional, após saneamento dos
três braços. Existe modelo facial RTMPose-m Face6 e demo oficial com detector de face.
Ele permite investigar se outra arquitetura melhora cobertura lateral e estabilidade.
Isso é uma hipótese, não uma vantagem comprovada neste dataset.

Fontes oficiais consultadas em 14/09/2026:
- https://mmpose.readthedocs.io/en/latest/model_zoo/face_2d_keypoint.html
- https://mmpose.readthedocs.io/en/latest/demos.html

Antes do piloto: congelar checkpoint e detector, verificar licença dos pesos/datasets,
mapear os 22 pontos anatomicamente (não copiar índices do InsightFace), testar GPU e medir
custo do detector junto com landmarks. Piloto na amostra congelada; ampliar só se trouxer
informação útil sobre falhas dos três atuais. Não adicionar framework corporal como substituto
da avaliação de olhos/boca.

## Validação desta auditoria

Matriz G2 conferida nos CSVs de execução, registro final agrupado por status, integridade
dos 12 caminhos de extração auditada. Dez testes existentes de contrato, parser OpenFace
e estatística passaram. Nenhuma nova extração ou rodada de treinamento foi iniciada.
