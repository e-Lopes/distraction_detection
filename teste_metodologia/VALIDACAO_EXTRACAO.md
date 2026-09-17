# Etapa 0: validar a extracao

## Aviso da investigacao angular

A auditoria descobriu que `CAP_PROP_POS_FRAMES` nao localizou os mesmos frames
da leitura sequencial nestes videos. O comparador foi corrigido para avancar
sequencialmente. As taxas por estado do comparativo inicial precisam ser refeitas;
as imagens CPU/GPU eram iguais entre si, mas podiam receber o rotulo de outro frame.
O caso referido antes como video_03/frame 28673 corresponde ao frame sequencial
28545, timestamp 1686.299 s. O frame 28673 real fica em 1692.499 s.
Isso nao afeta a copia dos CSVs completos nem o teste de janelas importadas,
que nao usaram seek de video.

`python teste_metodologia/investigate_angles.py` audita saltos dos CSVs completos,
reextrai CPU/GPU sequencialmente com verificacao de timestamp, salva landmarks,
imagens e compara solvePnP iterativo/SQPnP. Nao altera as medidas originais.

## Importacao dos resultados completos e teste GPU

Os resultados GPU completos da fase_1 foram copiados, sem modificar a origem, para
`results/imported_fase_1/20260916_204539_018100/`. A copia inclui os quatro CSVs,
estatisticas, diagnosticos, configuracao e auditoria GPU. Os hashes de cada arquivo
ficam nos manifestos das execucoes abaixo. Uma copia existente diferente provoca
erro, nunca sobrescrita silenciosa.

Sequencia segura na raiz do repositorio:

```powershell
python teste_metodologia/run_imported_gpu_smoke.py
python teste_metodologia/run_imported_gpu_smoke.py --window-len 15
python -m unittest teste_metodologia.test_imported_gpu_smoke teste_metodologia.test_compare_extraction_backends
```

O primeiro comando verifica as janelas de 90 frames; o segundo e exclusivamente
um teste funcional curto. Ambos geram diretorios novos. **Nao executar a sequencia
numerada legada como continuacao desta etapa:** ela ainda usa o CSV anterior,
preenche lacunas e o terceiro script meta-treina nos videos proprios.

Resultados de 2026-09-16:

| Video | Alert / Not-Alert, 90 frames | Alert / Not-Alert, 15 frames |
|---|---:|---:|
| 1 | 12 / 0 | 359 / 7 |
| 2 | 9 / 0 | 396 / 0 |
| 3 | 10 / 1 | 518 / 8 |
| 4 | 3 / 0 | 242 / 7 |

- 122337 frames importados; dois sem rotulo manual mantidos como unknown.
- Janelas sem overlap, sem preenchimento, totalmente validas e homogeneas na
  classe binaria. As contagens NAO sao comparaveis as antigas janelas preenchidas
  com stride 15/min_valid_ratio 0.5. Exclusoes por motivo e cobertura por estado
  estao salvas para nao esconder o vies desse filtro estrito.
- 90 frames: nenhum episodio viavel com K=1/Q=1.
- 15 frames: tres episodios, videos 1/3/4, seis forwards auditados em cuda:0
  (RTX 2060 SUPER). Video 2 inelegivel; oito testes automatizados passaram.
- Encoder aleatorio, BatchNorm em eval, normalizacao apenas no suporte; nenhuma
  atualizacao de pesos ou checkpoint. Acuracia nao foi divulgada como avaliacao.
- Janelas adjacentes ainda podem ser correlacionadas; ausencia de overlap nao
  demonstra independencia temporal. Definir separacao temporal na avaliacao final.
- A geometria angular continua pendente. Nao se adotou uma correcao automatica nem
  suavizacao para esconder saltos. Nao se alterou o plano para janelas de 15 frames.

Relatorios: [90 frames](results/imported_gpu_smoke/20260916_221328_887283/analysis.md)
e [teste GPU de 15 frames](results/imported_gpu_smoke/20260916_221345_687931/analysis.md).

## Comparacao controlada dos extratores

Esta etapa antecede o tratamento dos dados e a comparacao das pipelines A/B do
`plano_teste_metodologia.md`. Nao modifica `fase_1`, nao imputa sinais e nao treina.
Importa as formulas legadas e reutiliza, somente para leitura, os pesos e helpers
ja preparados em `fase_1/.gpu_runtime`. O comparador precisa de PyTorch CUDA,
MediaPipe, OpenCV, pandas e das dependencias do visualizador instalado.

Na raiz do repositorio:

```powershell
python teste_metodologia/compare_extraction_backends.py
python -m unittest teste_metodologia.test_compare_extraction_backends
```

O padrao e ate 60 frames consecutivos no centro do maior trecho de cada estado
em cada video. Estados ausentes no video nao geram trechos; intervalos curtos
nao sao completados artificialmente. Para ampliar: `--frames-per-clip 120`.
CPU e GPU sao executadas deliberadamente no teste pareado; nao e fallback.
Sem CUDA, o teste para. O visualizador continua com sua politica anterior.

Cada execucao cria `results/backend_comparison/<data_hora>/`:

- `metadata.json`: trechos, ROI, hashes dos modelos/helpers e rotulos, versoes,
  configuracao numerica e marcador de conclusao.
- `frames.csv`: quatro backends por frame, indicadores, mascaras, confianca
  de presenca do novo modelo, tempo e motivos de falha.
- `coverage.csv`: cobertura por backend/estado.
- `paired_comparison.csv`: discordancias de deteccao e diferencas por indicador,
  somente nos pares finitos, com contagens explicitas.
- `analysis.md`: protocolo, resultados e limites da comparacao.

Primeira execucao: [analise e decisao](results/backend_comparison/20260916_220506_284029/analysis.md).
Foram 795 frames, 14 trechos e quatro testes automatizados aprovados.
CPU/GPU concordaram na deteccao, mas houve um salto angular pontual relevante;
o novo modelo ainda nao deve substituir o legado no experimento.

## Limites e proximo passo

Nao ha filtro de pose corporal: estamos isolando a medicao facial. Cada referencia
MediaPipe reinicia por trecho, sem aquecimento temporal anterior. Inicializacao e
leitura de video ficam fora do tempo de inferencia. Nenhum backend e ground truth.
A amostra por estado nao representa a prevalencia real dos videos inteiros.
Ainda falta comparar CUDA graphs, usados na extracao completa, com o modo eager.

Antes de tratar dados, salvar landmarks e erro de reprojecao e revisar visualmente
frames problematicos, sobretudo video_03/frame 28673 e sua vizinhanca. Acompanhar
tambem falhas em distraction/fatigue. Nao corrigir saltos por suavizacao cega.
Depois de fixar a medicao, definir lacunas curtas/longas, preservar mascaras e
recontar janelas. Calibracao e normalizacao nao podem usar query. Separar
support/query por intervalos sem frames compartilhados. Meta-treino permanece nos
datasets publicos; estes videos sao destinados a meta-avaliacao, com transparencia
sobre qualquer trecho utilizado para desenvolver ou ajustar o extrator.
