# Migracao GPU: validacao inicial

Data: 2026-09-16. GPU: NVIDIA GeForce RTX 2060 SUPER, 8 GB.

## Execucao unica

Todo o codigo de preparacao, adaptacao e extracao esta em `fase_1/visualizar_cinco_series.py`.
Os pesos e o codigo externo versionado ficam no cache ignorado pelo Git `fase_1/.gpu_runtime/`.
Nao e necessario executar instaladores separados. Pacotes do ambiente usados: PyTorch
2.5.1+cu121, Ultralytics 8.3.65, OpenCV 4.10.0, NumPy e ONNX Runtime (importado pelos
helpers externos, sem criar sessoes de inferencia ONNX).

```powershell
python fase_1/visualizar_cinco_series.py --background
```

O programa sempre tenta CUDA primeiro. Se a inicializacao nao for possivel, informa
o erro e pergunta `Deseja continuar a extracao em CPU? [s/N]`. Apenas uma resposta
afirmativa permite CPU. Enter, resposta negativa ou terminal sem entrada cancelam.
`--require-gpu` cancela sem perguntar, apropriado para execucoes automatizadas.

## O que roda em cada dispositivo

- GPU: YOLO11n-Pose, BlazeFace short-range e FaceLandmarker de 478 pontos.
- CPU: decodificacao de video, recortes, pos-processamento, EAR/MAR/solvePnP,
  graficos, estatisticas e gravacao de arquivos.
- O backend CPU de referencia usa MediaPipe Pose e FaceMesh com refinamento e tracking.
- O backend GPU utiliza CUDA Graphs nas redes faciais, float32 e uma face por frame.
- CUDA e obrigatorio nas redes do backend GPU; entradas/saidas e parametros sao verificados.
- `gpu_audit.json` registra chamadas das tres redes, incluindo aquecimento. `config.json`
  registra dispositivo, versoes, modelos, hashes dos pesos, codigo externo e parametros.

## Teste real: primeiros 120 frames de cada video

Executados sequencialmente em CPU e GPU, sem janela. Contagens e indicadores
comparados por video/frame; nao houve extracao completa nesta validacao.

| Video | FPS CPU no loop | FPS GPU no loop (arquivo unificado) | Cinco indicadores CPU | Cinco indicadores GPU |
|---|---:|---:|---:|---:|
| 1 | 20,60 | 27,45 | 104/120 | 65/120 |
| 2 | 18,75 | 26,02 | 120/120 | 84/120 |
| 3 | 20,76 | 26,65 | 120/120 | 104/120 |
| 4 | 20,64 | 27,45 | 111/120 | 78/120 |

Lote unificado GPU: aproximadamente 18 segundos para 480 frames, contra 24 segundos
na referencia CPU, excluindo a inicializacao anterior ao loop. Amostra curta, sem
repeticoes estatisticas: nao representa garantia de desempenho nos quatro videos completos.

Auditoria GPU do lote unificado: 482 chamadas YOLO-Pose, 481 BlazeFace e 481 FaceLandmarker,
incluindo aquecimento. Todos os arquivos de series, estatisticas e diagnostico foram gerados.

## Diferencas que impedem equivalencia automatica

Os modelos GPU nao sao os mesmos da execucao historica CPU. O novo modelo facial
preserva a topologia dos indices usados nos indicadores, mas nao e o attention mesh
legado; tambem nao aplica o tracking temporal do MediaPipe Solutions. A deteccao
corporal muda para YOLO11n-Pose. Os limiares iguais numericamente nao significam
confiancas calibradas iguais.

Na primeira amostra GPU otimizada, o erro absoluto medio frente a CPU, usando somente
frames com os cinco indicadores em ambos e diferenca circular para angulos, foi:

| Video | Frames pareados | EAR | MAR | Pitch (graus) | Yaw (graus) | Roll (graus) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0,1462 | 0,0212 | 5,6089 | 4,8826 | 6,4892 |
| 2 | 84 | 0,1570 | 0,0781 | 5,6748 | 8,3384 | 12,4706 |
| 3 | 104 | 0,1435 | 0,0479 | 3,7615 | 5,0136 | 8,9318 |
| 4 | 71 | 0,1280 | 0,0554 | 2,5368 | 5,9470 | 7,1202 |

Portanto: inferencia GPU funcional e mais rapida nesta amostra, mas com menor cobertura
e diferencas relevantes nas medidas. Nao considerar migracao metodologicamente
equivalente. Antes de substituir resultados da pesquisa, validar visualmente landmarks,
investigar confianca/crop/tracking e recalibrar criterios em dados adequados ao protocolo.
Nenhum ajuste de limiar foi feito para esconder a queda de cobertura.

## Rastreabilidade

- CPU: `.gpu_runtime/validation/cpu/20260916_203853_551833/`.
- GPU antes de consolidar o codigo: `.gpu_runtime/validation/gpu_graph/20260916_203816_275761/`.
- GPU consolidada: `.gpu_runtime/validation/unified_gpu/20260916_204241_167527/`.
- Cinco testes de regressao passaram: combinacoes de detectores, falhas individuais,
  prioridade GPU, consentimento CPU e cancelamento sem entrada interativa.
- Codigo facial externo: [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx),
  commit `add50e0f486405c96695812f9a9b9b89a485892b`, licenca Apache-2.0.
- Peso corporal: [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0),
  `yolo11n-pose.pt`. Consulte a licenca do Ultralytics ao redistribuir.

Os scripts auxiliares temporarios de preparacao e backend foram incorporados ao
visualizador e removidos; nenhuma serie historica foi sobrescrita.
