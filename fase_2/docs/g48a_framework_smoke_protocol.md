# G48A — protocolo ativo do smoke de frameworks faciais

Esta é a etapa ativa para a comparação exploratória. O protocolo G48 anterior e o
experimento de anotação CVAT/400 frames permanecem preservados, porém pausados; seus
artefatos não integram a amostra nem as métricas desta G48A.

## Escopo da G1

- baseline congelada: `mp.solutions.face_mesh.FaceMesh` 0.10.21, `refine_landmarks=True`,
  rastreamento e 478 pontos;
- candidatos: InsightFace 0.7.3 `buffalo_l/2d106det.onnx` em ONNX Runtime CPU e OpenFace
  da linha 2.0 em contêiner;
- 30 frames congelados: 10 fáceis, 10 intermediários e 10 difíceis, cobrindo os quatro
  vídeos; sem anotação manual;
- contrato comum de 22 pontos para EAR, MAR e pose; ausência representada por NaN e
  máscaras explícitas;
- resultado limitado à viabilidade do mapeamento e decisão de avanço. A G2 não faz
  parte desta execução.

O Face Mesh recebe aproximadamente um segundo de contexto cronológico antes de cada
alvo. InsightFace e OpenFace são reiniciados por imagem; isso evita transportar estado
entre frames não contíguos e vídeos diferentes. A face do operador é restringida à
região espacial plausível da cadeira para rejeitar o trabalhador ao fundo.

## Reprodutibilidade

- manifesto: `fase_2/data/manifests/g48a_smoke_30.csv` (seed 42);
- InsightFace: ambiente Conda `g48_insightface`, Python 3.11, CPU;
- `2d106det.onnx`: SHA-256 `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf`;
- detector `det_10g.onnx`: SHA-256 `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91`;
- OpenFace: `algebr/openface@sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197`.

A imagem OpenFace foi criada em 2018 e não expõe uma revisão Git interna verificável.
Por isso, ela é registrada como “2.0-era”, fixada pelo digest, e não como 2.2.0. Os
avisos sobre modelos sintéticos dos olhos não afetam os 68 pontos faciais usados aqui.
