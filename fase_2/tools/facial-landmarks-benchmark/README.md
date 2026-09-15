# Benchmark comparativo de frameworks de extração de landmarks faciais

Comparação de **eficiência computacional** entre três frameworks de extração
de landmarks faciais — **MediaPipe** (Face Mesh, 468/478 pts), **OpenFace 2.2.0**
(CE-CLM, 68 pts) e **InsightFace** (RetinaFace + 2D-106) — em datasets públicos
de sonolência ao volante (**NTHU-DDD**, **UTA-RLDD**) e, na sequência, em dados
próprios mais desafiadores.

Métricas coletadas por vídeo processado: FPS médio, latência de inferência
(p50/p90/p95/máx), taxa de detecção de face, uso médio de CPU, pico de RAM e,
quando há GPU NVIDIA disponível, utilização média de GPU e pico de memória de
GPU.

## Estrutura do projeto

```
facial-landmarks-benchmark/
├── docker-compose.yml
├── Makefile
├── data/                  # coloque os vídeos aqui (ver seção "Datasets")
│   ├── nthu_ddd/
│   ├── uta_rldd/
│   └── custom/
├── results/                # saída: CSVs por frame, sumário agregado, gráficos
├── docker/
│   ├── mediapipe/Dockerfile
│   ├── openface/Dockerfile     # build do OpenFace a partir do código-fonte
│   ├── insightface/Dockerfile  # base CUDA, requer GPU NVIDIA
│   └── analysis/Dockerfile
└── src/
    ├── common/          # schema de resultados, monitor de recursos, métricas
    ├── runners/         # run_mediapipe.py, run_openface.py, run_insightface.py
    └── analysis/        # aggregate.py, plots.py
```

## Pré-requisitos

- Docker Desktop com **Docker Compose v2** (`docker compose ...`, sem hífen).
- Para o InsightFace com GPU: NVIDIA Container Toolkit configurado no host e,
  no Docker Desktop, GPU habilitada (Windows: WSL2 + driver NVIDIA atualizado
  com suporte a CUDA; Docker Desktop → Settings → Resources → verificar que a
  integração WSL2 está ativa). Teste com:
  ```
  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
  ```
  Se isso não mostrar sua GPU, ajuste o toolkit antes de rodar o serviço
  `insightface` — sem isso o `docker-compose.yml` falha ao reservar o device.

### Setup específico: Windows 10 + WSL2 + GPU NVIDIA (ex.: RTX 2080 Super)

A RTX 2080 Super (Turing, compute capability 7.5, 8GB) roda a imagem
`docker/insightface` (CUDA 11.8) sem nenhuma alteração no Dockerfile. O que
precisa de atenção é a configuração do lado do Windows:

1. **Versão do Windows**: `winver` deve mostrar build 21H2 (19044) ou mais
   recente — é o mínimo estável para GPU Paravirtualization no WSL2 sem
   precisar de build Insider.
2. **Driver NVIDIA**: instale o driver mais recente para Windows (Game Ready
   ou Studio, tanto faz) direto do site da NVIDIA. **Não** instale nenhum
   driver NVIDIA dentro do WSL2/Ubuntu — o driver do Windows já expõe a GPU
   para dentro do WSL2 automaticamente.
3. **WSL2 atualizado**: no PowerShell (admin), `wsl --update` e depois
   `wsl --shutdown` para garantir o kernel mais recente.
4. **Docker Desktop**: em Settings → General, confirme "Use the WSL 2 based
   engine" ativado; em Settings → Resources → WSL Integration, confirme que
   sua distro padrão está habilitada. Com driver e WSL2 atualizados, o Docker
   Desktop expõe a GPU automaticamente — **não é necessário** instalar o
   NVIDIA Container Toolkit manualmente dentro da distro quando se usa o
   Docker Desktop (isso só é preciso em setups com Docker Engine nativo em
   Linux, não é o caso aqui).
5. **Teste** (dentro do WSL2 ou PowerShell, com Docker Desktop rodando):
   ```
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
   Deve listar a RTX 2080 Super. Se der erro `could not select device driver
   "nvidia"`, o mais comum é o Docker Desktop precisar ser reiniciado depois
   de atualizar o driver/WSL2.
6. **Recursos do WSL2**: os datasets são grandes (NTHU-DDD ~9,5h de vídeo,
   UTA-RLDD ~96GB). Se notar lentidão de I/O ou falta de memória, crie/edite
   `%UserProfile%\.wslconfig` para aumentar o limite de RAM/swap da WSL2, e
   prefira manter a pasta do projeto (e `data/`) dentro do filesystem nativo
   do WSL2 (ex.: `\\wsl$\Ubuntu\home\...`) em vez de `/mnt/c/...` — o I/O de
   `/mnt/c` é bem mais lento para ler milhares de frames de vídeo.
7. **`docker-compose.yml`** já está com a reserva de GPU (`deploy.resources.
   reservations.devices`) só no serviço `insightface` — MediaPipe e OpenFace
   continuam em CPU normalmente, não precisam de GPU.

## Datasets

Nenhum dos dois datasets pode ser baixado automaticamente pelo projeto —
ambos exigem passar por um formulário/termo de uso dos autores.

- **NTHU-DDD**: solicitar acesso ao Computer Vision Lab da National Tsing Hua
  University em <http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/>
  (uso mediante acordo de não divulgação/NDA com os autores). Após liberado,
  extraia os vídeos `.avi` mantendo a estrutura de pastas por sujeito/cenário
  dentro de `data/nthu_ddd/`.
- **UTA-RLDD**: página oficial em
  <https://sites.google.com/view/utarldd/home> (The University of Texas at
  Arlington Real-Life Drowsiness Dataset — ~30h de vídeo, 60 participantes,
  3 classes: alert / low_vigilant / drowsy). Baixe e extraia em
  `data/uta_rldd/`.
- **Dados próprios**: depois de validar o pipeline nos dois datasets acima,
  coloque seus vídeos mais desafiadores em `data/custom/`.

O runner descobre vídeos recursivamente (extensões `.avi .mp4 .mov .mkv .wmv
.mpg .mpeg`), então a estrutura de subpastas de cada dataset pode ser mantida
como está — o `video_id` de cada resultado é derivado do caminho relativo,
evitando colisão de nomes entre sujeitos.

## Como rodar

```bash
docker compose build

# teste rápido primeiro (limita a 300 frames/vídeo) para validar o pipeline
# antes de rodar tudo:
docker compose run --rm mediapipe --input /data/nthu_ddd --dataset nthu_ddd \
    --output-dir /results --max-frames 300

# execução completa, um framework de cada vez, por dataset:
make nthu     # mediapipe + openface + insightface sobre data/nthu_ddd
make uta      # idem sobre data/uta_rldd
make custom   # idem sobre data/custom (dados próprios)

# depois de rodar os frameworks desejados, gera estatísticas + gráficos:
make analysis
```

Os comandos `make` são apenas atalhos para `docker compose run --rm <serviço>
--input ... --dataset ... --output-dir /results`; rode os comandos manualmente
se quiser mais controle (ex.: `--max-frames`, `--refine-landmarks` no
mediapipe, `--model-pack buffalo_sc` no insightface para uma variante mais
leve).

Cada execução **acrescenta** linhas a `results/aggregate_summary.csv` — então
dá pra rodar os frameworks em qualquer ordem, em máquinas diferentes, ou
retomar depois de uma pausa, e só gerar os gráficos comparativos no final com
`make analysis` (ou `docker compose run --rm analysis`).

## Saídas

- `results/<framework>/<dataset>/<video_id>__frames.csv` — uma linha por
  frame processado (latência, detecção, nº de landmarks, confiança).
- `results/aggregate_summary.csv` — uma linha por vídeo processado, com FPS
  médio, percentis de latência, taxa de detecção e uso de CPU/RAM/GPU.
- `results/grouped_stats.csv` — estatísticas agregadas por
  (framework, dataset), geradas por `src/analysis/aggregate.py`.
- `results/plots/*.png` — gráficos comparativos (FPS, boxplot de latência,
  taxa de detecção, uso de CPU/RAM, utilização de GPU).

## Limitação importante: granularidade do OpenFace

O binário `FeatureExtraction` do OpenFace processa o vídeo inteiro
internamente (C++), sem expor um callback por frame para medição de latência.
Por isso, o runner do OpenFace mede o **tempo total de parede do vídeo** e
deriva um FPS médio e uma latência aproximada (`tempo_total / nº_frames`) —
isso fica marcado na coluna `granularity` (`"video"` vs `"frame"` para
MediaPipe/InsightFace) tanto no CSV de sumário quanto no `run_metadata.json`
de cada framework. O boxplot de latência por frame também exibe essa ressalva
no título. Isso é relevante para justificar a metodologia no capítulo de
resultados da dissertação: FPS é diretamente comparável entre os três, mas a
*distribuição* de latência por frame só é comparável entre MediaPipe e
InsightFace.

## Build do OpenFace

O Dockerfile de `docker/openface` clona e compila o OpenFace 2.2.0 a partir
do código-fonte oficial — é historicamente o build mais frágil dos três
(projeto sem manutenção ativa; dependências de sistema mudam entre versões
do Ubuntu). Se `docker compose build openface` falhar, confira a
[wiki oficial do projeto](https://github.com/TadasBaltrusaitis/OpenFace/wiki)
para a combinação atual de versões de OpenCV/Boost e ajuste o Dockerfile.

## Próximos passos sugeridos

1. Validar o pipeline com `--max-frames 300` em 2-3 vídeos de cada dataset.
2. Rodar a bateria completa em NTHU-DDD e UTA-RLDD (`make nthu uta`).
3. Gerar `make analysis` e revisar `results/grouped_stats.csv` e os gráficos.
4. Repetir em `data/custom/` com os dados próprios mais desafiadores.
5. Se quiser correlacionar eficiência com *qualidade* dos landmarks (não só
   velocidade), um próximo passo natural é anotar um subconjunto com
   landmarks de referência (ex.: via ferramenta de anotação manual) e
   calcular NME (normalized mean error) por framework — isso não está
   implementado aqui, que foca em eficiência computacional conforme pedido.
