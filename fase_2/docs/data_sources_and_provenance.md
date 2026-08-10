# Fontes de dados e proveniência

Este documento registra os ativos encontrados, como foram produzidos e quais usos são
metodologicamente válidos na fase 2. O inventário resumido está em
[`data/manifests/data_sources.csv`](../data/manifests/data_sources.csv).

## 1. Vídeos de teleoperação

- Quatro vídeos locais, com aproximadamente 30 minutos cada.
- São a fonte primária para extração de EAR, MAR, pitch, yaw, roll e qualidade da detecção.
- Permanecem em `data/raw/videos/`, ignorados pelo Git.
- Não podem ser copiados, renomeados, convertidos em frames persistentes ou publicados.
- FPS, resolução, número de frames e duração ainda precisam ser registrados no manifesto
  real; `videos.example.csv` não constitui evidência.

## 2. Anotações temporais da fase 2

A segunda validação manual foi realizada assistindo individualmente aos quatro vídeos e
marcando intervalos em resolução de segundos.

| Propriedade | Valor verificado |
|---|---:|
| Intervalos | 71 |
| Vídeos cobertos | 4 |
| Lacunas internas | 0 |
| Sobreposições | 0 |
| Alerta | 4.440 s |
| Fadiga | 84 s |
| Distração | 825 s |
| Operador ausente | 1.794 s |

Os limites são inclusivos e usam `MM:SS`. As grafias portuguesas foram preservadas para
auditoria e normalizadas para `alert`, `fatigue` e `distraction`. `Ausente` foi separado como
`operator_absent`, sem target comportamental.

O arquivo normalizado está no caminho local configurado em `configs/data/base.yaml` e é
ignorado pelo Git. A conversão para frames dependerá do FPS real de cada vídeo. O término
anotado também deverá ser comparado com a duração do container para detectar caudas sem
rótulo.

### Implicação para modelagem

Há forte desbalanceamento: fadiga possui apenas 84 segundos. Nenhum fold pode ser aceito sem
relatar a presença de fadiga em treino, validação e teste. Class weights, sampling e
augmentation pertencem a experimentos posteriores e só podem atuar sobre treino.

## 3. Frames por estado da fase 1

O pipeline legado processou um a cada três frames, atribuiu estados por regras/detectores e
salvou imagens em subpastas. Um contador global gerou nomes de `000001.jpg` a
`040778.jpg`.

| Estado predito/pasta | Imagens |
|---|---:|
| NORMAL | 21.479 |
| NAO_DETECTADO | 8.809 |
| MAOS_FORA | 8.302 |
| POSTURA_RUIM | 2.039 |
| CELULAR | 149 |
| **Total** | **40.778** |

O arquivo versionado `fase_1/validacao_manual.csv` revisou manualmente as 40.778 imagens e
contém `arquivo`, `estado_predito` e `estado_real`.

### Distribuição do rótulo manual

| Estado real | Imagens |
|---|---:|
| NORMAL | 22.264 |
| NAO_DETECTADO | 10.701 |
| MAOS_FORA | 4.020 |
| CELULAR | 2.906 |
| POSTURA_RUIM | 887 |

### Limitações

- As classes descrevem corpo, presença e celular, não fadiga/distração temporal.
- O contador é global e não registra diretamente o vídeo nem o frame original.
- O script não persistiu os limites do contador entre os quatro vídeos.
- As imagens contêm overlays do detector; usá-las como entrada de classificação pode causar
  vazamento visual do rótulo.
- A sequência foi subamostrada e não equivale à sequência completa dos vídeos.

Esses rótulos podem apoiar análise de presença, postura, mãos e celular. É proibido mapear
automaticamente `NORMAL → alert`, estados corporais → `distraction` ou `NAO_DETECTADO →
fatigue`.

## 4. Bounding boxes de celular produzidas no LabelImg

O LabelImg foi configurado com uma classe, `celular`, e produziu labels YOLO normalizados.
As 2.906 imagens usadas correspondem exatamente às linhas com `estado_real = CELULAR` da
validação manual.

| Item | Quantidade |
|---|---:|
| Imagens | 2.906 |
| Arquivos de anotação associados a imagens | 2.878 |
| Imagens com uma bounding box | 2.870 |
| Labels vazios | 8 |
| Imagens sem arquivo de label | 28 |
| Imagens tratadas como background | 36 |
| Boxes inválidos | 0 |

Cada linha não vazia segue `class_id x_center y_center width height`, com coordenadas entre
zero e um e `class_id = 0`.

### Treinamentos encontrados

Foram executados YOLO11n/s/m/l/x com validação cruzada e existem pesos e relatórios locais.
O relatório principal apresenta mAP@0.5 próximo de 99,5%, mas a divisão usou embaralhamento
aleatório e `KFold(shuffle=True)`. Frames consecutivos e muito semelhantes podem aparecer
em treino e validação, portanto essas métricas não demonstram generalização temporal ou
entre vídeos.

O conjunto também possui poucos backgrounds, pois foi formado a partir de frames já
classificados como celular. Isso limita a avaliação de falsos positivos.

### Reconciliação com o documento de qualificação

O `DocQualificacao.pdf` informa 2.879 imagens no dataset específico de celular. A auditoria
dos ativos locais encontrou 2.906 imagens, 2.878 labels associados às imagens e 2.870 labels
não vazios. A provável origem do número 2.879 é a contagem de todos os `.txt` do pacote,
incluindo `classes.txt`, mas isso ainda deve ser confirmado antes de corrigir ou reutilizar a
estatística acadêmica. Até lá, o manifesto distingue explicitamente imagens, arquivos de label
e bounding boxes.

O pipeline da fase 1 carrega um YOLO COCO e consulta a classe 67, enquanto o detector
customizado usa classe 0. Os pesos customizados não devem ser considerados integrados sem
uma configuração explícita e um teste de inferência.

### Uso permitido na fase 2

- Evidência opcional de celular para late fusion, somente depois do núcleo facial-temporal.
- Novo split por vídeo ou bloco temporal, nunca o KFold aleatório legado.
- Inclusão de negativos representativos e relatório de falsos positivos.
- Pesos, imagens e labels permanecem locais e fora do Git.

## 5. Fluxo de dados da fase 2

```text
vídeos + intervalos temporais
        │
        ├── indicadores faciais por frame + flags de qualidade
        │                  │
        │                  └── janelas 30/60/150 → modelos principais
        │
        └── evidências legadas de presença/celular
                           │
                           └── late fusion opcional, após validação própria
```

O target principal sempre vem das anotações temporais da fase 2. As fontes legadas nunca
alteram silenciosamente esse target.

## 6. Regras para importação futura

1. Não editar nem mover as fontes originais.
2. Receber a origem por argumento/configuração local não versionada.
3. Validar contagens, nomes, formatos, classes e hashes antes de importar.
4. Gravar cópias somente sob `fase_2/data/external/phase1/`, que é ignorado.
5. Gerar um manifesto anonimizado, sem caminhos absolutos ou identificação interna.
6. Preservar `source_id`, nome original do item e transformação aplicada.
7. Não usar nenhum ativo em treino até definir seu vídeo/bloco temporal e split.

## 7. Artefatos faciais pendentes de transferência

O documento de qualificação relata a execução do baseline Face Mesh e apresenta médias de
EAR, MAR e pitch, além de taxas de detecção facial por vídeo. A auditoria desta máquina não
encontrou CSV, Parquet ou outro dataset persistido com EAR, MAR, pitch, yaw e roll por frame,
mas o artefato pode estar na máquina usada na extração original.

Por isso, a prioridade é recuperar e auditar a extração original. Os vídeos só devem ser
reprocessados se o material não puder ser localizado ou falhar nas validações. As tabelas
acadêmicas são evidência histórica, mas não substituem o dataset por frame.

### Pacote mínimo a recuperar

- Arquivos tabulares com os cinco indicadores por frame.
- Identificação do vídeo, índice do frame e/ou timestamp.
- Informação explícita de detecção facial ou regra documentada para reconhecer falhas.
- Código e configuração usados na extração, incluindo landmarks e ROI.
- Versões de Python, OpenCV e MediaPipe, quando disponíveis.
- Logs, resumos ou tabelas que permitam reproduzir as taxas publicadas.

### Validação após a transferência

1. Copiar para `fase_2/data/interim/legacy_extraction/`, que permanece fora do Git.
2. Registrar nomes, tamanhos e SHA-256 em manifesto anonimizado.
3. Verificar unicidade e ordenação de `(video_id, frame_idx)`.
4. Comparar quantidade de frames e duração com o manifesto real dos vídeos.
5. Validar faixas, NaN, infinitos e semântica dos zeros.
6. Recalcular as taxas de detecção e comparar com 57,0%, 83,7%, 67,5% e 42,0%.
7. Confirmar que EAR/MAR/head pose foram calculados com a mesma definição acadêmica.
8. Somente após aprovação, promover uma versão normalizada para `data/interim/`.
