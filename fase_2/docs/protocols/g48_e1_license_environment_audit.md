# G4.8-E1 — Auditoria inicial de licenças e ambientes

**Data da verificação:** 01/09/2026  
**Estado:** parcial; gate de licença aberto  
**Registro estruturado:** `../data/manifests/g48_extractor_registry.csv`

Este documento registra uma triagem técnica, não um parecer jurídico. Código, pesos e datasets
de treinamento são auditados separadamente. A licença do repositório não é automaticamente a
licença de todos os checkpoints distribuídos por ele.

## Resultado executivo

| Extrator | Código | Pesos | Pesquisa | Transferência à parceira |
|---|---|---|---|---|
| MediaPipe | Apache-2.0 | empacotados | elegível | sem bloqueio identificado nesta triagem |
| InsightFace | MIT | pesquisa acadêmica não comercial | elegível | bloqueada sem licença separada |
| OpenFace 2.0 | licença própria não comercial | restrições do toolkit/datasets | elegível | bloqueada sem licença comercial |
| MMPose/RTMW | Apache-2.0 | auditoria pendente por checkpoint/dataset | pendente | pendente |
| OpenSeeFace | BSD-2-Clause | BSD-2-Clause | elegível | elegível preservando avisos |

## Evidência e decisão por candidato

### MediaPipe Face Mesh

O repositório oficial usa Apache-2.0. Para preservar a comparação já congelada, a G4.8 mantém
inicialmente `mediapipe==0.10.21` e a API `solutions.face_mesh`, em vez de migrar o baseline no
meio do experimento.

Fonte: <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>

**Decisão E1:** baseline elegível e já operacional.

### InsightFace

A biblioteca Python declara código MIT, mas os modelos pré-treinados públicos — inclusive os
baixados automática ou manualmente — são limitados a pesquisa não comercial. O pacote
`buffalo_l` fornece alinhamento 2D de 106 pontos e 3D de 68 pontos, sendo o candidato inicial.

Fontes:

- <https://github.com/deepinsight/insightface/tree/master/python-package#license>
- <https://github.com/deepinsight/insightface/tree/master/model_zoo>
- <https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md>

**Decisão E1:** elegível para o experimento acadêmico; não promovível para transferência sem
licença/autorização separada. O arquivo `MODEL.LICENSE` do pacote efetivamente usado deve ser
arquivado por hash antes do smoke.

### OpenFace 2.0

A licença oficial restringe o software a pesquisa interna não comercial por instituição
acadêmica ou sem fins lucrativos. Também existem obrigações sobre derivados e necessidade de
respeitar licenças dos datasets/componentes. O projeto aponta uma via própria para licença
comercial.

Fontes:

- <https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt>
- <https://github.com/TadasBaltrusaitis/OpenFace>

**Decisão E1:** elegível somente para pesquisa não comercial. Antes de modificar ou transferir
qualquer componente, revisar as obrigações da licença com a instituição responsável.

### MMPose/RTMW

O código MMPose é Apache-2.0. O candidato inicial é RTMPose whole-body `m` em 256×192, que produz
pontos corporais, faciais e das mãos. O checkpoint oficial declara treinamento em
COCO-WholeBody; a variante RTMW Cocktail14 mistura 14 datasets e amplia muito a superfície de
licenciamento. Por isso, a G4.8 não usará Cocktail14 até concluir auditoria dataset a dataset.

Fontes:

- <https://github.com/open-mmlab/mmpose/blob/main/LICENSE>
- <https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose_coco-wholebody.yml>
- <https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md>

**Decisão E1:** código elegível; checkpoint ainda pendente. Congelar URL, SHA-256, config,
detector associado e licenças do conjunto de treinamento antes do primeiro resultado oficial.

### OpenSeeFace

O projeto declara código e modelos sob BSD-2-Clause.

Fontes:

- <https://github.com/emilianavt/OpenSeeFace/blob/master/LICENSE>
- <https://github.com/emilianavt/OpenSeeFace>

**Decisão E1:** elegível, preservando os avisos exigidos; integração permanece após os smokes dos
quatro candidatos principais, conforme o protocolo.

## Ambientes propostos

Cada stack fica isolada para impedir conflitos de CUDA, PyTorch, ONNX Runtime e OpenCV:

| Ambiente | Conteúdo inicial | Backend |
|---|---|---|
| `g48_mediapipe` | MediaPipe 0.10.21 + contrato atual | CPU |
| `g48_insightface` | InsightFace 0.7.3 + ONNX Runtime | CPU e CUDA separados |
| `g48_openface` | binário/release OpenFace 2.2.0 | CPU |
| `g48_mmpose` | MMPose 1.3.2 + MMEngine/MMCV compatíveis | CPU e CUDA |
| `g48_openseeface` | release 1.20.4 + ONNX Runtime | CPU primeiro |

Versões propostas só se tornam congeladas depois de um smoke que registre versão, modelo,
licença local, hash do peso, hardware e um frame válido. Não serão instaladas juntas na `.venv`
canônica da Fase 2.

## Pendências para fechar o gate E1

- concluir a licença do checkpoint RTMPose e do detector usado em top-down;
- decidir institucionalmente se InsightFace/OpenFace podem constar apenas como baselines de
  pesquisa ou se devem ser excluídos por inviabilidade de transferência;
- baixar cada artefato em ambiente isolado e registrar SHA-256/licença local;
- executar smoke nos mesmos frames frontal, lateral e ocluído;
- aprovar visualmente o mapeamento anatômico antes de gerar métricas.

