# Fase 2 — Detecção temporal de fadiga e distração

Esta pasta contém a etapa atual do projeto de mestrado, dedicada à classificação temporal dos estados **Alert**, **Fatigue** e **Distraction** a partir de indicadores faciais extraídos de vídeos reais de teleoperação.

## Objetivo

Comparar regras fixas, classificadores clássicos e modelos temporais para verificar se a evolução dos indicadores faciais melhora a identificação de fadiga e distração em relação a abordagens sem contexto temporal.

## Indicadores de entrada

São extraídos cinco indicadores por frame:

- **EAR:** abertura dos olhos;
- **MAR:** abertura da boca;
- **Pitch:** inclinação vertical da cabeça;
- **Yaw:** rotação horizontal da cabeça;
- **Roll:** inclinação lateral da cabeça.

O pipeline utiliza MediaPipe Face Mesh para localizar os pontos faciais e OpenCV para estimar a orientação da cabeça.

## Representação temporal

Os indicadores são organizados em janelas de diferentes durações:

- 30 frames;
- 60 frames;
- 150 frames.

A distribuição das classes deve ser recalculada para cada combinação de tamanho de janela, stride e regra de rotulagem.

Quando forem utilizadas flags de qualidade, a entrada poderá ser representada por uma matriz `N × 8`:

```text
[EAR, MAR, Pitch, Yaw, Roll,
 FaceDetected, WasInterpolated, MissingDurationSoFar]
```

## Pré-processamento

Serão comparadas duas abordagens principais:

1. **Zero-fill:** tratamento já utilizado, mantido como baseline.
2. **Interpolação de gaps curtos:** preenchimento de pequenas falhas, acompanhado de flags que indicam a origem do valor.

Falhas longas, oclusões e ausência do operador não devem ser interpoladas como observações faciais reais. Esses casos são registrados como estados operacionais separados.

## Modelos

### Baselines

- regras fixas baseadas em EAR, MAR e pitch;
- XGBoost com features agregadas por janela.

### Modelos temporais principais

- LSTM;
- TCN.

### Comparações opcionais

- Random Forest;
- SVM;
- Transformer.

O Transformer e a fusão multimodal são experimentos complementares e não devem impedir a conclusão dos modelos centrais.

## Divisão dos dados

A avaliação principal utiliza **leave-one-video-out**:

- um vídeo completo é reservado para teste;
- os vídeos restantes são utilizados para treino e validação;
- a validação utiliza blocos temporais contínuos;
- janelas sobrepostas não podem ficar em subconjuntos diferentes;
- transformações e hiperparâmetros são ajustados somente com treino e validação.

Esse protocolo evita que janelas temporalmente próximas provoquem resultados artificialmente elevados.

## Desbalanceamento

Serão comparadas as seguintes estratégias:

1. distribuição original;
2. class weights;
3. weighted sampling;
4. augmentation leve das séries temporais.

Augmentation e sampling são aplicados somente ao conjunto de treinamento. Validação e teste mantêm a distribuição real.

## Avaliação

A métrica principal é o **Macro F1-score**. Também serão analisados:

- precision, recall e F1 por classe;
- balanced accuracy;
- matriz de confusão;
- falsos alertas por hora ou sessão;
- estabilidade dos resultados entre folds e seeds;
- tempo de inferência e tamanho do modelo.

## Explicabilidade

A análise das decisões inclui:

- permutation importance;
- SHAP para modelos clássicos, quando aplicável;
- ablação dos grupos ocular, oral, head pose e missingness;
- estudos de caso com verdadeiros positivos, falsos positivos e falsos negativos;
- análise específica de trechos com alta taxa de falha na detecção facial.

## Estrutura da pasta

```text
fase_2/
├── configs/          configurações dos experimentos
├── data/             dados locais não versionados
├── docs/             protocolo e decisões metodológicas
├── notebooks/        análises exploratórias
├── outputs/          métricas, figuras e artefatos
├── reports/          resultados utilizados na dissertação
├── src/              implementação do pipeline
├── tests/            testes de consistência e vazamento
├── .gitignore
├── pyproject.toml
└── README.md
```

## Fluxo experimental

1. validar as anotações e o manifesto dos vídeos;
2. gerar o diagnóstico de classes e missingness;
3. criar splits temporais sem vazamento;
4. gerar janelas de 30, 60 e 150 frames;
5. comparar zero-fill e interpolação com flags;
6. treinar regras fixas, XGBoost, LSTM e TCN;
7. avaliar estratégias de desbalanceamento;
8. executar ablação, explicabilidade e estudos de caso;
9. avaliar late fusion somente após concluir o núcleo experimental.

## Comandos atuais da fundação

Execute sempre a partir da raiz do repositório:

### Ambiente isolado

```bash
python -m venv fase_2/.venv
source fase_2/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'fase_2[dev]'
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.5,<3'
```

O PyTorch CPU evita baixar bibliotecas CUDA enquanto os modelos temporais ainda não estão em
execução. Em uma máquina com GPU, a instalação deve ser substituída pela variante compatível
com o driver/CUDA local. O ambiente `.venv` é local e ignorado pelo Git.

### Pipeline de dados

```bash
python -m fase_2.src.data manifest
python -m fase_2.src.data validate-annotations
python -m fase_2.src.data convert-annotations
python -m fase_2.src.data diagnose
python -m fase_2.src.data diagnose-missingness
python -m fase_2.src.data generate-splits
python -m fase_2.src.training.dummy_baseline
python -m fase_2.src.training.classical_baselines --xgb-device cuda
python -m fase_2.src.training.classical_grid_search --xgb-device cuda
python -m pytest fase_2/tests -q
```

Entradas sensíveis permanecem em `fase_2/data/raw/`. Manifestos contêm somente IDs,
caminhos relativos, metadados e índices temporais. Métricas agregadas são gravadas em
`fase_2/outputs/metrics/`.

## Plano pós-banca

- [Documentação da fase 2](docs/README.md): hierarquia entre qualificação, plano vigente e registros técnicos.
- [Linha de base da qualificação](docs/qualification_baseline.md): contribuições formalizadas, evidências disponíveis e pendências de reconciliação.
- [Plano de ações pós-banca revisado](docs/plano_pos_banca.md): referência complementar; não substitui o `PlanoPósBanca.pdf` vigente.
- [Mês 1 — Dados e protocolo](docs/months/mes_01_dados_protocolo.md): checklist operacional, entregáveis, bloqueios e critérios de aceite da etapa atual.
- [Plano incremental de implementação](docs/IMPLEMENTATION_PLAN.md): estado atual, arquitetura, próximos marcos, riscos e critérios de aceite do pipeline de ML.

## Fontes de dados e legado da fase 1

- [Inventário e proveniência](docs/data_sources_and_provenance.md): relação entre vídeos, anotações temporais, frames por estado, validação manual e bounding boxes de celular.
- `data/manifests/data_sources.csv`: inventário anonimizado e versionável dos ativos conhecidos.

Os dados da fase 1 são evidência auxiliar para presença, postura, mãos e celular. Eles não
substituem os rótulos temporais `alert`, `fatigue` e `distraction` usados como target principal
na fase 2. Imagens, labels completos, ZIPs e pesos permanecem fora do Git.

### Extração das séries faciais

O extrator canônico da Fase 2 recebe o diretório local dos vídeos sem registrar esse caminho no repositório. Exemplo:

```bash
python -m fase_2.src.features.extract_facial_series \
  --video-dir "D:/dados/TELEOP/videos" \
  --roi-config fase_2/configs/preprocessing/legacy_roi.json \
  --workers 4
```

Por padrão, são procurados `1.mp4` a `4.mp4`, associados a `video_01` a `video_04`, e os CSVs são gravados em `fase_2/data/interim/legacy_extraction/`. A saída contém EAR, MAR, pitch, yaw, roll, `face_detected` e estado operacional por frame. Campos faciais ausentes ficam vazios; zero-fill só pode ser aplicado posteriormente, na entrada do modelo.

Para uma execução curta de validação, use `--max-frames 100`. O modo sintético só é ativado com `--demo`; ausência de vídeos ou MediaPipe gera erro. Arquivos existentes não são substituídos sem `--overwrite`.

No Windows, o MediaPipe utilizado pelo projeto não oferece delegate GPU. `--workers 4` processa os quatro vídeos em processos CPU independentes, preservando o mesmo Face Mesh e reduzindo o tempo total sem alterar o método de extração.

`legacy_heuristic_state` é preservado exclusivamente para confrontar a nova extração com o relatório histórico. Ele não é o target da Fase 2 e não substitui as anotações temporais manuais.

## Reprodutibilidade

Cada execução deve registrar:

- hash do commit;
- configuração utilizada;
- versão do dataset e das anotações;
- fold e seed;
- tamanho da janela e stride;
- estratégia de pré-processamento;
- métricas globais e por classe;
- matriz de confusão;
- previsões por janela;
- tempo de treinamento e inferência.

## Privacidade

Os vídeos e dados operacionais são sensíveis. Não devem ser versionados:

- vídeos ou frames extraídos;
- anotações completas com informações internas;
- pesos de modelos;
- caches e previsões individuais;
- arquivos com identificação do operador ou da empresa.
