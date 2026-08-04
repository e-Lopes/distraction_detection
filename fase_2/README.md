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

