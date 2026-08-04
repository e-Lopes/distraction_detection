# Fatigue and Distraction Detection for Heavy Machinery Teleoperation

Repositório experimental do projeto de mestrado de Eduardo Lamy Lopes (PUCPR).

O objetivo é comparar regras fixas, classificadores clássicos e modelos temporais para classificar **Alerta**, **Fadiga** e **Distração** a partir de EAR, MAR, pitch, yaw e roll, considerando falhas de detecção facial, diferentes janelas temporais e desbalanceamento das classes.

## Princípios do projeto

- Nenhum vídeo, frame extraído ou dado sensível da empresa deve ser versionado no Git.
- Splits são definidos por vídeo/sessão e nunca por janelas aleatórias.
- Toda transformação aprende parâmetros apenas no conjunto de treino.
- Augmentation ocorre somente no treino.
- O teste permanece intocado durante seleção de janela, features e hiperparâmetros.
- Cada execução gera configuração, métricas e identificador reproduzível.

## Fluxo principal

1. Validar anotações e gerar manifesto dos vídeos.
2. Definir folds leave-one-video-out e blocos temporais de validação.
3. Gerar janelas de 30, 60 e 150 frames com stride configurável.
4. Comparar zero-fill com interpolação curta e flags de missingness.
5. Comparar baseline de regras, XGBoost, LSTM e TCN.
6. Avaliar class weights, weighted sampling e augmentation leve.
7. Executar ablação, permutation importance e estudos de caso.
8. Avaliar late fusion apenas após concluir o núcleo experimental.

## Comandos previstos

```bash
python -m src.data.validate_annotations --config configs/data/base.yaml
python -m src.data.make_splits --config configs/splits/leave_one_video_out.yaml
python -m src.data.build_windows --config configs/experiment/core.yaml
python -m src.training.train --config configs/experiment/core.yaml model=xgboost
python -m src.evaluation.evaluate --run-id <run_id>
```

Os comandos são contratos de interface para a implementação. Os módulos iniciais contêm pontos de entrada mínimos e devem ser preenchidos conforme o pipeline existente for migrado.

## Estrutura

```text
configs/       configurações versionadas de dados, splits e experimentos
data/          dados locais não versionados; apenas README e manifestos vazios
docs/          decisões metodológicas, protocolo e dicionário de dados
notebooks/     exploração numerada; não contém lógica definitiva do pipeline
outputs/       artefatos reproduzíveis, métricas, figuras e modelos
scripts/       atalhos operacionais
src/           implementação reutilizável do pipeline
tests/         testes de rotulagem, splits, interpolação e métricas
```

## Reprodutibilidade

Cada execução deve registrar:

- hash do commit;
- arquivo de configuração;
- fold, seed, janela e stride;
- versão do dataset/anotações;
- métricas globais e por classe;
- matriz de confusão;
- previsões por janela;
- tempo de treinamento e inferência.

