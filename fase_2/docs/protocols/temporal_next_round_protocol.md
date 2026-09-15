# Próxima rodada — LSTM, TCN e Transformer leve

**Estado:** planejado, aguardando saneamento facial conforme
`../../results/audits/experimental_readiness_2026-09-14.md`. Este protocolo não substitui resultados G1–G4
nem a seleção de confirmação já prevista em `configs/final_experiment.yaml`.

## Entrada e desenho

Congelar primeiro um manifesto de séries validadas, com hash, extrator, pesos, hardware,
ROI e versão anatômica. O runner temporal lê `legacy_extraction` por padrão; não assume
automaticamente as séries G48A. Validar conversão para o schema de `temporal_data` antes de treino.

Manter tarefa primária Alert/Fatigue/Distraction, quatro folds leave-one-video-out,
validação em blocos, purge 150, stride 15, maioria 60% e scaler ajustado só no treino.
Reportar suporte por classe/fold, principalmente ausência de fadiga em algumas janelas de teste.

Separar duas perguntas para identificar a origem de qualquer ganho:
1. Efeito do extrator: mesmo classificador de controle, folds, representação e balanceamento.
2. Efeito da arquitetura: mesma entrada facial congelada para LSTM, TCN e Transformer.

## Matriz arquitetural a preparar após os gates

Triagem controlada: três modelos × 30/60/150 frames × quatro folds × seed 42 = 36 runs
por entrada nova. Reutilizar G2 se a entrada/configuração forem exatamente as históricas;
não chamar reuso de nova evidência. Todos os três modelos devem receber o mesmo balanceamento
na comparação pareada. `class_weights` é o candidato já presente na confirmação vigente;
confrontá-lo ao controle SVM com a mesma estratégia.

| Modelo | Configuração pequena de referência |
|---|---|
| LSTM | hidden 64, duas camadas, unidirecional, dropout 0,25, pooling last |
| TCN | canais 32/64/64, kernel 3, dropout 0,25, pooling mean |
| Transformer | dimensão 32, quatro heads, duas camadas, FFN 64, dropout 0,25, pooling mean, max_length 150 |

Usar limites históricos de 150 épocas, patience 15, minimum_delta 0,002, LR 0,001,
weight decay 0,0001 e gradient clip 1. Documentar batch size e memória; reduzir batch
somente com configuração nova e mesma regra entre braços. Seleção por Macro F1 de validação,
com F1/recall de Fatigue e Distraction, falsos episódios/hora e pior vídeo como complementos.
Não selecionar janela pelo teste externo nem declarar vitória por um fold isolado.

Confirmação posterior: janela e balanceamento congelados, seeds 42/123/456/789/2026 e
quatro folds. Se a pergunta for confirmar as três arquiteturas, são 60 runs temporais;
se houver promoção de finalistas, registrar o subconjunto antes de acessar teste.
O template `temporal_multiseed.yaml` atual inclui apenas LSTM/TCN; a confirmação vigente
inclui LSTM e SVM. Portanto nenhum dos dois já representa confirmação dos três modelos.

## Execução, progresso e organização

O dry-run de 14/09/2026 encontrou `torch 2.13.0+cpu` na `.venv` usada pelo comando abaixo.
Isso permite inspeção da matriz, mas esse ambiente precisa de uma instalação PyTorch CUDA
validada antes do treino na GTX 1060. ONNX Runtime CUDA no ambiente InsightFace não habilita
automaticamente CUDA no PyTorch. O dry-run de 60 frames listou corretamente 12 runs.

O pesquisador executa os treinos no terminal. Usar sempre `python -u`, progresso por época,
modelo/fold/seed, loss, Macro F1, early stopping e checkpoints. Conservar `outputs/models`,
`outputs/logs`, `outputs/predictions`, `outputs/metrics` e `outputs/figures` em geração nova.
Salvar config resolvida e fingerprint; CSV final por run, nunca promover arquivo parcial.

Inspeção da matriz histórica, sem treinamento (executável agora):

```bash
fase_2/.venv/bin/python -u -m fase_2.src.training.temporal_multiseed \
  --experiment-config fase_2/configs/experiment/g2_temporal_r0_w60.yaml --dry-run
```

Critérios de liberação: OpenFace saneado; relatório de três braços/quatro vídeos;
critério de operador auditado; mapeamento anatômico conferido; origem das séries congelada;
conversão para entrada temporal testada; matriz resolvida validada sem treino; smoke curto
de cada arquitetura em saída própria. Até lá, configuração de treino nova não é promovida.
