# Ampliação de famílias temporais — v1, 14/09/2026

## Estado e pergunta

Testar se novas famílias melhoram Macro F1 e identificação de Fatigue entre sessões. Nenhum treinamento oficial foi executado nesta implementação. A máquina de execução será a RTX 4060 do usuário; seus tempos, RAM e VRAM ainda não foram medidos. A máquina de desenvolvimento tem i7-8700, 16 GiB RAM e GTX 1060 6 GiB, mas o ambiente PyTorch local é CPU.

Usa a CLI `python -m fase_2`, `src/pipeline.py` e `src/reporting.py`, sem segundo pipeline. `configs/modern_experiment.yaml` herda `final_experiment.yaml` e isola resultados em `outputs/modern_v1`. Consulte também `reports/experimental_readiness_2026-09-14.md` e `docs/temporal_next_round_protocol.md`.

## Auditoria e validade

O histórico G2 contém 36 execuções: LSTM, TCN e Transformer × janelas 30/60/150 × quatro folds, seed 42. Isso não significa que todas as representações e estratégias de balanceamento tenham sido cruzadas. O screening atual de características usa 60 frames; MiniROCKET contempla três comprimentos. Confirmação depende de promoção explícita.

Entradas: EAR, MAR, pitch, yaw, roll; janelas 30/60/150, stride 15, maioria mínima 0,60 e exclusão de janelas mistas conforme configuração herdada. LOVO reserva um vídeo; validação interna em blocos contíguos, fração 0,20 e purge 150. São quatro sessões de um operador, não quatro operadores independentes. Testes históricos já foram examinados: novas avaliações desses vídeos não serão chamadas de teste intocado.

R0 preenche falhas com zero. R1 interpola até 15 frames e usa medianas; R2 acrescenta flags. Interpolação R1/R2 é offline e não prova viabilidade causal. Nesta rodada fixa-se R0, cinco canais, para não confundir ganhos de arquitetura com tratamento de falhas. Normalização, transformações e pesos de classe são ajustados apenas no treino do fold. Mantêm-se transições e exclusões iguais entre modelos.

MediaPipe e InsightFace possuem quatro arquivos completos; a comparação OpenFace permanece pendente de reextração/verificação dos arquivos antigos com falhas de parsing. Não misturar automaticamente extratores nem processar vídeos nesta rodada. Nenhum quarto extrator é necessário antes de resolver essa validade.

Resultados anteriores são referências descritivas; não reutilizados como novos fits por coincidência de nome. Fingerprints modernos incluem configuração resolvida, conteúdo das séries, anotações/splits/manifests, código Python, versões relevantes e revisão do encoder.

## Matriz executável de desenvolvimento

Uma configuração por modelo, janela 60, seed 42, quatro folds, class weights. Sem busca interna de alpha/C: Ridge alpha=1 e SVM C=1 linear são pré-especificados.

| Família | Entrada / implementação | Fits | Recurso principal |
|---|---|---:|---|
| SVM linear | R0_flat: trajetória achatada; **não** estatísticas agregadas | 4 | CPU |
| MultiROCKET + Ridge | aeon 1.3.0 multivariado, 840 kernels | 4 | CPU/RAM |
| HYDRA + MultiROCKET + Ridge | aeon 1.3.0, 16 grupos HYDRA; concatenação de features | 4 | CPU/RAM |
| Mantis congelado + Ridge | mantis-tsfm 1.1.0; cinco embeddings concatenados | 4 | CUDA + CPU |
| LSTM | 2 camadas, hidden 64 | 4 | CUDA |
| TCN | canais 32/64/64 | 4 | CUDA |
| Transformer leve | dimensão 32, 2 camadas, 4 heads | 4 | CUDA |
| Inception individual | implementação PyTorch, 6 módulos multiescala | 4 | CUDA |

Total: **32 fits**, incluindo 16 redes. Inception individual é inspirado na arquitetura InceptionTime, não o ensemble original nem reprodução certificada de seus resultados. LITEMV, ensemble InceptionTime, MantisV2, MOMENT, HIVE-COTE2 e TimEE ficam opcionais/desabilitados para limitar custo e adaptações.

MultiROCKET/HYDRA recebem N×5×T, nunca cinco canais unidos no tempo. Comprimento mínimo aceito pelo adaptador é 10; testes cobrem 30, 60 e 150. StandardScaler é train-only; HYDRA usa o sparse scaler da versão fixada do aeon. Kernels reduzidos caracterizam a versão com orçamento limitado, não os hiperparâmetros dos rankings publicados.

Mantis: `paris-noah/Mantis-8M`, revisão `bc7d5ab40c02133386a28e2c127f35c17c86901d`. Encoder congelado, eval/inference mode; cada canal é redimensionado linearmente para 512 pontos e codificado independentemente; concatenação no eixo de features. É uma adaptação multivariada explícita, não treinamento de interações entre canais no encoder. Fine-tuning não implementado/habilitado. Checkpoint persistido como state_dict junto ao estimador; retomada não depende de baixar pesos novamente. Nenhum cache compartilhado de embeddings entre folds foi habilitado.

## Seleção e confirmação

O perfil entregue executa **desenvolvimento**, salvando avaliação de validação, não avaliação externa. Não selecionar hiperparâmetros pelos resultados do vídeo reservado. Macro F1 de validação é primário; F1/recall de Fatigue e custo são critérios descritivos secundários. Não ajustar a grade após consultar testes.

Para avaliação externa, criar um perfil herdado com `modern_protocol.evaluation: external` e congelar `selection_lock` por fold. Cada chave textual do fold deve mapear exatamente `model`, `window`, `representation`, `balancing`, `parameters` do candidato selecionado pela validação daquele fold. Sem correspondência exata, a execução fica bloqueada, inclusive com `--force`. Preservar o perfil de desenvolvimento e seus resultados. Usar outro diretório de outputs para a confirmação.

Seeds confirmatórias pré-especificadas: 42, 123, 456, 789, 2024. O screening usa `training.screening.seed`; perfis de confirmação externa podem repetir a execução mudando somente esse campo. O candidato selecionado permanece congelado para todas as seeds. Não repetir SVM/encoder+Ridge determinísticos cinco vezes como evidência adicional. Limite planejado: dois finalistas estocásticos × quatro folds × cinco seeds = 40 fits; se o finalista for determinístico, apenas quatro. Essa seleção depende dos resultados futuros e não pode ser preenchida honestamente nesta etapa.

Ablation de comprimento somente após a comparação principal: três comprimentos × até dois finalistas × quatro folds × uma seed = 24 fits (16 adicionais se w60 compatível puder ser reutilizado). Confirmação usa a janela escolhida na validação interna, não aquela com maior teste. Nenhuma busca de stride, missingness e balanceamento simultânea.

## Orçamento e persistência

Executar sequencialmente; CPU 2 threads, batch neural 64, Mantis 16 janelas (80 canais), até 150 épocas, patience 15, AMP herdado. Metas iniciais: até 10 GiB RAM de processo, 5 GiB VRAM, até 30 min/fit. **Esses campos são metas de planejamento, não limites automaticamente impostos.** Limite operacional conservador de agenda: 32 × 30 min = 16 h para desenvolvimento; não é estimativa medida nem garantia. Monitorar o primeiro fold na RTX 4060 e interromper/reduzir batch se necessário; alteração deve gerar novo fingerprint. MultiROCKET pode demandar compilação inicial demorada.

Não há um único checkpoint por vídeo para classificação: cada unidade é modelo/janela/fold/seed. Redes preservam best/last pelo runner existente; transformações e classificador são persistidos juntos por fit. Saídas incluem registro de estado, métricas e predições; rerun compatível retoma/pula unidades completas. Não usar `--force` para retomada normal. Não mover ambientes virtuais entre máquinas.

## Métricas e relatório

Relatório existente estendido com seção moderna: fases separadas, famílias, resultado descritivo e falhas. Preservar métricas por classe, matriz de confusão e outputs de redes existentes. Adaptadores modernos salvam scores e Average Precision um-contra-resto/curvas PR quando existem positivos e negativos. AP não é área trapezoidal; Ridge não fornece probabilidades calibradas. Tempos de transformação e classificador são separados nos resumos. Extração facial não está incluída no tempo de classificação.

Limitações atuais de instrumentação: não há medição automática uniforme de pico RAM/VRAM nem benchmark de inferência sincronizado/aquecido para todas as famílias; figuras pareadas e agregação externa multi-seed ainda exigem resultados de confirmação e revisão. Não preencher custo ausente com zero. Reportar média por sessão separada de métricas de predições concatenadas; seeds e janelas sobrepostas não são réplicas independentes. Com quatro sessões, priorizar deltas descritivos pareados, não alegações fortes de significância. Resultados negativos permanecem no registro.

## Preparar e executar posteriormente na RTX 4060

Na raiz do repositório, crie um ambiente novo e instale PyTorch com CUDA conforme o seletor oficial https://pytorch.org/get-started/locally/ e o driver da máquina. Depois:

```bash
python3 -m venv fase_2/.venv
fase_2/.venv/bin/python -m pip install --upgrade pip
fase_2/.venv/bin/python -m pip install -e './fase_2[modern,foundation,dev]'
fase_2/.venv/bin/python -m pip check
fase_2/.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
fase_2/.venv/bin/python -u -m fase_2 train --plan --scope screening --config fase_2/configs/modern_experiment.yaml
bash fase_2/scripts/run_modern_experiment.sh
```

Copiar séries faciais, anotações, splits e manifests referenciados pela configuração; não basta copiar código. O wrapper verifica CUDA antes de preparar/treinar, executa prepare → train → report sequencialmente, imprime progresso e salva log. O primeiro Mantis requer internet para os pesos públicos fixados. Reexecutar o mesmo comando para retomar. Apenas relatório:

```bash
fase_2/.venv/bin/python -u -m fase_2 report --config fase_2/configs/modern_experiment.yaml
```

## Evidência e referências primárias

## Verificações da implementação

29 testes passaram envolvendo adaptadores, CLI e splits temporais; MultiROCKET/HYDRA testados com cinco canais e comprimentos 30/60/150, normalização train-only e roundtrip joblib; Inception com backward sintético e recarga. Mantis foi verificado separadamente com três séries sintéticas, checkpoint público fixado, encoder congelado e roundtrip de persistência offline. Nenhum desses resultados é métrica científica. Script shell passou em `bash -n`. Foi acrescentado teste de relatório sem resultados científicos.

No ambiente local de desenvolvimento, instalar aeon 1.3.0 ajustou pandas para 2.3.3, scikit-learn para 1.7.2 e scipy para 1.15.3; mantis-tsfm 1.1.0 também foi instalado. Não assumir compatibilidade de arquivos joblib históricos entre versões. Preparar ambiente novo na máquina de execução e preservar o ambiente antigo, se necessário.

## Fontes

- [Bake off redux](https://arxiv.org/abs/2304.13029): preprint 2023, artigo 2024; benchmark UCR univariado, não prova superioridade nestas séries multivariadas.
- [HYDRA](https://arxiv.org/abs/2203.13652): preprint 2022; [código oficial](https://github.com/angus924/hydra). Transformação competitiva no benchmark dos autores.
- [MultiROCKET na implementação aeon](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MultiRocket.html): API multivariada; versão utilizada fixada em 1.3.0, não seguir mudanças de stable sem revalidar.
- [InceptionTime](https://arxiv.org/abs/1909.04939), [código dos autores](https://github.com/hfawaz/InceptionTime): ensemble de redes; aqui somente membro individual para orçamento comparável.
- [LITE/LITEMV](https://arxiv.org/abs/2409.02869): preprint 2024, revisão 2025; [código oficial](https://github.com/MSD-IRIMAS/LITE). Alternativa leve, adiada para não introduzir outra stack neural nesta rodada.
- [Mantis](https://arxiv.org/abs/2502.15637): preprint 2025, revisão 2026; [código oficial](https://github.com/vfeofanov/mantis). [Model card dos pesos](https://huggingface.co/paris-noah/Mantis-8M) indica Apache-2.0; conferir licenças de código e pesos ao redistribuir.
- [MantisV2](https://arxiv.org/abs/2602.17868): preprint fevereiro 2026; comparações UCR/UEA/HAR/EEG dos autores, não uma garantia universal.
- [MOMENT](https://arxiv.org/abs/2402.03885): ICML 2024, avaliação multitarefa; fallback, não instalado como candidato.
- [TimEE](https://arxiv.org/abs/2607.07500): preprint julho 2026; método in-context com suporte rotulado. Compatibilidade multivariada deste problema não verificada; não habilitado.

As fontes motivam famílias, não ganhos esperados. Nenhum ranking publicado substitui validação local.
