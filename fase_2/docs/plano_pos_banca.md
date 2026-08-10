# Plano de Ações Pós-Banca

> **Status documental:** referência complementar derivada de
> `Plano_de_Acoes_Pos_Banca_Atualizado.pdf`. A execução atual segue
> `PlanoPósBanca.pdf` e as decisões mais recentes registradas em `docs/decisions/`.

## Versão revisada e protocolo de execução experimental

**Projeto:** Fatigue and Distraction Detection for Heavy Machinery Teleoperation  
**Pesquisador:** Eduardo Lamy Lopes  
**Programa:** Mestrado em Informática — PUCPR  
**Orientador:** Marcelo Eduardo Pellenz  
**Coorientador:** Marco Antonio Simões Teixeira  
**Horizonte:** seis meses

> **Objetivo do plano:** converter as recomendações da banca em um protocolo reproduzível, viável e alinhado à hipótese H3, preservando a fusão multimodal como extensão complementar.

Este Markdown é a versão de trabalho navegável do documento. O PDF `Plano_de_Acoes_Pos_Banca_Atualizado.pdf`, na raiz do repositório, permanece como snapshot oficial.

## 1. Direcionamento e escopo

O núcleo da dissertação avaliará como diferentes representações e modelos temporais classificam Alerta, Fadiga e Distração a partir de indicadores faciais, considerando falhas de detecção, duração das janelas e desbalanceamento. A comparação será realizada sobre uma entrada comum, permitindo confrontar regras fixas, classificadores clássicos e arquiteturas temporais.

> **Prioridade:** concluir um protocolo facial-temporal robusto e explicável. A fusão entre face, corpo e celular não deve bloquear os experimentos centrais.

### 1.1 Entregas obrigatórias e complementares

| Nível | Itens |
|---|---|
| Obrigatório | Dataset congelado; protocolo de janelas; splits sem vazamento; zero-fill e interpolação; XGBoost, LSTM e TCN; desbalanceamento; ablação; estudos de caso. |
| Baseline | Regras fixas; zero-fill; distribuição real; classificador clássico sem modelagem temporal. |
| Complementar | Random Forest, SVM e Transformer; late fusion; SHAP aprofundado em modelos temporais. |
| Trabalho futuro | Generalização para novos operadores/câmeras; modelos end-to-end sobre vídeo bruto; implantação Edge completa. |

## 2. Dataset e protocolo de anotação

O conjunto atual contém quatro vídeos reais de teleoperação, aproximadamente 122.337 frames e eventos naturais. Os indicadores faciais são EAR, MAR, pitch, yaw e roll. As classes comportamentais permanecem Alerta, Fadiga e Distração.

### 2.1 Separar classe comportamental de estado operacional

| Dimensão | Valores | Uso |
|---|---|---|
| Classe comportamental | Alert, Fatigue, Distraction | Target principal dos modelos. |
| Estado operacional | Valid, FaceMissing, Occlusion, OperatorAbsent | Qualidade/condição da observação; não deve ser confundida com o target. |

Operador ausente será tratado primeiro como estado operacional. Uma eventual saída específica de ausência pertence à etapa de fusão e não deve alterar a taxonomia principal antes de sua validação.

### 2.2 Concordância das anotações

- Selecionar uma amostra estratificada de segmentos de todas as classes, transições, vídeos e condições de iluminação.
- Solicitar revisão independente de aproximadamente 10% a 20% dos segmentos, de acordo com a disponibilidade do segundo anotador.
- Calcular concordância por classe e registrar divergências nas fronteiras temporais dos eventos.
- Usar Cohen's Kappa quando aplicável e documentar o procedimento de consenso.

## 3. Construção e rotulagem das janelas

Serão avaliadas janelas de 30, 60 e 150 frames. O stride deverá ser explícito em toda execução e a distribuição das classes será recalculada para cada combinação de tamanho, stride e regra de rotulagem.

### 3.1 Regra primária

1. A classe candidata é a classe majoritária entre os frames válidos da janela.
2. A classe candidata deve atingir a proporção mínima configurada, inicialmente 60%.
3. Se não houver predominância suficiente ou houver empate, a janela recebe a marcação Transition/Mixed.
4. Janelas Transition/Mixed ficam fora do treinamento principal e são mantidas para análise complementar.
5. Janelas com missing rate acima do limite definido no treino são analisadas separadamente; o limite não será escolhido olhando o teste.

### 3.2 Diagnóstico obrigatório

- Distribuição frame-level por vídeo e classe.
- Distribuição window-level por tamanho de janela, stride e regra de rótulo.
- Missing rate por vídeo, classe, janela e estado operacional.
- Distribuição das durações dos gaps para orientar o limite de interpolação curta.
- Quantidade de janelas mistas e efeito das transições sobre as classes raras.

## 4. Divisão temporal e generalização

A avaliação externa usará leave-one-video-out, produzindo quatro rodadas. Em cada rodada, um vídeo inteiro será reservado para teste e os três restantes serão usados para treino e validação.

| Rodada | Teste | Treino e validação |
|---:|---|---|
| 1 | Vídeo 1 | Vídeos 2, 3 e 4 |
| 2 | Vídeo 2 | Vídeos 1, 3 e 4 |
| 3 | Vídeo 3 | Vídeos 1, 2 e 4 |
| 4 | Vídeo 4 | Vídeos 1, 2 e 3 |

### 4.1 Regras contra vazamento

- A validação interna usará blocos temporais contínuos dentro dos vídeos de treino.
- Janelas sobrepostas não podem atravessar treino, validação e teste.
- Será aplicado purge gap de pelo menos 150 frames entre subconjuntos adjacentes.
- Normalização, imputação, seleção de features, class weights e hiperparâmetros serão estimados somente no treino.
- O fold de teste não será usado para escolher janela, stride, limiar de gap, augmentation ou arquitetura.

> **Limite da conclusão:** se os quatro vídeos representarem o mesmo operador, o estudo demonstrará generalização entre sessões e condições do domínio estudado, não entre novos operadores.

## 5. Pré-processamento e representação

### 5.1 Condições de observação

| Condição | Tratamento | Informação adicional |
|---|---|---|
| Detecção válida | Usar os cinco indicadores observados. | `face_detected = 1` |
| Gap curto | Comparar interpolação linear, forward fill e mediana móvel; selecionar no treino/validação. | `was_interpolated = 1` |
| Gap longo | Não interpolar como dado real; usar mediana do treino ou zero no baseline. | `missing_duration_so_far` |
| Operador ausente | Registrar como estado operacional separado. | Sinal corporal/ausência |

### 5.2 Representação sequencial

A entrada principal com flags será uma matriz N × 8: `[EAR, MAR, Pitch, Yaw, Roll, FaceDetected, WasInterpolated, MissingDurationSoFar]`. A duração acumulada do gap é preferida à duração total, pois não utiliza informação futura e é compatível com inferência em tempo real.

### 5.3 Features para classificadores clássicos

- Média, mediana, desvio-padrão, mínimo, máximo e amplitude.
- Inclinação temporal e estatísticas dos deltas.
- Proporção de valores acima/abaixo de limiares de referência.
- Missing ratio, quantidade de gaps e maior gap na janela.

LSTM e TCN receberão prioritariamente sinais e flags frame a frame. Features agregadas não serão repetidas em cada frame sem uma comparação específica, para evitar redundância.

## 6. Estratégia de modelagem

| Papel | Modelo | Objetivo |
|---|---|---|
| Baseline de domínio | Regras fixas | Comparação com thresholds isolados e sem aprendizado temporal. |
| Baseline clássico principal | XGBoost | Avaliar features agregadas e oferecer explicabilidade direta. |
| Temporal recorrente | LSTM | Modelar dependências sequenciais. |
| Temporal convolucional | TCN | Modelar padrões locais e longos com treinamento eficiente. |
| Opcional | RF, SVM, Transformer | Comparações complementares sem bloquear o núcleo. |

### 6.1 Execução em duas etapas

1. Usar uma configuração de referência para selecionar pré-processamento, janela e representação sem explorar todas as combinações com todos os modelos.
2. Comparar os modelos principais usando a configuração selecionada e repetir os resultados em todos os folds e seeds definidos.

## 7. Desbalanceamento e data augmentation

A comparação será controlada e realizada somente depois de estabilizar o pipeline principal.

| Cenário | Descrição |
|---|---|
| A — Original | Treinamento com a distribuição real. |
| B — Class weights | Maior penalização para erros nas classes raras. |
| C — Weighted sampling | Maior frequência de janelas raras durante o treino. |
| D — Augmentation leve | Jitter, pequena variação de escala e masking curto, preservando faixas físicas. |

- Augmentation e sampling serão aplicados apenas ao treino.
- Validação e teste manterão a distribuição observada.
- Não combinar weighted sampling e class weights no experimento principal sem hipótese específica.
- Avaliar se o ganho em Macro F1 aumenta falsos alertas por hora.

## 8. Métricas e análise estatística

| Categoria | Métricas |
|---|---|
| Principal | Macro F1. |
| Por classe | Precision, recall e F1 para Alert, Fatigue e Distraction. |
| Balanceamento | Balanced accuracy e matriz de confusão. |
| Operacional | Falsos alertas por hora/sessão e latência de detecção quando mensurável. |
| Robustez | Média, desvio-padrão e intervalo de confiança entre folds/seeds. |
| Eficiência | Tempo de treino, tempo de inferência e tamanho do modelo. |

Além da avaliação por janela, eventos completos deverão ser inspecionados para evitar que muitas janelas sobrepostas produzam uma impressão exagerada do tamanho amostral.

## 9. Explicabilidade

### 9.1 Importância global

- SHAP e importância interna para XGBoost, quando adequados.
- Permutation importance por indicador e por grupo.
- Relatar estabilidade da importância entre folds, e não apenas uma execução.

### 9.2 Ablação por grupo

| Grupo | Indicadores |
|---|---|
| Ocular | EAR |
| Oral | MAR |
| Head pose | Pitch, yaw e roll |
| Missingness | Face detected, interpolated e missing duration |
| Multimodal opcional | Phone, hands, posture e absence |

A ablação principal deverá retreinar o modelo após remover o grupo. Apenas zerar uma feature no teste pode criar entradas fora da distribuição.

### 9.3 Estudos de caso

- Verdadeiro positivo de fadiga.
- Verdadeiro positivo de distração.
- Falso positivo e falso negativo.
- Caso com alto missing rate para verificar se o modelo aprendeu o comportamento ou a falha do detector.

## 10. Fusão hierárquica opcional

A late fusion combinará probabilidades do modelo facial-temporal, evidência de celular e sinais corporais. A implementação ocorrerá somente após a conclusão dos experimentos centrais.

1. Verificar presença do operador e qualidade da observação.
2. Obter probabilidades de alerta, fadiga e distração do módulo facial-temporal.
3. Integrar probabilidade de celular e sinais corporais em um meta-classificador simples.
4. Avaliar a decisão final e comparar com o melhor modelo facial isolado.

> **Prevenção de vazamento:** o meta-classificador deverá ser treinado com probabilidades out-of-fold dos modelos-base; não com previsões geradas sobre os mesmos exemplos usados para treinar esses modelos.

## 11. Cronograma revisado

| Mês | Foco | Critério de conclusão |
|---:|---|---|
| 1 | Dados e protocolo | Dataset/anotações congelados; regra de janela; diagnóstico; folds e registro de decisões. |
| 2 | Pré-processamento | Zero-fill, interpolação, flags e features implementados e testados sem vazamento. |
| 3 | Modelos centrais | Regras, XGBoost, LSTM e TCN executados; janela e configuração principal selecionadas. |
| 4 | Desbalanceamento | Original, class weights, sampling e augmentation comparados com métricas operacionais. |
| 5 | Explicabilidade | Ablação, importance e estudos de caso concluídos; fusão apenas se o núcleo estiver fechado. |
| 6 | Síntese e defesa | Testes finais, estatística, limitações, texto revisado e materiais da defesa. |

### 11.1 Escrita contínua

A escrita não será concentrada no sexto mês. Ao final de cada mês, deverão ser atualizados metodologia, registro de decisões, tabelas de resultados e limitações. O mês 6 ficará reservado à integração e revisão.

## 12. Critérios de decisão e riscos

| Risco | Mitigação |
|---|---|
| Poucos vídeos | Leave-one-video-out, intervalos de confiança e conclusões restritas ao domínio estudado. |
| Fadiga rara | Macro F1, recall por classe, pesos/sampling/augmentation e análise de falsos alertas. |
| Falha do Face Mesh | Flags, análise de missingness, interpolação curta e estudo de caso específico. |
| Escopo excessivo | XGBoost, LSTM e TCN como núcleo; Transformer e fusão condicionados ao tempo. |
| Vazamento temporal | Splits por vídeo/blocos, purge gap, fit de transformações apenas no treino e testes automatizados. |
| Viés de anotação | Segundo anotador em amostra estratificada e análise de concordância. |

## 13. Justificativa da abordagem baseada em indicadores

Embora abordagens end-to-end sobre vídeo bruto sejam possíveis, o conjunto disponível é limitado para o treinamento confiável de modelos espaço-temporais de alta capacidade. Os indicadores faciais reduzem a dimensionalidade, incorporam conhecimento do domínio, diminuem o custo computacional e tornam as decisões mais interpretáveis. Essa representação comum também permite comparar regras fixas, classificadores clássicos e modelos temporais de forma controlada.

A limitação é a dependência da qualidade da detecção facial, observada nas diferentes taxas de detecção entre os vídeos. Por isso, missingness e estados operacionais fazem parte explícita do protocolo, em vez de serem tratados apenas como ruído.

## Apêndice A — Estrutura recomendada do repositório Git

O repositório deve ser organizado por etapas reproduzíveis, e não por notebooks ou modelos isolados. Dados sensíveis e artefatos pesados permanecem fora do Git.

```text
configs/
  data/ splits/ preprocessing/ experiment/
data/
  raw/ interim/ processed/ manifests/
docs/
  methodology/ decisions/ data_dictionary.md
notebooks/
outputs/
  metrics/ figures/ models/ predictions/ logs/
reports/
scripts/
src/
  data/ preprocessing/ features/ models/
  training/ evaluation/ explainability/ fusion/ utils/
tests/
```

### A.1 Regras de versionamento

- Não versionar vídeos, frames, modelos treinados, dados pessoais, caminhos internos ou arquivos da empresa.
- Versionar configurações, código, manifestos anonimizados, testes, documentação metodológica e métricas consolidadas quando permitido.
- Registrar hash do commit, configuração, fold, seed, versão do dataset e métricas em cada execução.
- Manter notebooks apenas para exploração; lógica reutilizável deve migrar para `src/`.
- Criar testes para rotulagem das janelas, ausência de sobreposição temporal, interpolação e cálculo das métricas.
