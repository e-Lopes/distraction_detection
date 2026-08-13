# Mês 1 — Dados e protocolo

Este documento operacionaliza o primeiro mês do [Plano de Ações Pós-Banca](../plano_pos_banca.md). Ele é um checklist de execução: marcar um item como concluído exige a evidência indicada, não apenas código ou arquivos de exemplo.

## Objetivo e critério de conclusão

Estabelecer a fundação reproduzível do pipeline temporal, com dataset e anotações congelados, manifesto validado, diagnóstico de classes e missingness, regra de janelamento, quatro folds sem vazamento e decisões metodológicas registradas.

O Mês 1 termina somente quando todos os itens obrigatórios estiverem concluídos ou quando uma pendência externa estiver identificada como bloqueio, com responsável e evidência documentados. Modelos, augmentation, balanceamento, explicabilidade e late fusion ficam fora desta etapa.

## Estado inicial verificado

- [x] Existem quatro vídeos locais: `1.mp4`, `2.mp4`, `3.mp4` e `4.mp4`.
- [x] Os quatro vídeos são ignorados por `fase_2/.gitignore`, pela regra `data/raw/**`.
- [x] Nenhum vídeo está rastreado pelo Git.
- [x] A estrutura existente de `fase_2` será reutilizada; não existe `fase_2/fase_2`.
- [x] As anotações temporais manuais dos quatro vídeos foram localizadas e normalizadas em 71 intervalos contínuos, versão `v1`.
- [x] O protótipo da outra máquina foi recuperado e auditado. Como ele mantinha EAR, MAR, pitch, yaw e roll somente em memória, as séries foram regeneradas com o mesmo Face Mesh e schema canônico explícito.
- [x] OpenCV headless 4.10 foi disponibilizado para leitura não destrutiva dos metadados.

`fase_1/validacao_manual.csv` foi identificado como fonte auxiliar: contém 40.778 revisões
manuais de estados corporais/celular, mas não é o target temporal da fase 2. Suas classes
`NORMAL`, `CELULAR`, `POSTURA_RUIM`, `MAOS_FORA` e `NAO_DETECTADO` não podem ser
convertidas silenciosamente em Alert/Fatigue/Distraction.

## Entradas necessárias

| Entrada | Local esperado/configurado | Condição de aceite |
|---|---|---|
| Vídeos | `fase_2/data/raw/videos/` | Quatro arquivos presentes, somente leitura e ignorados pelo Git. |
| Anotações | Definido em `fase_2/configs/data/base.yaml` após localizar o formato real | Formato, versão, IDs e cobertura temporal documentados; nenhuma conversão silenciosa. |
| Indicadores faciais | `fase_2/data/interim/legacy_extraction/` | Proveniência, hashes e semântica de missing/zero registrados. |
| Configuração | `fase_2/configs/` | Caminhos relativos à raiz e parâmetros metodológicos centralizados. |

O inventário completo das fontes conhecidas está em
[`data/manifests/data_sources.csv`](../../data/manifests/data_sources.csv), com análise de
proveniência em [`docs/data_sources_and_provenance.md`](../data_sources_and_provenance.md).

## 1. Segurança e organização

- [ ] Executar `git status --short --branch` antes de qualquer alteração e preservar mudanças existentes.
- [ ] Executar `git check-ignore -v fase_2/data/raw/videos/{1,2,3,4}.mp4` antes e depois da implementação.
- [ ] Confirmar que `git ls-files` não lista vídeos, frames, anotações brutas ou indicadores individuais sensíveis.
- [ ] Não modificar, copiar, renomear nem decodificar os vídeos em arquivos derivados nesta etapa.
- [ ] Não modificar `fase_1/`.
- [ ] Não criar `fase_2/fase_2`, uma segunda árvore de código ou diretórios equivalentes aos já existentes.
- [ ] Não introduzir caminhos absolutos ou específicos da máquina.

## 2. Manifesto dos vídeos

### Tarefas

- [x] Implementar leitura não destrutiva de metadados dos quatro MP4.
- [x] Mapear os nomes físicos `1.mp4`–`4.mp4` para `video_01`–`video_04`.
- [x] Gerar `fase_2/data/manifests/videos.csv` com `video_id`, `relative_path`, `session_id`, `fps`, `width`, `height`, `num_frames`, `duration_seconds` e `annotation_version`.
- [x] Registrar limitações do container e divergências entre duração, FPS e contagem de frames.

### Critérios de aceite

- [x] FPS, resolução, número de frames e duração foram extraídos para todos os vídeos.
- [x] A consistência aproximada `duration_seconds ≈ num_frames / fps` foi verificada e eventuais diferenças foram explicadas.
- [x] Nenhum valor veio de `videos.example.csv`; valores zerados, fictícios ou copiados não contam como resultado.
- [x] Os caminhos do manifesto são relativos e não expõem operador, empresa ou diretórios internos.

### Comando esperado

```bash
python -m fase_2.src.data manifest --config fase_2/configs/data/base.yaml
```

## 3. Anotações e congelamento do dataset

### Tarefas

- [x] Localizar e documentar o formato real antes de escrever o adaptador.
- [x] Relacionar nomes/IDs das anotações aos quatro registros do manifesto.
- [x] Validar campos obrigatórios, classes, frames negativos, intervalos invertidos, limites do vídeo, sobreposições conflitantes e trechos sem rótulo.
- [x] Separar classe comportamental (`alert`, `fatigue`, `distraction`) de estado operacional (`valid`, `face_missing`, `occlusion`, `operator_absent`).
- [x] Gerar relatório sem corrigir silenciosamente a fonte.
- [ ] Registrar versão, data de congelamento e procedimento de consenso.

### Critérios de aceite

- [ ] Todas as anotações possuem proveniência e versão.
- [ ] Os quatro vídeos têm cobertura explicitamente medida; lacunas são reportadas.
- [ ] Erros invalidantes impedem os diagnósticos seguintes.
- [ ] A revisão independente de 10% a 20% e o cálculo de concordância estão planejados ou executados conforme disponibilidade do segundo anotador.

### Comando esperado

```bash
python -m fase_2.src.data validate-annotations --config fase_2/configs/data/base.yaml
```

## 4. Diagnóstico frame-level e window-level

### Tarefas

- [x] Gerar distribuição frame-level por vídeo e classe, com frames, duração e percentuais por vídeo e dataset.
- [x] Construir janelas de 30, 60 e 150 frames com stride configurável, inicialmente 15.
- [x] Rotular pela maioria dos frames válidos, exigindo proporção mínima de 60%.
- [x] Marcar empate ou predominância insuficiente como `mixed`/`transition`.
- [x] Manter janelas mistas nas estatísticas e excluí-las apenas do treinamento principal.
- [x] Impedir que uma janela atravesse vídeos.
- [x] Gerar distribuição por vídeo, tamanho, classe e quantidade de janelas mistas.

### Critérios de aceite

- [ ] Totais e percentuais fecham dentro da tolerância numérica documentada.
- [x] A regra de rótulo está isolada, configurável e coberta por testes sintéticos.
- [x] Casos de empate, 60% exatos, vídeo menor que a janela, início, final e stride estão testados.

### Comandos esperados

```bash
python -m fase_2.src.data diagnose-frames --config fase_2/configs/data/base.yaml
python -m fase_2.src.data diagnose-windows --config fase_2/configs/data/base.yaml
```

## 5. Missingness

### Tarefas

- [x] Localizar o extrator existente e documentar como falhas eram codificadas.
- [x] Não interpretar zero como missing: a nova tabela usa campos vazios e `face_detected=0`.
- [ ] Calcular missing rate por vídeo, classe e tamanho de janela.
- [x] Calcular a distribuição de gaps consecutivos e quantidades de gaps curtos e longos por vídeo.
- [ ] Se os indicadores não existirem, implementar somente a interface e documentar o formato necessário; não reprocessar os quatro vídeos.

### Critérios de aceite

- [x] Cálculos estão cobertos por dados sintéticos, inclusive zeros válidos e gaps nas bordas.
- [x] O limite entre gap curto e longo é configurado em 15 frames, sem observar o fold de teste.
- [ ] Resultados reais são omitidos, não inventados, enquanto os indicadores estiverem ausentes.

### Comando esperado

```bash
python -m fase_2.src.data diagnose-missingness --config fase_2/configs/data/base.yaml
```

## 6. Splits temporais

### Tarefas

- [x] Gerar quatro folds leave-one-video-out, reservando um vídeo inteiro para teste em cada fold.
- [x] Dividir os vídeos restantes em blocos temporais contínuos de treino e validação.
- [x] Aplicar purge gap de pelo menos 150 frames entre blocos adjacentes.
- [x] Excluir janelas que toquem limites ou purge gaps.
- [x] Persistir somente IDs e índices temporais, sem dados sensíveis.

### Testes obrigatórios contra vazamento

- [x] Um teste falha quando intervalos de treino e validação se sobrepõem.
- [x] Um teste falha quando o purge gap é menor que 150 frames.
- [x] Um teste falha quando uma janela alcança dois subconjuntos.
- [x] Um teste falha quando o vídeo reservado aparece em treino ou validação.
- [x] Um teste confirma que janelas sobrepostas do mesmo trecho permanecem no mesmo subconjunto.
- [x] Os testes usam somente dados sintéticos e não carregam os MP4.

Os placeholders foram substituídos por testes sintéticos de isolamento e purge gap. A posição
dos blocos de validação ainda precisa ser revista porque alguns subconjuntos não contêm fadiga.

### Comando esperado

```bash
python -m fase_2.src.data generate-splits --config fase_2/configs/data/base.yaml --split-config fase_2/configs/splits/leave_one_video_out.yaml
```

## 7. Entregáveis e evidências

- [ ] Manifesto anonimizado validado.
- [ ] Relatório de validação das anotações.
- [ ] CSVs e resumo Markdown das distribuições frame-level e window-level.
- [x] Relatório inicial de missingness por vídeo; recortes por classe e janela permanecem pendentes.
- [ ] Quatro arquivos de splits temporais versionáveis.
- [x] Testes automatizados de rótulos, anotações, missingness e vazamento.
- [ ] Registros em `fase_2/docs/decisions/` para regra de janela, stride/60%, splits/purge gap e estados operacionais.
- [ ] README com comandos executáveis a partir da raiz.

## 8. Fechamento do mês

Antes de declarar o Mês 1 concluído:

```bash
git status --short --branch
git check-ignore -v fase_2/data/raw/videos/{1,2,3,4}.mp4
git ls-files '*.mp4' '*.avi' '*.mkv' '*.mov'
pytest fase_2/tests
ruff check fase_2
test ! -e fase_2/fase_2
```

- [ ] Todos os comandos terminam com o resultado esperado.
- [ ] Bloqueios restantes têm evidência, responsável e próximo passo.
- [ ] Metodologia, decisões, tabelas e limitações foram atualizadas.
- [ ] Nenhum resultado individual sensível foi adicionado ao Git.
