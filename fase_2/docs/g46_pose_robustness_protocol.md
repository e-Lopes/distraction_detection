# G4.6 — robustez geométrica à câmera lateral

**Status:** protocolo implementado; dados públicos ainda não adquiridos  
**Escopo:** seed 42, quatro folds, somente treino/validação

A G4.6 isola o efeito da câmera lateral antes da G5. O schema v1 e os resultados G0–G4.5
permanecem imutáveis. A extração v2 registra os olhos separadamente, qualidade geométrica,
assimetria e pose pela decomposição Euler do OpenCV.

São comparadas três representações com o mesmo Random Forest de controle: baseline,
geometria normalizada e geometria com EAR corrigido por pose. Em cada fold, uma regressão
Huber quadrática aprende o EAR aberto esperado em função de pitch/yaw usando exclusivamente
frames `Alert` válidos dos blocos de treino. O limiar exploratório de fechamento é 70% desse
EAR esperado. PERCLOS causal é calculado em 30 e 60 segundos, apenas com frames válidos e
cobertura mínima de 50%.

Os thresholds de `PERCLOS_Thresholds.png` são somente faixas visuais até que sua proveniência,
fórmula de `S` e janela original sejam documentadas. O teste externo não participa da seleção.

NTHU-DDD e YawDD terão manifestos locais com split por `subject_id`. Os encoders auxiliares
produzem `prob_eye_closed_public_encoder` e `prob_yawn_public_encoder`; essas probabilidades
serão atributos, não substitutos do target operacional. A execução externa permanece bloqueada
até aquisição autorizada dos datasets e registro de licenças/proveniência.

## Comandos

```bash
python -m fase_2.src.features.extract_facial_series \
  --schema-version v2 --video-dir fase_2/data/raw/videos \
  --roi-config fase_2/configs/preprocessing/legacy_roi.json --workers 4 --overwrite

python -m fase_2.src.training.g46_pose_robustness --dry-run
python -m fase_2.src.training.g46_pose_robustness
```

## Promoção

A representação corrigida só avança se reduzir a dependência EAR–pose, melhorar validação em
mais de um fold e não elevar falsos episódios de fadiga de forma desproporcional. Só então será
repetida em cinco seeds.
