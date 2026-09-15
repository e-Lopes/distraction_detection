# G4.8 — Guia de anotação dos 22 landmarks no CVAT

## Pacote local

O comando abaixo materializa as 400 ROIs congeladas em `data/external/G48/cvat_sample/`:

```bash
python -m fase_2.src.data.g48_preparation export-cvat
```

O diretório é sensível e ignorado pelo Git. Ele contém `images/`, `package_manifest.csv` com
SHA-256 e dimensões, e `cvat_labels.json` com o contrato. O ZIP local pode ser enviado ao CVAT,
mas não deve ser publicado nem anexado ao repositório.

## Projeto e tarefa

1. Criar projeto `G48_face_landmarks_v1`.
2. Criar um label skeleton `face_22` com os pontos abaixo exatamente nesta ordem.
3. Criar uma tarefa com ordenação lexicográfica dos arquivos e sem embaralhamento.
4. Enviar `g48_cvat_images.zip` ou os arquivos de `images/`.
5. Manter os nomes originais; eles codificam somente `video_id` e índice do frame.
6. Exportar em COCO Keypoints após concluir e revisar.

## Ordem anatômica congelada

Esquerda/direita são sempre da pessoa observada, não do anotador nem da imagem.

| Índice | Ponto |
|---:|---|
| 0 | left_eye_outer |
| 1 | left_eye_upper_outer |
| 2 | left_eye_upper_inner |
| 3 | left_eye_inner |
| 4 | left_eye_lower_inner |
| 5 | left_eye_lower_outer |
| 6 | right_eye_outer |
| 7 | right_eye_upper_outer |
| 8 | right_eye_upper_inner |
| 9 | right_eye_inner |
| 10 | right_eye_lower_inner |
| 11 | right_eye_lower_outer |
| 12 | mouth_left_corner |
| 13 | mouth_upper_left |
| 14 | mouth_upper_center |
| 15 | mouth_upper_right |
| 16 | mouth_right_corner |
| 17 | mouth_lower_right |
| 18 | mouth_lower_center |
| 19 | mouth_lower_left |
| 20 | nose_tip |
| 21 | chin_center |

## Regras de marcação

- Marcar o centro anatômico do ponto, não a borda do pixel mais contrastante.
- Não inferir através de oclusão completa. Marcar o ponto como `outside` quando ele não puder ser
  localizado; usar `occluded` quando sua posição ainda puder ser estimada visualmente.
- Óculos, reflexos e sombras não tornam um ponto automaticamente ausente.
- Se somente um olho estiver visível, anotar o olho visível e marcar os pontos realmente
  invisíveis do outro conforme a regra anterior.
- Não deslocar pontos para fazer a forma parecer simétrica em vista lateral.
- Manter uma única face por frame: o operador dentro da ROI, não pessoas ao fundo.
- Frames sem operador ou com identidade ambígua devem receber a tag de revisão `exclude`, sem
  inventar landmarks.

## Controle de qualidade

- Executar piloto com 12 frames: um frontal, lateral, ocluído e falha do baseline por vídeo,
  quando disponíveis.
- Revisar visualmente o piloto antes de continuar os 388 restantes.
- Reanotar cegamente pelo menos 10% da amostra.
- Não alterar a topologia após iniciar a anotação oficial; qualquer correção cria `v2`.
- Conferir nomes, contagem e hashes contra `package_manifest.csv` antes da importação e depois da
  exportação.

O export COCO Keypoints será convertido pelo pipeline somente se a categoria contiver os 22
nomes nesta ordem. Falhas e pontos invisíveis permanecem ausentes; não são convertidos para zero.

