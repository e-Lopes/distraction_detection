# Dados locais

Vídeos, frames, anotações completas, indicadores extraídos, labels de detecção e pesos de
modelos são sensíveis e não devem ser versionados. Somente manifestos anonimizados, schemas
e métricas consolidadas autorizadas entram no Git.

## Organização esperada

```text
data/
├── raw/
│   ├── videos/                       # 1.mp4 a 4.mp4
│   └── annotations/                  # intervalos temporais manuais
├── external/
│   └── phase1/                       # ativos auxiliares, se importados localmente
│       ├── frames_by_state/
│       └── cellphone_detection/
├── interim/                          # indicadores por frame
├── processed/                        # janelas e tabelas para treino
└── manifests/                        # somente metadados anonimizados versionáveis
```

Não é obrigatório duplicar os ativos da fase 1. Uma futura rotina de importação deve aceitar
uma origem local configurável, validar hashes/contagens e produzir dados em `external/`, que
já é ignorado. Symlinks e caminhos absolutos não devem aparecer em arquivos versionados.

Consulte [Inventário e proveniência](../docs/data_sources_and_provenance.md) para saber o papel
de cada fonte e as limitações de reutilização.
