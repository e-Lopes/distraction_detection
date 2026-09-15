"""Report appendix for the modern-family protocol; no claims without measured runs."""
from pathlib import Path
import csv
from collections import defaultdict


def modern_report(config, metrics):
    lines = ['\n## Ampliação de famílias — protocolo v1', '',
        'Pergunta: novas famílias melhoram Macro F1 e identificação de Fatigue em sessões reservadas?',
        f"Fase configurada: **{config['modern_protocol']['evaluation']}**. Desenvolvimento e teste externo não são agrupados.",
        'Quatro vídeos, operador compartilhado: generalização entre sessões, sem alegação de novos operadores.',
        'Resultados históricos consultados anteriormente são exploratórios; não constituem teste intocado.', '',
        '| Modelo | Representação | Janelas | Balanceamento |', '|---|---|---|---|']
    for section in config['models']['screening'].values():
        if section.get('enabled'):
            for item in section['candidates']:
                lines.append(f"| {item['model']} | {section['representation']} | {section['windows']} | {section.get('balancing', 'none')} |")
    groups = defaultdict(list)
    for row in metrics:
        if row['metric'] in ('macro_f1', 'f1'):
            groups[(row['model'], row['subset'], row['metric'], row.get('label', ''))].append(float(row['value']))
    lines += ['', '### Resultados por fase', '']
    if not groups:
        lines.append('Sem resultados científicos novos. Não há ranking nem evidência de ganho.')
    for key, values in sorted(groups.items()):
        lines.append(f'- {key}: média descritiva={sum(values)/len(values):.4f}; n={len(values)}. '
                     'Não interpretar repetições de seed como sessões independentes.')
    lines += ['', '### Custos, falhas e resultados negativos', '',
        'Tempos de transformação e classificador, scores e PR são preservados nos CSVs por run.',
        'Average Precision não é PR-AUC trapezoidal. PR indefinida sem positivos ou sem negativos.',
        'Custo não medido permanece ausente; não inclui extração facial. Inferência exige aquecimento e sincronização CUDA.',
        'Métricas por episódio permanecem secundárias, pendentes de thresholds temporais congelados.', '']
    registry = Path(config['outputs']['registry'])
    if registry.exists():
        with registry.open() as stream:
            for row in csv.DictReader(stream):
                if row['status'] in ('failed', 'blocked', 'dependency_missing'):
                    lines.append(f"- {row['run_id']}: {row['status']} — {row.get('message', '')}")
    lines += ['', '### Limites e conclusão', '',
        'Média entre sessões e Macro F1 de predições externas concatenadas respondem a perguntas distintas.',
        'Deltas devem parear mesmo fold/seed/entrada. Não usar janelas sobrepostas como réplicas independentes.',
        'Com quatro sessões, a inferência estatística é frágil. Nenhuma conclusão de superioridade sem confirmação.',
        'Protocolo, orçamento e referências: `docs/protocols/modern_families_protocol.md`.', '']
    return '\n'.join(lines)
