"""Reports exclusively from compatible fits trained on the binary target."""
from collections import defaultdict
from pathlib import Path
import json
import math


def generate(config_path):
    from .pipeline import load_config, build_plan, _read_csv, _write_csv_atomic
    from .reporting import consolidate_metrics, METRIC_FIELDS
    config = load_config(config_path)
    plan = build_plan(config_path, scope='all')
    completed = [r for r in plan if r.status == 'completed']
    current = {r.run_id for r in completed}
    current.update(Path(r.artifact).parent.name for r in completed if Path(r.artifact).suffix == '.pt')
    metrics = [r for r in consolidate_metrics(config) if r['run_id'] in current]
    for row in metrics:
        row['scientific_status'] = config['target']['evaluation']
        row['source'] = 'binary_training'
    _write_csv_atomic(Path(config['outputs']['metrics']), metrics, METRIC_FIELDS)
    grouped = defaultdict(dict)
    for row in metrics:
        if row['scope'] not in {'global','class'} or row['metric'] not in {'macro_f1','f1','recall','precision'}:
            continue
        try:
            value = float(row['value'])
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        key = (row['model'],row['representation'],str(row['window_size_frames']),row['subset'])
        metric = row['metric'] + (f"_{row['label']}" if row['label'] else '')
        grouped[key].setdefault(metric, []).append(value)
    summary = []
    for (model, representation, window, subset), values in sorted(grouped.items()):
        row = dict(model=model,representation=representation,window=window,subset=subset)
        for key, samples in values.items():
            row[key] = sum(samples)/len(samples)
        row['evaluations'] = len(values.get('macro_f1',[]))
        summary.append(row)
    root = Path(config['outputs']['root'])
    _write_csv_atomic(root/'binary_summary.csv', summary,
        ['model','representation','window','subset','macro_f1','f1_attention','f1_distraction',
         'recall_attention','recall_distraction','precision_attention','precision_distraction','evaluations'])
    display = config['target']['display_names']
    lines = ['# Atenção × Distração — treinamento binário', '',
        f"Experimento: **{config['name']}**. Fase: **{config['target']['evaluation']}**.", '',
        'Atenção = alert. Distração = fatigue + distraction, conforme definição operacional do pesquisador.',
        'Rótulos agrupados por frame antes da maioria das janelas. Modelos ajustados diretamente com duas classes.',
        'Ausência do operador e falhas de observação não recebem automaticamente o rótulo Distração.', '',
        f'Execuções compatíveis concluídas: **{len(completed)}/{len(plan)}**.', '',
        '| Modelo | Entrada | Janela | Avaliação | N | Macro F1 | F1 Atenção | F1 Distração |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['model']} | {row['representation']} | {row['window']} | {row['subset']} | "
                     f"{row['evaluations']} | {row.get('macro_f1',float('nan')):.4f} | "
                     f"{row.get('f1_attention',float('nan')):.4f} | {row.get('f1_distraction',float('nan')):.4f} |")
    if not summary:
        lines += ['', 'Nenhum resultado binário compatível disponível.']
    lines += ['', '## Protocolo e limites', '',
        'Quatro sessões do mesmo operador; janelas sobrepostas não são amostras independentes.',
        'Médias descritivas por execução. Validação e teste são apresentados separadamente.',
        'A seleção usa somente validação. Testes históricos já consultados não são testes intocados.',
        'Macro F1 binário não é comparável diretamente ao ternário. O agrupamento não prova melhora na detecção de fadiga.',
        'A análise por episódio exige matching e parâmetros temporais próprios; não são importados proxies históricos de fadiga.',
        f"Early stopping: {config['training']['early_stopping']}; máximo {config['training']['max_epochs']} épocas.", '',
        '## Execuções pendentes ou impedidas', '']
    for run in plan:
        if run.status != 'completed':
            lines.append(f'- {run.model}/{run.representation}/w{run.window_size_frames}/fold{run.fold}/seed{run.seed}: {run.status}. {run.reason}')
    figures = Path(config['outputs']['figures']); figures.mkdir(parents=True, exist_ok=True)
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = [r for r in summary if r['subset'] == 'validation']
        if rows:
            fig, ax = plt.subplots(figsize=(10,max(4,.32*len(rows))))
            ax.barh([f"{r['model']} · {r['representation']} · w{r['window']}" for r in rows],
                    [r.get('macro_f1',0) for r in rows])
            ax.set(xlim=(0,1),xlabel='Macro F1 de validação · duas classes')
            fig.tight_layout();fig.savefig(figures/'binary_macro_f1.png',dpi=140);plt.close(fig)
            lines += ['', '![Macro F1 binário](figures/binary_macro_f1.png)']
    Path(config['outputs']['report']).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    _write_csv_atomic(Path(config['outputs']['event_metrics']), [], ['model','target_class','metric','value'])
    print(f"[REPORT] {config['outputs']['report']} — {len(completed)}/{len(plan)} concluídas",flush=True)
    return 0
