"""Measurement preparation/audit artifacts inside the public pipeline."""
from collections import defaultdict, Counter
import json
import math
from pathlib import Path

import numpy as np


def validate_raw(path, expected_rows):
    from .pipeline import _read_csv
    rows = _read_csv(path)
    if len(rows) != expected_rows:
        raise ValueError(f'{path}: incomplete raw extraction ({len(rows)}/{expected_rows})')
    last = -np.inf
    for index, row in enumerate(rows):
        if row.get('schema_version') != 'measurement_v1' or int(row['frame_index']) != index or row['video_id'] != path.stem:
            raise ValueError(f'{path}: incompatible raw schema/frame identity')
        timestamp = float(row['timestamp_seconds'])
        if not math.isfinite(timestamp) or timestamp <= last: raise ValueError('Invalid decoder timestamp')
        last = timestamp
        for column in ('ear_left','ear_right','mar','pitch','yaw','roll','valid_left','valid_right','valid_mouth','valid_pose','tracking_status','track_id','rotation_matrix','landmarks_px'):
            if column not in row: raise ValueError(f'Missing raw column: {column}')
        for column in ('ear_left','ear_right','mar','pitch','yaw','roll'):
            if math.isinf(float(row[column])): raise ValueError('Inf is not a measurement')
    meta = json.loads(path.with_suffix('.json').read_text())
    if not meta['complete']: raise ValueError('Sample raw extraction cannot train official experiment')
    from .features.extract_facial_series import sha256_file
    if meta['raw_sha256'] != sha256_file(path): raise ValueError('Raw artifact hash mismatch')
    return rows


def prepare_measurements(config_path, config):
    from .pipeline import _read_csv, _write_csv_atomic, pipeline_fingerprint
    from .data_audit import audit
    from .data.splits import SplitBlock
    from .preprocessing.missingness import expand_behavior_labels
    from .preprocessing.measurement_quality import SIGNALS, treat_block
    from .training.measurement_data import build_measurement_fold
    checked = audit(config_path)
    if checked['errors']: raise ValueError('\n'.join(checked['errors']))
    root = Path(config['outputs']['root'])
    videos = _read_csv(Path(config['data']['manifest']))
    series = {v['video_id']: validate_raw(Path(config['data']['facial_series'])/f"{v['video_id']}.csv", int(v['num_frames'])) for v in videos}
    for video in series:
        meta = json.loads((Path(config['data']['facial_series'])/f'{video}.json').read_text())
        expected_settings = json.loads(json.dumps(config['measurement_protocol']['extraction']))
        if meta['quality'] != config['measurement_protocol']['quality'] or meta['extraction_settings'] != expected_settings:
            raise ValueError('Raw extraction diagnostics/settings do not match frozen configuration; do not reuse stale quality flags')
        from .features.extract_facial_series import sha256_file
        if meta.get('extraction_code_sha256') != sha256_file(Path(__file__).parent/'features'/'measurement_raw.py'):
            raise ValueError('Raw extraction code changed; review provenance before reusing measurements')
    labels = expand_behavior_labels({k:len(v) for k,v in series.items()}, _read_csv(Path(config['data']['annotations'])))
    blocks = [SplitBlock(int(r['fold']), r['subset'], r['video_id'], int(r['start_frame']), int(r['end_frame'])) for r in _read_csv(Path(config['splits']['manifest']))]
    coverage, window_rows, rejection_rows, gap_rows = [], [], [], []
    variants = config['measurement_protocol']['variants']
    for name, variant in variants.items():
        if not variant.get('enabled') or name not in config['preprocessing']['historical_representations']: continue
        pre = config['preprocessing']['historical_representations'][name]
        for fold in config['splits']['folds']:
            print(f'[QUALITY] variant={name} fold={fold} coverage/windows', flush=True)
            for b in [b for b in blocks if b.fold == fold and b.subset != 'test']:
                rows = treat_block(series[b.video_id][b.start_frame:b.end_frame+1], pre['measurement_policy'])
                fields = ['video_id','frame_index','timestamp_seconds','ear_left','ear_right','ear_source','tracking_status','track_id']
                fields += [prefix+s for prefix in ('raw_','','valid_','rejected_','interpolated_','gap_seconds_','requires_imputation_') for s in SIGNALS]
                _write_csv_atomic(root/'treated_frames'/name/f'fold_{fold}'/f'{b.subset}_{b.video_id}_{b.start_frame}.csv',
                                  [{k:r[k] for k in fields} for r in rows],fields)
                from .preprocessing.measurement_quality import gaps
                for signal in SIGNALS:
                    observed = np.array([r['valid_'+signal] == '1' for r in rows])
                    for start,stop in gaps(observed):
                        gap_rows.append(dict(variant=name,fold=fold,subset=b.subset,video_id=b.video_id,
                            indicator=signal,start_frame=rows[start]['frame_index'],end_frame=rows[stop-1]['frame_index'],
                            gap_seconds=rows[start]['gap_seconds_'+signal],bounded=int(start>0 and stop<len(rows))))
                grouped = defaultdict(list)
                for r in rows: grouped[labels[b.video_id][int(r['frame_index'])] or 'unannotated'].append(r)
                for label, group in grouped.items():
                    for indicator in SIGNALS:
                        flags = [int(r['valid_'+indicator]) for r in group]
                        durations = [float(r['gap_seconds_'+indicator]) for r in group if not int(r['valid_'+indicator])]
                        coverage.append(dict(variant=name,fold=fold,subset=b.subset,video_id=b.video_id,label=label,
                            indicator=indicator,frames=len(group),valid_fraction=np.mean(flags),
                            interpolated_fraction=np.mean([int(r['interpolated_'+indicator]) for r in group]),
                            rejected_fraction=np.mean([int(r['rejected_'+indicator]) for r in group]),
                            imputed_fraction=np.mean([int(r['requires_imputation_'+indicator]) for r in group]),
                            max_gap_seconds=max(durations,default=0)))
                    reasons = Counter(f'{region}:{r["reason_"+region]}' for r in group for region in ('left','right','mouth','pose') if r['reason_'+region])
                    reasons.update('tracking:'+r['tracking_status'] for r in group if r['tracking_status'] != 'tracked')
                    switches = sum(a['ear_source'] != z['ear_source'] and 'missing' not in (a['ear_source'],z['ear_source'])
                                   and int(z['frame_index']) == int(a['frame_index'])+1 for a,z in zip(group,group[1:]))
                    reasons['ear_source_change_diagnostic'] = switches
                    rejection_rows.extend(dict(variant=name,fold=fold,subset=b.subset,video_id=b.video_id,label=label,reason=k,count=v) for k,v in reasons.items())
            splits, _, _ = build_measurement_fold(series, labels, blocks, pre, fold=fold,
                size_frames=60, stride_frames=config['windowing']['stride_frames'],
                minimum_proportion=config['windowing']['minimum_target_proportion'], representation=name)
            for subset in ('train','validation'):
                for item in splits[subset].metadata:
                    window_rows.append(dict(variant=name,fold=fold,subset=subset,video_id=item.video_id,
                        start_frame=item.start_frame,end_frame=item.end_frame,label=item.label,
                        missing_ratio=item.missing_ratio,interpolated_ratio=item.interpolated_ratio,
                        no_usable_measurement=int(item.missing_ratio == 1)))
    _write_csv_atomic(root/'quality_coverage.csv', coverage)
    _write_csv_atomic(root/'quality_windows.csv', window_rows)
    _write_csv_atomic(root/'quality_rejections.csv', rejection_rows)
    _write_csv_atomic(root/'quality_gaps.csv', gap_rows)
    # Assert identical retained windows, never drop difficult ones in a treatment.
    sets = defaultdict(set)
    for row in window_rows: sets[row['variant']].add(tuple(row[k] for k in ('fold','subset','video_id','start_frame','end_frame','label')))
    if sets and any(v != next(iter(sets.values())) for v in sets.values()): raise ValueError('Unequal evaluation windows across treatments')
    root.mkdir(parents=True, exist_ok=True)
    state = {'status':'completed','fingerprint':pipeline_fingerprint(Path(config_path),config), 'mode':'development_coverage_only'}
    temp = root/'prepare_state.json.tmp'
    temp.write_text(json.dumps(state,indent=2)); temp.replace(root/'prepare_state.json')
    print('[QUALITY PREPARED] No classifier was fitted.', flush=True)
    return 0


def sample_figure(rows, previews, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,12))
    grid = fig.add_gridspec(5,3)
    for j,(frame,row) in enumerate(previews[:3]):
        ax = fig.add_subplot(grid[0:2,j]); ax.imshow(frame[:,:,::-1])
        geometry = json.loads(row['geometry']); x,y,w,h = geometry['roi']
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x,y),w,h,fill=False,color='yellow'))
        points = json.loads(row['landmarks_px'])
        if points:
            p = np.array(list(points.values())); ax.scatter(p[:,0],p[:,1],s=5,c='lime')
        rotation = json.loads(row['rotation_matrix'])
        if rotation:
            import cv2
            rv = cv2.Rodrigues(np.array(rotation))[0]
            axes = cv2.projectPoints(np.array([[0,0,0],[.03,0,0],[0,.03,0],[0,0,.03]]),rv,
                np.array(json.loads(row['translation'])),np.array(json.loads(row['camera_matrix'])),np.zeros(4))[0].reshape(-1,2)
            for end,color in zip(axes[1:],('red','cyan','blue')):
                ax.plot([axes[0,0],end[0]],[axes[0,1],end[1]],color=color,lw=1)
        ax.set_xlim(max(0,x-60),min(frame.shape[1],x+w+60)); ax.set_ylim(min(frame.shape[0],y+h+60),max(0,y-60))
        ax.set_title(f"f={row['frame_index']} t={row['timestamp_seconds']:.3f}s\n{row['tracking_status']}",fontsize=9)
    times = [float(r['timestamp_seconds']) for r in rows]
    for i,signal in enumerate(('ear_left','ear_right','mar','pitch','yaw','roll')):
        ax = fig.add_subplot(grid[2+i//3,i%3])
        ax.plot(times,[float(r[signal]) for r in rows],lw=1)
        ax.set_title(signal); ax.grid(alpha=.2)
    ax = fig.add_subplot(grid[4,:])
    masks = np.array([[int(r['valid_'+k]) for r in rows] for k in ('left','right','mouth','pose')])
    ax.imshow(masks,aspect='auto',interpolation='nearest',vmin=0,vmax=1,extent=(times[0],times[-1],4,0))
    ax.set_yticks(np.arange(4)+.5,['left','right','mouth','pose']); ax.set_xlabel('decoder timestamp (seconds); geometry heuristics, NOT identity/occlusion confidence')
    fig.suptitle('DEVELOPMENT SAMPLE — raw candidates, no classification labels; no interpolation applied')
    fig.tight_layout(); fig.savefig(target,dpi=120); plt.close(fig)


def measurement_figures(config, metrics):
    """Only actual development results, paired by fold; no historical fallback."""
    from .pipeline import _write_csv_atomic
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(config['outputs']['root'])
    target = Path(config['outputs']['figures']); target.mkdir(parents=True,exist_ok=True)
    rows = [r for r in metrics if r['subset'] == 'validation']
    baseline = {(r['model'],str(r['fold']),str(r['seed']),r['metric'],r.get('label','')):float(r['value'])
                for r in rows if r['representation']=='QA'}
    deltas=[]
    for r in rows:
        key=(r['model'],str(r['fold']),str(r['seed']),r['metric'],r.get('label',''))
        if r['representation'] != 'QA' and key in baseline:
            deltas.append(dict(model=r['model'],variant=r['representation'],fold=r['fold'],seed=r['seed'],
                subset='validation',metric=r['metric'],label=r.get('label',''),delta_vs_QA=float(r['value'])-baseline[key]))
    _write_csv_atomic(root/'quality_paired_deltas.csv',deltas,
                      ['model','variant','fold','seed','subset','metric','label','delta_vs_QA'])
    output=[]
    for metric,label in (('macro_f1',''),('f1','fatigue')):
        selected=[r for r in rows if r['metric']==metric and r.get('label','')==label and r['model'] in ('svm','lstm')]
        if not selected: continue
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,model in zip(axes,('svm','lstm')):
            values=[r for r in selected if r['model']==model]
            variants=sorted({r['representation'] for r in values})
            groups=defaultdict(list)
            for r in values:groups[(r['fold'],r['seed'])].append(r)
            for key,group in sorted(groups.items()):
                group.sort(key=lambda r:r['representation'])
                ax.plot([variants.index(r['representation']) for r in group],[float(r['value']) for r in group],marker='o',label=f'fold/seed {key}')
            ax.set_xticks(range(len(variants)),variants);ax.set_title(model);ax.set_ylim(0,1)
            if values:ax.legend(fontsize=7)
        fig.suptitle(f'Desenvolvimento — {metric} {label}; pontos por fold, não por janela')
        fig.tight_layout(); path=target/f'quality_{metric}_{label or "all"}.png'
        fig.savefig(path,dpi=140);plt.close(fig);output.append(str(path))
    return output


def measurement_report(config, metrics):
    from .pipeline import _read_csv
    root = Path(config['outputs']['root'])
    coverage = _read_csv(root/'quality_coverage.csv')
    lines = ['# Tratamento controlado dos indicadores — v1', '',
        'Pergunta: tratar a qualidade das medições melhora Macro F1 e identificação de Fatigue, mantendo as mesmas janelas?',
        'Relatório gerado pelo pipeline existente. O relatório histórico e suas seções/referências permanecem preservados; não importar conclusões históricas como evidência desta rodada.', '',
        'Desenvolvimento somente. Quatro sessões de um operador; testes históricos já examinados.',
        'SVM linear sobre trajetória achatada e LSTM, w60/stride15/seed42/class weights. Nenhuma janela excluída por qualidade.',
        'QA: referência pareada (5 canais); demais: 5 valores + 5 máscaras de observação + 5 flags de interpolação = 15 canais.',
        'QA usa medidas do mesmo extrator auditável; não é reprodução bit a bit dos arquivos antigos.',
        'Diagnóstico missingness_logistic: somente médias das máscaras/flags (10 features); associação não implica causalidade.', '',
        '| Variante | Habilitada | Política |', '|---|---|---|']
    for name,v in config['measurement_protocol']['variants'].items():
        lines.append(f"| {name} | {v['enabled']} | `{v['policy']}` |")
    lines += ['', '### Cobertura e resultados', '']
    if not coverage: lines.append('Extração completa/preparação pendentes; nenhuma cobertura científica inferida da amostra.')
    else:
        lines += ['Tabelas: quality_coverage.csv (indicador/classe/sessão/fold), quality_windows.csv (conjunto comum), quality_rejections.csv (motivos).',
                  'Comparar Fatigue e Distraction com Alert, incluindo janelas totalmente ausentes. Não confundir flags geométricas com oclusão conhecida.']
    lines += ['', '### Classificação por tratamento, fold e seed', '']
    measured = [r for r in metrics if r['subset'] == 'validation']
    if not measured:
        lines.append('Nenhum treinamento oficial desta rodada. Sem ranking, ganho ou resultado de classificação.')
    else:
        lines += ['| Modelo | Variante | Fold | Seed | Classe | Métrica | Valor |', '|---|---|---|---|---|---|---|']
        lines += [f"| {r['model']} | {r['representation']} | {r['fold']} | {r['seed']} | {r.get('label','')} | {r['metric']} | {r['value']} |" for r in measured]
    lines += ['', '### Comparações pareadas e cobertura', '',
        'Arquivos de deltas pareiam modelo/fold/seed/métrica/classe com QA. QB0 versus QB deve ser examinado para separar qualidade de representação. Não usar janelas como réplicas.',
        'Confusões e predições estão nos artefatos por run, sem limiar de abstenção. Cobertura de Fatigue/Distraction deve acompanhar qualquer ganho de F1.', '',
        '### Falhas e custo', '']
    for row in _read_csv(Path(config['outputs']['registry'])):
        if row['status'] in ('failed','blocked','dependency_missing'):
            lines.append(f"- {row['run_id']}: {row['status']}; {row.get('message','')}")
    lines += ['Custo de classificação ainda não medido quando não há runs. A projeção da amostra não inclui revisão humana nem treinos.', '', '### Limitações', '',
        'ROI e continuidade espacial não certificam identidade. Após perda/ambiguidade, exige âncora humana para retomar.',
        'Olho predominante não definido; C/E bloqueados até verificação. D desabilitada: sem referência operacional defensável.',
        'Limiar geométrico é heurística de desenvolvimento; EAR baixo e MAR alto não são rejeitados pelo seu valor.',
        'Pose com intrínsecos aproximados; erro baixo de reprojeção não comprova acurácia.',
        'Interpolação usa futuro, não representa fechamento ocular observado. Mediana trailing 0,1s pode apagar eventos curtos e precisa de inspeção.',
        'Protocolos e comandos: docs/protocols/measurement_quality_protocol.md.', '']
    return '\n'.join(lines)
