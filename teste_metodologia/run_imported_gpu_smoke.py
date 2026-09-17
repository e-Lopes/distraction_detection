"""Copia auditavel da fase_1 e teste de inferencia GPU, sem meta-treino."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
FEATURES = ['ear', 'mar', 'pitch', 'yaw', 'roll']
KEYS = ['video_id', 'frame_index']
LABELS = {'alert': 0, 'distraction': 1, 'fatigue': 1}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(source):
    destination = BASE / 'results/imported_fase_1' / source.name
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        target = destination / path.name
        expected = digest(path)
        if target.exists() and digest(target) != expected:
            raise ValueError(f'Copia existente diferente; nao sobrescrever: {target}')
        if not target.exists():
            shutil.copy2(path, target)
        if digest(target) != expected:
            raise RuntimeError(f'Copia nao verificada: {target}')
        manifest.append(dict(source=str(path.resolve()), copy=str(target.resolve()), sha256=expected))
    return destination, manifest


def join_labels(frames, labels):
    if frames.duplicated(KEYS).any() or labels.duplicated(KEYS).any():
        raise ValueError('Chaves duplicadas nos resultados ou rotulos.')
    # Never import the old indicator columns from the label file.
    joined = frames.merge(labels[KEYS + ['state']], on=KEYS, how='left', validate='one_to_one', indicator=True)
    missing = int(joined['_merge'].eq('left_only').sum())
    joined['state'] = joined.state.fillna('unknown').str.strip().str.lower()
    return joined.drop(columns='_merge'), missing


def strict_windows(frame, length):
    """Fixed disjoint intervals, homogeneous binary labels, no imputation."""
    if length < 1:
        raise ValueError('window_len deve ser positivo.')
    arrays, records, excluded = [], [], []
    for video, group in frame.groupby('video_id', sort=True):
        group = group.sort_values('frame_index').set_index('frame_index')
        if group.index.has_duplicates or group.index.min() < 0:
            raise ValueError('Indices de frames invalidos.')
        for start in range(0, int(group.index.max()) + 2 - length, length):
            chunk = group.reindex(range(start, start + length))
            values = chunk[FEATURES].to_numpy(dtype=np.float32)
            labels = chunk.state.map(LABELS)
            reason = None
            if labels.isna().any():
                reason = 'absent_unknown_or_missing_frame'
            elif labels.nunique() != 1:
                reason = 'mixed_binary_labels'
            elif not chunk.all_indicators_valid.eq(1).all() or not np.isfinite(values).all():
                reason = 'missing_indicator'
            item = dict(video_id=video, start=start, end_exclusive=start+length)
            if reason:
                excluded.append(dict(**item, reason=reason))
            else:
                records.append(dict(**item, label=int(labels.iloc[0]), window_index=len(arrays)))
                arrays.append(values)
    features = np.stack(arrays) if arrays else np.empty((0, length, 5), dtype=np.float32)
    return features, pd.DataFrame(records, columns=['video_id', 'start', 'end_exclusive', 'label', 'window_index']), pd.DataFrame(excluded)


def select_episode(windows, video, k, q, seed):
    rng = np.random.default_rng(seed)
    support, query = [], []
    for label in (0, 1):
        indices = windows.loc[windows.video_id.eq(video) & windows.label.eq(label), 'window_index'].to_numpy()
        if len(indices) < k + q:
            raise ValueError(f'{video}: classe {label} sem janelas suficientes para K={k}, Q={q}.')
        chosen = rng.permutation(indices)[:k+q]
        support.extend(chosen[:k].tolist())
        query.extend(chosen[k:].tolist())
    chosen = windows.set_index('window_index')
    for s in support:
        for qidx in query:
            a, b = chosen.loc[s], chosen.loc[qidx]
            if a.start < b.end_exclusive and b.start < a.end_exclusive:
                raise ValueError('Sobreposicao support/query.')
    return support, query


def support_normalize(support, query):
    mean = support.mean(axis=(0, 1), keepdims=True)
    std = support.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (support-mean)/std, (query-mean)/std, mean, std


def gpu_smoke(features, windows, k, q):
    from prototypical_network import TemporalEncoder1D
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA obrigatoria para o smoke test; nao ha fallback CPU.')
    torch.manual_seed(42)
    encoder = TemporalEncoder1D().cuda().eval()
    if not all(p.is_cuda for p in encoder.parameters()):
        raise RuntimeError('Pesos fora da GPU.')
    calls = []

    def audit(module, inputs, output):
        if not inputs[0].is_cuda or not output.is_cuda:
            raise RuntimeError('Forward fora da GPU.')
        calls.append(str(output.device))

    handle = encoder.register_forward_hook(audit)
    results = []
    try:
        for video in sorted(windows.video_id.unique()):
            try:
                support, query = select_episode(windows, video, k, q, 42)
            except ValueError as error:
                results.append(dict(video_id=video, status='insufficient_windows', reason=str(error)))
                continue
            sx, qx, mean, std = support_normalize(features[support], features[query])
            with torch.inference_mode():
                # eval() freezes BatchNorm: query cannot influence support statistics.
                zs = encoder(torch.from_numpy(sx).cuda())
                zq = encoder(torch.from_numpy(qx).cuda())
                prototypes = zs.reshape(2, k, -1).mean(dim=1)
                distances = torch.cdist(zq, prototypes).square()
                logits = -distances
                target = torch.arange(2, device='cuda').repeat_interleave(q)
                loss = torch.nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise RuntimeError('Saida numerica invalida.')
                torch.cuda.synchronize()
            results.append(dict(video_id=video, status='ok', support_indices=support,
                                query_indices=query, loss=float(loss), inference_device=str(logits.device),
                                support_mean=mean.tolist(), support_std=std.tolist(),
                                note='Random untrained encoder. Functional test only; no accuracy claim.'))
    finally:
        handle.remove()
    return dict(gpu=torch.cuda.get_device_name(0), cuda_forward_calls=len(calls),
                optimizer_steps=0, checkpoint_saved=False, episodes=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE.parent / 'fase_1/resultados_legados/extracao_ao_vivo/20260916_204539_018100')
    parser.add_argument('--window-len', type=int, default=90)
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--q-query', type=int, default=1)
    args = parser.parse_args()
    if min(args.window_len, args.k_shot, args.q_query) < 1:
        parser.error('Tamanhos devem ser positivos.')
    source = args.source.resolve()
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    if config.get('inference_backend') != 'torch_cuda':
        raise ValueError('Esta sequencia espera os resultados da extracao GPU.')
    copied, manifest = snapshot(source)
    run = BASE / 'results/imported_gpu_smoke' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    label_path = BASE / 'results/classificacoes_frames_exatos.csv'
    frames = []
    for i in range(1, 5):
        video = f'video_{i:02d}'
        group = pd.read_csv(copied / f'{video}.csv')
        if group.empty or not group.video_id.eq(video).all():
            raise ValueError(f'Video incorreto no arquivo {video}.csv')
        frames.append(group)
    merged, missing = join_labels(pd.concat(frames, ignore_index=True), pd.read_csv(label_path, usecols=KEYS+['state']))
    merged.to_csv(run / 'features_with_labels.csv', index=False)
    features, windows, excluded = strict_windows(merged, args.window_len)
    windows.to_csv(run / 'windows.csv', index=False)
    excluded.to_csv(run / 'excluded_windows.csv', index=False)
    np.savez_compressed(run / 'windows.npz', features=features)
    counts = windows.groupby(['video_id', 'label']).size().unstack(fill_value=0).reindex(
        index=[f'video_{i:02d}' for i in range(1,5)], columns=[0,1], fill_value=0)
    counts.to_csv(run / 'window_counts.csv')
    coverage = merged.groupby(['video_id','state']).agg(frames=('frame_index','size'), valid=('all_indicators_valid','sum'))
    coverage.to_csv(run / 'frame_coverage.csv')
    metadata = dict(source_files=manifest, labels_sha256=digest(label_path), rows=len(merged),
                    unlabeled_frames=missing, window_len=args.window_len, stride=args.window_len,
                    k_shot=args.k_shot, q_query=args.q_query, imputation=False,
                    strict_windows=len(windows), angle_quality_validated=False, complete=False,
                    script_sha256=digest(Path(__file__)), torch_version=torch.__version__)
    (run / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Importacao: {len(merged)} frames, {missing} sem rotulo.\n{counts}', flush=True)
    result = gpu_smoke(features, windows, args.k_shot, args.q_query)
    (run / 'gpu_smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    metadata.update(complete=True, smoke_executed=bool(result['cuda_forward_calls']))
    (run / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = ['# Importacao e teste GPU', '', f'Origem: `{source}`.',
              f'Frames: {len(merged)}; sem rotulo: {missing} (mantidos como unknown).',
              f'Janelas estritas: {len(windows)}, comprimento/stride {args.window_len}.', '',
              '```text', counts.to_string(), '```', '',
              f'GPU: {result["gpu"]}; forwards auditados: {result["cuda_forward_calls"]}.',
              'Encoder aleatorio em eval; normalizacao apenas no suporte. Nenhum treino ou checkpoint.',
              'K/Q deste teste nao substituem a avaliacao K=1/5/10 planejada.',
              'Janelas sem overlap, sem NaN e homogeneas na classe binaria. Sem imputacao.',
              'Filtro estrito introduz vies de selecao; consultar frame_coverage e excluded_windows.',
              'Angulos ainda nao validados. Nao interpretar este teste como resultado cientifico.', '',
              '## Proximo passo',
              'Auditar landmarks e erro de reprojecao antes de tratar lacunas. Fixar extrator,',
              'preparar dados publicos para meta-treino e usar os videos proprios so na meta-avaliacao.',
              'Os scripts numerados legados nao foram executados: usam outro CSV/preenchimento',
              'e o terceiro treina nos videos proprios, em desacordo com o plano.']
    (run / 'analysis.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Relatorio: {run / "analysis.md"}', flush=True)


if __name__ == '__main__':
    main()
