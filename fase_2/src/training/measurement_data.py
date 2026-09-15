"""Quality representations through the existing sequence/fold contract."""
from __future__ import annotations

import numpy as np
from ..data.windowing import build_windows
from ..data.splits import validate_split_blocks
from ..preprocessing.measurement_quality import SIGNALS, FEATURES, treat_block, number
from .dummy_baseline import CLASSES


def build_measurement_fold(series, labels_by_video, blocks, preprocessing, *, fold,
                           size_frames, stride_frames, minimum_proportion, representation, classes=CLASSES):
    from .temporal_data import SequenceSplit, SequenceScaler, SequenceMetadata
    if validate_split_blocks(blocks, purge_gap_frames=0):
        raise ValueError('Invalid or overlapping split blocks')
    policy = preprocessing['measurement_policy']
    baseline = representation == 'QA'
    n_features = 5 if baseline else 15
    selected = [b for b in blocks if b.fold == fold]
    treated = {}
    fit_rows = []
    for block in selected:
        original = series[block.video_id][block.start_frame:block.end_frame+1]
        if len(original) != block.end_frame-block.start_frame+1:
            raise ValueError('Incomplete raw series; samples cannot be used as full experiment inputs')
        rows = treat_block(original, policy)
        if baseline:
            for row, source in zip(rows, original):
                for name in SIGNALS:
                    key = 'legacy_'+name if name in SIGNALS[2:] else name
                    value = number(source[key]) if source['tracking_status'] == 'tracked' else np.nan
                    row[name] = str(value if np.isfinite(value) else 0.)
        treated[(block.video_id, block.start_frame, block.end_frame)] = rows
        if block.subset == 'train': fit_rows.extend(rows)
    mean, scale = np.zeros(n_features), np.ones(n_features)
    for j, name in enumerate(SIGNALS):
        values = np.array([number(row[name]) for row in fit_rows
                           if baseline or row['valid_'+name] == '1'])
        values = values[np.isfinite(values)]
        # Unobservable training channel: fixed zero imputation, no held-out statistics.
        if len(values):
            mean[j], scale[j] = values.mean(), max(values.std(), 1e-8)
    scaler = SequenceScaler(tuple(mean), tuple(scale))
    xs = {s: [] for s in ('train', 'validation', 'test')}
    ys, meta = {s: [] for s in xs}, {s: [] for s in xs}
    windows = build_windows(labels_by_video, size_frames=size_frames, stride_frames=stride_frames,
                            behavior_classes=set(classes), minimum_proportion=minimum_proportion)
    for w in windows:
        if w.label == 'mixed': continue
        matches = [b for b in selected if b.video_id == w.video_id and b.start_frame <= w.start_frame and w.end_frame <= b.end_frame]
        if not matches: continue
        if len(matches) != 1: raise ValueError('Window belongs to multiple partitions')
        b = matches[0]
        rows = treated[(b.video_id, b.start_frame, b.end_frame)][w.start_frame-b.start_frame:w.end_frame-b.start_frame+1]
        names = SIGNALS if baseline else FEATURES
        x = np.array([[number(r[n]) for n in names] for r in rows], dtype=np.float32)
        if not baseline:
            x = scaler.transform(x)
        x[~np.isfinite(x)] = 0 # Numeric model placeholder AFTER observed-only scaling.
        xs[b.subset].append(x)
        ys[b.subset].append(classes.index(w.label))
        valid = np.array([[r['valid_'+s] == '1' for s in SIGNALS] for r in rows])
        ip = np.array([[r['interpolated_'+s] == '1' for s in SIGNALS] for r in rows])
        meta[b.subset].append(SequenceMetadata(w.video_id, w.start_frame, w.end_frame,
                                             w.label, 1-valid.mean(), ip.mean()))
    if any(not v for v in xs.values()): raise ValueError('Empty experimental split')
    if baseline:
        # Faithful R0 scaling: overlapping training windows weight repeated frames.
        flat = np.asarray(xs['train']).reshape(-1,5).astype(np.float64)
        mean,scale = flat.mean(axis=0),flat.std(axis=0)
        scale[scale == 0] = 1
        scaler = SequenceScaler(tuple(mean),tuple(scale))
        xs = {s:[scaler.transform(x) for x in values] for s,values in xs.items()}
    result = {s: SequenceSplit(np.array(xs[s], dtype=np.float32), np.array(ys[s], dtype=np.int64), tuple(meta[s])) for s in xs}
    return result, scaler, {}
