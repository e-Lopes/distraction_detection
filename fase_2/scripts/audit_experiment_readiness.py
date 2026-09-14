"""Audita artefatos reais sem treinar nem alterar extrações. Executar com python -u -m."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path('fase_2')
    output = root / 'outputs/metrics/experimental_audit'
    output.mkdir(parents=True, exist_ok=True)
    videos = pd.read_csv(root / 'data/manifests/videos.csv')
    rows = []
    for extractor in ('mediapipe', 'insightface', 'openface'):
        for video in videos.itertuples():
            path = root / f'outputs/G48A_all_frames/metrics/{extractor}/{video.video_id}.csv'
            row = dict(extractor=extractor, video_id=video.video_id,
                       expected_frames=video.num_frames, frames=0, detected=0,
                       status='missing_final', artifact=str(path))
            if path.exists():
                frame = pd.read_csv(path)
                detected = frame.face_detected.astype(str).str.lower().isin(['true', '1'])
                valid = (len(frame) == video.num_frames
                         and np.array_equal(frame.frame_id.to_numpy(), np.arange(video.num_frames))
                         and frame.video_id.eq(video.video_id).all()
                         and np.allclose(frame.timestamp_ms, frame.frame_id * 1000 / video.fps))
                row.update(frames=len(frame), detected=int(detected.sum()),
                           status=('structurally_complete' if detected.any() else 'suspect_zero_detection')
                           if valid else 'invalid_alignment')
            rows.append(row)
            print(f"{extractor}/{video.video_id}: {row['status']} ({row['frames']}/{video.num_frames})", flush=True)
    pd.DataFrame(rows).to_csv(output / 'extractor_integrity.csv', index=False)
    registry = pd.read_csv(root / 'outputs/final/run_registry.csv')
    registry.groupby(['scope', 'phase', 'model', 'representation', 'window_size_frames', 'status'],
                     dropna=False).agg(runs=('run_id', 'size'), folds=('fold', 'nunique'),
                                       seeds=('seed', 'nunique')).reset_index().to_csv(
                                           output / 'window_coverage.csv', index=False)
    g2 = pd.read_csv(root / 'outputs/metrics/G2/g2_execution_table.csv')
    coverage = []
    for model in ('lstm', 'tcn', 'transformer'):
        for window in (30, 60, 150):
            selected = g2[(g2.model == model) & (g2.window_size_frames == window) & (g2.seed == 42)]
            complete = (len(selected) == 4 and set(selected.fold) == {1, 2, 3, 4}
                        and selected.best_validation_macro_f1.notna().all())
            coverage.append(dict(model=model, window=window, seed=42, runs=len(selected), complete=complete))
    pd.DataFrame(coverage).to_csv(output / 'g2_matrix.csv', index=False)
    state = dict(historical_g2_complete=all(r['complete'] for r in coverage),
                 extractor_structure_complete=all(r['status'] == 'structurally_complete' for r in rows),
                 temporal_promotion_ready=False,
                 note='Estrutura completa não comprova precisão geométrica nem proveniência; promoção exige revisão científica.')
    (output / 'readiness.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
