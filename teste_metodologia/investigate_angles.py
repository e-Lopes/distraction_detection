"""Auditoria geometrica offline; nao corrige nem substitui indicadores."""
import json
import math
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

from compare_extraction_backends import BASE, FacialBackend, RUNTIME, sha256
from fase_2.src.features.extract_facial_series import HEAD_POSE_IDS, HEAD_POSE_3D


def angles(rotation):
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    return np.degrees([math.atan2(-rotation[2, 0], sy),
                       math.atan2(rotation[1, 0], rotation[0, 0]),
                       math.atan2(rotation[2, 1], rotation[2, 2])])


def solve(points, camera, method):
    ok, rv, tv = cv2.solvePnP(HEAD_POSE_3D, np.asarray(points, np.float64), camera,
                             np.zeros((4, 1)), flags=method)
    if not ok:
        raise RuntimeError('solvePnP falhou')
    rotation = cv2.Rodrigues(rv)[0]
    projected = cv2.projectPoints(HEAD_POSE_3D, rv, tv, camera, np.zeros((4, 1)))[0].reshape(-1, 2)
    depths = (rotation @ HEAD_POSE_3D.T + tv)[2]
    return dict(pitch=angles(rotation)[0], yaw=angles(rotation)[1], roll=angles(rotation)[2],
                rmse_px=float(np.sqrt(np.mean(np.sum((points-projected)**2, axis=1)))),
                min_depth=float(depths.min()), all_positive_depth=bool((depths>0).all())), rotation, projected


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA obrigatoria nesta auditoria pareada.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = BASE / 'results/angle_audit' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    source = BASE / 'results/imported_fase_1/20260916_204539_018100'
    targets = {(3, 28673)}
    original = pd.read_csv(source / 'video_03.csv')
    # The previous seek-based audit read timestamp 1686.299, not frame 28673.
    bad_pose_frame = int(original.loc[(original.timestamp_seconds-1686.299).abs().idxmin(), 'frame_index'])
    targets.add((3, bad_pose_frame))
    summaries, events = [], []
    for video in range(1, 5):
        df = pd.read_csv(source / f'video_{video:02d}.csv').sort_values('frame_index')
        cols = ['pitch', 'yaw', 'roll']
        raw = df[cols].diff().abs()
        circular = abs((df[cols].diff()+180)%360-180)
        adjacent = df.frame_index.diff().eq(1) & df.all_indicators_valid.eq(1) & df.all_indicators_valid.shift().eq(1)
        jump = circular.max(axis=1).where(adjacent)
        summaries.append(dict(video=video, adjacent_valid_pairs=int(adjacent.sum()),
                              circular_over30=int(jump.gt(30).sum()), circular_over90=int(jump.gt(90).sum()),
                              wrap_only_over180=int((adjacent & raw.max(axis=1).gt(180) & jump.le(30)).sum())))
        selected = df.loc[jump.nlargest(3).index]
        for idx, row in selected.iterrows():
            targets.add((video, int(row.frame_index)))
        for idx in df.index[jump.gt(30)]:
            events.append(dict(video=video, frame_index=int(df.loc[idx, 'frame_index']),
                               max_circular_delta=float(jump.loc[idx])))
    pd.DataFrame(summaries).to_csv(out / 'full_video_jump_counts.csv', index=False)
    pd.DataFrame(events).to_csv(out / 'full_video_jump_events.csv', index=False)
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    x, y, w, h = config['roi']
    camera = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], np.float64)
    backends = {'cpu': FacialBackend('cpu'), 'cuda': FacialBackend('cuda:0')}
    rows, paired, stored = [], [], {}
    rng = np.random.default_rng(42)
    perturbations = []
    regions = {}
    timestamps = {}
    for video in range(1, 5):
        wanted = {i for v, center in targets if v == video for i in range(center-1, center+2)}
        expected = pd.read_csv(source / f'video_{video:02d}.csv').set_index('frame_index')
        cap = cv2.VideoCapture(str(BASE.parent / f'fase_2/data/raw/{video}.mp4'))
        try:
            for index in range(max(wanted)+1):
                if not cap.grab():
                    raise RuntimeError('Falha na decodificacao sequencial')
                if index in wanted:
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise RuntimeError('Falha no retrieve')
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if abs(timestamp - expected.loc[index, 'timestamp_seconds']) > .002:
                        raise RuntimeError(f'Timestamp nao corresponde ao CSV: {video}/{index}')
                    regions[video,index] = frame[y:y+h, x:x+w].copy()
                    timestamps[video,index] = timestamp
        finally:
            cap.release()
        print(f'Video {video}: frames sequenciais conferidos por timestamp', flush=True)
    for video, center in sorted(targets):
        for index in range(center-1, center+2):
            region = regions[video,index]
            results, panels = {}, []
            for name, backend in backends.items():
                landmarks, detected, presence = backend.detect(region)
                panel = region.copy()
                cv2.putText(panel, f'{video}/{index} {name} presence={presence:.3f}', (8,20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                if landmarks is None:
                    rows.append(dict(video=video, frame_index=index, backend=name, method='missing_face', presence=presence))
                    panels.append(panel)
                    continue
                points = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in landmarks])
                stored[f'v{video}_f{index}_{name}'] = points
                image_points = points[HEAD_POSE_IDS, :2]
                for p in points:
                    cv2.circle(panel, tuple(np.round(p[:2]).astype(int)), 1, (60,200,60), -1)
                for method, flag in [('iterative', cv2.SOLVEPNP_ITERATIVE), ('sqpnp', cv2.SOLVEPNP_SQPNP)]:
                    try:
                        values, rotation, projected = solve(image_points, camera, flag)
                        rows.append(dict(video=video, frame_index=index, backend=name, method=method, presence=presence, **values))
                        results[(name, method)] = (rotation, values, image_points)
                        if method == 'iterative':
                            for p, pred in zip(image_points, projected):
                                cv2.circle(panel, tuple(np.round(p).astype(int)), 3, (0,255,255), -1)
                                cv2.drawMarker(panel, tuple(np.round(pred).astype(int)), (0,0,255), cv2.MARKER_CROSS, 8, 1)
                            cv2.putText(panel, f'RMSE={values["rmse_px"]:.2f} depth={values["min_depth"]:.3f}', (8,42), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    except cv2.error as error:
                        rows.append(dict(video=video, frame_index=index, backend=name, method=method, error=str(error)))
                if video == 3 and index == bad_pose_frame and name == 'cpu':
                    for trial in range(100):
                        perturbed = image_points + rng.normal(0, .01, image_points.shape)
                        for method, flag in [('iterative', cv2.SOLVEPNP_ITERATIVE), ('sqpnp', cv2.SOLVEPNP_SQPNP)]:
                            values, _, _ = solve(perturbed, camera, flag)
                            perturbations.append(dict(trial=trial, method=method, noise_sigma_px=.01, **values))
                panels.append(panel)
            for method in ('iterative', 'sqpnp'):
                if ('cpu', method) in results and ('cuda', method) in results:
                    a, av, ap = results[('cpu', method)]
                    b, bv, bp = results[('cuda', method)]
                    geodesic = np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1)))
                    paired.append(dict(video=video, frame_index=index, method=method,
                                       head_point_max_delta_px=float(np.linalg.norm(ap-bp, axis=1).max()),
                                       rotation_delta_degrees=float(geodesic)))
            if index == center:
                cv2.imwrite(str(out / f'v{video}_f{index}.jpg'), np.concatenate(panels, axis=1))
        print(f'Auditado video {video}, frame {center}', flush=True)
    pd.DataFrame(rows).drop_duplicates(['video','frame_index','backend','method']).to_csv(out/'poses.csv', index=False)
    pd.DataFrame(paired).drop_duplicates(['video','frame_index','method']).to_csv(out/'paired.csv', index=False)
    pd.DataFrame(perturbations).to_csv(out/'perturbations.csv', index=False)
    np.savez_compressed(out/'landmarks.npz', **stored)
    metadata = dict(targets=sorted(targets), camera=camera.tolist(), camera_calibrated=False,
                    decoding='sequential grab/retrieve; timestamps matched to full extraction within 2ms',
                    known_unstable_frame=bad_pose_frame,
                    frame_timestamps={f'{v}/{i}':t for (v,i),t in timestamps.items()},
                    seed=42, precision='FP32 eager, TF32 off', torch=torch.__version__, opencv=cv2.__version__,
                    model_manifest=json.loads((RUNTIME/'manifest.json').read_text(encoding='utf-8')),
                    script_sha256=sha256(__import__('pathlib').Path(__file__)),
                    source_hashes={p.name: sha256(p) for p in source.glob('video_??.csv')},
                    note='Diagnostic sample selected for extreme jumps, not prevalence of geometric failures. No production correction.')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__':
    main()
