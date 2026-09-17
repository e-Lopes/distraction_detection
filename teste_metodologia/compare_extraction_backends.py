"""Auditoria facial pareada; nao treina nem altera os CSVs do experimento."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from fase_1.visualizar_cinco_series import (
    RUNTIME, REVISION, EXPECTED_SHA256, _load_face_modules, measure_frame, sha256,
)

FEATURES = ['ear', 'mar', 'pitch', 'yaw', 'roll']


def select_clips(labels, length):
    """Centro do maior trecho continuo por video/estado, sem consultar deteccoes."""
    if length < 1 or labels.duplicated(['video_id', 'frame_index']).any():
        raise ValueError('Tamanho invalido ou chaves de frames duplicadas.')
    clips = []
    for video, group in labels.groupby('video_id', sort=True):
        group = group.sort_values('frame_index').copy()
        group['run'] = (group.state.ne(group.state.shift()) |
                        group.frame_index.diff().ne(1)).cumsum()
        for state in ('alert', 'distraction', 'fatigue', 'absent'):
            runs = [g for _, g in group[group.state.eq(state)].groupby('run')]
            if not runs:
                continue
            run = max(runs, key=len)
            count = min(length, len(run))
            start = int(run.frame_index.iloc[0]) + (len(run) - count) // 2
            clips.append(dict(video_id=str(video), state=state, start=start, count=count))
    return clips


def difference(a, b, angular=False):
    delta = np.asarray(a) - np.asarray(b)
    return np.abs((delta + 180) % 360 - 180 if angular else delta)


class Session:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.calls = 0

    def run(self, outputs, inputs):
        import torch
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(next(iter(inputs.values())))).to(self.device)
            result = self.model(batch)
            if batch.device != self.device or any(t.device != self.device for t in result):
                raise RuntimeError('Inferencia no dispositivo incorreto.')
            self.calls += 1
            return [t.cpu().numpy() for t in result]


class FacialBackend:
    def __init__(self, device):
        import torch
        nets, detector_helpers, mesh_helpers = _load_face_modules()
        self.device = torch.device(device)

        def session(cls, name):
            net = cls()
            net.load_state_dict(torch.load(RUNTIME / 'weights' / name,
                                           map_location='cpu', weights_only=True))
            net.eval().to(self.device)
            if any(p.device != self.device for p in net.parameters()):
                raise RuntimeError('Pesos no dispositivo incorreto.')
            return Session(net, self.device)

        self.detector = detector_helpers.BlazeFace.__new__(detector_helpers.BlazeFace)
        self.detector.session = session(nets.BlazeFaceNet, 'face_detection_short_range.pt')
        self.detector.input_name = 'image'
        self.detector.anchors = detector_helpers._generate_face_anchors()
        self.mesh = mesh_helpers.FaceMesh.__new__(mesh_helpers.FaceMesh)
        self.mesh.session = session(nets.FaceLandmarkerNet, 'face_landmarker_256x256.pt')
        self.mesh.input_name = 'image'
        self.mesh.input_size, self.mesh.num_landmarks = 256, 478
        for _ in range(3):
            self.detector.session.run(None, {'image': np.zeros((1, 3, 128, 128), np.float32)})
            self.mesh.session.run(None, {'image': np.zeros((1, 3, 256, 256), np.float32)})

    def detect(self, region):
        boxes, keys, scores = self.detector.detect(region, threshold=.5)
        points, presence = None, np.nan
        if len(boxes):
            result, confidence = self.mesh.predict(region, boxes[:1], margin=.25, keypoints=keys[:1])
            presence = float(confidence[0])
            if presence >= .5 and np.isfinite(result[0]).all():
                h, w = region.shape[:2]
                points = [SimpleNamespace(x=float(p[0]/w), y=float(p[1]/h), z=float(p[2]/w))
                          for p in result[0]]
        return points, bool(len(boxes)), presence


def summarize(rows):
    frame = pd.DataFrame(rows)
    coverage = frame.groupby(['backend', 'state']).agg(
        frames=('frame_index', 'size'), faces=('face_mesh_detected', 'sum'),
        valid=('all_indicators_valid', 'sum'), mean_ms=('elapsed_ms', 'mean')).reset_index()
    coverage['valid_fraction'] = coverage.valid / coverage.frames
    comparisons = []
    keys = ['video_id', 'frame_index']
    cpu = frame[frame.backend.eq('torch_cpu')]
    for backend in ('torch_cuda', 'mediapipe_tracking', 'mediapipe_static'):
        pair = cpu.merge(frame[frame.backend.eq(backend)], on=keys, suffixes=('_cpu', '_other'), validate='one_to_one')
        item = dict(reference='torch_cpu', compared=backend, frames=len(pair),
                    face_disagreements=int((pair.face_mesh_detected_cpu != pair.face_mesh_detected_other).sum()),
                    common_valid=int((pair.all_indicators_valid_cpu.eq(1) & pair.all_indicators_valid_other.eq(1)).sum()))
        for feature in FEATURES:
            valid = np.isfinite(pair[f'{feature}_cpu']) & np.isfinite(pair[f'{feature}_other'])
            delta = difference(pair.loc[valid, f'{feature}_cpu'], pair.loc[valid, f'{feature}_other'], feature in FEATURES[2:])
            item[f'{feature}_paired_n'] = int(valid.sum())
            item[f'{feature}_mae'] = float(delta.mean()) if len(delta) else None
            item[f'{feature}_max'] = float(delta.max()) if len(delta) else None
        comparisons.append(item)
    return frame, coverage, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-per-clip', type=int, default=60)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--gpu-index', type=int, default=0)
    args = parser.parse_args()
    import torch
    import mediapipe as mp
    if not torch.cuda.is_available():
        raise RuntimeError('Comparacao exige CUDA; nenhum fallback silencioso.')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = json.loads((RUNTIME / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision'] != REVISION:
        raise RuntimeError('Revisao externa inesperada.')
    for name, expected in manifest['source_hashes'].items():
        if sha256(RUNTIME / 'face_mesh' / name) != expected:
            raise RuntimeError(f'Codigo externo modificado: {name}')
    for name in ('face_detection_short_range.pt', 'face_landmarker_256x256.pt'):
        if sha256(RUNTIME / 'weights' / name) != EXPECTED_SHA256[name]:
            raise RuntimeError(f'Peso modificado: {name}')
    label_path = BASE / 'results/classificacoes_frames_exatos.csv'
    labels = pd.read_csv(label_path, usecols=['video_id', 'frame_index', 'state'])
    clips = select_clips(labels, args.frames_per_clip)
    roi_path = BASE.parent / 'fase_1/roi_config.json'
    x, y, w, h = json.loads(roi_path.read_text(encoding='utf-8'))['roi_cadeira']
    out = BASE / 'results/backend_comparison' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    metadata = dict(arguments=vars(args), clips=clips, roi=[x,y,w,h], label_sha256=sha256(label_path),
                    models=manifest, torch=torch.__version__, mediapipe=mp.__version__,
                    opencv=cv2.__version__, gpu=torch.cuda.get_device_name(args.gpu_index),
                    execution='FP32 eager, TF32 off; same facial helpers and weights; no body gate',
                    tracking_reset='each clip, no preceding warmup frames',
                    decoding='sequential grab before clip; no CAP_PROP_POS_FRAMES seek', complete=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    backends = {'torch_cpu': FacialBackend('cpu'), 'torch_cuda': FacialBackend(f'cuda:{args.gpu_index}')}
    rows, done = [], 0
    started = time.perf_counter()
    total = sum(c['count'] for c in clips)
    for clip_index, clip in enumerate(clips):
        number = int(clip['video_id'].split('_')[-1])
        path = BASE.parent / f'fase_2/data/raw/{number}.mp4'
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'Video indisponivel: {path}')
        # VFR videos can seek to a different timestamp while reporting the requested index.
        for _ in range(clip['start']):
            if not cap.grab():
                cap.release()
                raise RuntimeError('Falha ao avancar sequencialmente para o trecho.')
        tracking = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1,
                                                  min_detection_confidence=.5, min_tracking_confidence=.5)
        static = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                                max_num_faces=1, min_detection_confidence=.5)
        try:
            for offset in range(clip['count']):
                index = clip['start'] + offset
                if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != index:
                    raise RuntimeError('Seek nao corresponde ao frame solicitado.')
                ok, image = cap.read()
                if not ok or x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                    raise RuntimeError('Falha de leitura ou ROI fora do video.')
                region = image[y:y+h, x:x+w].copy()
                rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                names = list(backends) if offset % 2 == 0 else list(reversed(backends))
                for name in names + ['mediapipe_tracking', 'mediapipe_static']:
                    tick = time.perf_counter()
                    if name in backends:
                        landmarks, detected, presence = backends[name].detect(region)
                    else:
                        result = (tracking if name == 'mediapipe_tracking' else static).process(rgb)
                        landmarks = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
                        detected, presence = None, np.nan
                    elapsed = (time.perf_counter() - tick) * 1000
                    values, _, eye, diagnostic = measure_frame(True, landmarks, w, h)
                    rows.append(dict(video_id=clip['video_id'], frame_index=index, clip_id=clip_index,
                                     state=clip['state'], timestamp_seconds=cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                                     backend=name, detector_found=detected, presence=presence,
                                     elapsed_ms=elapsed, selected_eye=str(eye), **dict(zip(FEATURES, values)), **diagnostic))
                done += 1
        finally:
            cap.release()
            tracking.close()
            static.close()
        elapsed = time.perf_counter() - started
        print(f'{done}/{total} frames | {clip["video_id"]} {clip["state"]} | ETA {elapsed/done*(total-done):.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / 'frames.csv', index=False)
    frame, coverage, comparisons = summarize(rows)
    coverage.to_csv(out / 'coverage.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out / 'paired_comparison.csv', index=False)
    metadata.update(complete=True, elapsed_seconds=time.perf_counter()-started,
                    calls={name: dict(detector=b.detector.session.calls, mesh=b.mesh.session.calls)
                           for name,b in backends.items()}, warmup_calls_per_network=3)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = ['# Validacao facial CPU/GPU', '',
              f'Amostra diagnostica: {done} frames em {len(clips)} trechos. Nao e avaliacao final de classificacao.', '',
              '## Protocolo',
              '- Maior intervalo continuo por video/estado, recorte central; selecao independente das deteccoes.',
              '- CPU/CUDA: mesmos pesos, preprocessing, FP32 eager, limiares 0.5; sem tracking e sem filtro corporal.',
              '- MediaPipe original: referencias com/sem tracking, reiniciadas por trecho. Nao sao ground truth.',
              '- EAR usa apenas o olho mais a esquerda da imagem; angulos usam a convencao legada, sem recalibracao.',
              '- Tempos excluem leitura, inicializacao e calculo dos indicadores; nao estimam a execucao completa.', '',
              '## Cobertura por estado', '', '```text', coverage.to_string(index=False), '```', '',
              '## Diferencas pareadas', '', '```json', json.dumps(comparisons, indent=2), '```', '',
              'MAE angular usa distancia circular em graus; EAR/MAR sao adimensionais. Pares finitos por indicador.',
              'Concordancia CPU/GPU nao demonstra acuracia anatomica. Cobertura tambem nao e acuracia.', '',
              '## Continuidade do experimento',
              '1. Inspecionar visualmente landmarks em trechos independentes, sobretudo falhas em distraction/fatigue.',
              '2. Fixar extrator e convencoes; manter NaN e mascaras individuais. Nao interpolar lacunas longas.',
              '3. Ajustar normalizacao/tratamento apenas no treino ou suporte permitido; nao usar query para calibrar.',
              '4. Recontar janelas Alert/Not-Alert, excluir absent, separar support/query sem frames compartilhados.',
              '5. Meta-treinar nos dados publicos; reservar os quatro videos para meta-avaliacao das pipelines A/B.',
              '', 'Esta auditoria nao modifica os dados de entrada, gera janelas ou executa treinamento.']
    (out / 'analysis.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f'Relatorio: {out / "analysis.md"}', flush=True)


if __name__ == '__main__':
    main()
