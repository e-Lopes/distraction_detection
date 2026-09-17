"""Extracao GPU com video ou --background; CPU somente apos confirmacao.

Os modelos sao preparados automaticamente em .gpu_runtime na primeira execucao.
GPU: YOLO11n-Pose + BlazeFace + FaceLandmarker 478 em PyTorch/CUDA.
CPU: referencia MediaPipe Solutions com attention mesh e tracking.
Os modelos diferem: nao misturar estatisticas ou reutilizar limiares sem validar.
Espaco pausa; Q/Esc encerra e salva. --require-gpu cancela se CUDA falhar.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import importlib
import importlib.util
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from fase_2.src.features.extract_facial_series import (
    LEFT_EYE, RIGHT_EYE, compute_ear, compute_mar, compute_head_pose,
)

RUNTIME = Path(__file__).resolve().parent / '.gpu_runtime'
REPOSITORY = 'https://github.com/yakhyo/mediapipe-face-mesh-onnx.git'
REVISION = 'add50e0f486405c96695812f9a9b9b89a485892b'
FACE_RELEASE = 'https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/'
ASSETS = {
    'face_detection_short_range.pt': FACE_RELEASE + 'face_detection_short_range.pt',
    'face_landmarker_256x256.pt': FACE_RELEASE + 'face_landmarker_256x256.pt',
    'yolo11n-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt',
}
EXPECTED_SHA256 = {
    'face_detection_short_range.pt': '7eb4d030f2458413a3382c10db7aaac46c4bc91725a12b531c94b557cb92bfe3',
    'face_landmarker_256x256.pt': 'fe4527aed53fe16c9b5825055787c707781022f7b4a76960dfafe7a5441f23ec',
    'yolo11n-pose.pt': '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0',
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_gpu_assets():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Instale PyTorch com CUDA compativel com sua GPU antes de continuar.')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    code = RUNTIME / 'face_mesh'
    if not code.exists():
        subprocess.run(['git', 'clone', REPOSITORY, str(code)], check=True)
    actual = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        dirty = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True)
        if dirty.strip():
            raise RuntimeError(f'Codigo externo modificado em {code}; preserve as alteracoes antes de atualizar.')
        subprocess.run(['git', '-C', str(code), 'fetch', 'origin', REVISION], check=True)
        subprocess.run(['git', '-C', str(code), 'checkout', '--detach', REVISION], check=True)
    weights = RUNTIME / 'weights'
    weights.mkdir(exist_ok=True)
    source_hashes = {str(path.relative_to(code)): sha256(path)
                     for path in sorted((code / 'models').glob('*.py'))}
    manifest = dict(repository=REPOSITORY, revision=REVISION, source_hashes=source_hashes, weights={})
    for name, url in ASSETS.items():
        path = weights / name
        if not path.exists():
            print(f'Baixando {name}...', flush=True)
            temporary = path.with_suffix('.download')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        digest = sha256(path)
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f'Hash do modelo difere da versao validada: {path}')
        manifest['weights'][name] = dict(url=url, sha256=digest)
    with (RUNTIME / 'manifest.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print(f'Preparado para {torch.cuda.get_device_name(0)}: {RUNTIME}', flush=True)



@dataclass
class Detection:
    pose_detected: bool
    landmarks: list | None
    pose_confidence: float = float('nan')
    face_confidence: float = float('nan')


class CpuBackend:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.metadata = dict(inference_device='cpu', inference_backend='mediapipe.solutions',
                             mediapipe_version=mp.__version__, pose_model='MediaPipe Pose full',
                             face_model='MediaPipe FaceMesh attention', tracking=True)

    def __enter__(self):
        self.pose = self.mp.solutions.pose.Pose(min_detection_confidence=.7, min_tracking_confidence=.5)
        try:
            self.face = self.mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                           min_detection_confidence=.5,
                                                           min_tracking_confidence=.5)
        except BaseException:
            self.pose.close()
            raise
        return self

    def __exit__(self, *args):
        self.pose.close()
        self.face.close()

    def detect(self, region):
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        pose = self.pose.process(rgb)
        face = self.face.process(rgb)
        return Detection(pose.pose_landmarks is not None,
                         face.multi_face_landmarks[0].landmark if face.multi_face_landmarks else None)


def _load_face_modules():
    # Isolate the upstream package instead of inserting a generic 'models' package.
    package = '_extraction_facemesh_models'
    root = RUNTIME / 'face_mesh/models'
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, root / '__init__.py',
                                                     submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return tuple(importlib.import_module(f'{package}.{name}') for name in ('model', 'blazeface', 'onnx_model'))


class TorchSession:
    """Reuse upstream crop/decoding helpers; execute networks with torch CUDA only."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cuda_calls = 0
        self.graph = None

    def capture(self, shape):
        """Fixed one-face batches avoid hundreds of Python/kernel launches per frame."""
        import torch
        self.static_input = torch.zeros(shape, dtype=torch.float32, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                self.model(self.static_input)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.static_output = self.model(self.static_input)
        torch.cuda.synchronize(self.device)

    def run(self, outputs, inputs):
        import torch
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(next(iter(inputs.values())))).to(self.device)
            if self.graph is None:
                result = self.model(batch)
            else:
                if batch.shape != self.static_input.shape:
                    raise ValueError(f'Batch inesperado: {batch.shape}; esperado {self.static_input.shape}')
                self.static_input.copy_(batch)
                self.graph.replay()
                result = self.static_output
            if not all(tensor.is_cuda for tensor in result):
                raise RuntimeError('Inferencia facial fora da GPU; nenhum fallback permitido.')
            self.cuda_calls += 1
            return [tensor.cpu().numpy() for tensor in result]


class CudaBackend:
    def __init__(self, device='cuda:0'):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA indisponivel. Use PyTorch CUDA ou --backend cpu explicitamente.')
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise ValueError(f'GPU inexistente: {device}')
        manifest_path = RUNTIME / 'manifest.json'
        if not manifest_path.exists():
            prepare_gpu_assets()
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['revision'] != REVISION:
            raise RuntimeError('Versao dos modelos diferente da esperada; execute novamente este visualizador.')
        for name, expected in EXPECTED_SHA256.items():
            if sha256(RUNTIME / 'weights' / name) != expected:
                raise RuntimeError(f'Peso alterado desde a preparacao: {name}')
        if not manifest.get('source_hashes'):
            raise RuntimeError('Manifesto antigo; execute novamente este visualizador.')
        for name, expected in manifest['source_hashes'].items():
            if sha256(RUNTIME / 'face_mesh' / name) != expected:
                raise RuntimeError(f'Codigo externo alterado desde a preparacao: {name}')
        nets, detector_helpers, mesh_helpers = _load_face_modules()
        def load(net_class, name):
            net = net_class()
            state = torch.load(RUNTIME / 'weights' / name, map_location='cpu', weights_only=True)
            net.load_state_dict(state)
            return net.eval().to(self.device)
        self.detector_net = load(nets.BlazeFaceNet, 'face_detection_short_range.pt')
        self.mesh_net = load(nets.FaceLandmarkerNet, 'face_landmarker_256x256.pt')

        # The upstream classes only depend on session.run for neural inference.
        class TorchDetector(detector_helpers.BlazeFace):
            def __init__(self, network, device):
                self.session = TorchSession(network, device)
                self.input_name = 'image'
                self.anchors = detector_helpers._generate_face_anchors()

        class TorchMesh(mesh_helpers.FaceMesh):
            def __init__(self, network, device):
                self.session = TorchSession(network, device)
                self.input_name = 'image'
                self.input_size = 256
                self.num_landmarks = 478

        self.detector = TorchDetector(self.detector_net, self.device)
        self.mesh = TorchMesh(self.mesh_net, self.device)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNTIME / 'ultralytics'))
        from ultralytics import YOLO, __version__ as yolo_version
        self.pose = YOLO(str(RUNTIME / 'weights/yolo11n-pose.pt'), task='pose')
        self.pose.to(self.device)
        self.pose_cuda_calls = 0

        def require_cuda(module, inputs):
            if not inputs[0].is_cuda:
                raise RuntimeError('YOLO recebeu entrada CPU; inferencia cancelada.')
            self.pose_cuda_calls += 1

        self.pose.model.register_forward_pre_hook(require_cuda)
        for network in (self.detector_net, self.mesh_net, self.pose.model):
            if not all(parameter.is_cuda for parameter in network.parameters()):
                raise RuntimeError('Ha pesos fora da GPU; inferencia cancelada.')
        # Exercise every neural model, even when the first video frame has no face.
        with torch.inference_mode():
            self.detector.session.run(None, {'image': np.zeros((1, 3, 128, 128), np.float32)})
            self.mesh.session.run(None, {'image': np.zeros((1, 3, 256, 256), np.float32)})
            self.pose.predict(np.zeros((491, 538, 3), np.uint8), device=str(self.device),
                              imgsz=640, conf=.7, max_det=1, verbose=False)
        torch.cuda.synchronize(self.device)
        self.detector.session.capture((1, 3, 128, 128))
        self.mesh.session.capture((1, 3, 256, 256))
        self.metadata = dict(inference_device=str(self.device), inference_backend='torch_cuda',
                             gpu_name=torch.cuda.get_device_name(self.device), torch_version=torch.__version__,
                             torch_cuda_version=torch.version.cuda, ultralytics_version=yolo_version,
                             pose_model='YOLO11n-pose', face_detector='BlazeFace short range',
                             face_model='FaceLandmarker 478 (not legacy attention)', tracking=False,
                             comparability_note='Different models and tracking from CPU reference; validate coverage and indicator thresholds before scientific use.',
                             pose_confidence_threshold=.7, face_detection_threshold=.5,
                             face_presence_threshold=.5, weights=manifest['weights'],
                             source_revision=REVISION,
                             source_hashes=manifest['source_hashes'],
                             facial_execution='CUDA graphs, float32, batch=1',
                             cpu_stages=['video decoding', 'crop/resize', 'postprocessing',
                                         'EAR/MAR/solvePnP', 'plots', 'CSV/statistics'],
                             cuda_warmup_verified=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def detect(self, region):
        result = self.pose.predict(region, device=str(self.device), imgsz=640, conf=.7,
                                   max_det=1, verbose=False)[0]
        pose_ok = bool(result.boxes is not None and len(result.boxes) > 0 and
                       result.keypoints is not None and len(result.keypoints.data) > 0)
        confidence = float(result.boxes.conf[0].item()) if pose_ok else float('nan')
        boxes, keypoints, scores = self.detector.detect(region, threshold=.5)
        landmarks = None
        face_confidence = float('nan')
        if len(boxes):
            # Match the old single-face contract: highest detector confidence in ROI.
            points, presence = self.mesh.predict(region, boxes[:1], margin=.25, keypoints=keypoints[:1])
            face_confidence = float(presence[0])
            if face_confidence >= .5 and np.isfinite(points[0]).all():
                h, w = region.shape[:2]
                landmarks = [SimpleNamespace(x=float(p[0]/w), y=float(p[1]/h), z=float(p[2]/w))
                             for p in points[0]]
        return Detection(pose_ok, landmarks, confidence, face_confidence)

    def audit(self):
        return dict(pose_cuda_calls=self.pose_cuda_calls,
                    face_detector_cuda_calls=self.detector.session.cuda_calls,
                    face_landmarker_cuda_calls=self.mesh.session.cuda_calls,
                    warmup_included=True)




def select_backend(gpu_index=0, require_gpu=False):
    print(f'Inicializando GPU cuda:{gpu_index}...', flush=True)
    try:
        return CudaBackend(f'cuda:{gpu_index}')
    except Exception as error:
        print(f'Nao foi possivel iniciar a extracao em GPU: {type(error).__name__}: {error}', flush=True)
        if require_gpu:
            raise RuntimeError('GPU obrigatoria; execucao encerrada sem usar CPU.') from error
        while True:
            try:
                answer = input('Deseja continuar a extracao em CPU? [s/N]: ').strip().lower()
            except EOFError:
                raise RuntimeError('Sem resposta interativa; extracao cancelada. CPU nao autorizada.') from error
            if answer in ('s', 'sim', 'y', 'yes'):
                backend = CpuBackend()
                backend.metadata['cpu_fallback_authorized'] = True
                backend.metadata['gpu_initialization_error'] = f'{type(error).__name__}: {error}'
                return backend
            if answer in ('', 'n', 'nao', 'não', 'no'):
                raise SystemExit('Extracao cancelada pelo usuario.')
            print('Responda s para continuar em CPU ou n para encerrar.', flush=True)

SIGNALS = ('ear', 'mar', 'pitch', 'yaw', 'roll')
WINDOW = 'Video e indicadores'


def format_duration(seconds):
    if seconds is None:
        return 'calculando'
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self, total_frames):
        self.total_frames = total_frames
        self.frames = 0
        self.started = time.perf_counter()
        self.last_report = self.started

    def update(self, video_id, video_frames, video_total, force=False):
        now = time.perf_counter()
        if not force and now - self.last_report < 10:
            return
        elapsed = now - self.started
        rate = self.frames / elapsed if elapsed > 0 else 0
        ready = self.frames >= 30 and rate > 0
        remaining = max(0, self.total_frames - self.frames)
        eta = remaining / rate if ready else None
        video_eta = max(0, video_total - video_frames) / rate if ready else None
        percent = 100 * self.frames / self.total_frames if self.total_frames else 0
        print(f'[{video_id}] {video_frames}/{video_total} frames | '
              f'Total: {percent:.1f}% ({self.frames}/{self.total_frames}) | '
              f'{rate:.1f} FPS | Decorrido: {format_duration(elapsed)} | '
              f'Restante video: {format_duration(video_eta)} | '
              f'Restante lote: {format_duration(eta)} | '
              f'Duracao total prevista: {format_duration(elapsed + eta if eta is not None else None)}',
              flush=True)
        self.last_report = now

    def exclude_pause(self, seconds):
        self.started += seconds
        self.last_report += seconds
DIAGNOSTIC_FIELDS = (
    'face_mesh_detected', 'indicators_accepted', 'all_indicators_valid',
    'detection_status', 'ear_valid', 'mar_valid', 'head_pose_valid',
    'ear_status', 'mar_status', 'head_pose_status',
)


def measure_frame(pose_ok, landmarks, width, height, reference=None):
    """Validity means successful finite computation, not anatomical accuracy."""
    face_mesh_ok = landmarks is not None
    accepted = bool(pose_ok and face_mesh_ok)
    detection_status = ('both_detected' if accepted else
                        'pose_missing' if face_mesh_ok else
                        'face_mesh_missing' if pose_ok else 'both_missing')
    diagnostics = dict(face_mesh_detected=int(face_mesh_ok), indicators_accepted=int(accepted),
                       all_indicators_valid=0, detection_status=detection_status)
    values = [np.nan] * 5
    raw_angles = [np.nan] * 3
    selected_eye = None
    for name in ('ear', 'mar', 'head_pose'):
        diagnostics[f'{name}_valid'] = 0
        diagnostics[f'{name}_status'] = 'not_evaluated_' + detection_status
    if accepted:
        for name in ('ear', 'mar', 'head_pose'):
            try:
                if name == 'ear':
                    selected_eye = image_left_eye(landmarks)
                    computed = [compute_ear(landmarks, selected_eye, width, height)]
                elif name == 'mar':
                    computed = [compute_mar(landmarks, width, height)]
                else:
                    computed = list(compute_head_pose(landmarks, width, height))
                finite = bool(np.isfinite(computed).all())
                diagnostics[f'{name}_valid'] = int(finite)
                diagnostics[f'{name}_status'] = 'ok' if finite else 'nonfinite_result'
                if finite:
                    if name == 'head_pose':
                        raw_angles = computed
                        values[2:] = computed if reference is None else relative_angles(computed, reference)
                    else:
                        values[0 if name == 'ear' else 1] = computed[0]
            except (RuntimeError, cv2.error, ValueError, IndexError, FloatingPointError) as error:
                diagnostics[f'{name}_status'] = 'calculation_error:' + type(error).__name__
    diagnostics['all_indicators_valid'] = int(all(diagnostics[f'{name}_valid']
                                                  for name in ('ear', 'mar', 'head_pose')))
    return values, raw_angles, selected_eye, diagnostics


def save_detection_diagnostics(rows, output, video_id, complete):
    from collections import Counter
    suffix = '' if complete else '_parcial'
    path = output / f'{video_id}{suffix}_diagnostico_deteccao.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['video_id', 'complete', 'field', 'value', 'frames', 'fraction'])
        writer.writeheader()
        for field in ('pose_detected', *DIAGNOSTIC_FIELDS):
            for value, count in sorted(Counter(row[field] for row in rows).items()):
                writer.writerow(dict(video_id=video_id, complete=complete, field=field,
                                     value=value, frames=count, fraction=count/len(rows)))
    print(f'Diagnostico: {path}', flush=True)


def image_left_eye(landmarks):
    """Select by image position, not anatomical left/right naming."""
    return min((LEFT_EYE, RIGHT_EYE),
               key=lambda indices: np.mean([landmarks[i].x for i in indices]))


def relative_angles(angles, reference):
    # Circular component differences; not a calibrated anatomical rotation.
    return (np.asarray(angles) - np.asarray(reference) + 180) % 360 - 180


def save_statistics(rows, output, video_id, complete, source_fps, processing_seconds):
    """Mode uses fixed bins; missing detections never contribute to summaries."""
    summaries = []
    valid_rows = [row for row in rows if row['face_detected']]
    for name in (*SIGNALS, 'pitch_raw', 'yaw_raw', 'roll_raw'):
        values = np.asarray([row[name] for row in valid_rows], dtype=float)
        values = values[np.isfinite(values)]
        step = .01 if name in ('ear', 'mar') else 1.0
        summary = dict(video_id=video_id, complete=complete, indicator=name,
                       total_frames=len(rows), valid_samples=len(values),
                       face_detection_ratio=len(valid_rows)/len(rows) if rows else 0,
                       source_fps=source_fps, processing_seconds=processing_seconds,
                       processing_fps=len(rows)/processing_seconds if processing_seconds else 0,
                       mean=None, median=None, std=None, minimum=None, p05=None,
                       p95=None, maximum=None, mode_bin_width=step,
                       mode_bin_centers='[]', mode_count=0, mode_fraction=None)
        if len(values):
            centers, counts = np.unique(np.floor(values/step + .5).astype(np.int64), return_counts=True)
            modes = (centers[counts == counts.max()] * step).tolist()
            summary.update(mean=float(values.mean()), median=float(np.median(values)),
                           std=float(values.std()), minimum=float(values.min()),
                           maximum=float(values.max()), p05=float(np.percentile(values, 5)),
                           p95=float(np.percentile(values, 95)), mode_bin_centers=json.dumps(modes),
                           mode_count=int(counts.max()), mode_fraction=float(counts.max()/len(values)))
        summaries.append(summary)
    suffix = '' if complete else '_parcial'
    path = output / f'{video_id}{suffix}_estatisticas.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'Estatisticas: {path}', flush=True)


class Charts:
    def __init__(self, seconds):
        self.seconds = seconds
        self.samples = deque()
        self.last_draw = float('-inf')
        self.cached_image = None
        self.figure = Figure(figsize=(6.4, 8), dpi=100)
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(5, 1, sharex=True)
        self.lines = []
        for ax, name, color in zip(self.axes, SIGNALS,
                                   ('#1676b8', '#258b45', '#b34040', '#7954a3', '#a86c10')):
            ax.set_ylabel(name.upper())
            ax.grid(alpha=.25)
            self.lines.append(ax.plot([], [], color=color, linewidth=1)[0])
        self.axes[-1].set_xlabel('Tempo do video (s)')
        self.figure.subplots_adjust(left=.16, right=.97, top=.97, bottom=.07, hspace=.30)

    def update(self, timestamp, values, graph_fps=5):
        self.samples.append((timestamp, *values))
        while self.samples and self.samples[0][0] < timestamp - self.seconds:
            self.samples.popleft()
        now = time.perf_counter()
        if self.cached_image is not None and now - self.last_draw < 1 / graph_fps:
            return self.cached_image
        data = np.asarray(self.samples)
        for i, (ax, line) in enumerate(zip(self.axes, self.lines)):
            y = data[:, i + 1].copy()
            # Break angular wrap jumps instead of drawing misleading vertical lines.
            if i >= 2 and len(y) > 1:
                y[1:][np.abs(np.diff(y)) > 180] = np.nan
            line.set_data(data[:, 0], y)
            finite = y[np.isfinite(y)]
            if i < 2:
                ax.set_ylim(0, max(.6, float(finite.max()) * 1.1) if len(finite) else .6)
            else:
                ax.set_ylim(-90 if i == 2 else -180, 90 if i == 2 else 180)
        self.axes[-1].set_xlim(max(0, timestamp - self.seconds), max(self.seconds, timestamp))
        self.canvas.draw()
        self.cached_image = cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        self.last_draw = time.perf_counter()
        return self.cached_image


def process_video(path, video_id, roi, output, seconds, headless=False, max_frames=None,
                  angle_reference=None, graph_fps=5, fast=False, progress=None, backend=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f'Nao foi possivel abrir: {path}')
    rows = []
    complete = False
    stop = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    charts = None if headless else Charts(seconds)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(total, max_frames) if max_frames is not None else total
    processing_seconds = 0.0
    try:
        with (backend if backend is not None else select_backend()) as detector:
            while True:
                started = time.perf_counter()
                if max_frames is not None and len(rows) >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    complete = True
                    break
                index = len(rows)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not np.isfinite(timestamp) or (rows and timestamp <= rows[-1]['timestamp_seconds']):
                    timestamp = max(index / fps, rows[-1]['timestamp_seconds'] + 1 / fps) if rows else 0
                x, y, w, h = roi
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    raise ValueError(f'ROI {roi} invalida para {frame.shape[1]}x{frame.shape[0]}')
                region = frame[y:y+h, x:x+w]
                detection = detector.detect(region)
                pose_ok, lm = detection.pose_detected, detection.landmarks
                values, raw_angles, selected_eye, diagnostics = measure_frame(pose_ok, lm, w, h, angle_reference)
                face_ok = bool(diagnostics['indicators_accepted'])
                eye_name = ('' if selected_eye is None else
                            'LEFT_EYE' if selected_eye is LEFT_EYE else 'RIGHT_EYE')
                if face_ok:
                    if not headless:
                        for point in lm:
                            cv2.circle(region, (int(point.x*w), int(point.y*h)), 1, (60, 230, 80), -1)
                        for eye_index in selected_eye or []:
                            point = lm[eye_index]
                            cv2.circle(region, (int(point.x*w), int(point.y*h)), 3, (0, 230, 255), -1)
                rows.append(dict(video_id=video_id, frame_index=index, timestamp_seconds=timestamp,
                                 **dict(zip(SIGNALS, values)), face_detected=int(face_ok),
                                 pose_detected=int(pose_ok), ear_eye=eye_name,
                                 pose_confidence=detection.pose_confidence,
                                 face_presence_confidence=detection.face_confidence,
                                 **diagnostics,
                                 **dict(zip(('pitch_raw', 'yaw_raw', 'roll_raw'), raw_angles))))
                if index % 300 == 0:
                    metrics = ' | '.join(f'{name}={value:.3f}' for name, value in zip(SIGNALS, values))
                    print(f'[{video_id}] {index+1}/{total} frames | {timestamp:.1f}s | '
                          f'{diagnostics["detection_status"]} | {metrics}', flush=True)
                if headless:
                    processing_seconds += time.perf_counter() - started
                    if progress is not None:
                        progress.frames += 1
                        progress.update(video_id, len(rows), target)
                    continue
                graph = charts.update(timestamp, values, graph_fps)
                panel = np.full((800, 800, 3), 24, dtype=np.uint8)
                scale = min(800/w, 510/h)
                preview = cv2.resize(region, None, fx=scale, fy=scale)
                ph, pw = preview.shape[:2]
                panel[70:70+ph, (800-pw)//2:(800+pw)//2] = preview
                cv2.putText(panel, f'{video_id} | {timestamp:.1f}s | frame {index}', (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2)
                for i, (name, value) in enumerate(zip(SIGNALS, values)):
                    label = f'{name.upper()}: {value:.3f}' if np.isfinite(value) else f'{name.upper()}: N/D'
                    cv2.putText(panel, label, (25, 610+i*32), cv2.FONT_HERSHEY_SIMPLEX,
                                .65, (230, 230, 230), 1)
                display = np.hstack((panel, graph))
                cv2.imshow(WINDOW, display)
                elapsed = time.perf_counter() - started
                processing_seconds += elapsed
                delay = 1 if fast else max(1, int(1000 * (1/fps - elapsed)))
                key = cv2.waitKey(delay) & 0xff
                if key == 32:
                    pause_started = time.perf_counter()
                    while True:
                        key = cv2.waitKey(100) & 0xff
                        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            key = 27
                        if key in (32, 27, ord('q')):
                            break
                    if progress is not None:
                        progress.exclude_pause(time.perf_counter() - pause_started)
                if progress is not None:
                    progress.frames += 1
                    progress.update(video_id, len(rows), target)
                if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    stop = True
                    break
    finally:
        cap.release()
        if charts is not None:
            charts.figure.clear()
        suffix = '' if complete else '_parcial'
        destination = output / f'{video_id}{suffix}.csv'
        with destination.open('w', newline='', encoding='utf-8') as stream:
            fields = ['video_id', 'frame_index', 'timestamp_seconds', *SIGNALS, 'face_detected',
                      'pose_detected', 'ear_eye', 'pitch_raw', 'yaw_raw', 'roll_raw',
                      'pose_confidence', 'face_presence_confidence', *DIAGNOSTIC_FIELDS]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Salvo: {destination} ({len(rows)} frames)', flush=True)
        save_statistics(rows, output, video_id, complete, fps, processing_seconds)
        save_detection_diagnostics(rows, output, video_id, complete)
        if backend is not None and hasattr(backend, 'audit'):
            with (output / 'gpu_audit.json').open('w', encoding='utf-8') as stream:
                json.dump(backend.audit(), stream, indent=2)
        if progress is not None:
            progress.update(video_id, len(rows), target, force=True)
    return stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos-dir', type=Path, default=BASE.parent / 'fase_2/data/raw')
    parser.add_argument('--roi-config', type=Path, default=BASE / 'roi_config.json')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'resultados_legados/extracao_ao_vivo')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--graph-fps', type=float, default=5,
                        help='Redesenhos dos graficos por segundo; extracao continua em todos os frames')
    parser.add_argument('--fast', action='store_true', help='Reproduzir sem limitar ao FPS original')
    parser.add_argument('--gpu-index', type=int, default=0, help='Indice da GPU CUDA (padrao: 0)')
    parser.add_argument('--require-gpu', action='store_true',
                        help='Interromper se a inferencia nao puder rodar em GPU')
    parser.add_argument('--headless', '--background', action='store_true',
                        help='Somente terminal, sem renderizacao ou espera de reproducao')
    parser.add_argument('--angle-reference', type=float, nargs=3, metavar=('PITCH', 'YAW', 'ROLL'),
                        help='Referencia neutra na convencao historica; exibe diferencas angulares em graus')
    parser.add_argument('--max-frames', type=int, help='Limite de frames por video para verificacao')
    args = parser.parse_args()
    if args.gpu_index < 0:
        parser.error('gpu-index deve ser nao negativo')
    if not np.isfinite(args.graph_fps) or args.graph_fps <= 0:
        parser.error('graph-fps deve ser finito e positivo')
    if args.seconds <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error('seconds e max-frames devem ser positivos')
    if args.angle_reference is not None and not np.isfinite(args.angle_reference).all():
        parser.error('Referencia angular deve conter valores finitos')
    with args.roi_config.open(encoding='utf-8') as stream:
        roi = tuple(json.load(stream)['roi_cadeira'])
    paths = [args.videos_dir / f'{i}.mp4' for i in range(1, 5)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    backend = select_backend(args.gpu_index, args.require_gpu)
    print(f'Inferencia: {backend.metadata["inference_backend"]} em '
          f'{backend.metadata["inference_device"]}. '
          'Leitura, indicadores e gravacao em CPU.', flush=True)
    if backend.metadata['inference_backend'] == 'torch_cuda':
        print('Modelos GPU diferentes da referencia CPU: valide cobertura e limiares dos indicadores.',
              flush=True)
    total_frames = 0
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f'Nao foi possivel abrir {path}')
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError(f'Contagem de frames indisponivel: {path}')
            total_frames += min(count, args.max_frames) if args.max_frames is not None else count
        finally:
            capture.release()
    output = args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    with (output / 'config.json').open('w', encoding='utf-8') as stream:
        json.dump(dict(roi=roi, videos=[str(p) for p in paths],
                       **backend.metadata, opencv_version=cv2.__version__,
                       schema_version=3, acceptance_rule='pose_detected AND face_mesh_detected',
                       face_detected_semantics='Legacy alias of indicators_accepted',
                       validity_semantics='Finite successful calculation, not a visibility or anatomical quality score; not evaluated when acceptance fails',
                       ear_selection='minimum_mean_image_x', angle_reference=args.angle_reference,
                       angle_convention='historical: pitch=Y, yaw=Z, roll=X',
                       headless=args.headless, graph_fps=args.graph_fps, fast=args.fast,
                       statistics_mode_bin_degrees=1.0, statistics_mode_bin_ratio=.01,
                       statistics_note='Frame-weighted; finite detected samples only. Linear angle summaries; modes are not neutral-pose calibration.'), stream, indent=2)
    controls = 'Ctrl+C: salvar parcial e encerrar.' if args.headless else 'Espaco: pausa. Q/Esc: encerrar.'
    print(f'Resultados: {output}\n{controls}', flush=True)
    print(f'Lote: 4 videos, {total_frames} frames. Previsao atualizada a cada 10 segundos; '
          'aguardando pelo menos 30 frames para estimar. Pausas manuais excluidas.', flush=True)
    progress = Progress(total_frames)
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1440, 800)
        for i, path in enumerate(paths, 1):
            if process_video(path, f'video_{i:02d}', roi, output, args.seconds,
                             args.headless, args.max_frames, args.angle_reference,
                             args.graph_fps, args.fast, progress, backend):
                break
    except KeyboardInterrupt:
        print('Extracao interrompida; dados parciais salvos.', flush=True)
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
