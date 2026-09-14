"""Auditable raw measurements, explicit operator lock, original-image geometry."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .extract_facial_series import (LEFT_EYE, RIGHT_EYE, MOUTH_TOP, MOUTH_BOTTOM,
    MOUTH_LR, HEAD_POSE_IDS, HEAD_POSE_3D, compute_ear, compute_mar, compute_head_pose,
    validate_roi, sha256_file)
from ..preprocessing.measurement_quality import legacy_angles, region_valid


def letterbox(frame, roi, target):
    x, y, w, h = roi
    validate_roi(tuple(roi), frame.shape[1], frame.shape[0])
    ratio = min(target[0]/w, target[1]/h)
    rw, rh = round(w*ratio), round(h*ratio)
    px, py = (target[0]-rw)//2, (target[1]-rh)//2
    canvas = np.zeros((target[1], target[0], 3), np.uint8)
    canvas[py:py+rh, px:px+rw] = cv2.resize(frame[y:y+h, x:x+w], (rw, rh))
    return canvas, {'roi': list(roi), 'scale_x': rw/w, 'scale_y': rh/h,
                    'padding': [px, py], 'target': list(target)}


def original_points(landmarks, geometry):
    x, y, _, _ = geometry['roi']
    px, py = geometry['padding']
    tw, th = geometry['target']
    return np.array([[(p.x*tw-px)/geometry['scale_x']+x,
                      (p.y*th-py)/geometry['scale_y']+y] for p in landmarks])


def bbox(points):
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    return np.r_[low, high-low]


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])).prod()
    return intersection/max(a[2:].prod()+b[2:].prod()-intersection, 1e-9)


class OperatorLock:
    """Conservative spatial continuity, NOT biometric identity recognition.

    Requires manually verified anchor. After ambiguity/loss, no automatic reacquisition.
    """
    def __init__(self, anchor, minimum_iou):
        self.box = anchor
        self.threshold = minimum_iou
        self.lost = False

    def select(self, candidates):
        if self.lost: return None, 'lost_requires_review'
        if self.box is None: return None, 'unverified_operator'
        # Reject multiple detections even if only one overlaps: no silent switching.
        if len(candidates) != 1 or iou(self.box, bbox(candidates[0])) < self.threshold:
            self.lost = True
            return None, 'ambiguous_or_lost'
        self.box = bbox(candidates[0])
        return candidates[0], 'tracked'


def measure(points, frame, roi, criteria):
    height, width = frame.shape[:2]
    normalized = [SimpleNamespace(x=p[0]/width, y=p[1]/height) for p in points]
    out = {'ear_left': compute_ear(normalized, LEFT_EYE, width, height),
           'ear_right': compute_ear(normalized, RIGHT_EYE, width, height),
           'mar': compute_mar(normalized, width, height)}
    out['ear'] = (out['ear_left']+out['ear_right'])/2
    for name, ids in (('left', LEFT_EYE), ('right', RIGHT_EYE)):
        valid, reason = region_valid(points[ids], roi, criteria['min_eye_width_px'])
        out['valid_'+name], out['reason_'+name] = int(valid), reason
        out['eye_width_'+name] = float(np.linalg.norm(points[ids[0]]-points[ids[3]]))
    mouth_ids = MOUTH_LR + MOUTH_TOP + MOUTH_BOTTOM
    mouth = points[mouth_ids]
    # region_valid uses indices 0 and 3 as horizontal anchors.
    mouth_check = np.vstack([mouth[0], mouth[2], mouth[3], mouth[1], mouth[4:]])
    valid, reason = region_valid(mouth_check, roi, criteria['min_mouth_width_px'])
    out['valid_mouth'], out['reason_mouth'] = int(valid), reason
    x, y, w, h = roi
    gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    out['blur_laplacian_roi'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    out.update(pitch=np.nan, yaw=np.nan, roll=np.nan, rotation_matrix='[]', translation='[]',
               reprojection_rmse_px=np.nan, valid_pose=0, reason_pose='pnp_failed')
    k = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], float)
    out['camera_matrix'] = json.dumps(k.tolist())
    try:
        success, rv, tv = cv2.solvePnP(HEAD_POSE_3D, points[HEAD_POSE_IDS].astype(float), k,
                                     np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            rotation = cv2.Rodrigues(rv)[0]
            projected = cv2.projectPoints(HEAD_POSE_3D, rv, tv, k, np.zeros(4))[0].reshape(-1, 2)
            error = float(np.sqrt(np.mean(np.sum((projected-points[HEAD_POSE_IDS])**2, axis=1))))
            out.update(zip(('pitch', 'yaw', 'roll'), legacy_angles(rotation)))
            out.update(rotation_matrix=json.dumps(rotation.tolist()), translation=json.dumps(tv.ravel().tolist()), reprojection_rmse_px=error)
            inside, why = region_valid(points[HEAD_POSE_IDS], roi, 0)
            valid = inside and np.isfinite(error) and error <= criteria['max_pose_rmse_px'] and tv[2, 0] > 0
            out.update(valid_pose=int(valid), reason_pose='' if valid else (why or 'reprojection_or_depth'))
    except cv2.error:
        pass
    # Paired reference reproduces the old crop-centred approximate intrinsics.
    old = [SimpleNamespace(x=(p[0]-x)/w, y=(p[1]-y)/h) for p in points]
    try:
        baseline_pose = compute_head_pose(old, w, h)
    except (RuntimeError, cv2.error):
        baseline_pose = [np.nan]*3
    out.update(zip(('legacy_pitch', 'legacy_yaw', 'legacy_roll'), baseline_pose))
    out['landmarks_px'] = json.dumps({str(i): points[i].tolist() for i in sorted(set(LEFT_EYE+RIGHT_EYE+mouth_ids+HEAD_POSE_IDS))})
    return out


def extract(config, *, video, start_frame=0, max_frames=90, full=False, output=None):
    import mediapipe as mp
    settings = config['measurement_protocol']['extraction']
    if start_frame < 0 or max_frames <= 0 or (full and start_frame != 0):
        raise ValueError('Use nonnegative start, positive sample size; full extraction starts at zero')
    if not full:
        from ..pipeline import _read_csv
        stop = start_frame + min(max_frames, settings['max_sample_frames']) - 1
        allowed = [b for b in _read_csv(Path(config['splits']['manifest']))
                   if int(b['fold']) == config['measurement_protocol']['development_fold']
                   and b['subset'] == 'train' and b['video_id'] == video
                   and int(b['start_frame']) <= start_frame <= stop <= int(b['end_frame'])]
        if not allowed: raise ValueError('Sample must lie entirely inside preregistered development training block')
    entry = settings['videos'][video]
    roi = entry['roi']
    anchors = {int(k): v for k, v in entry.get('verified_anchors', {}).items()}
    if full and 0 not in anchors:
        raise ValueError('Full extraction requires a manually verified frame-0 operator anchor')
    root = Path(output or (config['data']['facial_series'] if full else settings['sample_output']))
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'{video}.csv'
    if target.exists(): raise FileExistsError(f'Raw output exists; use a new output directory: {target}')
    cap = cv2.VideoCapture(entry['path'])
    if not cap.isOpened(): raise ValueError(f"Cannot open {entry['path']}")
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if start_frame >= source_count:
        cap.release()
        raise ValueError('Sample starts after end of video')
    tracker = OperatorLock(anchors.get(start_frame), settings['minimum_tracking_iou'])
    temporary = target.with_suffix('.csv.tmp')
    if temporary.exists():
        cap.release()
        raise FileExistsError(f'Partial raw output preserved: {temporary}; use a new directory or review it manually')
    started = time.perf_counter()
    records, previews = [], []
    processed = 0
    previous_time = -np.inf
    count = source_count-start_frame if full else min(max_frames, settings['max_sample_frames'])
    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3,
                refine_landmarks=True, min_detection_confidence=settings['min_detection_confidence']) as mesh, temporary.open('w', newline='') as stream:
            writer = None
            for offset in range(count):
                ok, frame = cap.read()
                if not ok: break
                index = start_frame+offset
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
                if not np.isfinite(timestamp) or timestamp <= previous_time:
                    raise ValueError('Decoder timestamps unavailable/nonmonotonic; use a PTS-capable decoder, never silently infer FPS timestamps')
                previous_time = timestamp
                image, geometry = letterbox(frame, roi, settings['resize'])
                result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                candidates = [original_points(f.landmark, geometry) for f in (result.multi_face_landmarks or [])]
                if index in anchors: tracker = OperatorLock(anchors[index], settings['minimum_tracking_iou'])
                points, status = tracker.select(candidates)
                # Unverified samples retain candidates for geometry inspection, never training-valid.
                diagnostic = points if points is not None else candidates[0] if len(candidates) == 1 else None
                if diagnostic is not None:
                    measurements = measure(diagnostic, frame, roi, config['measurement_protocol']['quality'])
                else:
                    # Fixed schema without invoking solvePnP on unavailable measurements.
                    measurements = {key: np.nan for key in ('ear','ear_left','ear_right','mar','pitch','yaw','roll',
                        'eye_width_left','eye_width_right','blur_laplacian_roi','reprojection_rmse_px','legacy_pitch','legacy_yaw','legacy_roll')}
                    measurements.update({f'valid_{k}': 0 for k in ('left','right','mouth','pose')})
                    measurements.update({f'reason_{k}': 'no_unambiguous_candidate' for k in ('left','right','mouth','pose')})
                    measurements.update(rotation_matrix='[]',translation='[]',camera_matrix='[]',landmarks_px='{}')
                row = dict(schema_version='measurement_v1', video_id=video, frame_index=index,
                    timestamp_seconds=timestamp, timestamp_source='opencv_decoder_pos_msec',
                    face_detected=int(points is not None), candidate_count=len(candidates),
                    tracking_status=status, track_id=f'{video}:{max((k for k in anchors if k<=index), default=-1)}',
                    bbox=json.dumps(bbox(diagnostic).tolist()) if diagnostic is not None else '[]',
                    geometry=json.dumps(geometry), extractor_version=mp.__version__, official_landmark_confidence='unavailable',
                    **measurements)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                processed += 1
                if not full: records.append(row)
                if offset in (0, count//2, count-1):
                    previews.append((frame.copy(), row))
                if (offset+1) % settings['progress_every'] == 0:
                    print(f'[RAW] {video} {offset+1}/{count} tracked={status}', flush=True)
        if processed == 0: raise ValueError('No decodable frames; raw output not promoted')
        temporary.replace(target)
    finally:
        cap.release()
    elapsed = time.perf_counter()-started
    info = dict(complete=full and processed == source_count, processed_frames=processed,
                start_frame=start_frame, seconds=elapsed, seconds_per_frame=elapsed/processed,
                source_num_frames=source_count, source_path=entry['path'],
                source_size_bytes=Path(entry['path']).stat().st_size,
                source_sha256=sha256_file(Path(entry['path'])) if full else 'not_hashed_for_bounded_sample',
                raw_sha256=sha256_file(target), extraction_settings=settings,
                extraction_code_sha256=sha256_file(Path(__file__)),
                definitions_code_sha256=sha256_file(Path(__file__).with_name('extract_facial_series.py')),
                quality=config['measurement_protocol']['quality'], opencv=cv2.__version__, mediapipe=mp.__version__)
    target.with_suffix('.json').write_text(json.dumps(info, indent=2))
    if full:
        from ..pipeline import _read_csv, _write_csv_atomic
        manifest = root/'extraction_manifest.csv'
        entries = [r for r in _read_csv(manifest) if r['video_id'] != video]
        entries.append(dict(video_id=video,output_sha256=info['raw_sha256'],source_sha256=info['source_sha256'],
                            processed_frames=info['processed_frames'],complete=info['complete'],schema='measurement_v1'))
        _write_csv_atomic(manifest,entries)
    if records:
        from ..measurement_reporting import sample_figure
        sample_figure(records, previews, root / f'{video}_audit.png')
    print(f'[RAW SAVED] {target}; {elapsed:.2f}s; complete={info["complete"]}', flush=True)
    return info
