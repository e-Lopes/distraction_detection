"""Versioned per-indicator treatment. No labels or cross-block state are accepted."""
from __future__ import annotations

import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SIGNALS = ('ear', 'mar', 'pitch', 'yaw', 'roll')
FEATURES = SIGNALS + tuple('valid_' + s for s in SIGNALS) + tuple('interpolated_' + s for s in SIGNALS)


def number(value):
    return float(value) if value not in (None, '') else float('nan')


def legacy_angles(matrix):
    """Historical columns: pitch=Y, yaw=Z, roll=X in Rz Ry Rx, degrees."""
    r = np.asarray(matrix)
    sy = np.hypot(r[0, 0], r[1, 0])
    return np.degrees([np.arctan2(-r[2, 0], sy),
                       np.arctan2(r[1, 0], r[0, 0]) if sy > 1e-6 else 0,
                       np.arctan2(r[2, 1], r[2, 2]) if sy > 1e-6 else np.arctan2(-r[1, 2], r[1, 1])])


def rotation_from_angles(angles):
    p, y, r = angles
    return Rotation.from_euler('xyz', [r, p, y], degrees=True)


def relative_rotation(matrix, reference):
    r0 = np.asarray(reference, dtype=float)
    if r0.shape != (3, 3) or not np.allclose(r0.T @ r0, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r0), 1):
        raise ValueError('Operational reference must be a proper rotation matrix')
    return r0.T @ np.asarray(matrix)


def region_valid(points, bounds, min_width_px):
    points = np.asarray(points, dtype=float)
    x, y, w, h = bounds
    if not np.isfinite(points).all():
        return False, 'nonfinite_landmark'
    if np.any((points[:, 0] < x) | (points[:, 0] >= x+w) | (points[:, 1] < y) | (points[:, 1] >= y+h)):
        return False, 'outside_crop'
    if np.linalg.norm(points[0] - points[3]) < min_width_px:
        return False, 'insufficient_horizontal_pixels'
    return True, ''


def gaps(valid):
    edges = np.diff(np.r_[True, valid, True].astype(int))
    return zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1))


def treat_block(rows, policy):
    """Return copies; timestamp-bounded, per-indicator offline interpolation.

    A gap duration is the interval between observed anchors (conservative).
    No repair spans an identity loss, even if both endpoints have detections.
    """
    if not rows:
        return []
    t = np.array([number(r['timestamp_seconds']) for r in rows])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Strictly increasing real timestamps required within each block')
    if len({r['video_id'] for r in rows}) != 1:
        raise ValueError('Treatment cannot cross sessions')
    if any(int(b['frame_index']) != int(a['frame_index']) + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError('Treatment cannot cross discontinuous frame indices')
    output = [dict(r) for r in rows]
    values = np.full((len(rows), 5), np.nan)
    rejected = np.zeros((len(rows), 5), bool)
    quality = policy.get('reject_quality', True)
    eye_policy = policy.get('eye', 'mean')
    if eye_policy not in ('mean', 'left', 'right', 'weighted'):
        raise ValueError('Explicit eye policy required')
    for i, row in enumerate(rows):
        if row.get('schema_version') != 'measurement_v1':
            raise ValueError('Raw per-eye measurement_v1 required; historical five-number CSV is insufficient')
        tracked = row.get('tracking_status') == 'tracked'
        left, right = number(row['ear_left']), number(row['ear_right'])
        vl = np.isfinite(left) and (not quality or row['valid_left'] == '1')
        vr = np.isfinite(right) and (not quality or row['valid_right'] == '1')
        origin = 'missing'
        if eye_policy in ('left', 'right'):
            v, ok = (left, vl) if eye_policy == 'left' else (right, vr)
            ear = v if ok else np.nan
            if ok: origin = eye_policy
        elif eye_policy == 'mean':
            ear = np.mean([v for v, ok in ((left, vl), (right, vr)) if ok]) if vl or vr else np.nan
            origin = 'both' if vl and vr else 'left' if vl else 'right' if vr else 'missing'
        else:
            weights = np.array([number(row.get('eye_width_left')), number(row.get('eye_width_right'))])
            weights[~np.array([vl, vr])] = 0
            ear = np.nansum(np.array([left, right]) * weights) / weights.sum() if weights.sum() > 0 else np.nan
            origin = ('weighted_both' if vl and vr else 'left' if vl else 'right') if np.isfinite(ear) else 'missing'
        original = [ear] + [number(row[s]) for s in SIGNALS[1:]]
        flags = [np.isfinite(ear), not quality or row['valid_mouth'] == '1'] + [not quality or row['valid_pose'] == '1']*3
        for j, (v, valid) in enumerate(zip(original, flags)):
            rejected[i, j] = np.isfinite(v) and not valid
            if tracked and valid: values[i, j] = v
        output[i]['ear_source'] = origin if tracked else 'missing'
        rejected[i, 0] = quality and ((np.isfinite(left) and not vl) or (np.isfinite(right) and not vr))
        if policy.get('reference_rotation') is not None and np.isfinite(values[i, 2:]).all():
            matrix = json.loads(row['rotation_matrix'])
            values[i, 2:] = legacy_angles(relative_rotation(matrix, policy['reference_rotation']))
    if policy.get('reference_r0'):
        for i,row in enumerate(rows):
            for j,signal in enumerate(SIGNALS):
                key = 'legacy_'+signal if j >= 2 else signal
                values[i,j] = number(row[key]) if row['tracking_status'] == 'tracked' else np.nan
    observed = np.isfinite(values)
    interp = np.zeros_like(observed)
    duration = np.zeros_like(values)
    maximum = float(policy.get('interpolation_seconds', 0))
    tracks = [r.get('track_id', '') for r in rows]
    for j in range(5):
        for start, stop in gaps(observed[:, j]):
            bounded = start > 0 and stop < len(rows)
            span = t[stop] - t[start-1] if bounded else t[stop-1] - t[start]
            duration[start:stop, j] = span
            if not bounded or maximum <= 0 or span > maximum + 1e-9:
                continue
            if any(r.get('tracking_status') != 'tracked' for r in rows[start-1:stop+1]) or len(set(tracks[start-1:stop+1])) != 1:
                continue
            if j >= 2:
                if not observed[[start-1, stop], 2:].all(): continue
                rotations = Rotation.concatenate([rotation_from_angles(values[start-1, 2:]), rotation_from_angles(values[stop, 2:])])
                matrices = Slerp([t[start-1], t[stop]], rotations)(t[start:stop]).as_matrix()
                values[start:stop, j] = [legacy_angles(m)[j-2] for m in matrices]
            else:
                values[start:stop, j] = np.interp(t[start:stop], t[[start-1, stop]], values[[start-1, stop], j])
            interp[start:stop, j] = True
    # Preserve raw rotations; unwrap treated Euler traces only within continuous identity segments.
    for j in (() if policy.get('reference_r0') else range(2, 5)):
        start = 0
        for stop in range(1, len(rows)+1):
            boundary = stop == len(rows) or tracks[stop] != tracks[stop-1] or not np.isfinite(values[stop-1, j]) or not np.isfinite(values[stop, j])
            if boundary:
                if np.isfinite(values[start:stop, j]).all():
                    values[start:stop, j] = np.degrees(np.unwrap(np.radians(values[start:stop, j])))
                start = stop
    smoothing = float(policy.get('smoothing_seconds', 0))
    # Causal trailing median, eyes/mouth only. Do not smooth Euler components.
    if smoothing > 0:
        source = values.copy()
        for j in range(2):
            segment = 0
            for i in range(len(rows)):
                if not np.isfinite(source[i, j]) or (i and (tracks[i] != tracks[i-1] or t[i]-t[i-1] > smoothing)):
                    segment = i
                if np.isfinite(source[i, j]):
                    start = max(segment, int(np.searchsorted(t, t[i]-smoothing)))
                    finite = source[start:i+1, j]
                    values[i, j] = np.median(finite[np.isfinite(finite)])
    for i, row in enumerate(output):
        for j, signal in enumerate(SIGNALS):
            row['raw_' + signal] = str(number(rows[i].get(signal)))
            row[signal] = str(0.0 if policy.get('reference_r0') and not np.isfinite(values[i,j]) else values[i, j])
            row['valid_' + signal] = str(int(observed[i, j]))
            row['rejected_' + signal] = str(int(rejected[i, j]))
            row['interpolated_' + signal] = str(int(interp[i, j]))
            row['gap_seconds_' + signal] = str(duration[i, j])
            row['requires_imputation_' + signal] = str(int(not np.isfinite(values[i, j])))
        row['was_interpolated'] = str(int(interp[i].any()))
    return output
