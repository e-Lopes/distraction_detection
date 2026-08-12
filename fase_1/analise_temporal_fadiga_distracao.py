"""
Fatigue & Distraction Analysis -- Presentation Output Generator
================================================================
Generates all charts, tables, and statistics needed for the seminar slides.

Recovered from the legacy TELEOP workspace on 2026-08-11. The original
analysis is preserved here with portable paths and CSV export added.

Outputs (saved to ./presentation_outputs/):
  1. fatigue_signals_<video>.png    -- EAR / MAR / Pitch over time (per video)
  2. state_distribution_<video>.png -- Pie + bar chart of state % (per video)
  3. state_distribution_combined.png-- Combined bar chart across all videos
  4. threshold_table.png            -- Visual table of heuristic rules
  5. summary_report.txt             -- Text summary with all numbers
  6. sample_frames_<video>/         -- Key annotated frames (alert/fatigue/distraction)
  7. series_temporais_<video>.csv   -- Per-frame EAR/MAR/head-pose series

Usage:
  1. Set VIDEO_DIR below to the folder containing your .mp4 files
  2. Optionally set ROI_CONFIG_PATH to your roi_config.json
  3. Run:  python fatigue_analysis_presentation.py
"""

import cv2
import numpy as np
import csv
import json
import math
import sys
import os
from collections import deque, Counter
from pathlib import Path
from datetime import timedelta

# -- Try importing optional deps -----------------------------------------------
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==============================================================================
# CONFIGURATION -- Edit these paths before running
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"             # Folder with 1.mp4, 2.mp4, 3.mp4, 4.mp4
ROI_CONFIG_PATH = BASE_DIR / "roi_config_temporal.example.json"
OUTPUT_DIR = BASE_DIR / "presentation_outputs"

# Explicit video list (processed in order)
VIDEO_FILES = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
VIDEO_LABELS = ["Vídeo 1", "Vídeo 2", "Vídeo 3", "Vídeo 4"]

# -- Thresholds (same as your original code) -----------------------------------
EAR_THRESHOLD      = 0.25
EAR_CONSEC_FRAMES  = 20
MAR_THRESHOLD      = 0.55
PITCH_THRESHOLD    = -15.0
HIST_LEN           = 150

# -- Landmark indices ----------------------------------------------------------
RIGHT_EYE      = [33, 160, 158, 133, 153, 144]
LEFT_EYE       = [362, 385, 387, 263, 373, 380]
MOUTH_TOP      = [82, 13, 312]
MOUTH_BOTTOM   = [87, 14, 317]
MOUTH_LR       = [78, 308]
HEAD_POSE_IDS  = [1, 152, 33, 263, 61, 291]
HEAD_POSE_3D   = np.array([
    [0.000,  0.000,  0.000],
    [0.000, -0.064, -0.013],
    [-0.043, 0.033, -0.026],
    [0.043,  0.033, -0.026],
    [-0.029,-0.029, -0.024],
    [0.029, -0.029, -0.024],
], dtype=np.float64)

# -- Colors for states ---------------------------------------------------------
STATE_COLORS = {
    'ALERT':       '#2ecc71',
    'FATIGUE':     '#e67e22',
    'DISTRACTION': '#e74c3c',
}
STATE_COLORS_BGR = {
    'ALERT':       (0, 220, 60),
    'FATIGUE':     (0, 165, 255),
    'DISTRACTION': (0, 0, 255),
}


# ==============================================================================
# METRIC FUNCTIONS (unchanged from your original code)
# ==============================================================================

def _dist(a, b):
    return np.linalg.norm(a - b)

def compute_ear(landmarks, eye_ids, w, h):
    p = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_ids])
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)

def compute_mar(landmarks, w, h):
    top = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in MOUTH_TOP], axis=0)
    bot = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in MOUTH_BOTTOM], axis=0)
    lc  = np.array([landmarks[MOUTH_LR[0]].x * w, landmarks[MOUTH_LR[0]].y * h])
    rc  = np.array([landmarks[MOUTH_LR[1]].x * w, landmarks[MOUTH_LR[1]].y * h])
    return _dist(top, bot) / (_dist(lc, rc) + 1e-6)

def compute_head_pose(landmarks, frame_w, frame_h):
    img_pts = np.array(
        [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in HEAD_POSE_IDS],
        dtype=np.float64,
    )
    focal = float(frame_w)
    cam = np.array([[focal, 0, frame_w / 2],
                    [0, focal, frame_h / 2],
                    [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_POSE_3D, img_pts, cam, np.zeros((4, 1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = 0.0
        roll  = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    return pitch, yaw, roll

def classify(ear_val, mar_val, pitch_val, ear_streak):
    if ear_streak >= EAR_CONSEC_FRAMES or mar_val > MAR_THRESHOLD:
        return 'FATIGUE'
    if pitch_val < PITCH_THRESHOLD:
        return 'DISTRACTION'
    return 'ALERT'


# ==============================================================================
# ROI LOADING
# ==============================================================================

def load_roi(config_path):
    if config_path is None:
        return None
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get('roi_cadeira')
    except Exception:
        return None


# ==============================================================================
# PROCESS A SINGLE VIDEO -- returns log dict
# ==============================================================================

def process_video(video_path, roi, face_mesh):
    """Process one video and return the full log of metrics per frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return None

    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s = n_tot / fps if fps > 0 else 0

    print(f"  Resolution: {fw}x{fh} @ {fps:.1f} fps | {n_tot} frames | {dur_s:.0f}s")

    log = dict(
        time=[], ear=[], mar=[], pitch=[], yaw=[], roll=[],
        face_detected=[], state=[], frame_n=[]
    )

    # Storage for sample frames (first occurrence of each state)
    sample_frames = {}

    ear_streak = 0
    frame_n    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ROI
        if roi:
            rx, ry, rw, rh = roi
            roi_bgr = frame[ry:ry+rh, rx:rx+rw]
        else:
            rx, ry, rw, rh = 0, 0, fw, fh
            roi_bgr = frame

        roi_h, roi_w = roi_bgr.shape[:2]
        results = face_mesh.process(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))

        ear_val = mar_val = pitch_val = yaw_val = roll_val = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lm = results.multi_face_landmarks[0].landmark
            ear_val   = (compute_ear(lm, RIGHT_EYE, roi_w, roi_h) +
                         compute_ear(lm, LEFT_EYE, roi_w, roi_h)) / 2.0
            mar_val   = compute_mar(lm, roi_w, roi_h)
            pitch_val, yaw_val, roll_val = compute_head_pose(lm, roi_w, roi_h)

            # Draw landmarks on frame for sample captures
            for idx in RIGHT_EYE + LEFT_EYE + MOUTH_LR + [1, 152]:
                px = int(lm[idx].x * roi_w) + rx
                py = int(lm[idx].y * roi_h) + ry
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        # EAR streak
        if 0 < ear_val < EAR_THRESHOLD:
            ear_streak += 1
        else:
            ear_streak = 0

        state = classify(ear_val, mar_val, pitch_val, ear_streak)

        # Log
        log['time'].append(ts)
        log['ear'].append(ear_val)
        log['mar'].append(mar_val)
        log['pitch'].append(pitch_val)
        log['yaw'].append(yaw_val)
        log['roll'].append(roll_val)
        log['face_detected'].append(face_detected)
        log['state'].append(state)
        log['frame_n'].append(frame_n)

        # Capture sample frames: most extreme non-zero cases
        # For FATIGUE: lowest EAR (eyes most closed) and highest MAR (biggest yawn)
        # For DISTRACTION: most negative pitch (biggest head drop)
        # For ALERT: any good frame with face detected
        if face_detected and ear_val > 0:
            def _annotate(frm, st, ear, mar, pitch, t_s, fn):
                ann = frm.copy()
                col = STATE_COLORS_BGR.get(st, (255, 255, 255))
                cv2.rectangle(ann, (rx, ry), (rx+rw, ry+rh), col, 3)
                oy = 30
                cv2.putText(ann, f'State: {st}', (20, oy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
                cv2.putText(ann, f'EAR: {ear:.3f}  MAR: {mar:.3f}  Pitch: {pitch:+.1f}',
                            (20, oy + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(ann, f't = {t_s:.1f}s  (frame {fn})',
                            (20, oy + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                return ann

            # FATIGUE - lowest EAR (most closed eyes, non-zero)
            if state == 'FATIGUE' and 0 < ear_val < 0.5:
                prev = sample_frames.get('FATIGUE_low_ear')
                if prev is None or ear_val < prev['ear']:
                    sample_frames['FATIGUE_low_ear'] = {
                        'frame': _annotate(frame, state, ear_val, mar_val, pitch_val, ts, frame_n),
                        'face': True, 'ts': ts, 'ear': ear_val
                    }

            # FATIGUE - highest MAR (biggest yawn, non-zero)
            if state == 'FATIGUE' and mar_val > 0.1:
                prev = sample_frames.get('FATIGUE_high_mar')
                if prev is None or mar_val > prev['mar']:
                    sample_frames['FATIGUE_high_mar'] = {
                        'frame': _annotate(frame, state, ear_val, mar_val, pitch_val, ts, frame_n),
                        'face': True, 'ts': ts, 'mar': mar_val
                    }

            # DISTRACTION - most negative pitch (biggest head drop, non-zero)
            if state == 'DISTRACTION' and pitch_val < -15 and pitch_val > -60:
                prev = sample_frames.get('DISTRACTION_low_pitch')
                if prev is None or pitch_val < prev['pitch']:
                    sample_frames['DISTRACTION_low_pitch'] = {
                        'frame': _annotate(frame, state, ear_val, mar_val, pitch_val, ts, frame_n),
                        'face': True, 'ts': ts, 'pitch': pitch_val
                    }

            # ALERT - any good frame (first one with face detected is fine)
            if state == 'ALERT' and 'ALERT' not in sample_frames:
                sample_frames['ALERT'] = {
                    'frame': _annotate(frame, state, ear_val, mar_val, pitch_val, ts, frame_n),
                    'face': True, 'ts': ts
                }

        # Progress
        if frame_n % 1000 == 0:
            pct = frame_n / n_tot * 100 if n_tot else 0
            print(f"    frame {frame_n:>6}/{n_tot} ({pct:.0f}%)  "
                  f"EAR={ear_val:.3f} MAR={mar_val:.3f} Pitch={pitch_val:+.1f} [{state}]")

    cap.release()

    log['fps'] = fps
    log['n_tot'] = n_tot
    log['duration_s'] = dur_s
    log['resolution'] = f"{fw}x{fh}"
    log['sample_frames'] = sample_frames

    return log


def save_time_series_csv(log, video_name, out_dir):
    """Persist the complete per-frame series for reuse by temporal models."""
    path = out_dir / f"series_temporais_{video_name}.csv"
    fields = [
        'frame_n', 'time_seconds', 'ear', 'mar', 'pitch', 'yaw', 'roll',
        'face_detected', 'state'
    ]
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index in range(len(log['frame_n'])):
            writer.writerow({
                'frame_n': log['frame_n'][index],
                'time_seconds': log['time'][index],
                'ear': log['ear'][index],
                'mar': log['mar'][index],
                'pitch': log['pitch'][index],
                'yaw': log['yaw'][index],
                'roll': log['roll'][index],
                'face_detected': int(log['face_detected'][index]),
                'state': log['state'][index],
            })
    print(f"  [OK] Série temporal: {path.name}")
    return path


# ==============================================================================
# CHART GENERATORS
# ==============================================================================

def generate_signal_plot(log, video_name, out_dir):
    """Generate the 3-panel EAR/MAR/Pitch time-series plot."""
    t = np.array(log['time'])
    if len(t) == 0:
        return

    # Clip outliers for clean visualization
    ear_clean   = np.clip(log['ear'],   0, 0.50)
    mar_clean   = np.clip(log['mar'],   0, 0.80)
    pitch_clean = np.clip(log['pitch'], -50, 50)

    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f'Visual Fatigue Indicators -- {video_name}',
                 fontsize=18, fontweight='bold', y=0.98)

    metrics = [
        ('EAR',       ear_clean,   '#2980b9', EAR_THRESHOLD,   'EAR < 0.25',  (0, 0.50)),
        ('MAR',       mar_clean,   '#27ae60', MAR_THRESHOLD,    'MAR > 0.55',  (0, 0.80)),
        ('Pitch',     pitch_clean, '#d35400', PITCH_THRESHOLD,  'Pitch < -15', (-50, 50)),
    ]

    for ax, (label, data, color, threshold, thresh_label, ylim) in zip(axes, metrics):
        ax.plot(t, data, color=color, lw=0.5, alpha=0.85)
        ax.axhline(threshold, color='#e74c3c', ls='--', lw=1.5, alpha=0.9,
                   label=f'Threshold ({thresh_label})')
        ax.set_ylabel(label, fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylim(*ylim)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.8)
        ax.tick_params(labelsize=13, pad=6)
        ax.grid(True, alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=16, fontweight='bold', labelpad=10)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    path = out_dir / f'fatigue_signals_{video_name}.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Signal plot: {path.name}")


def generate_state_distribution(log, video_name, out_dir):
    """Generate pie + bar chart for state distribution of one video."""
    counts = Counter(log['state'])
    total = len(log['state'])
    if total == 0:
        return counts

    labels = ['ALERT', 'FATIGUE', 'DISTRACTION']
    values = [counts.get(l, 0) for l in labels]
    pcts   = [100 * v / total for v in values]
    colors = [STATE_COLORS[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'State Distribution -- {video_name}',
                 fontsize=14, fontweight='bold')

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(10)
    ax1.set_title('Proportion', fontsize=11, pad=10)

    # Bar chart with frame counts
    bars = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Number of Frames', fontsize=11)
    ax2.set_title('Frame Count', fontsize=11, pad=10)

    for bar, v, p in zip(bars, values, pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{v:,}\n({p:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Info box
    dur = log.get('duration_s', 0)
    info_text = (f"Total frames: {total:,}\n"
                 f"Duration: {dur:.0f}s ({dur/60:.1f} min)\n"
                 f"Resolution: {log.get('resolution', 'N/A')}\n"
                 f"FPS: {log.get('fps', 'N/A'):.1f}")
    ax2.text(0.98, 0.98, info_text, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))

    ax2.grid(axis='y', alpha=0.2)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    path = out_dir / f'state_distribution_{video_name}.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] State distribution: {path.name}")

    return counts


def generate_combined_distribution(all_results, out_dir):
    """Generate a combined bar chart comparing state distribution across all videos."""
    if not all_results:
        return

    labels = ['ALERT', 'FATIGUE', 'DISTRACTION']
    video_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(video_names)*3), 6))
    fig.suptitle('State Distribution -- All Videos Combined',
                 fontsize=15, fontweight='bold')

    x = np.arange(len(video_names))
    width = 0.25

    for i, state in enumerate(labels):
        pcts = []
        for vname in video_names:
            total = sum(all_results[vname]['counts'].values())
            pcts.append(100 * all_results[vname]['counts'].get(state, 0) / total if total else 0)
        bars = ax.bar(x + i*width, pcts, width, label=state,
                     color=STATE_COLORS[state], edgecolor='white', linewidth=1)
        for bar, p in zip(bars, pcts):
            if p > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{p:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Percentage of Frames (%)', fontsize=11)
    ax.set_xticks(x + width)

    # Shorten video names for x-axis
    short_names = []
    for n in video_names:
        if len(n) > 25:
            short_names.append(n[:12] + '...' + n[-10:])
        else:
            short_names.append(n)
    ax.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)

    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = out_dir / 'state_distribution_combined.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] Combined distribution: {path.name}")


def generate_threshold_table(out_dir):
    """Generate a visual table image showing the heuristic rules."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    fig.suptitle('Heuristic System -- Classification Rules',
                 fontsize=14, fontweight='bold', y=0.95)

    table_data = [
        ['Indicator',   'Condition',                         'Classification', 'Source'],
        ['EAR',         f'< {EAR_THRESHOLD} for {EAR_CONSEC_FRAMES} consecutive frames', 'FATIGUE',      'Eye closure detection'],
        ['MAR',         f'> {MAR_THRESHOLD}',                'FATIGUE',      'Yawn detection'],
        ['Pitch',       f'< {PITCH_THRESHOLD}°',            'DISTRACTION',  'Head drop / look away'],
        ['Otherwise',   '--',                                 'ALERT',        'Normal operation'],
    ]

    colors_row = [
        ['#34495e'] * 4,                          # header
        [STATE_COLORS['FATIGUE']] * 4,
        [STATE_COLORS['FATIGUE']] * 4,
        [STATE_COLORS['DISTRACTION']] * 4,
        [STATE_COLORS['ALERT']] * 4,
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.38, 0.18, 0.22],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    for i, row in enumerate(table_data):
        for j in range(len(row)):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold', fontsize=11)
            else:
                cell.set_facecolor(colors_row[i][j] + '18')  # hex alpha for light tint
                cell.set_text_props(fontsize=10)
            cell.set_edgecolor('#bdc3c7')
            cell.set_linewidth(1.5)

    # Stack info
    fig.text(0.5, 0.05, 'Stack: MediaPipe Face Mesh + OpenCV (solvePnP) -- CPU-only',
            ha='center', fontsize=10, fontstyle='italic', color='#555555')

    path = out_dir / 'threshold_table.png'
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Threshold table: {path.name}")


def save_sample_frames(log, video_name, out_dir):
    """Save annotated sample frames for each detected state."""
    sample_frames = log.get('sample_frames', {})
    if not sample_frames:
        return

    frames_dir = out_dir / f'sample_frames_{video_name}'
    frames_dir.mkdir(exist_ok=True)

    for state, data in sample_frames.items():
        path = frames_dir / f'{state.lower()}_t{data["ts"]:.1f}s.png'
        cv2.imwrite(str(path), data['frame'])

    print(f"  [OK] Sample frames: {frames_dir.name}/ ({len(sample_frames)} states)")


def generate_summary_report(all_results, out_dir):
    """Generate a text summary report with all numbers for the presentation."""
    lines = []
    lines.append("=" * 70)
    lines.append("  FATIGUE & DISTRACTION ANALYSIS -- SUMMARY REPORT")
    lines.append("  For Seminar Presentation")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Heuristic Thresholds:")
    lines.append(f"    EAR threshold:    < {EAR_THRESHOLD} for {EAR_CONSEC_FRAMES} consecutive frames")
    lines.append(f"    MAR threshold:    > {MAR_THRESHOLD}")
    lines.append(f"    Pitch threshold:  < {PITCH_THRESHOLD}°")
    lines.append("")
    lines.append("-" * 70)

    total_frames_all = 0
    total_duration_all = 0
    combined_counts = Counter()

    for vname, data in all_results.items():
        log = data['log']
        counts = data['counts']
        total = sum(counts.values())
        dur = log.get('duration_s', 0)
        total_frames_all += total
        total_duration_all += dur
        combined_counts += counts

        lines.append(f"\n  Video: {vname}")
        lines.append(f"  Resolution: {log.get('resolution', 'N/A')}")
        lines.append(f"  FPS: {log.get('fps', 0):.1f}")
        lines.append(f"  Duration: {dur:.0f}s ({dur/60:.1f} min)")
        lines.append(f"  Total frames analysed: {total:,}")
        lines.append(f"")
        for state in ['ALERT', 'FATIGUE', 'DISTRACTION']:
            c = counts.get(state, 0)
            pct = 100 * c / total if total else 0
            lines.append(f"    {state:<14}: {c:>7,} frames  ({pct:5.1f}%)")

        # Metric statistics
        if log['ear']:
            ear_arr = [v for v in log['ear'] if v > 0]
            if ear_arr:
                lines.append(f"\n  EAR stats (when face detected):")
                lines.append(f"    mean={np.mean(ear_arr):.3f}  std={np.std(ear_arr):.3f}  "
                            f"min={np.min(ear_arr):.3f}  max={np.max(ear_arr):.3f}")

            mar_arr = [v for v in log['mar'] if v > 0]
            if mar_arr:
                lines.append(f"  MAR stats (when face detected):")
                lines.append(f"    mean={np.mean(mar_arr):.3f}  std={np.std(mar_arr):.3f}  "
                            f"min={np.min(mar_arr):.3f}  max={np.max(mar_arr):.3f}")

            pitch_arr = [v for v in log['pitch'] if v != 0]
            if pitch_arr:
                lines.append(f"  Pitch stats (when face detected):")
                lines.append(f"    mean={np.mean(pitch_arr):+.1f}°  std={np.std(pitch_arr):.1f}°  "
                            f"min={np.min(pitch_arr):+.1f}°  max={np.max(pitch_arr):+.1f}°")

            # Face detection rate
            face_frames = sum(1 for e in log['ear'] if e > 0)
            lines.append(f"\n  Face detection rate: {face_frames:,}/{total:,} "
                        f"({100*face_frames/total:.1f}%)")

        lines.append("-" * 70)

    # Overall summary
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  OVERALL SUMMARY")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Videos analysed:        {len(all_results)}")
    lines.append(f"  Total frames:           {total_frames_all:,}")
    lines.append(f"  Total duration:         {total_duration_all:.0f}s ({total_duration_all/60:.1f} min)")
    lines.append(f"")
    for state in ['ALERT', 'FATIGUE', 'DISTRACTION']:
        c = combined_counts.get(state, 0)
        pct = 100 * c / total_frames_all if total_frames_all else 0
        lines.append(f"  {state:<14}: {c:>8,} frames  ({pct:5.1f}%)")
    lines.append(f"{'=' * 70}")

    # Talking points for presentation
    lines.append(f"\n\n-- TALKING POINTS FOR PRESENTATION ----------------------")
    lines.append(f"")
    lines.append(f"Slide 10 (Threshold Table):")
    lines.append(f'  "O sistema heurístico usa thresholds fixos:')
    lines.append(f'   EAR abaixo de {EAR_THRESHOLD} por {EAR_CONSEC_FRAMES} frames = fadiga,')
    lines.append(f'   MAR acima de {MAR_THRESHOLD} = bocejo,')
    lines.append(f'   Pitch abaixo de {PITCH_THRESHOLD}° = distração."')
    lines.append(f"")
    lines.append(f"Slide 11-12 (Results):")

    # Generate specific talking points based on results
    if combined_counts:
        dominant = combined_counts.most_common(1)[0]
        lines.append(f'  "Nos {len(all_results)} vídeos analisados ({total_duration_all/60:.0f} min total),')
        lines.append(f'   o estado mais frequente foi {dominant[0]} com '
                     f'{100*dominant[1]/total_frames_all:.1f}% dos frames."')

        dist_pct = 100 * combined_counts.get('DISTRACTION', 0) / total_frames_all if total_frames_all else 0
        if dist_pct > 30:
            lines.append(f'  "A alta taxa de DISTRACTION ({dist_pct:.0f}%) reflete as limitações')
            lines.append(f'   do sistema heurístico -- posição lateral do operador gera falsos positivos."')

    report_text = "\n".join(lines)
    path = out_dir / 'summary_report.txt'
    path.write_text(report_text, encoding='utf-8')
    print(f"\n[OK] Summary report: {path.name}")

    # Also print to console
    print(report_text)


# ==============================================================================
# DEMO MODE -- Generate example outputs without MediaPipe / videos
# ==============================================================================

def generate_demo_outputs(out_dir):
    """Generate example charts using synthetic data (for testing without videos)."""
    print("\n" + "=" * 60)
    print("  DEMO MODE -- Generating examples with synthetic data")
    print("  (No MediaPipe or video files needed)")
    print("=" * 60 + "\n")

    np.random.seed(42)
    n_frames = 3000
    fps = 25.0
    t = np.arange(n_frames) / fps

    # Simulate realistic EAR/MAR/Pitch signals
    ear = 0.28 + 0.03 * np.random.randn(n_frames)
    mar = 0.15 + 0.05 * np.abs(np.random.randn(n_frames))
    pitch = -5.0 + 5.0 * np.random.randn(n_frames)

    # Add fatigue episodes (eye closure)
    for start in [500, 1200, 2400]:
        length = np.random.randint(25, 60)
        ear[start:start+length] = 0.15 + 0.02 * np.random.randn(length)

    # Add yawn episodes
    for start in [800, 1800]:
        length = np.random.randint(15, 30)
        mar[start:start+length] = 0.65 + 0.05 * np.random.randn(length)

    # Add distraction episodes (head turn)
    for start in [300, 1000, 1500, 2000, 2600]:
        length = np.random.randint(40, 100)
        pitch[start:start+length] = -25.0 + 3.0 * np.random.randn(length)

    # Zero-out some frames (simulate face not detected)
    for start in [600, 1400, 2200]:
        length = np.random.randint(20, 50)
        ear[start:start+length] = 0
        mar[start:start+length] = 0
        pitch[start:start+length] = 0

    # Classify
    ear_streak = 0
    states = []
    for e, m, p in zip(ear, mar, pitch):
        if 0 < e < EAR_THRESHOLD:
            ear_streak += 1
        else:
            ear_streak = 0
        states.append(classify(e, m, p, ear_streak))

    log = {
        'time': t.tolist(), 'ear': ear.tolist(), 'mar': mar.tolist(),
        'pitch': pitch.tolist(), 'yaw': [0]*n_frames, 'roll': [0]*n_frames,
        'state': states, 'frame_n': list(range(1, n_frames+1)),
        'fps': fps, 'n_tot': n_frames, 'duration_s': n_frames/fps,
        'resolution': '1920x1080', 'sample_frames': {},
    }

    vname = 'DEMO_synthetic'
    generate_signal_plot(log, vname, out_dir)
    counts = generate_state_distribution(log, vname, out_dir)
    generate_threshold_table(out_dir)

    all_results = {vname: {'log': log, 'counts': counts}}

    # Add a second synthetic video for combined chart demo
    ear2 = 0.30 + 0.02 * np.random.randn(2000)
    mar2 = 0.12 + 0.04 * np.abs(np.random.randn(2000))
    pitch2 = -3.0 + 4.0 * np.random.randn(2000)
    for start in [400, 1000]:
        length = 30
        pitch2[start:start+length] = -22.0 + 2.0 * np.random.randn(length)

    ear_streak2 = 0
    states2 = []
    for e, m, p in zip(ear2, mar2, pitch2):
        if 0 < e < EAR_THRESHOLD:
            ear_streak2 += 1
        else:
            ear_streak2 = 0
        states2.append(classify(e, m, p, ear_streak2))

    log2 = {
        'time': (np.arange(2000)/fps).tolist(), 'ear': ear2.tolist(),
        'mar': mar2.tolist(), 'pitch': pitch2.tolist(),
        'yaw': [0]*2000, 'roll': [0]*2000,
        'state': states2, 'frame_n': list(range(1,2001)),
        'fps': fps, 'n_tot': 2000, 'duration_s': 2000/fps,
        'resolution': '1920x1080', 'sample_frames': {},
    }
    counts2 = generate_state_distribution(log2, 'DEMO_synthetic_2', out_dir)
    all_results['DEMO_synthetic_2'] = {'log': log2, 'counts': counts2}

    generate_combined_distribution(all_results, out_dir)
    generate_summary_report(all_results, out_dir)

    print("\n[OK] Demo outputs generated! Check:", out_dir)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always generate the threshold table (no deps needed)
    generate_threshold_table(out_dir)

    # Build explicit video list: 1.mp4, 2.mp4, 3.mp4, 4.mp4
    video_dir = Path(VIDEO_DIR)
    videos = []
    labels = []
    for fname, label in zip(VIDEO_FILES, VIDEO_LABELS):
        vpath = video_dir / fname
        if vpath.exists():
            videos.append(vpath)
            labels.append(label)
        else:
            print(f"  [AVISO] Não encontrado: {vpath}")

    if not videos:
        print(f"\nNenhum vídeo encontrado em: {video_dir.resolve()}")
        print(f"Esperado: {', '.join(VIDEO_FILES)}")
        print("\nCertifique-se de que os vídeos estão na pasta correta.")
        print("Caminho esperado:")
        for f in VIDEO_FILES:
            print(f"  {video_dir.resolve() / f}")
        print("\nGerando outputs de demonstração com dados sintéticos...\n")
        generate_demo_outputs(out_dir)
        return

    if not HAS_MEDIAPIPE:
        print("\nMediaPipe não instalado. Instale com: pip install mediapipe")
        print("Gerando outputs de demonstração com dados sintéticos...\n")
        generate_demo_outputs(out_dir)
        return

    roi = load_roi(ROI_CONFIG_PATH)

    print(f"\n{'=' * 60}")
    print(f"  FATIGUE ANALYSIS -- PRESENTATION OUTPUT GENERATOR")
    print(f"{'=' * 60}")
    print(f"  ROI: {roi}")
    print(f"  Vídeos: {len(videos)} de {len(VIDEO_FILES)} encontrados")
    for i, (v, l) in enumerate(zip(videos, labels), 1):
        print(f"    {i}. {l} -> {v.name}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'=' * 60}\n")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_results = {}

    for vi, (vpath, vlabel) in enumerate(zip(videos, labels), 1):
        print(f"\n{'-' * 60}")
        print(f"  [{vi}/{len(videos)}] Processando: {vpath.name} ({vlabel})")
        print(f"{'-' * 60}")

        log = process_video(vpath, roi, face_mesh)
        if log is None:
            continue

        # Use the clean label (Vídeo 1, Vídeo 2, etc.) for file names
        safe_label = vlabel.replace(' ', '_').replace('í', 'i')

        generate_signal_plot(log, safe_label, out_dir)
        counts = generate_state_distribution(log, safe_label, out_dir)
        save_sample_frames(log, safe_label, out_dir)
        save_time_series_csv(log, safe_label, out_dir)

        all_results[vlabel] = {'log': log, 'counts': counts}

        print(f"  [OK] {vlabel} concluído!")

    face_mesh.close()

    # Combined outputs across all videos
    if all_results:
        generate_combined_distribution(all_results, out_dir)
        generate_summary_report(all_results, out_dir)

    print(f"\n{'=' * 60}")
    print(f"  CONCLUÍDO! Outputs salvos em: {out_dir.resolve()}")
    print(f"{'=' * 60}")
    print(f"\n  Arquivos gerados para a apresentação:")
    print(f"  -------------------------------------")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<50} ({size_kb:.0f} KB)")
        elif f.is_dir():
            n_files = len(list(f.iterdir()))
            print(f"    {f.name + '/':<50} ({n_files} frames)")


if __name__ == '__main__':
    main()
