"""Pose ablation with train-only calibration and rolling context reset per split block."""
import numpy as np
import pandas as pd
from ..data.windowing import build_windows
from ..features.pose_robustness import fit_pose_correction, apply_pose_correction, rolling_perclos
from .g46_pose_robustness import REPRESENTATIONS, _window_features


def datasets(series, labels, blocks, config, run):
    chosen = [b for b in blocks if b.fold == run.fold]
    frames = {}
    for video, rows in series.items():
        frame = pd.DataFrame(rows)
        for column in frame.columns:
            if column not in {'video_id','timestamp','behavior_label'}:
                frame[column] = pd.to_numeric(frame[column],errors='coerce')
        frame['video_id'] = video
        frame['behavior_label'] = ['alert' if label == 'attention' else label for label in labels[video]]
        frames[video] = frame
    training = pd.concat([frames[b.video_id].iloc[b.start_frame:b.end_frame+1]
                          for b in chosen if b.subset=='train'],ignore_index=True)
    correction = fit_pose_correction(training)
    windows=build_windows(labels,size_frames=run.window_size_frames,
                          stride_frames=config['windowing']['stride_frames'],
                          minimum_proportion=config['windowing']['minimum_target_proportion'],
                          behavior_classes=set(config['target']['classes']))
    prepared={}
    for b in chosen:
        value=apply_pose_correction(frames[b.video_id].iloc[b.start_frame:b.end_frame+1].reset_index(drop=True),correction)
        for seconds in (30,60):
            value=rolling_perclos(value,window_seconds=seconds)
        prepared[(b.video_id,b.start_frame)]=value
    result={}
    columns=REPRESENTATIONS[run.representation.removeprefix('pose_')]
    for subset in ('train','validation','test'):
        selected=[];values=[]
        for w in windows:
            if w.label=='mixed':continue
            match=[b for b in chosen if b.subset==subset and b.video_id==w.video_id
                   and b.start_frame<=w.start_frame and w.end_frame<=b.end_frame]
            if not match:continue
            b=match[0]
            values.append(_window_features(prepared[(b.video_id,b.start_frame)],
                                            w.start_frame-b.start_frame,w.end_frame-b.start_frame,columns))
            selected.append(w)
        if not selected:raise ValueError(f'Pose: subconjunto {subset} vazio')
        result[subset]=(np.asarray(values),np.asarray([w.label for w in selected]),selected)
    return result,correction
