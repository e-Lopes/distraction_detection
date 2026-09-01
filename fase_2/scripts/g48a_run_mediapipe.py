#!/usr/bin/env python3
"""Executa o braço MediaPipe Face Mesh do smoke G48A e congela os frames-alvo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fase_2.src.evaluation.g48a_contract import empty_result, result_from_landmarks
from fase_2.src.features.g47_extractors import MediaPipeFaceMeshExtractor


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in (*points[:, :2].min(axis=0), *points[:, :2].max(axis=0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("fase_2/data/manifests/g48a_smoke_30.csv"))
    parser.add_argument("--videos", type=Path, default=Path("fase_2/data/manifests/videos.csv"))
    parser.add_argument("--roi", type=Path, default=Path("fase_2/configs/preprocessing/legacy_roi.json"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke"))
    args = parser.parse_args()
    targets = pd.read_csv(args.manifest)
    videos = pd.read_csv(args.videos).set_index("video_id")
    roi = json.loads(args.roi.read_text(encoding="utf-8"))["roi_cadeira"]
    roi_x, roi_y, roi_width, roi_height = (int(value) for value in roi)
    frames_dir = args.output / "frames"
    metrics_dir = args.output / "metrics"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows, landmarks = [], {}
    for target in targets.itertuples(index=False):
        meta = videos.loc[target.video_id]
        capture = cv2.VideoCapture(str(meta.relative_path))
        start = max(0, int(target.frame_id) - round(float(meta.fps)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        detection = None
        target_frame = None
        with MediaPipeFaceMeshExtractor(minimum_confidence=0.5) as extractor:
            for frame_id in range(start, int(target.frame_id) + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                detection = extractor.detect(frame)
                if frame_id == int(target.frame_id):
                    target_frame = frame
        capture.release()
        if target_frame is None:
            result = empty_result(video_id=target.video_id, frame_id=int(target.frame_id),
                                  timestamp_ms=float(target.timestamp_ms), extractor="mediapipe_face_mesh",
                                  extractor_version="0.10.21", inference_time_ms=float("nan"),
                                  failure_reason="target_frame_unreadable")
        elif detection is None or detection.points is None:
            result = empty_result(video_id=target.video_id, frame_id=int(target.frame_id),
                                  timestamp_ms=float(target.timestamp_ms), extractor="mediapipe_face_mesh",
                                  extractor_version="0.10.21",
                                  inference_time_ms=float(detection.inference_ms if detection else np.nan),
                                  failure_reason=detection.failure if detection else "no_result")
        else:
            height, width = target_frame.shape[:2]
            result = result_from_landmarks(points=detection.points, width=width, height=height,
                                           video_id=target.video_id, frame_id=int(target.frame_id),
                                           timestamp_ms=float(target.timestamp_ms), extractor="mediapipe_face_mesh",
                                           extractor_version="0.10.21", face_bbox=_bbox(detection.points),
                                           confidence=float(detection.confidence),
                                           inference_time_ms=float(detection.inference_ms))
            landmarks[target.sample_id] = detection.points.tolist()
        rows.append(result.to_dict())
        if target_frame is not None:
            cv2.imwrite(str(frames_dir / f"{target.sample_id}.jpg"), target_frame)
    pd.DataFrame(rows).to_csv(metrics_dir / "mediapipe_face_mesh.csv", index=False)
    (metrics_dir / "mediapipe_face_mesh_landmarks.json").write_text(
        json.dumps(landmarks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"MediaPipe: {sum(row['face_detected'] for row in rows)}/{len(rows)} frames com face")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
