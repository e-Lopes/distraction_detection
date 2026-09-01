#!/usr/bin/env python3
"""Executa InsightFace 2d106 no ambiente Conda isolado da G48A."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from fase_2.src.evaluation.g48a_contract import (
    empty_result,
    map_to_canonical,
    result_from_landmarks,
    select_operator_face,
)

MAPPING = [93, 96, 95, 89, 90, 91, 35, 41, 42, 39, 37, 36, 67, 68, 71, 64, 52, 55, 53, 58, 86, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("fase_2/data/manifests/g48a_smoke_30.csv"))
    parser.add_argument("--frames", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke/frames"))
    parser.add_argument("--model-root", type=Path,
                        default=Path("fase_2/outputs/cache/G48A_framework_smoke/insightface"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke/metrics"))
    args = parser.parse_args()
    app = FaceAnalysis(name="buffalo_l", root=str(args.model_root),
                       allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    rows, stored = [], {}
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        targets = list(csv.DictReader(stream))
    for target in targets:
        frame = cv2.imread(str(args.frames / f"{target['sample_id']}.jpg"))
        started = time.perf_counter_ns()
        faces = app.get(frame) if frame is not None else []
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if not faces:
            result = empty_result(video_id=target["video_id"], frame_id=int(target["frame_id"]),
                                  timestamp_ms=float(target["timestamp_ms"]), extractor="insightface_2d106",
                                  extractor_version="0.7.3/buffalo_l", inference_time_ms=elapsed,
                                  failure_reason="face_missing" if frame is not None else "target_frame_unreadable")
        else:
            height, width = frame.shape[:2]
            try:
                index = select_operator_face(np.asarray([face.bbox for face in faces]), width=width, height=height)
            except ValueError:
                result = empty_result(video_id=target["video_id"], frame_id=int(target["frame_id"]),
                                      timestamp_ms=float(target["timestamp_ms"]), extractor="insightface_2d106",
                                      extractor_version="0.7.3/buffalo_l", inference_time_ms=elapsed,
                                      failure_reason="operator_face_missing")
                rows.append(result.to_dict())
                continue
            face = faces[index]
            raw = np.asarray(face.landmark_2d_106, dtype=float)
            canonical = map_to_canonical(raw, MAPPING)
            result = result_from_landmarks(points=canonical, width=width, height=height,
                                           video_id=target["video_id"], frame_id=int(target["frame_id"]),
                                           timestamp_ms=float(target["timestamp_ms"]), extractor="insightface_2d106",
                                           extractor_version="0.7.3/buffalo_l",
                                           face_bbox=tuple(float(v) for v in face.bbox),
                                           confidence=float(face.det_score), inference_time_ms=elapsed)
            stored[target["sample_id"]] = canonical.tolist()
        rows.append(result.to_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "insightface_2d106.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "insightface_2d106_landmarks.json").write_text(json.dumps(stored), encoding="utf-8")
    print(f"InsightFace: {sum(row['face_detected'] for row in rows)}/{len(rows)} frames com face")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
