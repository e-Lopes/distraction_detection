#!/usr/bin/env python3
"""Normaliza os CSVs de 68 pontos produzidos pelo OpenFace na G48A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fase_2.src.evaluation.g48a_contract import empty_result, map_to_canonical, result_from_landmarks

MAPPING = [45, 44, 43, 42, 47, 46, 36, 37, 38, 39, 40, 41, 54, 53, 51, 49, 48, 59, 57, 55, 30, 8]
DIGEST = "sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("fase_2/data/manifests/g48a_smoke_30.csv"))
    parser.add_argument("--raw", type=Path,
                        default=Path("fase_2/outputs/G48A_framework_smoke/openface_independent"))
    parser.add_argument("--frames", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke/frames"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/outputs/G48A_framework_smoke/metrics"))
    args = parser.parse_args()
    rows, stored = [], {}
    for target in pd.read_csv(args.manifest).itertuples(index=False):
        raw_path = args.raw / f"{target.sample_id}.csv"
        raw = pd.read_csv(raw_path, skipinitialspace=True) if raw_path.exists() else pd.DataFrame()
        frame = cv2.imread(str(args.frames / f"{target.sample_id}.jpg"))
        valid = len(raw) == 1 and int(raw.iloc[0]["success"]) == 1 and frame is not None
        if not valid:
            reason = "openface_tracking_failed" if raw_path.exists() else "openface_output_missing"
            result = empty_result(video_id=target.video_id, frame_id=int(target.frame_id),
                                  timestamp_ms=float(target.timestamp_ms), extractor="openface_68",
                                  extractor_version=f"2.0-era/{DIGEST}", inference_time_ms=float("nan"),
                                  failure_reason=reason)
        else:
            record = raw.iloc[0]
            points68 = np.column_stack(
                ([float(record[f"x_{i}"]) for i in range(68)],
                 [float(record[f"y_{i}"]) for i in range(68)])
            )
            canonical = map_to_canonical(points68, MAPPING)
            height, width = frame.shape[:2]
            mins, maxs = points68.min(axis=0), points68.max(axis=0)
            result = result_from_landmarks(points=canonical, width=width, height=height,
                                           video_id=target.video_id, frame_id=int(target.frame_id),
                                           timestamp_ms=float(target.timestamp_ms), extractor="openface_68",
                                           extractor_version=f"2.0-era/{DIGEST}",
                                           face_bbox=tuple(float(v) for v in (*mins, *maxs)),
                                           confidence=float(record["confidence"]),
                                           inference_time_ms=float("nan"))
            stored[target.sample_id] = canonical.tolist()
        rows.append(result.to_dict())
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "openface_68.csv", index=False)
    (args.output / "openface_68_landmarks.json").write_text(json.dumps(stored), encoding="utf-8")
    print(f"OpenFace: {sum(row['face_detected'] for row in rows)}/{len(rows)} frames com face")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
