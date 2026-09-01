#!/usr/bin/env python3
"""Valida automaticamente os artefatos comuns produzidos pela G1 G48A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("fase_2/outputs/G48A_framework_smoke/metrics")
EXTRACTORS = ("mediapipe_face_mesh", "insightface_2d106", "openface_68")
MEASURES = ("ear_left", "ear_right", "ear", "mar", "pitch", "yaw", "roll")


def main() -> int:
    expected = pd.read_csv("fase_2/data/manifests/g48a_smoke_30.csv")[["video_id", "frame_id"]]
    expected_keys = set(map(tuple, expected.to_numpy()))
    for extractor in EXTRACTORS:
        frame = pd.read_csv(ROOT / f"{extractor}.csv")
        assert len(frame) == 30 and not frame.duplicated(["video_id", "frame_id"]).any()
        assert set(map(tuple, frame[["video_id", "frame_id"]].to_numpy())) == expected_keys
        missing = ~frame.face_detected.astype(bool)
        assert frame.loc[missing, list(MEASURES)].isna().all().all(), f"{extractor}: ausência não é NaN"
        assert (~frame.loc[missing, ["ear_valid", "mar_valid", "head_pose_valid"]].astype(bool)).all().all()
        present = frame.face_detected.astype(bool)
        assert np.isfinite(frame.loc[present, list(MEASURES)].to_numpy(dtype=float)).all()
        assert (frame.loc[present, "failure_reason"].fillna("") == "").all()
    print("G48A: contrato, cardinalidade, NaNs e máscaras válidos para os 3 extratores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
