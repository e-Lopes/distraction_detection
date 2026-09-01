import numpy as np
import pandas as pd

from fase_2.src.data.g48_preparation import select_manifest_rows


def test_g48_sample_is_deterministic_excludes_absent_and_respects_gap():
    size = 120
    series = pd.DataFrame(
        {
            "face_detected": np.resize([1, 1, 0], size),
            "pitch": np.linspace(-30, 30, size),
            "yaw": np.resize([-30, 0, 30], size),
        }
    )
    labels = np.resize(["alert", "fatigue", "distraction", None], size).tolist()

    first = select_manifest_rows(
        series, labels, video_id="video_01", fps=2, count=12, minimum_gap_seconds=1, seed=42
    )
    second = select_manifest_rows(
        series, labels, video_id="video_01", fps=2, count=12, minimum_gap_seconds=1, seed=42
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["behavior_label"]) <= {"alert", "fatigue", "distraction"}
    assert np.diff(first["frame_index"]).min() >= 2
    assert first["sample_id"].is_unique
