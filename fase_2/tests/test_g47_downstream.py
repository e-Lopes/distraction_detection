import numpy as np
import pandas as pd

from fase_2.src.evaluation.g47_downstream import paired_to_series


def test_paired_series_preserves_missingness():
    frame = pd.DataFrame(
        {
            "video_id": ["video_01", "video_01"],
            "frame_index": [0, 1],
            "timestamp_seconds": [0.0, 0.1],
            "mp_detected": [1, 0],
            "yolo_detected": [1, 1],
            **{f"mp_{name}": [1.0, np.nan] for name in ("ear", "mar", "pitch", "yaw", "roll")},
            **{f"yolo_{name}": [2.0, 2.0] for name in ("ear", "mar", "pitch", "yaw", "roll")},
        }
    )
    rows = paired_to_series(frame, "mp")
    assert rows[0]["face_detected"] == "1"
    assert rows[1]["face_detected"] == "0"
    assert rows[1]["ear"] == ""
