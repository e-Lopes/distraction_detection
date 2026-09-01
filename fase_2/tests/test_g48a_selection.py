from pathlib import Path

import pandas as pd

from fase_2.src.data.g48a_smoke_selection import select_smoke_frames


def test_real_g48a_selection_is_reproducible_and_stratified():
    candidates = pd.read_csv(Path("fase_2/data/manifests/g48_annotation_sample.csv"))
    first = select_smoke_frames(candidates, seed=42)
    second = select_smoke_frames(candidates, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 30 and first.sample_id.is_unique
    assert first.difficulty.value_counts().to_dict() == {"easy": 10, "intermediate": 10, "hard": 10}
    assert (pd.crosstab(first.difficulty, first.video_id) >= 2).all().all()
    assert set(first.temporal_class) == {"alert", "fatigue", "distraction"}
