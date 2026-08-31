from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fase_2.src.data.g47_landmarks import parse_wflw_line, stratified_sample_indices
from fase_2.src.training.g47_yolo_face import loss_grid, pose_loss_diverged, validate_dataset_yaml


def test_stratified_sample_is_reproducible_and_respects_gap():
    frame = pd.DataFrame(
        {
            "face_detected": [1] * 100,
            "pitch": np.tile([-30, 0, 20, 5], 25),
            "yaw": np.tile([0, 30, 5, -20], 25),
        }
    )
    labels = (["alert", "fatigue", "distraction", None] * 25)[:100]
    first = stratified_sample_indices(frame, labels, count=20, minimum_gap_frames=3, seed=42)
    second = stratified_sample_indices(frame, labels, count=20, minimum_gap_frames=3, seed=42)
    assert first == second
    assert min(np.diff(first)) >= 3


def test_wflw_parser_requires_complete_record():
    values = [str(index) for index in range(196 + 4 + 6)] + ["folder/image.jpg"]
    points, metadata, relative = parse_wflw_line(" ".join(values))
    assert points.shape == (98, 2)
    assert len(metadata) == 10
    assert relative == "folder/image.jpg"
    with pytest.raises(ValueError):
        parse_wflw_line("1 2 3")


def test_dataset_yaml_and_loss_grid(tmp_path: Path):
    source = yaml.safe_load(Path("fase_2/configs/data/g47_yolo_dataset.example.yaml").read_text())
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    assert validate_dataset_yaml(path)["kpt_shape"] == [22, 3]
    assert len(loss_grid()) == 8
    source["flip_idx"][0] = 0
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="flip_idx"):
        validate_dataset_yaml(path)


def test_pose_loss_divergence_gate(tmp_path: Path):
    stable = tmp_path / "stable.csv"
    pd.DataFrame({"train/pose_loss": [2, 1.5, 1.2, 1.0]}).to_csv(stable, index=False)
    assert not pose_loss_diverged(stable)
    divergent = tmp_path / "divergent.csv"
    pd.DataFrame({"train/pose_loss": [1, 1, 1, 10]}).to_csv(divergent, index=False)
    assert pose_loss_diverged(divergent)
