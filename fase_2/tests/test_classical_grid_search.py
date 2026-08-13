from pathlib import Path

from fase_2.src.training.classical_grid_search import (
    candidate_id,
    load_candidate_result,
    parameter_grid,
    save_candidate_result,
)


def test_parameter_grid_is_cartesian_and_deterministic():
    grid = parameter_grid({"a": [1, 2], "b": ["x", "y"]})
    assert grid == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert candidate_id(grid[0]) == candidate_id({"b": "x", "a": 1})
    conditional = parameter_grid(
        [
            {"kernel": ["linear"], "C": [1, 10]},
            {"kernel": ["rbf"], "C": [1], "gamma": ["scale", 0.1]},
        ]
    )
    assert len(conditional) == 4
    assert not any(row["kernel"] == "linear" and "gamma" in row for row in conditional)


def test_candidate_checkpoint_requires_matching_fingerprint(tmp_path: Path):
    path = tmp_path / "candidate.json"
    result = {"fingerprint": "current", "validation_macro_f1": 0.5}
    save_candidate_result(path, result)
    assert load_candidate_result(path, "current") == result
    assert load_candidate_result(path, "stale") is None
