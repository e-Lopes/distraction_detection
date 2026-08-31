from pathlib import Path

import cv2
import numpy as np
import pytest

from fase_2.src.features.g47_schema import (
    DEFAULT_FLIP_INDEX,
    HEAD_MODEL_3D,
    LANDMARK_NAMES,
    POSE_INDICES,
    camera_matrix,
    compute_indicators,
    flip_landmarks,
    load_schema,
    normalized_mean_error,
    validate_flip_index,
)


def _points() -> np.ndarray:
    points = np.zeros((22, 3), dtype=float)
    left = np.array([[70, 50], [75, 47], [80, 47], [85, 50], [80, 53], [75, 53]])
    right = np.array([[30, 50], [35, 47], [40, 47], [45, 50], [40, 53], [35, 53]])
    mouth = np.array(
        [[65, 75], [60, 70], [50, 69], [40, 70], [35, 75], [40, 80], [50, 81], [60, 80]]
    )
    points[:6, :2] = left
    points[6:12, :2] = right
    points[12:20, :2] = mouth
    points[20, :2] = (50, 55)
    points[21, :2] = (50, 95)
    points[:, 2] = 1
    return points


def test_schema_and_flip_contract_are_exact():
    schema = load_schema(Path("fase_2/configs/features/g47_face_landmarks.yaml"))
    assert schema.names == LANDMARK_NAMES
    assert schema.flip_index == DEFAULT_FLIP_INDEX
    assert validate_flip_index(list(DEFAULT_FLIP_INDEX)) == DEFAULT_FLIP_INDEX
    invalid = list(range(22))
    invalid[19], invalid[20], invalid[21] = 20, 21, 19
    with pytest.raises(ValueError, match="involutivo"):
        validate_flip_index(invalid)


def test_two_horizontal_flips_restore_coordinates_and_visibility():
    normalized = _points()
    normalized[:, :2] /= 100
    restored = flip_landmarks(flip_landmarks(normalized, DEFAULT_FLIP_INDEX), DEFAULT_FLIP_INDEX)
    np.testing.assert_allclose(restored, normalized)


def test_shared_indicators_and_occlusion():
    points = _points()
    indicators = compute_indicators(points, 100, 100)
    assert indicators.ear == pytest.approx(0.4)
    assert indicators.mar == pytest.approx((80 + 81 + 80 - 70 - 69 - 70) / 3 / 30)
    points[1, 2] = 0
    occluded = compute_indicators(points, 100, 100)
    assert np.isnan(occluded.ear_left)
    assert np.isfinite(occluded.ear_right)


def test_solvepnp_recovers_neutral_synthetic_pose():
    rotation = np.zeros((3, 1), dtype=np.float64)
    translation = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        HEAD_MODEL_3D, rotation, translation, camera_matrix(640, 480), np.zeros((4, 1))
    )
    points = _points()
    points[POSE_INDICES, :2] = projected.reshape(-1, 2)
    pose = compute_indicators(points, 640, 480)
    assert pose.pitch == pytest.approx(0, abs=1e-4)
    assert pose.yaw == pytest.approx(0, abs=1e-4)
    assert pose.roll == pytest.approx(0, abs=1e-4)


def test_nme_uses_interocular_normalization():
    expected = _points()
    predicted = expected.copy()
    predicted[:, 0] += 2
    assert normalized_mean_error(expected, predicted) == pytest.approx(2 / 40)
