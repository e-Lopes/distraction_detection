import math

import numpy as np
import pytest

from fase_2.src.evaluation.g48a_contract import (
    empty_result,
    map_to_canonical,
    OPENFACE_MAPPING,
    select_operator_face,
)


def test_g48a_empty_result_uses_nan_and_validity_masks():
    result = empty_result(
        video_id="video_01",
        frame_id=1,
        timestamp_ms=10.0,
        extractor="test",
        extractor_version="1",
        inference_time_ms=2.0,
        failure_reason="no_face",
    )
    assert not result.face_detected
    assert not result.ear_valid and not result.mar_valid and not result.head_pose_valid
    assert math.isnan(result.ear) and result.failure_reason == "no_face"


def test_g48a_mapping_adds_visibility_and_checks_indices():
    points = np.arange(60, dtype=float).reshape(30, 2)
    mapped = map_to_canonical(points, list(range(22)))
    assert mapped.shape == (22, 3)
    np.testing.assert_array_equal(mapped[:, 2], 1)
    with pytest.raises(ValueError, match="22 índices"):
        map_to_canonical(points, [0])


def test_g48a_operator_selection_uses_roi_anchor_not_detection_order():
    boxes = np.asarray([[10, 10, 80, 100], [300, 70, 420, 220]], dtype=float)
    assert select_operator_face(boxes, width=538, height=491) == 1


def test_g48a_operator_selection_rejects_only_background_face():
    boxes = np.asarray([[195, -8, 255, 62]], dtype=float)
    with pytest.raises(ValueError, match="região espacial plausível"):
        select_operator_face(boxes, width=538, height=491)


def test_openface_mapping_uses_inner_lip_contour_for_mar():
    assert OPENFACE_MAPPING[12:20] == (64, 63, 62, 61, 60, 67, 66, 65)
