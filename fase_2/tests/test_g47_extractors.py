import sys
from types import SimpleNamespace

import pytest

from fase_2.src.features.g47_extractors import YOLOFacePoseExtractor


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch, keypoint_shape: list[int]) -> None:
    class FakeYOLO:
        def __init__(self, _path: str) -> None:
            self.model = SimpleNamespace(kpt_shape=keypoint_shape)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))


def test_yolo_extractor_accepts_coco_body_pose_weights(tmp_path, monkeypatch) -> None:
    model = tmp_path / "yolo26n-pose.pt"
    model.touch()
    _install_fake_ultralytics(monkeypatch, [17, 3])

    extractor = YOLOFacePoseExtractor(model)

    assert extractor.keypoint_shape == (17, 3)


def test_yolo_extractor_accepts_g47_facial_weights(tmp_path, monkeypatch) -> None:
    model = tmp_path / "yolo26n-face-custom.pt"
    model.touch()
    _install_fake_ultralytics(monkeypatch, [22, 3])

    extractor = YOLOFacePoseExtractor(model)

    assert extractor.model_path == model
