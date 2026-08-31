"""Adapters MediaPipe Face Mesh e YOLO26-Pose para o contrato G4.7."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .g47_schema import mediapipe_to_schema


@dataclass(frozen=True)
class LandmarkDetection:
    points: np.ndarray | None
    inference_ms: float
    confidence: float
    failure: str = ""


def _cuda_synchronize(device: str) -> None:
    if not str(device).startswith(("cuda", "0")):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MediaPipeFaceMeshExtractor:
    name = "mediapipe_face_mesh"

    def __init__(self, *, minimum_confidence: float = 0.5) -> None:
        import mediapipe as mp

        self.minimum_confidence = minimum_confidence
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=minimum_confidence,
            min_tracking_confidence=minimum_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def detect(self, bgr_frame: np.ndarray) -> LandmarkDetection:
        height, width = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        started = time.perf_counter_ns()
        result = self._mesh.process(rgb)
        inference_ms = (time.perf_counter_ns() - started) / 1e6
        if not result.multi_face_landmarks:
            return LandmarkDetection(None, inference_ms, 0.0, "face_missing")
        points = mediapipe_to_schema(result.multi_face_landmarks[0].landmark, width, height)
        return LandmarkDetection(points, inference_ms, 1.0)

    def __enter__(self) -> "MediaPipeFaceMeshExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class YOLOFacePoseExtractor:
    name = "yolo26_face_pose"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        image_size: int = 640,
        minimum_confidence: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Peso YOLO facial não encontrado: {self.model_path}")
        self.device = device
        self.image_size = image_size
        self.minimum_confidence = minimum_confidence
        self._model = YOLO(str(self.model_path))

    def close(self) -> None:
        return None

    def detect(self, bgr_frame: np.ndarray) -> LandmarkDetection:
        _cuda_synchronize(self.device)
        started = time.perf_counter_ns()
        results = self._model.predict(
            source=bgr_frame,
            device=self.device,
            imgsz=self.image_size,
            conf=self.minimum_confidence,
            max_det=1,
            verbose=False,
        )
        _cuda_synchronize(self.device)
        inference_ms = (time.perf_counter_ns() - started) / 1e6
        if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return LandmarkDetection(None, inference_ms, 0.0, "face_missing")
        data: Any = results[0].keypoints.data[0].detach().cpu().numpy()
        if data.shape[0] != 22 or data.shape[1] not in {2, 3}:
            return LandmarkDetection(
                None,
                inference_ms,
                0.0,
                f"invalid_keypoint_shape:{tuple(data.shape)}",
            )
        if data.shape[1] == 2:
            data = np.column_stack((data, np.ones(22, dtype=float)))
        confidence = float(np.nanmean(data[:, 2]))
        return LandmarkDetection(np.asarray(data, dtype=float), inference_ms, confidence)

    def __enter__(self) -> "YOLOFacePoseExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
