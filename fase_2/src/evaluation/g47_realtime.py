"""Visualização sincronizada MediaPipe × YOLO26 N/S/M em vídeo ou câmera."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ..features.extract_facial_series import load_roi
from ..features.g47_extractors import MediaPipeFaceMeshExtractor
from ..features.g47_schema import FacialIndicators, compute_coco_head_pose, compute_indicators

PANEL_SIZE = (640, 520)


def _crop(frame: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return frame
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI inválida")
    if x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError("ROI ultrapassa as dimensões do vídeo")
    return frame[y : y + height, x : x + width]


def _value(value: float) -> str:
    return f"{value:+6.1f}" if math.isfinite(value) else "   N/D"


def _header(
    panel: np.ndarray,
    title: str,
    latency_ms: float,
    fps: float,
    confidence: float | None = None,
) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 68), (18, 18, 18), -1)
    cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    performance = f"{latency_ms:6.1f} ms | {fps:4.1f} FPS"
    if confidence is not None and math.isfinite(confidence):
        performance += f" | pessoa {confidence:.1%}"
    cv2.putText(
        panel,
        performance,
        (14, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 220, 255),
        1,
    )


def _angles(panel: np.ndarray, indicators: FacialIndicators | None, note: str = "") -> None:
    height = panel.shape[0]
    cv2.rectangle(panel, (0, height - 82), (panel.shape[1], height), (18, 18, 18), -1)
    if indicators is None:
        line = "pitch: N/D   yaw: N/D   roll: N/D"
    else:
        line = (
            f"pitch {_value(indicators.pitch)}  yaw {_value(indicators.yaw)}  "
            f"roll {_value(indicators.roll)}"
        )
    cv2.putText(panel, line, (14, height - 49), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (90, 255, 120), 1)
    if note:
        cv2.putText(panel, note, (14, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 190, 255), 1)


def _draw_points(panel: np.ndarray, points: np.ndarray, source_shape: tuple[int, int]) -> None:
    source_height, source_width = source_shape
    scale_x = panel.shape[1] / source_width
    scale_y = panel.shape[0] / source_height
    for x, y, confidence in points:
        if confidence >= 0.5 and np.isfinite((x, y)).all():
            cv2.circle(panel, (round(x * scale_x), round(y * scale_y)), 3, (0, 255, 255), -1)


def _panel(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)


def _highest_confidence_detection(result: object) -> tuple[object, float | None]:
    """Mantém somente a pessoa cuja bounding box tem a maior confiança."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return result, None
    confidences = boxes.conf.detach().cpu().numpy().reshape(-1)
    selected_index = int(np.nanargmax(confidences))
    return result[selected_index], float(confidences[selected_index])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--roi-config", type=Path)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.05,
        help="Confiança mínima dos cinco pontos cefálicos COCO",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, help="Grava o painel em MP4")
    parser.add_argument("--no-display", action="store_true", help="Somente grava; não abre janela")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        args.model = [Path(f"fase_2/yolo_models/yolo26{scale}-pose.pt") for scale in "nsm"]
    if len(args.model) != 3:
        raise ValueError("Informe exatamente três pesos YOLO com --model (N, S e M)")
    if args.no_display and args.output is None:
        raise ValueError("--no-display exige --output")

    from ultralytics import YOLO

    models = [(path.stem, YOLO(str(path))) for path in args.model]
    shapes = {name: getattr(model.model, "kpt_shape", None) for name, model in models}
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    roi = load_roi(args.roi_config)
    writer: cv2.VideoWriter | None = None
    histories = {"MediaPipe": deque(maxlen=30), **{name: deque(maxlen=30) for name, _ in models}}
    processed = 0
    stop_requested = False

    try:
        with MediaPipeFaceMeshExtractor(minimum_confidence=args.confidence) as media_pipe:
            while args.max_frames is None or processed < args.max_frames:
                success, original = capture.read()
                if not success:
                    break
                frame = _crop(original, roi)
                height, width = frame.shape[:2]

                mp_result = media_pipe.detect(frame)
                histories["MediaPipe"].append(mp_result.inference_ms)
                mp_panel = _panel(frame)
                mp_indicators = None
                if mp_result.points is not None:
                    mp_indicators = compute_indicators(mp_result.points, width, height)
                    _draw_points(mp_panel, mp_result.points, (height, width))
                mp_mean = float(np.mean(histories["MediaPipe"]))
                _header(mp_panel, "MediaPipe Face Mesh", mp_result.inference_ms, 1000.0 / mp_mean)
                _angles(mp_panel, mp_indicators, "pose facial solvePnP (22 pontos)")
                panels = [mp_panel]

                for name, model in models:
                    started = time.perf_counter_ns()
                    raw_result = model.predict(
                        frame,
                        device=args.device,
                        imgsz=args.imgsz,
                        conf=args.confidence,
                        max_det=20,
                        verbose=False,
                    )[0]
                    result, person_confidence = _highest_confidence_detection(raw_result)
                    latency = (time.perf_counter_ns() - started) / 1e6
                    histories[name].append(latency)
                    panel = _panel(result.plot(labels=False, conf=False))
                    indicators = None
                    shape = shapes[name]
                    note = f"COCO corporal {shape}: angulos faciais indisponiveis"
                    if shape == [22, 3] and result.keypoints is not None and len(result.keypoints.data):
                        points = result.keypoints.data[0].detach().cpu().numpy()
                        indicators = compute_indicators(points, width, height)
                        note = "pose facial solvePnP (22 pontos)"
                    elif shape == [17, 3] and result.keypoints is not None and len(result.keypoints.data):
                        points = result.keypoints.data[0].detach().cpu().numpy()
                        pitch, yaw, roll = compute_coco_head_pose(
                            points, width, height, args.keypoint_confidence
                        )
                        indicators = FacialIndicators(
                            math.nan, math.nan, math.nan, math.nan, pitch, yaw, roll
                        )
                        note = "proxy COCO-5: nariz, olhos e orelhas"
                    mean_latency = float(np.mean(histories[name]))
                    _header(
                        panel,
                        name,
                        latency,
                        1000.0 / mean_latency,
                        person_confidence,
                    )
                    _angles(panel, indicators, note)
                    panels.append(panel)

                dashboard = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
                if args.output is not None and writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (dashboard.shape[1], dashboard.shape[0]),
                    )
                    if not writer.isOpened():
                        raise ValueError(f"Não foi possível criar {args.output}")
                if writer is not None:
                    writer.write(dashboard)
                if not args.no_display:
                    cv2.imshow("G4.7 | MediaPipe x YOLO26 | q=sair, espaco=pausar", dashboard)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        paused_key = cv2.waitKey(0) & 0xFF
                        if paused_key == ord("q"):
                            stop_requested = True
                processed += 1
                if stop_requested:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Visualização concluída: {processed} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
