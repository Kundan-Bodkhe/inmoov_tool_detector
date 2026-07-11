#!/usr/bin/env python3
"""
webcam_detect.py - Real-time mechanical-tool detection from a USB webcam.

This is the perception entry point intended to run on the InMoov robot
(camera mounted in the eye or on the chest). It continuously grabs frames,
runs YOLO inference, filters by confidence, draws results, and prints
structured Detection objects that a future manipulation module (robot arm
IK, grasp planner, ROS2 node) can consume.

Usage:
    python webcam_detect.py
    python webcam_detect.py --camera 1 --conf 0.5
    python webcam_detect.py --weights models/best.pt --device cpu
    python webcam_detect.py --no-display   # headless, just logs detections

Controls (when a display window is open):
    q   - quit
    +/- - increase / decrease confidence threshold
    s   - save current frame to reports/snapshots/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2

from config.config import config
from utils.logger import get_logger
from utils.device import get_device, describe_device
from utils.detection import Detection, results_to_detections, draw_detections, draw_overlay_text

logger = get_logger("camera", "camera.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam tool detection.")
    parser.add_argument("--weights", type=str, default=str(config.weights_path))
    parser.add_argument("--camera", type=int, default=config.camera_index)
    parser.add_argument("--conf", type=float, default=config.confidence_threshold)
    parser.add_argument("--iou", type=float, default=config.iou_threshold)
    parser.add_argument("--imgsz", type=int, default=config.image_size)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--width", type=int, default=config.window_width)
    parser.add_argument("--height", type=int, default=config.window_height)
    parser.add_argument("--no-display", action="store_true", help="Run headless (no OpenCV window); log detections only")
    parser.add_argument("--max-det", type=int, default=config.max_detections)
    return parser.parse_args()


class ToolDetector:
    """
    Wraps model loading + single-frame inference, producing structured
    Detection objects. Designed to be reused later by a ROS2 node or other
    non-webcam callers (e.g. a single still image from a depth camera).
    """

    def __init__(self, weights_path: str, device: str, conf: float, iou: float, imgsz: int, max_det: int):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from e

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. Train a model first (train.py) "
                "or point --weights at a valid .pt file."
            )

        self.device = get_device(device)
        logger.info(f"Loading model {weights_path} on device: {describe_device(self.device)}")
        self.model = YOLO(str(weights_path))
        self.class_names = self.model.names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        logger.info(f"Model loaded. Classes: {list(self.class_names.values())}")

    def infer(self, frame) -> tuple[List[Detection], float]:
        """Run inference on one BGR frame. Returns (detections, inference_ms)."""
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        detections = results_to_detections(results[0], self.class_names)
        return detections, inference_ms


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {index}. Check the camera is connected, "
            "not in use by another application, and try a different --camera index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> None:
    args = parse_args()

    try:
        detector = ToolDetector(
            weights_path=args.weights,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Camera {args.camera} opened successfully. Confidence threshold = {detector.conf}")
    logger.info("Press 'q' to quit, '+'/'-' to adjust confidence, 's' to save a snapshot.")

    snapshot_dir = config.reports_dir / "snapshots"
    frame_count = 0
    fps_smoothed = 0.0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to read frame from camera (camera disconnected?). Stopping.")
                break

            if config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            try:
                detections, inference_ms = detector.infer(frame)
            except Exception as e:
                logger.exception(f"Inference error on frame {frame_count}: {e}")
                detections, inference_ms = [], 0.0

            for det in detections:
                logger.debug(str(det))
                try:
                    print("=" * 60)
                    print(vars(det))
                    if all(hasattr(det, a) for a in ("x1", "y1", "x2", "y2")):
                        cx = (det.x1 + det.x2) / 2
                        cy = (det.y1 + det.y2) / 2
                        print(f"Tool       : {getattr(det, 'class_name', 'Unknown')}")
                        print(f"Confidence : {getattr(det, 'confidence', 0):.2f}")
                        print(f"BBox       : ({det.x1}, {det.y1}, {det.x2}, {det.y2})")
                        print(f"Center     : ({cx:.1f}, {cy:.1f})")
                except Exception:
                    pass

            now = time.time()
            instant_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            fps_smoothed = fps_smoothed * 0.9 + instant_fps * 0.1 if frame_count > 0 else instant_fps
            prev_time = now
            frame_count += 1

            if not args.no_display:
                draw_detections(
                    frame,
                    detections,
                    get_color=config.get_color,
                    box_thickness=config.box_thickness,
                    font_scale=config.font_scale,
                    font_thickness=config.font_thickness,
                )
                overlay = [
                    f"Device: {describe_device(detector.device)}",
                    f"FPS: {fps_smoothed:.1f}" if config.show_fps else "",
                    f"Inference: {inference_ms:.1f} ms" if config.show_inference_time else "",
                    f"Conf thresh: {detector.conf:.2f}",
                    f"Objects: {len(detections)}",
                ]
                overlay = [line for line in overlay if line]
                draw_overlay_text(frame, overlay)

                cv2.imshow(config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed. Exiting.")
                    break
                elif key == ord("+") or key == ord("="):
                    detector.conf = min(0.95, detector.conf + 0.05)
                    logger.info(f"Confidence threshold -> {detector.conf:.2f}")
                elif key == ord("-") or key == ord("_"):
                    detector.conf = max(0.05, detector.conf - 0.05)
                    logger.info(f"Confidence threshold -> {detector.conf:.2f}")
                elif key == ord("s"):
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    out_path = snapshot_dir / f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    logger.info(f"Saved snapshot to {out_path}")
            else:
                # Headless mode: just report periodically
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count}: {len(detections)} detections, {inference_ms:.1f} ms")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Shutting down.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("Camera released. Session ended.")


if __name__ == "__main__":
    main()
