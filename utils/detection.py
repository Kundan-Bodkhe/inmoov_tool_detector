"""
Detection data model and helpers.

This module defines the Detection dataclass — the structured object that
downstream robotics code (arm inverse kinematics, grasp planning, ROS2
publishers, etc.) will consume. It also provides helpers to convert raw
Ultralytics Results objects into a list of Detection instances, and to draw
them onto a frame with OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


@dataclass
class Detection:
    """
    A single detected tool, with everything a robotic arm / grasp planner
    would need downstream.

    Coordinates are in pixel space of the source frame.
    """

    class_name: str
    class_id: int
    confidence: float  # 0.0 - 1.0

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    width: int = 0
    height: int = 0
    center_x: int = 0
    center_y: int = 0
    area: int = 0

    def __post_init__(self) -> None:
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.center_x = self.xmin + self.width // 2
        self.center_y = self.ymin + self.height // 2
        self.area = self.width * self.height

    @property
    def confidence_pct(self) -> float:
        """Confidence expressed as a percentage, e.g. 98.5."""
        return round(self.confidence * 100, 1)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.class_name} ({self.confidence_pct}%) "
            f"bbox=({self.xmin},{self.ymin},{self.xmax},{self.ymax}) "
            f"center=({self.center_x},{self.center_y}) area={self.area}"
        )


def results_to_detections(results: "Results", class_names: dict) -> List[Detection]:
    """
    Convert a single Ultralytics Results object (one frame) into a list of
    Detection instances. Assumes NMS and confidence filtering have already
    been applied by the model call (conf=..., iou=... passed to predict()).
    """
    detections: List[Detection] = []
    if results.boxes is None:
        return detections

    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        detections.append(
            Detection(
                class_name=class_names.get(int(k), str(int(k))),
                class_id=int(k),
                confidence=float(c),
                xmin=int(round(x1)),
                ymin=int(round(y1)),
                xmax=int(round(x2)),
                ymax=int(round(y2)),
            )
        )
    return detections


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    get_color,
    box_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores onto a frame in-place."""
    for det in detections:
        color = get_color(det.class_name)
        cv2.rectangle(frame, (det.xmin, det.ymin), (det.xmax, det.ymax), color, box_thickness)

        label = f"{det.class_name} {det.confidence_pct}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        label_y = max(det.ymin - 8, text_h + 8)
        cv2.rectangle(
            frame,
            (det.xmin, label_y - text_h - baseline - 4),
            (det.xmin + text_w + 4, label_y + baseline - 4),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (det.xmin + 2, label_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (det.center_x, det.center_y), 3, color, -1)

    return frame


def draw_overlay_text(frame: np.ndarray, lines: List[str], origin=(10, 25), line_gap: int = 26) -> np.ndarray:
    """Draw a small stack of status text lines (FPS, inference time, device...) top-left."""
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame
