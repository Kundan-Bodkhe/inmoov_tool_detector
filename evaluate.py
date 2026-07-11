#!/usr/bin/env python3
"""
evaluate.py - Evaluate a trained YOLO model on the validation split.

Usage:
    python evaluate.py
    python evaluate.py --weights runs/detect/tool_detector/weights/best.pt
    python evaluate.py --data dataset/data.yaml --imgsz 640 --device cpu

Prints Precision, Recall, mAP50, mAP50-95, F1, average inference speed,
and GPU memory usage (if applicable). Also writes a text summary to
reports/evaluation_report.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config.config import config
from utils.logger import get_logger
from utils.device import get_device, describe_device, gpu_memory_usage

logger = get_logger("evaluation", "evaluation.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO tool-detection model.")
    parser.add_argument("--weights", type=str, default=str(config.weights_path), help="Path to trained weights (best.pt)")
    parser.add_argument("--data", type=str, default=str(config.data_yaml), help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=config.image_size)
    parser.add_argument("--batch", type=int, default=config.batch_size)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--conf", type=float, default=config.confidence_threshold)
    parser.add_argument("--iou", type=float, default=config.iou_threshold)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        logger.error(
            f"Weights not found at {weights_path}. Train a model first with train.py, "
            "or pass --weights pointing to a valid .pt file."
        )
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}.")
        sys.exit(1)

    device = get_device(args.device)
    logger.info(f"Using device: {describe_device(device)}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Failed to load weights '{weights_path}': {e}")
        sys.exit(1)

    logger.info(f"Evaluating {weights_path} on split='{args.split}' ...")
    start = time.time()
    try:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            conf=args.conf,
            iou=args.iou,
            split=args.split,
            project=str(config.runs_dir / "val"),
            name="eval",
            exist_ok=True,
            plots=True,
        )
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)
    elapsed = time.time() - start

    precision = float(metrics.box.mp)         # mean precision across classes
    recall = float(metrics.box.mr)            # mean recall across classes
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    speed = metrics.speed  # dict: preprocess, inference, loss, postprocess (ms/image)
    inference_ms = speed.get("inference", 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    allocated, reserved = gpu_memory_usage()

    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"Weights            : {weights_path}",
        f"Split              : {args.split}",
        f"Device             : {describe_device(device)}",
        f"Precision          : {precision:.4f}",
        f"Recall             : {recall:.4f}",
        f"mAP50              : {map50:.4f}",
        f"mAP50-95           : {map50_95:.4f}",
        f"F1 Score           : {f1:.4f}",
        f"Avg inference time : {inference_ms:.2f} ms/image",
        f"Avg FPS            : {fps:.2f}",
        f"GPU mem allocated  : {allocated:.2f} GB",
        f"GPU mem reserved   : {reserved:.2f} GB",
        f"Total eval time    : {elapsed:.2f} s",
        "=" * 60,
    ]

    # Per-class breakdown
    try:
        class_names = metrics.names
        maps = metrics.box.maps  # mAP50-95 per class
        lines.append("\nPer-class mAP50-95:")
        for idx, class_map in enumerate(maps):
            lines.append(f"  {class_names.get(idx, idx):<20s}: {class_map:.4f}")
    except Exception:
        pass

    report_text = "\n".join(lines)
    logger.info("\n" + report_text)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.reports_dir / "evaluation_report.txt"
    out_path.write_text(report_text)
    logger.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
