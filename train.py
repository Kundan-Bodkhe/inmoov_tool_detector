#!/usr/bin/env python3
"""
train.py - Train a YOLO model on the mechanical-tool dataset.

Usage:
    python train.py
    python train.py --epochs 200 --batch 32 --imgsz 640
    python train.py --data dataset/data.yaml --weights yolo11s.pt --device 0
    python train.py --skip-validation      # skip the pre-training dataset check

The training run is delegated to Ultralytics, which already saves best.pt,
last.pt, loss/precision/recall/mAP curves, and a confusion matrix under
runs/detect/<name>/. This script wraps that call with dataset validation,
structured logging, sensible defaults from config.py, and clear error
handling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.config import config
from utils.logger import get_logger
from utils.device import get_device, describe_device
from scripts.validate_dataset import validate_dataset

logger = get_logger("training", "training.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the mechanical tool dataset.")
    parser.add_argument("--data", type=str, default=str(config.data_yaml), help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default=config.pretrained_weights, help="Pretrained weights to start from")
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch", type=int, default=config.batch_size)
    parser.add_argument("--imgsz", type=int, default=config.image_size)
    parser.add_argument("--optimizer", type=str, default=config.optimizer)
    parser.add_argument("--lr0", type=float, default=config.learning_rate)
    parser.add_argument("--patience", type=int, default=config.patience)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument("--name", type=str, default="tool_detector", help="Run name under runs/detect/")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the pre-training dataset validation step")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint of this run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Starting training run")
    logger.info("=" * 70)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"data.yaml not found at {data_yaml}. Populate dataset/data.yaml and try again.")
        sys.exit(1)

    if not args.skip_validation:
        logger.info("Running dataset validation before training...")
        report = validate_dataset(data_yaml=data_yaml)
        if not report.is_valid:
            logger.error(
                "Dataset validation found issues. Fix them or re-run with --skip-validation "
                "to proceed anyway. See reports/dataset_validation_report.txt for details."
            )
            sys.exit(1)
        logger.info("Dataset validation passed.")
    else:
        logger.warning("Skipping dataset validation (--skip-validation set).")

    device = get_device(args.device)
    logger.info(f"Using device: {describe_device(device)}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        if args.resume:
            checkpoint = config.runs_dir / "detect" / args.name / "weights" / "last.pt"
            if not checkpoint.exists():
                logger.error(f"Checkpoint not found: {checkpoint}")
                sys.exit(1)

            logger.info(f"Resuming training from checkpoint: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            logger.info(f"Loading pretrained weights: {args.weights}")
            model = YOLO(args.weights)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=config.lr_final_factor,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=str(config.runs_dir / "detect"),
        name=args.name,
        seed=config.seed,
        resume=args.resume,
        exist_ok=True,
        plots=True,
        augment=not args.no_augment,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        shear=config.shear,
        flipud=config.flipud,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        mixup=config.mixup,
    )

    logger.info(f"Training config: {train_kwargs}")

    try:
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C). Partial checkpoints saved under runs/.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    logger.info(f"Training complete. Results saved to: {run_dir}")
    logger.info(f"  best.pt -> {best}")
    logger.info(f"  last.pt -> {last}")

    # Convenience copy into models/ so downstream scripts (evaluate.py,
    # webcam_detect.py) find it via the default config.weights_path.
    if best.exists():
        config.models_dir.mkdir(parents=True, exist_ok=True)
        dest = config.models_dir / "best.pt"
        dest.write_bytes(best.read_bytes())
        logger.info(f"Copied best.pt to {dest} (default inference weights path).")

    logger.info(
        "Next steps:\n"
        f"  Evaluate : python evaluate.py --weights {best}\n"
        f"  Webcam   : python webcam_detect.py --weights {best}"
    )


if __name__ == "__main__":
    main()
