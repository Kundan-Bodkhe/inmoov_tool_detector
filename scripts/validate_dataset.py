"""
Dataset validation for a YOLOv8-format object detection dataset.

Checks performed:
  - every image has a matching label file (and vice versa)
  - label files parse correctly (right number of columns, numeric values)
  - class IDs fall within the range declared in data.yaml
  - bounding boxes are normalized (0-1) and inside image boundaries
  - duplicate images (identical content, via MD5 hash)
  - corrupted / unreadable images

Produces a human-readable report (reports/dataset_validation_report.txt)
and returns a structured summary dict. Run standalone:

    python scripts/validate_dataset.py

or import `validate_dataset(...)` from train.py before training starts.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("dataset_validation", "dataset_validation.log")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ValidationReport:
    total_images: int = 0
    total_labels: int = 0
    images_missing_labels: List[str] = field(default_factory=list)
    labels_missing_images: List[str] = field(default_factory=list)
    corrupted_images: List[str] = field(default_factory=list)
    corrupted_labels: List[str] = field(default_factory=list)
    invalid_class_ids: List[str] = field(default_factory=list)
    out_of_bounds_boxes: List[str] = field(default_factory=list)
    duplicate_images: List[List[str]] = field(default_factory=list)
    empty_labels: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(
            [
                self.images_missing_labels,
                self.labels_missing_images,
                self.corrupted_images,
                self.corrupted_labels,
                self.invalid_class_ids,
                self.out_of_bounds_boxes,
            ]
        )

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "DATASET VALIDATION REPORT",
            "=" * 70,
            f"Total images found       : {self.total_images}",
            f"Total label files found  : {self.total_labels}",
            f"Images missing labels    : {len(self.images_missing_labels)}",
            f"Labels missing images    : {len(self.labels_missing_images)}",
            f"Corrupted images         : {len(self.corrupted_images)}",
            f"Corrupted label files    : {len(self.corrupted_labels)}",
            f"Invalid class IDs        : {len(self.invalid_class_ids)}",
            f"Out-of-bounds boxes      : {len(self.out_of_bounds_boxes)}",
            f"Empty label files        : {len(self.empty_labels)}",
            f"Duplicate image groups   : {len(self.duplicate_images)}",
            "-" * 70,
            f"RESULT: {'PASS - dataset looks valid' if self.is_valid else 'FAIL - issues found, see details below'}",
            "=" * 70,
        ]

        def _section(title: str, items: List[str], limit: int = 25) -> None:
            if not items:
                return
            lines.append(f"\n[{title}] ({len(items)} total, showing up to {limit})")
            for item in items[:limit]:
                lines.append(f"  - {item}")

        _section("Images missing labels", self.images_missing_labels)
        _section("Labels missing images", self.labels_missing_images)
        _section("Corrupted images", self.corrupted_images)
        _section("Corrupted label files", self.corrupted_labels)
        _section("Invalid class IDs", self.invalid_class_ids)
        _section("Out-of-bounds boxes", self.out_of_bounds_boxes)
        _section("Empty label files", self.empty_labels)

        if self.duplicate_images:
            lines.append(f"\n[Duplicate image groups] ({len(self.duplicate_images)} total)")
            for group in self.duplicate_images[:25]:
                lines.append(f"  - {group}")

        return "\n".join(lines)


def _md5(path: Path, block_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_num_classes(data_yaml: Path) -> int:
    if not data_yaml.exists():
        logger.warning(f"data.yaml not found at {data_yaml}; class-ID validation will be skipped.")
        return -1
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    return len(names)


def _validate_split(images_dir: Path, labels_dir: Path, num_classes: int, report: ValidationReport, check_duplicates: bool) -> None:
    if not images_dir.exists() or not labels_dir.exists():
        logger.warning(f"Skipping split, missing dir: {images_dir} or {labels_dir}")
        return

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    label_files = sorted([p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"])

    report.total_images += len(image_files)
    report.total_labels += len(label_files)

    label_stems = {p.stem for p in label_files}
    image_stems = {p.stem for p in image_files}

    for img in image_files:
        if img.stem not in label_stems:
            report.images_missing_labels.append(str(img))

    for lbl in label_files:
        if lbl.stem not in image_stems:
            report.labels_missing_images.append(str(lbl))

    hashes: Dict[str, List[str]] = {}

    for img in image_files:
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                width, height = im.size
        except Exception as e:
            report.corrupted_images.append(f"{img} ({e})")
            continue

        if check_duplicates:
            try:
                h = _md5(img)
                hashes.setdefault(h, []).append(str(img))
            except Exception:
                pass

        lbl_path = labels_dir / f"{img.stem}.txt"
        if not lbl_path.exists():
            continue

        try:
            lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        except Exception as e:
            report.corrupted_labels.append(f"{lbl_path} ({e})")
            continue

        if not lines:
            report.empty_labels.append(str(lbl_path))
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                report.corrupted_labels.append(f"{lbl_path}:{line_no} (expected 5 fields, got {len(parts)})")
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h_ = (float(v) for v in parts[1:])
            except ValueError:
                report.corrupted_labels.append(f"{lbl_path}:{line_no} (non-numeric values)")
                continue

            if num_classes >= 0 and not (0 <= cls_id < num_classes):
                report.invalid_class_ids.append(f"{lbl_path}:{line_no} (class_id={cls_id}, num_classes={num_classes})")

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h_ <= 1.0):
                report.out_of_bounds_boxes.append(f"{lbl_path}:{line_no} (cx={cx},cy={cy},w={w},h={h_})")
                continue

            # Check the box actually stays within the image once denormalized
            xmin = (cx - w / 2) * width
            xmax = (cx + w / 2) * width
            ymin = (cy - h_ / 2) * height
            ymax = (cy + h_ / 2) * height
            if xmin < -1 or ymin < -1 or xmax > width + 1 or ymax > height + 1:
                report.out_of_bounds_boxes.append(f"{lbl_path}:{line_no} (pixel bbox exceeds image size {width}x{height})")

    if check_duplicates:
        for h, paths in hashes.items():
            if len(paths) > 1:
                report.duplicate_images.append(paths)


def validate_dataset(
    dataset_dir: Path = None,
    data_yaml: Path = None,
    check_duplicates: bool = True,
    write_report: bool = True,
) -> ValidationReport:
    dataset_dir = dataset_dir or config.dataset_dir
    data_yaml = data_yaml or config.data_yaml

    logger.info(f"Validating dataset at {dataset_dir}")
    num_classes = _load_num_classes(data_yaml)

    report = ValidationReport()

    splits_found = False
    for split in ("train", "val", "test"):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if images_dir.exists():
            splits_found = True
            logger.info(f"Validating split '{split}'...")
            _validate_split(images_dir, labels_dir, num_classes, report, check_duplicates)

    if not splits_found:
        # Fall back to flat images/ and labels/ (no train/val subfolders)
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if images_dir.exists():
            logger.info("No train/val/test subfolders found; validating flat images/labels structure.")
            _validate_split(images_dir, labels_dir, num_classes, report, check_duplicates)
        else:
            logger.error(f"No images found under {dataset_dir}. Expected images/ or images/train, images/val, etc.")

    summary_text = report.summary()
    logger.info("\n" + summary_text)

    if write_report:
        config.reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = config.reports_dir / "dataset_validation_report.txt"
        out_path.write_text(summary_text)
        logger.info(f"Report written to {out_path}")

    return report


if __name__ == "__main__":
    result = validate_dataset()
    sys.exit(0 if result.is_valid else 1)
