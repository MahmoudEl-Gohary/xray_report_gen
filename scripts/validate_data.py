"""Validate that all images referenced in dataset manifests actually exist.

Reports statistics per dataset and flags missing files so you can fix
data issues before starting a multi-hour training run.

Usage:
    python scripts/validate_data.py --config configs/base.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from xray_pipeline.core.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_manifest(
    manifest_path: Path,
    image_dir: Path,
) -> Tuple[int, int, int, List[str]]:
    """Validate a single manifest file.

    Args:
        manifest_path: Path to the JSON manifest.
        image_dir: Directory where images should be located.

    Returns:
        Tuple of (total_studies, valid_studies, total_images, missing_files).
    """
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 0, 0, 0, [str(manifest_path)]

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_studies = len(data)
    valid_studies = 0
    total_images = 0
    missing: List[str] = []

    for entry in data:
        study_id = entry.get("id", "unknown")
        raw_images = entry.get("image") or entry.get("images") or []
        if isinstance(raw_images, str):
            raw_images = [raw_images]

        study_valid = True
        for img_name in raw_images:
            total_images += 1
            basename = Path(img_name).name

            img_path = image_dir / basename
            # Check .dicom -> .png fallback
            if not img_path.exists() and basename.lower().endswith(
                (".dicom", ".dcm")
            ):
                png_name = basename.rsplit(".", 1)[0] + ".png"
                img_path = image_dir / png_name

            if not img_path.exists():
                missing.append(f"{study_id}: {basename}")
                study_valid = False

        if study_valid and raw_images:
            valid_studies += 1

    return total_studies, valid_studies, total_images, missing


def _count_image_files(image_dir: Path) -> Dict[str, int]:
    """Count image files by extension in the image directory.

    Args:
        image_dir: Directory to scan.

    Returns:
        Dict mapping extension to count.
    """
    counts: Dict[str, int] = {}
    if not image_dir.exists():
        return counts

    for f in image_dir.iterdir():
        if f.is_file():
            ext = f.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1
    return counts


def main(config_path: Path) -> None:
    """Run validation across all configured datasets.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = PipelineConfig.from_yaml(config_path)
    all_ok = True

    for name, entry in config.datasets.items():
        image_dir = Path(entry.image_dir)
        logger.info("=" * 60)
        logger.info("Dataset: %s", name)
        logger.info("Image dir: %s", image_dir)

        # Count actual files on disk
        file_counts = _count_image_files(image_dir)
        total_files = sum(file_counts.values())
        logger.info("Files on disk: %d %s", total_files, dict(file_counts))

        # Validate train manifest
        for split, manifest_path in [
            ("train", Path(entry.train_manifest)),
            ("test", Path(entry.test_manifest)),
        ]:
            total, valid, imgs, missing = _validate_manifest(
                manifest_path, image_dir
            )
            status = "OK" if not missing else "ISSUES"
            logger.info(
                "  [%s] %s: %d/%d studies valid, %d images referenced",
                status,
                split,
                valid,
                total,
                imgs,
            )
            if missing:
                all_ok = False
                # Show first 10 missing files
                for m in missing[:10]:
                    logger.warning("    MISSING: %s", m)
                if len(missing) > 10:
                    logger.warning(
                        "    ... and %d more missing files", len(missing) - 10
                    )

    logger.info("=" * 60)
    if all_ok:
        logger.info("All datasets validated successfully.")
    else:
        logger.warning(
            "Some datasets have missing images. Fix before training."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate dataset images and manifests"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
