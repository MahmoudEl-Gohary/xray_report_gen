"""Preprocess raw datasets for the training pipeline.

This script performs two operations:
1. **DICOM-to-PNG conversion**: Reads raw DICOM files using ``pydicom``,
   normalizes pixel values to 8-bit, and saves as PNG in the target
   ``image_dir``. This is required for the Spine dataset whose raw data
   is in ``.dicom`` format.
2. **Manifest normalization**: Rewrites image paths in the JSON manifest
   to use basenames with ``.png`` extensions so the training pipeline can
   resolve them against ``image_dir``.

This script must be run **offline, before training**. It only needs to
be run once per dataset.

Usage -- Convert DICOMs for the spine dataset:
    python scripts/prepare_manifests.py \\
        --dataset spine \\
        --source-dir /path/to/vindr-spinexr/.../train_images \\
        --config configs/base.yaml

Usage -- Normalize manifest only (images are already PNGs):
    python scripts/prepare_manifests.py \\
        --dataset knee \\
        --config configs/base.yaml \\
        --skip-conversion

Dependencies:
    uv pip install pydicom numpy Pillow
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _dicom_to_png(dicom_path: Path, output_path: Path) -> bool:
    """Convert a single DICOM file to an 8-bit PNG.

    Uses ``pydicom`` to read the pixel array. Applies MONOCHROME1 inversion
    if needed (common in X-rays) and min-max normalization to [0, 255].

    Args:
        dicom_path: Path to the source ``.dicom`` file.
        output_path: Path where the ``.png`` will be saved.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    try:
        import pydicom
    except ImportError:
        logger.error(
            "pydicom is not installed. Run: uv pip install pydicom"
        )
        raise

    try:
        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Invert if MONOCHROME1 (white = low density)
        photometric = getattr(ds, "PhotometricInterpretation", "")
        if photometric == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

        # Min-max normalization to 8-bit range
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax - pmin > 0:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)

        img = Image.fromarray(pixel_array.astype(np.uint8))

        # Convert to RGB if grayscale (some models expect 3 channels)
        if img.mode != "RGB":
            img = img.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return True

    except Exception as e:
        logger.error("Failed to convert %s: %s", dicom_path, e)
        return False


def _find_dicom_file(
    source_dir: Path, filename: str
) -> Optional[Path]:
    """Locate a DICOM file in the source directory, searching recursively.

    Handles cases where the manifest has a nested path like
    ``vindr-spinexr/.../train_images/xxx.dicom`` but the actual file
    is at ``source_dir/xxx.dicom``.

    Args:
        source_dir: Root directory to search for DICOMs.
        filename: The basename of the DICOM file (e.g., ``xxx.dicom``).

    Returns:
        Path to the found file, or None if not found.
    """
    # Direct match
    candidate = source_dir / filename
    if candidate.exists():
        return candidate

    # Try without extension variations
    stem = Path(filename).stem
    for ext in (".dicom", ".dcm", ".DICOM", ".DCM"):
        candidate = source_dir / (stem + ext)
        if candidate.exists():
            return candidate

    # Recursive search (slower, fallback)
    for match in source_dir.rglob(filename):
        return match

    return None


def _convert_dataset_dicoms(
    manifest_path: Path,
    source_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """Convert all DICOMs referenced in a manifest to PNGs.

    Args:
        manifest_path: Path to the JSON manifest file.
        source_dir: Directory containing the raw DICOM files.
        output_dir: Directory where PNGs will be saved.

    Returns:
        Tuple of (converted_count, skipped_count).
    """
    if not manifest_path.exists():
        logger.warning("Manifest not found, skipping: %s", manifest_path)
        return 0, 0

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for entry in data:
        study_id = entry.get("id", "unknown")
        raw_images = entry.get("image") or entry.get("images") or []
        if isinstance(raw_images, str):
            raw_images = [raw_images]

        for img_ref in raw_images:
            basename = Path(img_ref).name
            stem = Path(basename).stem
            png_name = stem + ".png"
            png_path = output_dir / png_name

            # Skip if PNG already exists (idempotent)
            if png_path.exists():
                logger.debug("Already converted, skipping: %s", png_name)
                converted += 1
                continue

            # Only convert if source is a DICOM
            if basename.lower().endswith((".dicom", ".dcm")):
                dicom_file = _find_dicom_file(source_dir, basename)
                if dicom_file is None:
                    logger.warning(
                        "Study %s: DICOM not found in source dir: %s",
                        study_id,
                        basename,
                    )
                    skipped += 1
                    continue

                if _dicom_to_png(dicom_file, png_path):
                    converted += 1
                else:
                    skipped += 1
            else:
                # Source is already PNG/JPG -- copy to output dir
                src_file = _find_dicom_file(source_dir, basename)
                if src_file is not None:
                    import shutil

                    shutil.copy2(src_file, png_path)
                    converted += 1
                else:
                    logger.warning(
                        "Study %s: image not found in source dir: %s",
                        study_id,
                        basename,
                    )
                    skipped += 1

    return converted, skipped


def _normalize_manifest(manifest_path: Path) -> int:
    """Rewrite a manifest so all image paths are basenames with .png extension.

    This modifies the manifest file in-place.

    Args:
        manifest_path: Path to the JSON manifest file.

    Returns:
        Number of entries modified.
    """
    if not manifest_path.exists():
        logger.warning("Manifest not found, skipping: %s", manifest_path)
        return 0

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    modified = 0
    for entry in data:
        for key in ("image", "images"):
            if key not in entry:
                continue

            raw = entry[key]
            if isinstance(raw, str):
                raw = [raw]

            normalized = []
            for img_path in raw:
                basename = Path(img_path).name
                stem = Path(basename).stem

                # Ensure .png extension
                if basename.lower().endswith((".dicom", ".dcm")):
                    basename = stem + ".png"
                elif not basename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    basename = stem + ".png"

                normalized.append(basename)

            original = entry[key] if isinstance(entry[key], list) else [entry[key]]
            if normalized != original:
                modified += 1

            # Preserve original key format (list vs string)
            if isinstance(entry[key], str) and len(normalized) == 1:
                entry[key] = normalized[0]
            else:
                entry[key] = normalized

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return modified


def main() -> None:
    """Entry point for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw datasets: convert DICOMs to PNGs and "
            "normalize manifest image paths."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to process (must match a key in config.datasets)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the raw source images (DICOMs or PNGs). "
            "Required unless --skip-conversion is set."
        ),
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help=(
            "Skip DICOM-to-PNG conversion. Use when images are already "
            "PNGs in the target image_dir. Only normalizes the manifest."
        ),
    )
    args = parser.parse_args()

    # Late import to avoid requiring all deps just for --help
    from xray_pipeline.core.config import PipelineConfig

    config = PipelineConfig.from_yaml(args.config)

    if args.dataset not in config.datasets:
        available = ", ".join(config.datasets.keys())
        raise ValueError(
            f"Dataset '{args.dataset}' not found in config. "
            f"Available: {available}"
        )

    ds_entry = config.datasets[args.dataset]
    image_dir = Path(ds_entry.image_dir)

    # -- Step 1: Convert images (unless skipped) ---------------------------
    if not args.skip_conversion:
        if args.source_dir is None:
            raise ValueError(
                "--source-dir is required when converting images. "
                "Use --skip-conversion if images are already PNGs."
            )

        if not args.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {args.source_dir}"
            )

        logger.info(
            "Converting images for '%s': %s -> %s",
            args.dataset,
            args.source_dir,
            image_dir,
        )

        for split_name, manifest_path in [
            ("train", Path(ds_entry.train_manifest)),
            ("test", Path(ds_entry.test_manifest)),
        ]:
            converted, skipped = _convert_dataset_dicoms(
                manifest_path=manifest_path,
                source_dir=args.source_dir,
                output_dir=image_dir,
            )
            logger.info(
                "  [%s] Converted: %d, Skipped: %d",
                split_name,
                converted,
                skipped,
            )
    else:
        logger.info(
            "Skipping image conversion for '%s' (--skip-conversion)",
            args.dataset,
        )

    # -- Step 2: Normalize manifests ---------------------------------------
    logger.info("Normalizing manifests for '%s'...", args.dataset)
    for split_name, manifest_path in [
        ("train", Path(ds_entry.train_manifest)),
        ("test", Path(ds_entry.test_manifest)),
    ]:
        modified = _normalize_manifest(manifest_path)
        logger.info(
            "  [%s] %d entries normalized in %s",
            split_name,
            modified,
            manifest_path,
        )

    logger.info("Done. Run 'python scripts/validate_data.py --config %s' to verify.", args.config)


if __name__ == "__main__":
    main()
