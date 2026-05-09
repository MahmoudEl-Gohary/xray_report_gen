import json
import logging
from pathlib import Path
from typing import Iterator

from xray_pipeline.core.schema import StudyRecord

logger = logging.getLogger(__name__)


class ManifestReader:
    """Reads a JSON manifest and yields validated StudyRecord objects.

    Args:
        manifest_path: Path to the JSON manifest file.
        image_dir: Directory containing the image files.
        dataset_name: Logical name of the dataset (e.g., "spine", "knee").
        split: Data split identifier (e.g., "train", "test").
    """

    def __init__(
        self,
        manifest_path: Path,
        image_dir: Path,
        dataset_name: str,
        split: str,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_dir = image_dir
        self.dataset_name = dataset_name
        self.split = split

    def read_all(self) -> Iterator[StudyRecord]:
        """Parse manifest and yield StudyRecords with resolved image paths.

        Studies are skipped (with a warning) if no valid images are found
        or if the assistant report is missing.
        """
        if not self.manifest_path.exists():
            logger.error(
                "Manifest not found: %s", self.manifest_path
            )
            return

        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data)
        loaded = 0
        skipped_no_images = 0
        skipped_no_report = 0

        for entry in data:
            study_id = entry.get("id", "unknown_id")

            # Handle "image" (Knee format) or "images" (Spine format)
            raw_images = entry.get("image") or entry.get("images") or []
            if isinstance(raw_images, str):
                raw_images = [raw_images]

            # Resolve each image path
            resolved_paths = []
            for img_name in raw_images:
                # Use only the basename so nested paths in manifests
                # (e.g., "vindr-spinexr-.../train_images/xxx.dicom")
                # resolve against image_dir correctly.
                basename = Path(img_name).name
                img_path = self.image_dir / basename

                # If manifest says .dicom/.dcm but images were preprocessed to .png
                if not img_path.exists() and basename.lower().endswith(
                    (".dicom", ".dcm")
                ):
                    png_name = basename.rsplit(".", 1)[0] + ".png"
                    png_path = self.image_dir / png_name
                    if png_path.exists():
                        img_path = png_path

                if img_path.exists():
                    resolved_paths.append(img_path)
                else:
                    logger.debug(
                        "Image not found for study %s: %s", study_id, img_path
                    )

            if not resolved_paths:
                skipped_no_images += 1
                logger.warning(
                    "Skipping study %s: no valid images found", study_id
                )
                continue

            # Extract conversation roles
            conversations = entry.get("conversations", [])
            system_prompt, user_prompt, report_html = "", "", ""

            for turn in conversations:
                role = turn.get("from") or turn.get("role")
                val = turn.get("value") or turn.get("content") or ""

                if role == "system":
                    system_prompt = val
                elif role in ("user", "human"):
                    # Remove <image> tags; Unsloth/Qwen handles vision tokens
                    user_prompt = val.replace("<image>", "").strip()
                elif role in ("gpt", "assistant"):
                    report_html = val

            if not report_html:
                skipped_no_report += 1
                logger.warning(
                    "Skipping study %s: no assistant report in conversations",
                    study_id,
                )
                continue

            loaded += 1
            yield StudyRecord(
                study_id=study_id,
                image_paths=resolved_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                report_html=report_html,
                dataset_name=self.dataset_name,
                split=self.split,
            )

        logger.info(
            "[%s/%s] Loaded %d/%d studies (skipped: %d no images, %d no report)",
            self.dataset_name,
            self.split,
            loaded,
            total,
            skipped_no_images,
            skipped_no_report,
        )