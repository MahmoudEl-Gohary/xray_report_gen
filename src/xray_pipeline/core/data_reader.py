import json
from pathlib import Path
from typing import Iterator

from xray_pipeline.core.schema import StudyRecord


class ManifestReader:
    """Reads ShareGPT-style JSON manifests and yields StudyRecords."""

    def __init__(
        self,
        manifest_path: Path,
        image_dir: Path,
        dataset_name: str,
        split: str
    ) -> None:
        self.manifest_path = manifest_path
        self.image_dir = image_dir
        self.dataset_name = dataset_name
        self.split = split

    def read_all(self) -> Iterator[StudyRecord]:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            study_id = entry.get("id", "unknown_id")
            
            # Handle images (could be a string or a list)
            raw_images = entry.get("image") or entry.get("images") or []
            if isinstance(raw_images, str):
                raw_images = [raw_images]

            # Validate image existence
            resolved_paths = []
            for img_name in raw_images:
                img_path = self.image_dir / img_name
                if img_path.exists():
                    resolved_paths.append(img_path)

            if not resolved_paths:
                continue

            # Extract the ground truth report
            conversations = entry.get("conversations", [])
            report_html = ""
            for turn in conversations:
                role = turn.get("from") or turn.get("role")
                if role in ("gpt", "assistant"):
                    report_html = turn.get("value") or turn.get("content") or ""
                    break

            if not report_html:
                continue

            yield StudyRecord(
                study_id=study_id,
                image_paths=resolved_paths,
                report_html=report_html,
                dataset_name=self.dataset_name,
                split=self.split
            )