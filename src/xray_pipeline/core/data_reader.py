import json
from pathlib import Path
from typing import Iterator

from xray_pipeline.core.schema import StudyRecord

class ManifestReader:
    def __init__(self, manifest_path: Path, image_dir: Path, dataset_name: str, split: str) -> None:
        self.manifest_path = manifest_path
        self.image_dir = image_dir
        self.dataset_name = dataset_name
        self.split = split

    def read_all(self) -> Iterator[StudyRecord]:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            study_id = entry.get("id", "unknown_id")
            
            # Handle "image" (Knee) or "images" (Spine)
            raw_images = entry.get("image") or entry.get("images") or []
            if isinstance(raw_images, str):
                raw_images = [raw_images]

            # Validate and map image existence (Handles .dicom -> .png mapping)
            resolved_paths = []
            for img_name in raw_images:
                img_path = self.image_dir / img_name
                
                # If JSON says .dicom but we preprocessed to .png offline
                if not img_path.exists() and (img_name.endswith(".dicom") or img_name.endswith(".dcm")):
                    png_name = img_name.rsplit(".", 1)[0] + ".png"
                    png_path = self.image_dir / png_name
                    if png_path.exists():
                        img_path = png_path

                if img_path.exists():
                    resolved_paths.append(img_path)

            if not resolved_paths:
                continue

            # Extract exact conversation roles
            conversations = entry.get("conversations", [])
            system_prompt, user_prompt, report_html = "", "", ""
            
            for turn in conversations:
                role = turn.get("from") or turn.get("role")
                val = turn.get("value") or turn.get("content") or ""
                
                if role == "system":
                    system_prompt = val
                elif role == "user":
                    # Remove the <image> tags, Unsloth/Qwen handles vision tokens automatically
                    user_prompt = val.replace("<image>", "").strip() 
                elif role in ("gpt", "assistant"):
                    report_html = val

            if not report_html:
                continue

            yield StudyRecord(
                study_id=study_id,
                image_paths=resolved_paths,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                report_html=report_html,
                dataset_name=self.dataset_name,
                split=self.split
            )