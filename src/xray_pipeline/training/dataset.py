from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset

from xray_pipeline.core.schema import StudyRecord
from xray_pipeline.core.data_reader import ManifestReader

class RadiologyDataset(Dataset):
    """Formats radiology studies into Unsloth-compatible messages."""

    def __init__(self, manifest_path: Path, image_dir: Path, dataset_name: str, split: str = "train") -> None:
        reader = ManifestReader(
            manifest_path=manifest_path,
            image_dir=image_dir,
            dataset_name=dataset_name,
            split=split
        )
        self.records: List[StudyRecord] = list(reader.read_all())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        
        user_content = []
        # Append all images 
        for img_path in record.image_paths:
            user_content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
            
        # Append the specific text prompt for this image
        user_content.append({"type": "text", "text": record.user_prompt})
        
        # Build Unsloth/Qwen strict conversation format
        messages = []
        if record.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": record.system_prompt}]})
            
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": record.report_html}]})
        
        return {"messages": messages}