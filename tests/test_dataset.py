import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xray_pipeline.training.dataset import RadiologyDataset


def test_radiology_dataset_and_loader(tmp_path: Path):
    # 1. Setup Dummy Data
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "chest1.png").touch()
    (image_dir / "chest2.png").touch()
    
    manifest_path = tmp_path / "dataset.json"
    manifest_data = [
        {
            "id": "study_001",
            "images": ["chest1.png"],
            "conversations": [{"from": "gpt", "value": "Report 1"}]
        },
        {
            "id": "study_002",
            "images": ["chest2.png"],
            "conversations": [{"from": "gpt", "value": "Report 2"}]
        }
    ]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # 2. Initialize Dataset
    dataset = RadiologyDataset(
        manifest_path=manifest_path,
        image_dir=image_dir,
        dataset_name="mimic",
        split="train"
    )

    assert len(dataset) == 2
    assert dataset[0].study_id == "study_001"
    
    # 3. Test with PyTorch DataLoader (custom collate needed to bypass default tensor stacking)
    def simple_collate(batch):
        return batch
        
    loader = DataLoader(dataset, batch_size=2, collate_fn=simple_collate)
    batch = next(iter(loader))
    
    assert len(batch) == 2
    assert batch[1].report_html == "Report 2"