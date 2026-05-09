import json
from pathlib import Path
from PIL import Image

from xray_pipeline.training.dataset import RadiologyDataset

def test_unsloth_dataset_formatting(tmp_path: Path):
    # 1. Setup Dummy Data
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create a valid dummy image so PIL can open it
    img_path = image_dir / "chest1.png"
    Image.new("RGB", (10, 10)).save(img_path)
    
    manifest_path = tmp_path / "dataset.json"
    manifest_data = [
        {
            "id": "study_001",
            "images": ["chest1.png"],
            "conversations": [
                {"from": "system", "value": "You are a radiologist."},
                {"from": "user", "value": "<image> Analyze this."},
                {"from": "assistant", "value": "<p>Normal chest</p>"}
            ]
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

    # 3. Validate Unsloth Format
    assert len(dataset) == 1
    item = dataset[0]
    
    assert "messages" in item
    messages = item["messages"]
    
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    
    # Validate the image is loaded correctly
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image"
    assert isinstance(user_content[0]["image"], Image.Image)