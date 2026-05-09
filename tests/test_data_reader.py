import json
from pathlib import Path

from xray_pipeline.core.data_reader import ManifestReader


def test_manifest_reader(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create dummy image files
    img1 = image_dir / "chest1.png"
    img1.touch()
    
    manifest_path = tmp_path / "dataset.json"
    manifest_data = [
        {
            "id": "study_001",
            "images": ["chest1.png"],
            "conversations": [
                {"from": "human", "value": "Analyze this."},
                {"from": "assistant", "value": "Normal finding."}
            ]
        },
        {
            "id": "study_002",
            "images": ["missing.png"],
            "conversations": [
                {"from": "human", "value": "Analyze this."},
                {"from": "assistant", "value": "Will be skipped."}
            ]
        }
    ]
    
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    reader = ManifestReader(
        manifest_path=manifest_path,
        image_dir=image_dir,
        dataset_name="test_db",
        split="train"
    )
    
    records = list(reader.read_all())

    # Only study_001 should be loaded because study_002 has a missing image
    assert len(records) == 1
    assert records[0].study_id == "study_001"
    assert records[0].report_html == "Normal finding."
    assert len(records[0].image_paths) == 1