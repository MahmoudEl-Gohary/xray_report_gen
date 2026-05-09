import json
import logging
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
                {"from": "user", "value": "Analyze this."},
                {"from": "assistant", "value": "Normal finding."},
            ],
        },
        {
            "id": "study_002",
            "images": ["missing.png"],
            "conversations": [
                {"from": "user", "value": "Analyze this."},
                {"from": "assistant", "value": "Will be skipped."},
            ],
        },
    ]

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    reader = ManifestReader(
        manifest_path=manifest_path,
        image_dir=image_dir,
        dataset_name="test_db",
        split="train",
    )

    records = list(reader.read_all())

    # Only study_001 should load; study_002 has a missing image
    assert len(records) == 1
    assert records[0].study_id == "study_001"
    assert records[0].report_html == "Normal finding."
    assert len(records[0].image_paths) == 1


def test_manifest_reader_nested_paths(tmp_path: Path):
    """Test that nested paths in manifests resolve via basename extraction."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    img = image_dir / "abc123.png"
    img.touch()

    manifest_path = tmp_path / "dataset.json"
    manifest_data = [
        {
            "id": "study_nested",
            # Spine-style: nested path in manifest
            "images": ["vindr-spinexr/train_images/abc123.png"],
            "conversations": [
                {"from": "system", "value": "You are a radiologist."},
                {"from": "user", "value": "<image> Analyze."},
                {"from": "assistant", "value": "Normal spine."},
            ],
        }
    ]

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    reader = ManifestReader(
        manifest_path=manifest_path,
        image_dir=image_dir,
        dataset_name="spine",
        split="train",
    )

    records = list(reader.read_all())
    assert len(records) == 1
    assert records[0].study_id == "study_nested"
    assert records[0].image_paths[0] == img


def test_manifest_reader_dicom_to_png_fallback(tmp_path: Path):
    """Test .dicom -> .png fallback resolution."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    png = image_dir / "scan001.png"
    png.touch()

    manifest_path = tmp_path / "dataset.json"
    manifest_data = [
        {
            "id": "study_dicom",
            "images": ["scan001.dicom"],
            "conversations": [
                {"from": "user", "value": "Analyze."},
                {"from": "assistant", "value": "Normal."},
            ],
        }
    ]

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    reader = ManifestReader(
        manifest_path=manifest_path,
        image_dir=image_dir,
        dataset_name="test",
        split="train",
    )

    records = list(reader.read_all())
    assert len(records) == 1
    assert records[0].image_paths[0] == png


def test_manifest_reader_missing_file(tmp_path: Path, caplog):
    """Test that a missing manifest logs an error and yields nothing."""
    reader = ManifestReader(
        manifest_path=tmp_path / "nonexistent.json",
        image_dir=tmp_path,
        dataset_name="test",
        split="train",
    )

    with caplog.at_level(logging.ERROR):
        records = list(reader.read_all())

    assert len(records) == 0
    assert "Manifest not found" in caplog.text