import logging
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from xray_pipeline.core.schema import StudyRecord
from xray_pipeline.core.data_reader import ManifestReader
from xray_pipeline.core.config import PipelineConfig

logger = logging.getLogger(__name__)


class RadiologyDataset(Dataset):
    """Formats radiology studies into Unsloth-compatible messages.

    Each item returns a dict with a single key ``"messages"`` containing
    the system/user/assistant conversation in the format expected by
    ``UnslothVisionDataCollator``.
    """

    def __init__(
        self,
        manifest_path: Path,
        image_dir: Path,
        dataset_name: str,
        split: str = "train",
    ) -> None:
        reader = ManifestReader(
            manifest_path=manifest_path,
            image_dir=image_dir,
            dataset_name=dataset_name,
            split=split,
        )
        self.records: List[StudyRecord] = list(reader.read_all())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        user_content: List[Dict[str, Any]] = []

        # Load images with error handling
        for img_path in record.image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                user_content.append({"type": "image", "image": img})
            except Exception:
                logger.warning(
                    "Failed to open image %s in study %s, skipping image",
                    img_path,
                    record.study_id,
                )

        # Append the text prompt
        user_content.append({"type": "text", "text": record.user_prompt})

        # Build Unsloth/Qwen conversation format
        messages: List[Dict[str, Any]] = []
        if record.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": record.system_prompt}],
                }
            )

        messages.append({"role": "user", "content": user_content})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record.report_html}],
            }
        )

        return {"messages": messages}


def build_training_dataset(config: PipelineConfig) -> ConcatDataset:
    """Build a combined training dataset from all configured dataset entries.

    Args:
        config: The pipeline configuration containing dataset entries.

    Returns:
        A ``ConcatDataset`` that merges all individual datasets.
    """
    datasets: List[RadiologyDataset] = []

    for name, entry in config.datasets.items():
        manifest_path = Path(entry.train_manifest)
        image_dir = Path(entry.image_dir)

        ds = RadiologyDataset(
            manifest_path=manifest_path,
            image_dir=image_dir,
            dataset_name=name,
            split="train",
        )

        if len(ds) > 0:
            datasets.append(ds)
            logger.info("Dataset '%s': %d training studies loaded", name, len(ds))
        else:
            logger.warning("Dataset '%s': 0 studies loaded, skipping", name)

    if not datasets:
        raise RuntimeError(
            "No training data loaded from any configured dataset. "
            "Check manifest paths and image directories."
        )

    combined = ConcatDataset(datasets)
    logger.info("Total training studies: %d", len(combined))
    return combined