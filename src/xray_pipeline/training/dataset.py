from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from xray_pipeline.core.schema import StudyRecord
from xray_pipeline.core.data_reader import ManifestReader


class RadiologyDataset(Dataset):
    """
    PyTorch Dataset that loads radiology studies into memory as StudyRecords.
    Actual image loading (PIL) is deferred to the Collator for faster multiprocessing.
    """

    def __init__(
        self,
        manifest_path: Path,
        image_dir: Path,
        dataset_name: str,
        split: str = "train"
    ) -> None:
        reader = ManifestReader(
            manifest_path=manifest_path,
            image_dir=image_dir,
            dataset_name=dataset_name,
            split=split
        )
        # Load all records into memory (StudyRecord is just text/paths, so it is lightweight)
        self.records: List[StudyRecord] = list(reader.read_all())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> StudyRecord:
        return self.records[idx]