from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class StudyRecord:
    """
    Standard representation of a single radiology study.
    Passed from the Dataset to the Collator.
    """
    study_id: str
    image_paths: List[Path]
    system_prompt: str
    user_prompt: str
    report_html: str
    dataset_name: str
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceResult:
    """
    Standard representation of a model prediction.
    Saved during inference and read during evaluation.
    """
    study_id: str
    predicted_report: str
    reference_report: str
    dataset_name: str
    checkpoint_step: int
    metadata: Dict[str, Any] = field(default_factory=dict)