from pathlib import Path
from xray_pipeline.core.schema import StudyRecord, InferenceResult

def test_study_record_creation():
    record = StudyRecord(
        study_id="mimic_001",
        image_paths=[Path("/data/img1.png"), Path("/data/img2.png")],
        report_html="<p>Normal heart size.</p>",
        dataset_name="mimic_cxr",
        split="train"
    )
    
    assert record.study_id == "mimic_001"
    assert len(record.image_paths) == 2
    assert record.metadata == {}

def test_inference_result_creation():
    result = InferenceResult(
        study_id="mimic_001",
        predicted_report="<p>Clear lungs.</p>",
        reference_report="<p>Normal heart size.</p>",
        dataset_name="mimic_cxr",
        checkpoint_step=1000
    )
    
    assert result.checkpoint_step == 1000
    assert result.dataset_name == "mimic_cxr"