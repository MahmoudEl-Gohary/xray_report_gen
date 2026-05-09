from pathlib import Path

from xray_pipeline.core.schema import InferenceResult
from xray_pipeline.core.io import ResultsWriter, ResultsReader


def test_io_writer_reader(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    writer = ResultsWriter(run_dir)

    record = InferenceResult(
        study_id="test_study_1",
        predicted_report="Normal chest.",
        reference_report="No acute cardiopulmonary process.",
        dataset_name="mimic",
        checkpoint_step=500
    )

    writer.write(record)

    reader = ResultsReader(run_dir)
    results = list(reader.read_all())

    assert len(results) == 1
    assert results[0].study_id == "test_study_1"
    assert results[0].predicted_report == "Normal chest."
    assert results[0].checkpoint_step == 500