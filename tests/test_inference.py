import sys
from pathlib import Path
from unittest.mock import MagicMock

def test_inference_script_imports(tmp_path: Path):
    # Mock the unsloth module completely before importing the inference script
    sys.modules['unsloth'] = MagicMock()
    
    # Test that the script loads cleanly without import errors
    from xray_pipeline.training.inference import run_inference
    assert callable(run_inference)
    
    # Clean up the mock
    del sys.modules['unsloth']