from pathlib import Path
from xray_pipeline.core.config import PipelineConfig

def test_config_loader():
    config_path = Path("configs/base.yaml")
    config = PipelineConfig.from_yaml(config_path)
    
    assert config.project.name == "multianatomy-xray-report-generation"
    assert config.model_id == "unsloth/Qwen2.5-VL-3B-Instruct"
    assert config.project.seed == 3407
    
    # Check that the new nested configs loaded correctly
    assert config.lora.r == 16
    assert config.training.batch_size == 2