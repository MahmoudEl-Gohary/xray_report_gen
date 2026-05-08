from pathlib import Path
from xray_pipeline.core.config import PipelineConfig

def test_config_loader():
    config_path = Path("configs/base.yaml")
    config = PipelineConfig.from_yaml(config_path)
    
    assert config.project.name == "multianatomy-xray-report-generation"
    assert config.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config.project.seed == 42