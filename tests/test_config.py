from pathlib import Path
from xray_pipeline.core.config import PipelineConfig


def test_config_loader():
    config_path = Path("configs/base.yaml")
    config = PipelineConfig.from_yaml(config_path)

    assert config.project.name == "multianatomy-xray-report-generation"
    assert config.model_id == "unsloth/Qwen2.5-VL-2B-Instruct"
    assert config.project.seed == 3407

    # Check nested configs
    assert config.lora.r == 16
    assert config.training.batch_size == 2
    assert config.training.save_steps == 50
    assert config.training.report_to == "mlflow"

    # Check multi-dataset config
    assert "spine" in config.datasets
    assert "knee" in config.datasets
    assert config.datasets["spine"].image_dir == "data/spine/images"

    # Check inference config
    assert config.inference.max_new_tokens == 512
    assert config.inference.temperature == 0.1


def test_config_overrides():
    config_path = Path("configs/base.yaml")
    overrides = [
        "training.learning_rate=1e-5",
        "lora.r=32",
        "training.batch_size=4",
        "training.load_in_4bit=false",
    ]
    config = PipelineConfig.from_yaml_with_overrides(config_path, overrides)

    assert config.training.learning_rate == 1e-5
    assert config.lora.r == 32
    assert config.training.batch_size == 4
    assert config.training.load_in_4bit is False

    # Unmodified fields should remain at their YAML defaults
    assert config.project.seed == 3407
    assert config.model_id == "unsloth/Qwen2.5-VL-2B-Instruct"