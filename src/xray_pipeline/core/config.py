from pathlib import Path
import yaml
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    name: str
    experiment_name: str
    seed: int = 42

class PathsConfig(BaseModel):
    data_root: str
    results_dir: str
    mlflow_tracking_uri: str
    eval_resources_dir: str = "./.eval_cache"

class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision: bool = False
    finetune_language: bool = True

class TrainingConfig(BaseModel):
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    max_steps: int = 60
    warmup_steps: int = 5
    logging_steps: int = 5
    load_in_4bit: bool = True

class PipelineConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    model_id: str
    lora: LoraConfig
    training: TrainingConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)