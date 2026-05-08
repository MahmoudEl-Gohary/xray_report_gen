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

class PipelineConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    model_id: str
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)