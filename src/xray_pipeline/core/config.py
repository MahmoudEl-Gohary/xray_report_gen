from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Project-level metadata."""

    name: str
    experiment_name: str
    seed: int = 42


class DatasetEntry(BaseModel):
    """Paths for a single dataset (spine, knee, private, etc.)."""

    train_manifest: str
    test_manifest: str
    image_dir: str


class PathsConfig(BaseModel):
    """Global path settings."""

    results_dir: str
    mlflow_tracking_uri: str
    eval_resources_dir: str = "./.eval_cache"


class LoraConfig(BaseModel):
    """LoRA adapter hyperparameters."""

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision: bool = False
    finetune_language: bool = True


class TrainingConfig(BaseModel):
    """SFTTrainer and training-loop hyperparameters."""

    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 means use num_train_epochs
    warmup_steps: int = 5
    logging_steps: int = 5
    save_steps: int = 50
    save_strategy: str = "steps"
    load_in_4bit: bool = True
    max_seq_length: int = 2048
    report_to: str = "mlflow"


class InferenceConfig(BaseModel):
    """Generation parameters for inference."""

    max_new_tokens: int = 512
    temperature: float = 0.1


class EvalConfig(BaseModel):
    """Evaluation configuration for RadEval metrics.

    Each metric can be toggled individually via its boolean flag.
    Only metrics set to ``true`` will be computed during evaluation.

    Metric categories:
        Lexical:   bleu, rouge
        Semantic:  bertscore, radeval_bertscore, ratescore, srrbert
        Clinical:  f1chexbert, f1radbert_ct, radgraph, radgraph_radcliq,
                   radcliq, temporal
        LLM-based: green, mammo_green, crimson, radfact_ct
                   (require OPENAI_API_KEY / GEMINI_API_KEY)
    """

    # -- Lexical metrics ---------------------------------------------------
    bleu: bool = True
    rouge: bool = True

    # -- Semantic metrics --------------------------------------------------
    bertscore: bool = True
    radeval_bertscore: bool = False
    ratescore: bool = False
    srrbert: bool = False

    # -- Clinical metrics --------------------------------------------------
    f1chexbert: bool = True
    f1radbert_ct: bool = False
    radgraph: bool = False
    radgraph_radcliq: bool = False
    radcliq: bool = True
    temporal: bool = False

    # -- LLM-based metrics (require API keys) ------------------------------
    green: bool = False
    mammo_green: bool = False
    crimson: bool = False
    radfact_ct: bool = False

    # -- Output modes ------------------------------------------------------
    per_sample: bool = False
    detailed: bool = False

    def enabled_metrics(self) -> list[str]:
        """Return the list of metric names where the flag is True.

        Returns:
            List of enabled metric name strings.
        """
        _non_metric_fields = {"per_sample", "detailed"}
        return [
            name
            for name in type(self).model_fields
            if name not in _non_metric_fields
            and isinstance(getattr(self, name), bool)
            and getattr(self, name)
        ]


class PipelineConfig(BaseModel):
    """Root configuration for the entire pipeline."""

    project: ProjectConfig
    paths: PathsConfig
    datasets: Dict[str, DatasetEntry]
    model_id: str
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    evaluation: EvalConfig = EvalConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load and validate config from a YAML file."""
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_with_overrides(
        cls,
        path: Path,
        overrides: Optional[list[str]] = None,
    ) -> "PipelineConfig":
        """Load config from YAML, then apply dot-notation overrides.

        Args:
            path: Path to the YAML config file.
            overrides: List of ``"key=value"`` strings where ``key`` uses
                dot-notation (e.g., ``"training.learning_rate=1e-5"``,
                ``"lora.r=32"``).

        Returns:
            Validated PipelineConfig with overrides applied.
        """
        with path.open("r") as f:
            data = yaml.safe_load(f)

        for override in overrides or []:
            if "=" not in override:
                raise ValueError(
                    f"Invalid override format: '{override}'. "
                    "Expected 'key=value' (e.g., 'training.learning_rate=1e-5')."
                )
            key, value = override.split("=", 1)
            _set_nested(data, key.strip(), _auto_cast(value.strip()))

        return cls.model_validate(data)


def _set_nested(d: dict, key: str, value: object) -> None:
    """Set a value in a nested dict using dot-notation key.

    Args:
        d: The dictionary to modify.
        key: Dot-separated key path (e.g., ``"training.learning_rate"``).
        value: The value to set.
    """
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def _auto_cast(value: str) -> object:
    """Cast a string value to int, float, or bool if possible.

    Args:
        value: The string value to cast.

    Returns:
        The cast value, or the original string if no cast applies.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value