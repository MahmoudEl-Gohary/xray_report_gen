import argparse
import logging
import subprocess
from pathlib import Path

from xray_pipeline.core.env import load_dotenv
load_dotenv()

import dagshub
import mlflow

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.training.dataset import build_training_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def train(config_path: Path, overrides: list[str] | None = None) -> None:
    """Run the full training pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional list of ``"key=value"`` overrides for config
            fields (e.g., ``["training.learning_rate=1e-5", "lora.r=32"]``).
    """
    # -- Load Configuration ------------------------------------------------
    config = PipelineConfig.from_yaml_with_overrides(config_path, overrides)
    logger.info("Config loaded: %s", config.project.name)

    # -- MLflow Setup (DagsHub) --------------------------------------------
    dagshub.init(
        repo_owner="MahmoudEl-Gohary",
        repo_name="xray_report_gen",
        mlflow=True,
    )
    mlflow.set_experiment(config.project.experiment_name)

    with mlflow.start_run(
        run_name=f"{config.project.experiment_name}_{_get_git_sha()}"
    ):
        # Log all hyperparameters
        mlflow.log_params(
            {
                "model_id": config.model_id,
                "seed": config.project.seed,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
                "finetune_vision": config.lora.finetune_vision,
                "finetune_language": config.lora.finetune_language,
                "lr": config.training.learning_rate,
                "batch_size": config.training.batch_size,
                "grad_accum": config.training.grad_accum,
                "num_train_epochs": config.training.num_train_epochs,
                "max_steps": config.training.max_steps,
                "max_seq_length": config.training.max_seq_length,
                "save_steps": config.training.save_steps,
                "git_sha": _get_git_sha(),
                "datasets": ",".join(config.datasets.keys()),
            }
        )

        # Log the config file as an artifact for reproducibility
        mlflow.log_artifact(str(config_path))

        # -- Deferred GPU imports -----------------------------------------
        # Imported here to avoid crashing on non-GPU environments.
        from unsloth import (
            FastVisionModel,
            is_bfloat16_supported,
            UnslothVisionDataCollator,
        )
        from trl import SFTTrainer, SFTConfig

        # -- 1. Load Model & Tokenizer ------------------------------------
        logger.info("Loading model: %s", config.model_id)
        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_id,
            load_in_4bit=config.training.load_in_4bit,
            use_gradient_checkpointing="unsloth",
            max_seq_length=config.training.max_seq_length,
        )

        # -- 2. Apply LoRA Adapters ----------------------------------------
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision=config.lora.finetune_vision,
            finetune_language=config.lora.finetune_language,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        # -- 3. Load Combined Dataset -------------------------------------
        train_dataset = build_training_dataset(config)
        logger.info("Total training samples: %d", len(train_dataset))

        # -- 4. Configure SFTTrainer ---------------------------------------
        training_args = SFTConfig(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.grad_accum,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.num_train_epochs,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.training.logging_steps,
            save_steps=config.training.save_steps,
            save_strategy=config.training.save_strategy,
            output_dir=config.paths.results_dir,
            optim="adamw_8bit",
            dataset_text_field="",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=config.project.seed,
            report_to=config.training.report_to,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            args=training_args,
        )

        # -- 5. Train & Save -----------------------------------------------
        logger.info("Starting training...")
        trainer.train()

        save_path = Path(config.paths.results_dir) / "lora_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Log the LoRA adapter directory (small, not the full base model)
        mlflow.log_artifact(str(save_path))
        logger.info("Training complete. Model saved to %s", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the X-ray report model")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help=(
            "Override config values with dot-notation. "
            "Example: --override training.learning_rate=1e-5 lora.r=32"
        ),
    )
    args = parser.parse_args()
    train(args.config, args.override)