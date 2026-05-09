import argparse
from pathlib import Path
import mlflow

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.training.dataset import RadiologyDataset

def train(config_path: Path):
    # Load Configuration
    config = PipelineConfig.from_yaml(config_path)
    
    # MLflow Setup
    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)
    
    with mlflow.start_run():
        # Log all hyperparameters as a flat dictionary
        mlflow.log_params({
            "model_id": config.model_id,
            "seed": config.project.seed,
            "lora_r": config.lora.r,
            "lr": config.training.learning_rate,
            "batch_size": config.training.batch_size,
        })

        # We import ML libraries here to avoid crashing fast metadata operations
        # on environments without GPUs (like local Windows).
        from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig

        # 1. Load Model & Tokenizer
        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_id,
            load_in_4bit=config.training.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )

        # 2. Apply LoRA Adapters
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision=config.lora.finetune_vision,
            finetune_language=config.lora.finetune_language,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

        # 3. Load Dataset
        data_root = Path(config.paths.data_root)
        train_dataset = RadiologyDataset(
            manifest_path=data_root / "manifest.json",
            image_dir=data_root / "images",
            dataset_name="training_data",
            split="train"
        )

        # 4. Configure SFTTrainer
        training_args = SFTConfig(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.grad_accum,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.training.logging_steps,
            output_dir=config.paths.results_dir,
            optim="adamw_8bit",
            dataset_text_field="", # Unsloth vision collator uses 'messages'
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=config.project.seed,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            args=training_args,
        )

        # 5. Train & Save
        trainer.train()
        
        save_path = Path(config.paths.results_dir) / "lora_model"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(save_path))
        print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)