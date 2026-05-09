import argparse
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.core.data_reader import ManifestReader
from xray_pipeline.core.io import ResultsWriter
from xray_pipeline.core.schema import InferenceResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_checkpoint_step(checkpoint_dir: Path) -> int:
    """Extract the step number from a checkpoint directory name.

    Matches patterns like ``checkpoint-500``, ``checkpoint_step_1000``.
    Returns -1 if no step number is found.

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Returns:
        The step number, or -1 if extraction fails.
    """
    name = checkpoint_dir.name
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return -1


def run_inference(
    config_path: Path,
    checkpoint_dir: Path,
    dataset_name: str,
) -> None:
    """Run inference on a specific dataset's test split.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_dir: Path to the trained LoRA adapter directory.
        dataset_name: Which dataset to run inference on (must match a key
            in ``config.datasets``).
    """
    config = PipelineConfig.from_yaml(config_path)

    # Validate dataset name
    if dataset_name not in config.datasets:
        available = ", ".join(config.datasets.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. "
            f"Available datasets: {available}"
        )

    ds_entry = config.datasets[dataset_name]

    # Deferred import to prevent CUDA crashes on non-GPU environments
    from unsloth import FastVisionModel

    # -- 1. Load Model with LoRA Adapters ----------------------------------
    logger.info("Loading checkpoint from %s", checkpoint_dir)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(checkpoint_dir),
        load_in_4bit=config.training.load_in_4bit,
    )
    FastVisionModel.for_inference(model)

    # -- 2. Setup Data Reader (test split) ---------------------------------
    reader = ManifestReader(
        manifest_path=Path(ds_entry.test_manifest),
        image_dir=Path(ds_entry.image_dir),
        dataset_name=dataset_name,
        split="test",
    )

    # -- 3. Setup Writer ---------------------------------------------------
    run_dir = (
        Path(config.paths.results_dir)
        / config.project.experiment_name
        / dataset_name
    )
    writer = ResultsWriter(run_dir)
    logger.info("Saving predictions to: %s/predictions.jsonl", run_dir)

    # -- 4. Inference Loop -------------------------------------------------
    checkpoint_step = _extract_checkpoint_step(checkpoint_dir)
    processed = 0
    errors = 0

    records = list(reader.read_all())
    logger.info("Running inference on %d studies...", len(records))

    with torch.no_grad():
        for record in tqdm(records, desc=f"Inference [{dataset_name}]"):
            try:
                images = [
                    Image.open(p).convert("RGB") for p in record.image_paths
                ]

                # Build prompt (no assistant answer)
                messages = []
                if record.system_prompt:
                    messages.append(
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": record.system_prompt}
                            ],
                        }
                    )

                user_content = [
                    {"type": "image", "image": img} for img in images
                ]
                user_content.append(
                    {"type": "text", "text": record.user_prompt}
                )
                messages.append({"role": "user", "content": user_content})

                # Apply chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = tokenizer(
                    text=[text],
                    images=images if images else None,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)

                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.inference.max_new_tokens,
                    use_cache=True,
                    temperature=config.inference.temperature,
                )

                # Extract only newly generated tokens
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0, input_length:]
                predicted_html = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                # Save result
                result = InferenceResult(
                    study_id=record.study_id,
                    predicted_report=predicted_html,
                    reference_report=record.report_html,
                    dataset_name=record.dataset_name,
                    checkpoint_step=checkpoint_step,
                )
                writer.write(result)
                processed += 1

            except Exception as e:
                errors += 1
                logger.error(
                    "Failed to process study %s: %s", record.study_id, e
                )

    logger.info(
        "Inference complete. Processed: %d, Errors: %d", processed, errors
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a test dataset"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config YAML"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained LoRA directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to run inference on (e.g., 'spine', 'knee')",
    )
    args = parser.parse_args()
    run_inference(args.config, args.checkpoint, args.dataset)