import argparse
from pathlib import Path
from PIL import Image

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.core.data_reader import ManifestReader
from xray_pipeline.core.io import ResultsWriter
from xray_pipeline.core.schema import InferenceResult

def run_inference(config_path: Path, checkpoint_dir: Path):
    config = PipelineConfig.from_yaml(config_path)
    
    # Deferred import to prevent CUDA crashes on local non-GPU environments
    from unsloth import FastVisionModel

    # 1. Load Model with LoRA Adapters
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(checkpoint_dir),
        load_in_4bit=config.training.load_in_4bit,
    )
    FastVisionModel.for_inference(model)

    # 2. Setup Data Reader (Targeting the 'test' split)
    data_root = Path(config.paths.data_root)
    reader = ManifestReader(
        manifest_path=data_root / "test_manifest.json",
        image_dir=data_root / "images",
        dataset_name="test_data",
        split="test"
    )
    
    # 3. Setup Writer
    run_dir = Path(config.paths.results_dir) / config.project.experiment_name
    writer = ResultsWriter(run_dir)
    print(f"Saving predictions to: {run_dir}/predictions.jsonl")

    # 4. Inference Loop
    for record in reader.read_all():
        images = [Image.open(p).convert("RGB") for p in record.image_paths]
        
        # Build prompt excluding the assistant's answer
        messages = []
        if record.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": record.system_prompt}]})
        
        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": record.user_prompt})
        messages.append({"role": "user", "content": user_content})

        # Apply chat template with add_generation_prompt=True
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.1,
        )
        
        # Extract only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]
        predicted_html = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 5. Save the result
        result = InferenceResult(
            study_id=record.study_id,
            predicted_report=predicted_html,
            reference_report=record.report_html,
            dataset_name=record.dataset_name,
            checkpoint_step=-1,
        )
        writer.write(result)
        print(f"Processed: {record.study_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained LoRA directory")
    args = parser.parse_args()
    run_inference(args.config, args.checkpoint)