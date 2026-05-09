import os
import argparse
from pathlib import Path
import mlflow

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.core.io import ResultsReader

def run_evaluation(config_path: Path):
    config = PipelineConfig.from_yaml(config_path)
    
    # 1. ENFORCE STRICT CACHE PATHS BEFORE IMPORTING ML LIBRARIES
    res_dir = Path(config.paths.eval_resources_dir).resolve()
    os.environ["NLTK_DATA"] = str(res_dir / "nltk_data")
    os.environ["STANZA_RESOURCES_DIR"] = str(res_dir / "stanza_resources")
    
    # Late import to respect the environment variables we just set
    from RadEval import RadEval

    # 2. Read Predictions
    run_dir = Path(config.paths.results_dir) / config.project.experiment_name
    reader = ResultsReader(run_dir)
    
    references = []
    predictions = []
    
    for result in reader.read_all():
        references.append(result.reference_report)
        predictions.append(result.predicted_report)
        
    if not references:
        print(f"No predictions found in {run_dir}/predictions.jsonl")
        return

    print(f"Loaded {len(references)} predictions. Running RadEval...")

    # 3. Evaluate
    evaluator = RadEval(
        do_radcliq=True,
        do_bleu=True,
        do_bertscore=True,
        do_chexbert=False, # Set to True if SembScore is needed
        do_radgraph=False, # Set to True if RadGraph is needed
    )
    
    scores = evaluator(refs=references, hyps=predictions)
    
    # 4. Log to MLflow
    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)
    
    with mlflow.start_run():
        mlflow.log_metrics(scores)
        print("Evaluation complete. Metrics logged to MLflow:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_evaluation(args.config)