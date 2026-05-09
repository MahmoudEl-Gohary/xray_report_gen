import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import mlflow

from xray_pipeline.core.config import PipelineConfig
from xray_pipeline.core.io import ResultsReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_evaluation(
    config_path: Path,
    dataset_name: str,
    run_id: Optional[str] = None,
) -> None:
    """Evaluate predictions against references using RadEval.

    Args:
        config_path: Path to the YAML configuration file.
        dataset_name: Which dataset's predictions to evaluate.
        run_id: Optional MLflow run ID to log metrics to. If provided,
            metrics are logged to the existing run (linking train + eval).
    """
    config = PipelineConfig.from_yaml(config_path)

    # Enforce strict cache paths BEFORE importing ML libraries
    res_dir = Path(config.paths.eval_resources_dir).resolve()
    os.environ["NLTK_DATA"] = str(res_dir / "nltk_data")
    os.environ["STANZA_RESOURCES_DIR"] = str(res_dir / "stanza_resources")

    # Late import to respect the environment variables we just set
    from RadEval import RadEval

    # -- Read Predictions --------------------------------------------------
    run_dir = (
        Path(config.paths.results_dir)
        / config.project.experiment_name
        / dataset_name
    )
    reader = ResultsReader(run_dir)

    references = []
    predictions = []

    for result in reader.read_all():
        references.append(result.reference_report)
        predictions.append(result.predicted_report)

    if not references:
        logger.error("No predictions found in %s/predictions.jsonl", run_dir)
        return

    logger.info(
        "Loaded %d predictions for dataset '%s'. Running RadEval...",
        len(references),
        dataset_name,
    )

    # -- Evaluate ----------------------------------------------------------
    evaluator = RadEval(
        do_radcliq=True,
        do_bleu=True,
        do_bertscore=True,
        do_chexbert=False,
        do_radgraph=False,
    )

    scores = evaluator(refs=references, hyps=predictions)

    # -- Log to MLflow -----------------------------------------------------
    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)

    # Flatten scores if nested, and ensure all values are numeric
    flat_scores = {}
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            flat_scores[f"eval_{dataset_name}_{key}"] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    flat_scores[f"eval_{dataset_name}_{key}_{sub_key}"] = sub_value

    if run_id:
        # Log to the same run as training
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(flat_scores)
            logger.info("Metrics logged to existing MLflow run: %s", run_id)
    else:
        # Create a new evaluation run
        with mlflow.start_run(run_name=f"eval_{dataset_name}"):
            mlflow.log_metrics(flat_scores)
            mlflow.set_tag("eval_dataset", dataset_name)
            logger.info("Metrics logged to new MLflow run.")

    logger.info("Evaluation complete:")
    for metric, value in flat_scores.items():
        logger.info("  %s: %.4f", metric, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated reports with RadEval"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate (e.g., 'spine', 'knee')",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to log metrics to (links eval to training run)",
    )
    args = parser.parse_args()
    run_evaluation(args.config, args.dataset, args.run_id)