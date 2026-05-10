"""Evaluate generated radiology reports using RadEval.

Uses the new RadEval 2.x API where metrics are passed as a list of
strings rather than individual boolean flags. The set of metrics to
compute is driven by ``config.evaluation.metrics`` in the YAML config.

Usage::

    python -m xray_pipeline.evaluation.evaluate \\
        --config configs/base.yaml \\
        --dataset spine

    # Link metrics to an existing training run:
    python -m xray_pipeline.evaluation.evaluate \\
        --config configs/base.yaml \\
        --dataset spine \\
        --run-id <mlflow-run-id>
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from xray_pipeline.core.env import load_dotenv
load_dotenv()

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

    # Late import: radeval is only available in the eval environment
    from radeval import RadEval

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
        "Loaded %d predictions for dataset '%s'.", len(references), dataset_name
    )

    # -- Build Evaluator from Config ---------------------------------------
    eval_cfg = config.evaluation
    logger.info("Running RadEval with metrics: %s", eval_cfg.metrics)

    evaluator = RadEval(
        metrics=eval_cfg.metrics,
        per_sample=eval_cfg.per_sample,
        detailed=eval_cfg.detailed,
    )

    scores = evaluator(refs=references, hyps=predictions)

    # -- Flatten & Log to MLflow -------------------------------------------
    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)

    flat_scores = _flatten_scores(scores, dataset_name)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(flat_scores)
            logger.info("Metrics logged to existing MLflow run: %s", run_id)
    else:
        with mlflow.start_run(run_name=f"eval_{dataset_name}"):
            mlflow.log_metrics(flat_scores)
            mlflow.set_tag("eval_dataset", dataset_name)
            mlflow.set_tag("eval_metrics", ",".join(eval_cfg.metrics))
            logger.info("Metrics logged to new MLflow run.")

    logger.info("Evaluation complete:")
    for metric, value in flat_scores.items():
        logger.info("  %s: %.4f", metric, value)


def _flatten_scores(
    scores: dict, dataset_name: str
) -> dict[str, float]:
    """Flatten potentially nested RadEval output into MLflow-compatible metrics.

    RadEval returns a flat dict in default mode (``{"bleu": 0.36, ...}``),
    but some metrics produce sub-keys when ``detailed=True``
    (e.g., ``{"bleu_1": 0.55, "bleu_2": 0.42}``).

    Per-sample mode returns lists which are averaged here for MLflow logging.

    Args:
        scores: Raw RadEval output dict.
        dataset_name: Dataset name prefix for metric keys.

    Returns:
        Dict of ``{eval_<dataset>_<metric>: float}``.
    """
    flat: dict[str, float] = {}

    for key, value in scores.items():
        prefixed_key = f"eval_{dataset_name}_{key}"

        if isinstance(value, (int, float)):
            flat[prefixed_key] = float(value)
        elif isinstance(value, list):
            # Per-sample mode: average for MLflow
            numeric = [v for v in value if isinstance(v, (int, float))]
            if numeric:
                flat[prefixed_key] = sum(numeric) / len(numeric)
        elif isinstance(value, dict):
            # Nested sub-scores (e.g., detailed mode)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    flat[f"{prefixed_key}_{sub_key}"] = float(sub_value)

    return flat


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