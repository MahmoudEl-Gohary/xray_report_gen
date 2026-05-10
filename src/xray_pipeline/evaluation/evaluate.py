"""Evaluate generated radiology reports using RadEval.

Uses the RadEval 2.x API. The set of metrics to compute is driven by
boolean toggles in ``config.evaluation`` (e.g., ``bleu: true``).

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
import json
import logging
import sys
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


def _patch_radeval_import() -> None:
    """Work around a case-sensitivity bug in radeval <= 2.2.x.

    Internally, some radeval modules import from ``RadEval.metrics...``
    (capital R/E). On case-sensitive filesystems (Linux), this fails
    because the installed package directory is ``radeval`` (lowercase).

    This registers the lowercase module as an alias so the uppercase
    imports resolve correctly. Safe to call multiple times.
    """
    import radeval
    sys.modules.setdefault("RadEval", radeval)

    # Also alias submodules that are lazily loaded internally
    for key, mod in list(sys.modules.items()):
        if key.startswith("radeval."):
            alias = "RadEval." + key[len("radeval."):]
            sys.modules.setdefault(alias, mod)


def run_evaluation(
    config_path: Path,
    dataset_name: str,
    run_id: Optional[str] = None,
) -> None:
    """Evaluate predictions against references using RadEval.

    Scores are saved both to MLflow and to a local JSON file at
    ``<results_dir>/<experiment>/<dataset>/eval_scores.json``.

    Args:
        config_path: Path to the YAML configuration file.
        dataset_name: Which dataset's predictions to evaluate.
        run_id: Optional MLflow run ID to log metrics to. If provided,
            metrics are logged to the existing run (linking train + eval).
    """
    config = PipelineConfig.from_yaml(config_path)

    # Patch case-sensitivity bug before importing RadEval internals
    _patch_radeval_import()

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
    enabled = eval_cfg.enabled_metrics()

    if not enabled:
        logger.error("No metrics enabled in config.evaluation. Nothing to do.")
        return

    logger.info("Running RadEval with metrics: %s", enabled)

    evaluator = RadEval(
        metrics=enabled,
        per_sample=eval_cfg.per_sample,
        detailed=eval_cfg.detailed,
    )

    scores = evaluator(refs=references, hyps=predictions)

    # -- Flatten scores for logging ----------------------------------------
    flat_scores = _flatten_scores(scores, dataset_name)

    # -- Save to disk ------------------------------------------------------
    _save_scores_to_disk(run_dir, scores, flat_scores, enabled)

    # -- Log to MLflow -----------------------------------------------------
    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.project.experiment_name)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(flat_scores)
            logger.info("Metrics logged to existing MLflow run: %s", run_id)
    else:
        with mlflow.start_run(run_name=f"eval_{dataset_name}"):
            mlflow.log_metrics(flat_scores)
            mlflow.set_tag("eval_dataset", dataset_name)
            mlflow.set_tag("eval_metrics", ",".join(enabled))
            logger.info("Metrics logged to new MLflow run.")

    logger.info("Evaluation complete:")
    for metric, value in flat_scores.items():
        logger.info("  %s: %.4f", metric, value)


def _flatten_scores(
    scores: dict, dataset_name: str
) -> dict[str, float]:
    """Flatten RadEval output into MLflow-compatible scalar metrics.

    Handles three output shapes:
    - Default: ``{"bleu": 0.36}`` -- flat scalars
    - Detailed: ``{"bleu_1": 0.55, "bleu_2": 0.42}`` -- sub-scores
    - Per-sample: ``{"bleu": [0.85, 0.40]}`` -- lists, averaged here

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
            # Per-sample mode: average for MLflow scalar logging
            numeric = [v for v in value if isinstance(v, (int, float))]
            if numeric:
                flat[prefixed_key] = sum(numeric) / len(numeric)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    flat[f"{prefixed_key}_{sub_key}"] = float(sub_value)

    return flat


def _save_scores_to_disk(
    run_dir: Path,
    raw_scores: dict,
    flat_scores: dict[str, float],
    enabled_metrics: list[str],
) -> None:
    """Write evaluation results to ``eval_scores.json`` in the run directory.

    The JSON contains:
    - ``metrics_used``: list of metric names that were evaluated
    - ``raw_scores``: the direct output from RadEval (may include lists
      in per_sample mode)
    - ``summary``: the flattened, prefixed scores (same as what goes to MLflow)

    Args:
        run_dir: Directory where predictions.jsonl lives.
        raw_scores: Direct RadEval output dict.
        flat_scores: Flattened dict for MLflow.
        enabled_metrics: List of metric names that were enabled.
    """
    # Convert any non-serializable values to strings
    serializable_raw = {}
    for k, v in raw_scores.items():
        if isinstance(v, (int, float, str, bool)):
            serializable_raw[k] = v
        elif isinstance(v, list):
            serializable_raw[k] = v
        elif isinstance(v, dict):
            serializable_raw[k] = v
        else:
            serializable_raw[k] = str(v)

    output = {
        "metrics_used": enabled_metrics,
        "raw_scores": serializable_raw,
        "summary": flat_scores,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "eval_scores.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Scores saved to: %s", output_path)


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