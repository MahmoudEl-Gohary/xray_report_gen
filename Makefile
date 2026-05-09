# ============================================================================
# X-Ray Report Generation Pipeline -- Makefile
# ============================================================================
# Uses `uv` for environment management. uv downloads the exact Python
# version needed, so no system Python installation is required.
#
# Credentials: Create a `.env` file in the project root with:
#   MLFLOW_TRACKING_USERNAME=your-dagshub-username
#   MLFLOW_TRACKING_PASSWORD=your-dagshub-token
# ============================================================================

# Load .env if it exists (credentials, custom overrides)
-include .env
export

TRAIN_VENV  := .venv
EVAL_VENV   := .venv_eval
CONFIG      ?= configs/base.yaml
CHECKPOINT  ?= results/lora_model
DATASET     ?= spine
PY_VERSION  ?= 3.11

# ---- Environment Setup -----------------------------------------------------

.PHONY: setup-train
setup-train: ## Create training venv and install deps
	uv venv $(TRAIN_VENV) --python $(PY_VERSION)
	uv pip install -e ".[dev,training]" --python $(TRAIN_VENV)/bin/python
	@echo "Training environment ready: source $(TRAIN_VENV)/bin/activate"

.PHONY: setup-eval
setup-eval: ## Create evaluation venv and install deps
	uv venv $(EVAL_VENV) --python $(PY_VERSION)
	uv pip install -e ".[evaluation]" --python $(EVAL_VENV)/bin/python
	$(EVAL_VENV)/bin/python -m xray_pipeline.evaluation.setup_resources --config $(CONFIG)
	@echo "Evaluation environment ready: source $(EVAL_VENV)/bin/activate"

.PHONY: setup-all
setup-all: setup-train setup-eval ## Setup both environments

.PHONY: setup-dirs
setup-dirs: ## Create required data and output directories
	mkdir -p data/spine/images
	mkdir -p data/knee/images
	mkdir -p data/private/images
	mkdir -p results
	mkdir -p .eval_cache

# ---- Data Preparation -------------------------------------------------------

.PHONY: validate-data
validate-data: ## Validate that all images in manifests exist
	$(TRAIN_VENV)/bin/python scripts/validate_data.py --config $(CONFIG)

.PHONY: prepare-manifests
prepare-manifests: ## Convert raw dataset JSONs to pipeline-expected format
	$(TRAIN_VENV)/bin/python scripts/prepare_manifests.py --config $(CONFIG)

# ---- Training ---------------------------------------------------------------

.PHONY: train
train: ## Run training with the training venv
	$(TRAIN_VENV)/bin/python -m xray_pipeline.training.train --config $(CONFIG)

# ---- Inference --------------------------------------------------------------

.PHONY: infer
infer: ## Run inference on a specific dataset (set DATASET=spine|knee|private)
	$(TRAIN_VENV)/bin/python -m xray_pipeline.training.inference \
		--config $(CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--dataset $(DATASET)

# ---- Evaluation -------------------------------------------------------------

.PHONY: evaluate
evaluate: ## Run evaluation on a specific dataset (set DATASET=spine|knee|private)
	$(EVAL_VENV)/bin/python -m xray_pipeline.evaluation.evaluate \
		--config $(CONFIG) \
		--dataset $(DATASET)

# ---- Full Pipeline ----------------------------------------------------------

.PHONY: pipeline
pipeline: train infer evaluate ## Run full pipeline: train -> infer -> evaluate
	@echo "Full pipeline complete."

# ---- Testing ----------------------------------------------------------------

.PHONY: test
test: ## Run unit tests
	$(TRAIN_VENV)/bin/python -m pytest tests/ -v

# ---- Utilities --------------------------------------------------------------

.PHONY: clean
clean: ## Remove build artifacts (not data or results)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info src/*.egg-info

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
