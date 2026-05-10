# xray_report_gen

A pipeline for training, inference, and evaluation of multi-anatomy X-ray report generation models using Unsloth + Qwen2.5-VL.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Step-by-Step Setup (After Cloning)](#step-by-step-setup-after-cloning)
  - [Step 1: Create Directories](#step-1-create-directories)
  - [Step 2: Install uv](#step-2-install-uv-if-not-already-installed)
  - [Step 3: Setup Training Environment](#step-3-setup-training-environment)
  - [Step 4: Setup Evaluation Environment](#step-4-setup-evaluation-environment)
  - [Step 5: Place Raw Datasets](#step-5-place-raw-datasets)
  - [Step 6: Preprocess Datasets](#step-6-preprocess-datasets)
  - [Step 7: Validate Data](#step-7-validate-data)
  - [Step 8: Configure DagsHub/MLflow](#step-8-configure-dagshubmlflow)
  - [Step 9: Edit Config Parameters](#step-9-edit-config-parameters)
  - [Step 10: Run Training](#step-10-run-training)
  - [Step 11: Run Inference](#step-11-run-inference)
  - [Step 12: Run Evaluation](#step-12-run-evaluation)
- [Configuration Reference](#configuration-reference)
- [Makefile Targets](#makefile-targets)
- [CLI Override for Hyperparameter Sweeps](#cli-override-for-hyperparameter-sweeps)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

This repository fine-tunes **Qwen2.5-VL-2B-Instruct** on multi-anatomy radiology datasets (Spine, Knee, Private) using **LoRA** adapters via the **Unsloth** framework. It produces structured HTML radiology reports from X-ray images.

The pipeline has three stages, each runnable independently:
1. **Training** -- Fine-tune the base model on combined datasets
2. **Inference** -- Generate reports on held-out test sets
3. **Evaluation** -- Score predictions with RadEval (BLEU, BERTScore, RadCliQ)

## Architecture

```
configs/base.yaml          <-- Single source of truth for all parameters
        |
        v
  [Training Stage]         <-- .venv (transformers 4.x, unsloth, trl)
    train.py
    dataset.py (ConcatDataset from all datasets)
        |
        v
  results/lora_model/      <-- Saved LoRA adapters + checkpoints
        |
        v
  [Inference Stage]        <-- .venv (same training env)
    inference.py --dataset spine|knee|private
        |
        v
  results/<experiment>/<dataset>/predictions.jsonl
        |
        v
  [Evaluation Stage]       <-- .venv_eval (transformers 5.x, radeval)
    evaluate.py --dataset spine|knee|private
        |
        v
  MLflow/DagsHub            <-- Metrics logged remotely
```

> **Important:** Training and evaluation require **separate virtual environments** because their dependencies conflict (`transformers 4.x` vs `transformers 5.x`). This is enforced by design.

## Requirements

- `uv` package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA GPU with CUDA 12.4+ (A100 80GB recommended)
- No specific system Python version needed -- `uv` downloads Python 3.11 automatically

---

## Step-by-Step Setup (After Cloning)

This section walks through **every step** from `git clone` to a completed training run on the A100 machine. Nothing is installed system-wide; everything stays inside the project directory.

### Step 1: Create Directories

```bash
git clone https://github.com/MahmoudEl-Gohary/xray_report_gen.git
cd xray_report_gen

# Create all required directories
mkdir -p data/spine/images
mkdir -p data/knee/images
mkdir -p data/private/images
mkdir -p results
mkdir -p .eval_cache
```

Or use the Makefile shortcut:

```bash
make setup-dirs
```

After this step, your tree should look like:

```
xray_report_gen/
  data/
    spine/images/          # Empty -- will hold preprocessed PNGs
    knee/images/           # Empty -- will hold PNGs
    private/images/        # Empty -- will hold PNGs
  results/                 # Empty -- training outputs go here
  .eval_cache/             # Empty -- RadEval model cache
  configs/
  src/
  scripts/
  tests/
```

### Step 2: Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

### Step 3: Setup Training Environment

`uv` downloads the exact Python version needed -- no system Python 3.11 required.

```bash
# Creates .venv with Python 3.11 (downloaded automatically)
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,training]"
```

Or via Makefile:

```bash
make setup-train
```

Verify it works:

```bash
source .venv/bin/activate
python -c "import unsloth; print('Unsloth OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 4: Setup Evaluation Environment

RadEval has its own model dependencies (CheXbert, RadGraph, etc.) so we keep it isolated
from the training venv to avoid dependency bloat.

```bash
uv venv .venv_eval --python 3.11
uv pip install -e ".[evaluation]" --python .venv_eval/bin/python
```

Or via Makefile:

```bash
make setup-eval
```

### Step 5: Place Raw Datasets

Copy your raw dataset files to the machine. The expected locations:

**Spine dataset (VinDr-SpineXR):**

```
# The raw DICOMs should be accessible somewhere on disk, e.g.:
/data/raw/vindr-spinexr/train_images/
  000f985efcb28afd281e3cd1b4d370ee.dicom
  001a2b3c4d5e6f7g8h9i0j1k2l3m4n5o.dicom
  ...

# Your preprocessed JSON manifest should be placed at:
data/spine/train_manifest.json
data/spine/test_manifest.json
```

**Knee dataset:**

```
# PNGs should be accessible somewhere on disk, e.g.:
/data/raw/knee_images/
  study001_img1.png
  study001_img2.png
  ...

# Manifest:
data/knee/train_manifest.json
data/knee/test_manifest.json
```

### Step 6: Preprocess Datasets

This is the critical step. The script converts DICOMs to PNGs using `pydicom` and normalizes manifest paths.

**For Spine (has DICOMs -- needs conversion):**

```bash
# Via Makefile:
make prepare-manifests DATASET=spine SOURCE_DIR=/data/raw/vindr-spinexr/train_images

# Or directly:
.venv/bin/python -m xray_pipeline.scripts.prepare_manifests \
    --dataset spine \
    --source-dir /data/raw/vindr-spinexr/train_images \
    --config configs/base.yaml
```

This does two things:
1. Reads each DICOM referenced in the manifest from `--source-dir`
2. Converts it to an 8-bit PNG and saves it to `data/spine/images/`
3. Rewrites the manifest so paths are basenames like `abc123.png`

**For Knee (already PNGs -- copy + normalize only):**

If knee images are in a different directory and need to be copied:

```bash
make prepare-manifests DATASET=knee SOURCE_DIR=/data/raw/knee_images
```

If knee images are already in `data/knee/images/`, just normalize the manifest:

```bash
# Without SOURCE_DIR, Makefile auto-adds --skip-conversion
make prepare-manifests DATASET=knee
```

### Step 7: Validate Data

After preprocessing, verify that all images referenced in the manifests exist:

```bash
make validate-data
```

Expected output:

```
============================================================
Dataset: spine
Image dir: data/spine/images
Files on disk: 4000 {'.png': 4000}
  [OK] train: 3200/3200 studies valid, 6400 images referenced
  [OK] test: 800/800 studies valid, 1600 images referenced
============================================================
Dataset: knee
Image dir: data/knee/images
Files on disk: 5600 {'.png': 5600}
  [OK] train: 4480/4480 studies valid, 8960 images referenced
  [OK] test: 1120/1120 studies valid, 2240 images referenced
============================================================
All datasets validated successfully.
```

If any images are missing, the output will show `[ISSUES]` with the specific study IDs and filenames. Fix them before proceeding.

### Step 8: Configure DagsHub/MLflow

Create a [DagsHub](https://dagshub.com) repository (or use an existing one).

Credentials are loaded from a `.env` file in the project root -- **no system environment variables needed**. This is safe for borrowed machines since nothing is written outside the project directory.

```bash
# Create the .env file (it is gitignored, will not be committed)
cat > .env << 'EOF'
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
EOF
```

The `.env` file is automatically loaded by:
- **Makefile targets** (via `-include .env` + `export`)
- **Python scripts** (via `xray_pipeline.core.env.load_dotenv()` at startup)

The tracking URI in `configs/base.yaml` is set to:

```
https://dagshub.com/MahmoudEl-Gohary/xray_report_gen.mlflow
```

Update this if your DagsHub repo name or username is different.

### Step 9: Edit Config Parameters

All parameters are in `configs/base.yaml`. Here are the most important ones to review before your first run:

| Parameter | Default | What to Check |
|---|---|---|
| `model_id` | `unsloth/Qwen2.5-VL-2B-Instruct` | Confirm this is the model you want |
| `training.batch_size` | `2` | Increase to 4 if A100 80GB allows |
| `training.grad_accum` | `4` | Effective batch = batch_size * grad_accum = 8 |
| `training.learning_rate` | `2e-4` | Standard for LoRA fine-tuning |
| `training.num_train_epochs` | `1` | Increase for more training |
| `training.max_steps` | `-1` | Set to a positive number for smoke tests (e.g., `60`) |
| `training.save_steps` | `50` | Checkpoint frequency |
| `training.max_seq_length` | `2048` | Increase if reports are long, decrease if OOM |
| `lora.r` | `16` | LoRA rank (8, 16, 32, 64) |
| `inference.max_new_tokens` | `512` | Max report length during generation |

**For a smoke test (to verify everything works):**

```bash
python -m xray_pipeline.training.train \
    --config configs/base.yaml \
    --override training.max_steps=10 training.logging_steps=1
```

### Step 10: Run Training

```bash
source .venv/bin/activate

# Full training
python -m xray_pipeline.training.train --config configs/base.yaml

# Or via Makefile
make train
```

Training will:
- Load all configured datasets and concatenate them
- Save checkpoints to `results/` every `save_steps` steps
- Log metrics to MLflow/DagsHub in real time
- Save final LoRA adapters to `results/lora_model/`

Monitor training on DagsHub: `https://dagshub.com/MahmoudEl-Gohary/xray_report_gen`

### Step 11: Run Inference

Run inference on a specific dataset's test split:

```bash
# Inference on spine test set
python -m xray_pipeline.training.inference \
    --config configs/base.yaml \
    --checkpoint results/lora_model \
    --dataset spine

# Inference on knee test set
python -m xray_pipeline.training.inference \
    --config configs/base.yaml \
    --checkpoint results/lora_model \
    --dataset knee

# Or via Makefile
make infer DATASET=spine
```

Predictions are saved to: `results/baseline-qwen/spine/predictions.jsonl`

### Step 12: Run Evaluation

Switch to the evaluation environment:

```bash
deactivate
source .venv_eval/bin/activate

# Evaluate spine predictions
python -m xray_pipeline.evaluation.evaluate \
    --config configs/base.yaml \
    --dataset spine

# Link metrics to training run (pass the run ID from MLflow)
python -m xray_pipeline.evaluation.evaluate \
    --config configs/base.yaml \
    --dataset spine \
    --run-id <mlflow-run-id-from-training>

# Or via Makefile
make evaluate DATASET=spine
```

---

## Configuration Reference

The full config is in `configs/base.yaml`. Key sections:

```yaml
project:
  name: "multianatomy-xray-report-generation"
  experiment_name: "baseline-qwen"     # MLflow experiment name
  seed: 3407                           # Random seed for reproducibility

paths:
  results_dir: "./results"
  mlflow_tracking_uri: "https://dagshub.com/MahmoudEl-Gohary/xray_report_gen.mlflow"
  eval_resources_dir: "./.eval_cache"

datasets:
  spine:
    train_manifest: "data/spine/train_manifest.json"
    test_manifest: "data/spine/test_manifest.json"
    image_dir: "data/spine/images"      # Where PNGs live (NOT raw DICOMs)
  knee:
    train_manifest: "data/knee/train_manifest.json"
    test_manifest: "data/knee/test_manifest.json"
    image_dir: "data/knee/images"

model_id: "unsloth/Qwen3.5-2B"

lora:
  r: 16                  # LoRA rank
  lora_alpha: 16
  lora_dropout: 0.0
  finetune_vision: false  # Freeze vision encoder
  finetune_language: true # Fine-tune language decoder

training:
  batch_size: 2
  grad_accum: 4           # Effective batch = 2 * 4 = 8
  learning_rate: 2.0e-4
  num_train_epochs: 1
  max_steps: -1           # -1 = use num_train_epochs. Set to 60 for smoke test.
  warmup_steps: 5
  logging_steps: 5
  save_steps: 50          # Save checkpoint every 50 steps
  save_strategy: "steps"
  load_in_4bit: true      # QLoRA 4-bit quantization
  max_seq_length: 2048    # Cap context length to control VRAM
  report_to: "mlflow"     # Auto-log metrics to MLflow

inference:
  max_new_tokens: 512     # Max tokens to generate per study
  temperature: 0.1        # Low temp = more deterministic output

# Available: bleu, rouge, bertscore, radeval_bertscore, f1chexbert,
#            f1radbert_ct, radgraph, ratescore, radgraph_radcliq,
#            radcliq, srrbert, temporal, green, mammo_green,
#            crimson, radfact_ct
evaluation:
  metrics: [bleu, rouge, bertscore, radcliq, f1chexbert]
  per_sample: false       # true = per-sample list, false = corpus average
  detailed: false         # true = sub-scores (BLEU-1/2/3/4, label F1s)
```

## Makefile Targets

| Target | Variables | Description |
|---|---|---|
| `make setup-dirs` | -- | Create all required directories |
| `make setup-train` | -- | Create `.venv` and install training deps |
| `make setup-eval` | -- | Create `.venv_eval` and install eval deps |
| `make setup-all` | -- | Setup both environments |
| `make prepare-manifests` | `DATASET=spine SOURCE_DIR=/path` | Convert DICOMs + normalize manifests |
| `make validate-data` | -- | Check all images exist |
| `make train` | -- | Run training |
| `make infer` | `DATASET=spine` | Run inference on a dataset |
| `make evaluate` | `DATASET=spine` | Run evaluation on a dataset |
| `make pipeline` | -- | Run train -> infer -> evaluate |
| `make test` | -- | Run unit tests |
| `make clean` | -- | Remove build artifacts |
| `make help` | -- | Show all targets |

## CLI Override for Hyperparameter Sweeps

Override any config parameter from the command line without editing the YAML:

```bash
# Change learning rate and LoRA rank
python -m xray_pipeline.training.train \
    --config configs/base.yaml \
    --override training.learning_rate=1e-5 lora.r=32

# Quick smoke test (10 steps only)
python -m xray_pipeline.training.train \
    --config configs/base.yaml \
    --override training.max_steps=10 training.logging_steps=1

# Change batch size and gradient accumulation
python -m xray_pipeline.training.train \
    --config configs/base.yaml \
    --override training.batch_size=4 training.grad_accum=2
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v

# Or via Makefile
make test
```

Tests validate the core pipeline logic (config loading, data reading, schema, I/O, overrides) without requiring a GPU.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` rejects Python version | `requires-python` constraint | Use `uv venv --python 3.11` (auto-downloads) |
| Dependency conflict on install | Training + eval deps mixed | Use separate venvs (`.venv` and `.venv_eval`) |
| "No training data loaded" | Manifest paths or image dir wrong | Run `uv run validate-data --config configs/base.yaml` |
| CUDA OOM during training | Batch too large or seq too long | Reduce `batch_size` or `max_seq_length` in config |
| MLflow connection refused | DagsHub credentials not set | Create a `.env` file (see Step 8) |
| Images not found by data reader | Nested paths in manifest | Run `uv run prepare-manifests --dataset spine --source-dir ... --config configs/base.yaml` |
| DICOM conversion fails | Missing pydicom | `uv pip install pydicom numpy Pillow` (included in base deps) |
| RadEval crashes on import | Wrong venv active | Must use `.venv_eval`, not `.venv` |
| `ModuleNotFoundError: unsloth` | Wrong venv active | Must use `.venv`, not `.venv_eval` |

## Project Structure

```
xray_report_gen/
  configs/
    base.yaml                   # All pipeline configuration
  src/xray_pipeline/
    core/
      config.py                 # Pydantic config models + CLI override
      schema.py                 # StudyRecord, InferenceResult dataclasses
      data_reader.py            # ManifestReader with logging + fallbacks
      env.py                    # .env file loader (zero-dependency)
      io.py                     # ResultsWriter / ResultsReader (JSONL)
    training/
      dataset.py                # RadiologyDataset + build_training_dataset()
      train.py                  # Training entrypoint
      inference.py              # Inference entrypoint
    evaluation/
      evaluate.py               # RadEval evaluation entrypoint
      setup_resources.py        # Download NLTK/Stanza resources
    scripts/
      prepare_manifests.py      # CLI: DICOM-to-PNG + manifest normalization
      validate_data.py          # CLI: check all images exist on disk
  tests/                        # Unit tests (12 tests)
  Makefile                      # Automation targets
  .env                          # NOT in git -- DagsHub credentials
  data/                         # NOT in git -- created after cloning
    spine/images/               # Preprocessed PNGs (separate from raw DICOMs)
    knee/images/
  results/                      # NOT in git -- training outputs
```

## License

MIT
