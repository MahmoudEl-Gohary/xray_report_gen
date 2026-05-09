# xray_report_gen

A small toolkit for training, inference and evaluation of chest X-ray report generation models.

## Table of contents
- [xray\_report\_gen](#xray_report_gen)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Project structure](#project-structure)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Running tests](#running-tests)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

This repository provides data loading, training, inference and evaluation utilities for an X-ray report generation pipeline. The core code lives under `src/xray_pipeline` and configuration files are in the `configs/` directory.

## Requirements

- Python 3.8+
- Recommended: create a virtual environment and install project dependencies.

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the package in editable mode:

```bash
pip install -e .
```

If your project uses additional dependencies not declared in pyproject, install them as needed.

## Quickstart

After installing the package, run training, inference or evaluation modules via Python's `-m` interface. Example (after `pip install -e .`):

```bash
python -m xray_pipeline.training.train --config configs/base.yaml
python -m xray_pipeline.training.inference --checkpoint path/to/checkpoint
python -m xray_pipeline.evaluation.evaluate --preds path/to/preds --refs path/to/refs
```

Adjust CLI flags according to the specific script options.

## Project structure

- [src/xray_pipeline](src/xray_pipeline): main package
  - [src/xray_pipeline/core/config.py](src/xray_pipeline/core/config.py): configuration helpers
  - [src/xray_pipeline/core/data_reader.py](src/xray_pipeline/core/data_reader.py): dataset reading utilities
  - [src/xray_pipeline/core/io.py](src/xray_pipeline/core/io.py): I/O helpers
  - [src/xray_pipeline/core/schema.py](src/xray_pipeline/core/schema.py): data schema and validation
  - [src/xray_pipeline/training/train.py](src/xray_pipeline/training/train.py): training entrypoint
  - [src/xray_pipeline/training/inference.py](src/xray_pipeline/training/inference.py): inference utilities
  - [src/xray_pipeline/evaluation/evaluate.py](src/xray_pipeline/evaluation/evaluate.py): evaluation routines

- [configs/base.yaml](configs/base.yaml): default configuration file
- [tests/](tests): unit tests (run with `pytest`)

## Configuration

Configuration is YAML-based. See [configs/base.yaml](configs/base.yaml) for available options (data paths, model, training hyperparameters). Use the `--config` flag of the entrypoint scripts to point to a YAML file.

## Usage

- Training:

```bash
python -m xray_pipeline.training.train --config configs/base.yaml
```

- Inference (generate reports / predictions):

```bash
python -m xray_pipeline.training.inference --checkpoint path/to/checkpoint --output preds.json
```

- Evaluation:

```bash
python -m xray_pipeline.evaluation.evaluate --preds preds.json --refs refs.json
```

Refer to the module docstrings for detailed argument names and behavior.

## Running tests

Run the test suite with `pytest` from the repository root:

```bash
pytest -q
```

## Contributing

Contributions are welcome. Please open issues or pull requests describing changes. Follow the existing code style and add tests for new functionality.

## License

Specify your license here (e.g., MIT, Apache-2.0). If you do not intend to publish, indicate that this repository is for private use.
