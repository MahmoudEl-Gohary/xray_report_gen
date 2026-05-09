import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from xray_pipeline.core.schema import InferenceResult


class ResultsWriter:
    """Writes inference results to a JSONL file."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "predictions.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: InferenceResult) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")


class ResultsReader:
    """Reads inference results from a JSONL file."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "predictions.jsonl"

    def read_all(self) -> Iterator[InferenceResult]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                yield InferenceResult(**json.loads(line))