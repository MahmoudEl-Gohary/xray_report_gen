"""Load environment variables from a ``.env`` file.

This is a minimal, zero-dependency implementation. It reads key=value
pairs from a ``.env`` file in the project root and injects them into
``os.environ``. Existing environment variables are NOT overwritten,
so system-level settings always take precedence.

The file is located by walking up from this module's location until a
``.env`` file is found (or the filesystem root is reached).
"""

import os
from pathlib import Path


def load_dotenv() -> None:
    """Load variables from the nearest ``.env`` file into ``os.environ``.

    - Lines starting with ``#`` are treated as comments.
    - Empty lines are skipped.
    - Values may be optionally quoted (single or double quotes).
    - Existing env vars are NOT overwritten.
    """
    env_path = _find_env_file()
    if env_path is None:
        return

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and blanks
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Strip surrounding quotes if present
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            # Do not overwrite existing env vars
            if key not in os.environ:
                os.environ[key] = value


def _find_env_file() -> Path | None:
    """Walk up from this file's directory to find a ``.env`` file.

    Returns:
        Path to the ``.env`` file, or None if not found.
    """
    current = Path(__file__).resolve().parent

    # Walk up at most 10 levels (src/xray_pipeline/core -> project root)
    for _ in range(10):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None
