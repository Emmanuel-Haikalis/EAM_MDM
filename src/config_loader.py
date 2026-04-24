"""Configuration loader — reads YAML files from the project config/ directory.

All modules should call load_config(name) rather than importing constants
directly, so organisations can tune vocabularies without touching source code.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

_CONFIG_DIR = pathlib.Path(__file__).parent.parent / "config"


def load_config(name: str) -> Dict[str, Any]:
    """Return the parsed contents of config/<name>.yaml, or {} if absent."""
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
