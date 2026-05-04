from __future__ import annotations

import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


LOGGER = logging.getLogger(__name__)
SECRET_KEY_PARTS = ("password", "token", "secret", "key")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def resolve_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_resolved_config(cfg: DictConfig, path: str | Path) -> None:
    target = ensure_parent(path)
    target.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit(root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=resolve_path(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        compound_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, compound_key))
        else:
            flattened[compound_key] = value
    return flattened


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        return {}
    return flatten_dict(payload)


def is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SECRET_KEY_PARTS)


def safe_mlflow_params(cfg: DictConfig) -> dict[str, str]:
    params: dict[str, str] = {}
    for key, value in flatten_config(cfg).items():
        if is_secret_key(key):
            continue
        value_text = str(value)
        if len(key) <= 250 and len(value_text) <= 500:
            params[key] = value_text
    return params
