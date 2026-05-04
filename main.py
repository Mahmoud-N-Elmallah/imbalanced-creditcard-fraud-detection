from __future__ import annotations

import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "Src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_fraud.pipeline import run_stage  # noqa: E402


@hydra.main(version_base=None, config_path="Config", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    run_stage(cfg)


if __name__ == "__main__":
    main()
