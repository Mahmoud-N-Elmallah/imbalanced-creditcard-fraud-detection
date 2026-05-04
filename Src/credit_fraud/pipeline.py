from __future__ import annotations

import logging

from omegaconf import DictConfig

from credit_fraud.data import download_data, split_data
from credit_fraud.evaluation import evaluate_model
from credit_fraud.features import build_feature_datasets
from credit_fraud.training import train_model
from credit_fraud.utils import configure_logging


LOGGER = logging.getLogger(__name__)
VALID_STAGES = {"all", "download", "split", "features", "train", "evaluate"}


def run_stage(cfg: DictConfig) -> None:
    configure_logging()
    stage = str(cfg.stage)
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Valid stages: {sorted(VALID_STAGES)}")

    LOGGER.info("Running stage: %s", stage)
    if stage in {"all", "download"}:
        download_data(cfg)
    if stage in {"all", "split"}:
        split_data(cfg)
    if stage in {"all", "features"}:
        build_feature_datasets(cfg)
    if stage in {"all", "train"}:
        train_model(cfg)
    if stage in {"all", "evaluate"}:
        evaluate_model(cfg)
