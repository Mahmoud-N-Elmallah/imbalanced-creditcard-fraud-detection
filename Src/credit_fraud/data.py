from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from credit_fraud.utils import ensure_dir, resolve_path, write_json


LOGGER = logging.getLogger(__name__)


def download_data(cfg: DictConfig) -> Path:
    """Ensure the raw Kaggle CSV exists without re-downloading existing data."""
    raw_path = resolve_path(cfg.paths.raw_data)
    raw_dir = ensure_dir(cfg.paths.raw_dir)

    if raw_path.exists():
        LOGGER.info("Raw data exists, skip download: %s", raw_path)
        return raw_path

    legacy_path = resolve_path(cfg.paths.legacy_data)
    if legacy_path.exists():
        shutil.copy2(legacy_path, raw_path)
        LOGGER.info("Copied legacy raw data into raw folder: %s", raw_path)
        return raw_path

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install kaggle package before downloading data.") from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(cfg.data.kaggle_dataset, path=str(raw_dir), unzip=True)

    if not raw_path.exists():
        candidates = list(raw_dir.rglob(str(cfg.data.raw_filename)))
        if candidates:
            shutil.move(str(candidates[0]), raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Kaggle download finished but {raw_path} was not found.")

    LOGGER.info("Downloaded raw data: %s", raw_path)
    return raw_path


def _class_ratio(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    counts = frame[target_column].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "total_rows": total,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "positive_ratio": float(counts.get(1, 0) / total) if total else 0.0,
    }


def split_data(cfg: DictConfig) -> tuple[Path, Path]:
    """Create leakage boundary: clean and split before feature engineering."""
    raw_path = resolve_path(cfg.paths.raw_data)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data missing: {raw_path}")

    df = pd.read_csv(raw_path)
    target_column = str(cfg.target_column)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from raw data.")

    rows_before = len(df)
    if bool(cfg.data.split.drop_duplicates):
        df = df.drop_duplicates().reset_index(drop=True)
    rows_after = len(df)

    stratify = df[target_column] if bool(cfg.data.split.stratify) else None
    train_df, test_df = train_test_split(
        df,
        test_size=float(cfg.data.split.test_size),
        random_state=int(cfg.seed),
        stratify=stratify,
        shuffle=bool(cfg.data.split.shuffle),
    )

    train_path = resolve_path(cfg.paths.intermediate_train)
    test_path = resolve_path(cfg.paths.intermediate_test)
    ensure_dir(train_path.parent)
    ensure_dir(test_path.parent)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    report = {
        "rows_before_drop_duplicates": rows_before,
        "rows_after_drop_duplicates": rows_after,
        "duplicates_removed": rows_before - rows_after,
        "train": _class_ratio(train_df, target_column),
        "test": _class_ratio(test_df, target_column),
    }
    write_json(resolve_path(cfg.paths.reports_dir) / "data_split.json", report)
    LOGGER.info("Wrote intermediate train/test data.")
    return train_path, test_path
