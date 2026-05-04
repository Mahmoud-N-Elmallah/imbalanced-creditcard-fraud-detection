from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from credit_fraud.data import download_data, split_data


def _base_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 42,
            "target_column": "Class",
            "paths": {
                "raw_dir": str(tmp_path / "Data" / "raw"),
                "raw_data": str(tmp_path / "Data" / "raw" / "creditcard.csv"),
                "legacy_data": str(tmp_path / "Data" / "creditcard.csv"),
                "intermediate_dir": str(tmp_path / "Data" / "intermediate"),
                "intermediate_train": str(tmp_path / "Data" / "intermediate" / "train.csv"),
                "intermediate_test": str(tmp_path / "Data" / "intermediate" / "test.csv"),
                "reports_dir": str(tmp_path / "Reports"),
            },
            "data": {
                "kaggle_dataset": "mlg-ulb/creditcardfraud",
                "raw_filename": "creditcard.csv",
                "split": {
                    "test_size": 0.25,
                    "stratify": True,
                    "shuffle": True,
                    "drop_duplicates": True,
                },
            },
        }
    )


def test_download_skips_existing_raw_file(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    raw = Path(cfg.paths.raw_data)
    raw.parent.mkdir(parents=True)
    raw.write_text("Time,Amount,Class\n0,1,0\n", encoding="utf-8")

    assert download_data(cfg) == raw.resolve()
    assert raw.read_text(encoding="utf-8") == "Time,Amount,Class\n0,1,0\n"


def test_split_preserves_class_ratio_and_non_overlapping_rows(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    raw = Path(cfg.paths.raw_data)
    raw.parent.mkdir(parents=True)
    rows = []
    for idx in range(80):
        rows.append({"Time": idx, "V1": float(idx), "Amount": idx + 1.0, "Class": 0})
    for idx in range(20):
        rows.append({"Time": idx + 100, "V1": float(idx), "Amount": idx + 2.0, "Class": 1})
    pd.DataFrame(rows).to_csv(raw, index=False)

    split_data(cfg)
    train = pd.read_csv(cfg.paths.intermediate_train)
    test = pd.read_csv(cfg.paths.intermediate_test)

    assert abs(train["Class"].mean() - 0.2) < 0.05
    assert abs(test["Class"].mean() - 0.2) < 0.05
    train_hash = set(pd.util.hash_pandas_object(train, index=False))
    test_hash = set(pd.util.hash_pandas_object(test, index=False))
    assert train_hash.isdisjoint(test_hash)
