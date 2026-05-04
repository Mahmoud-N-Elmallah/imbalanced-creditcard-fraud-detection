from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from credit_fraud.features import build_feature_datasets, build_feature_transformer


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 42,
            "target_column": "Class",
            "paths": {
                "intermediate_train": str(tmp_path / "Data" / "intermediate" / "train.csv"),
                "intermediate_test": str(tmp_path / "Data" / "intermediate" / "test.csv"),
                "final_train": str(tmp_path / "Data" / "final" / "train.csv"),
                "final_test": str(tmp_path / "Data" / "final" / "test.csv"),
                "preprocessor": str(tmp_path / "Artifacts" / "preprocessor.pkl"),
                "reports_dir": str(tmp_path / "Reports"),
            },
            "features": {
                "log_time": True,
                "log_amount": True,
                "amount_per_unit_time": True,
                "epsilon": 1.0e-9,
                "scale": True,
                "rolling_amount_mean": {"enabled": False, "window": 10},
                "kmeans": {"enabled": True, "n_clusters": 3, "feature_name": "cluster"},
            },
        }
    )


def _frame(offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": np.arange(offset, offset + 10),
            "V1": np.linspace(0, 1, 10),
            "Amount": np.linspace(1, 20, 10),
            "Class": [0, 1] * 5,
        }
    )


def test_feature_transformer_uses_train_fit_state_for_scaling(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    train = _frame(1)
    test = _frame(10_000)
    transformer = build_feature_transformer(cfg)

    transformer.fit(train.drop(columns=["Class"]), train["Class"])

    amount_index = transformer.feature_names_in_.index("Amount")
    expected_train_center = np.median(np.log1p(train["Amount"]))
    assert transformer.scaler_.center_[amount_index] == expected_train_center

    transformed_test = transformer.transform(test.drop(columns=["Class"]))
    assert "cluster" in transformed_test.columns


def test_build_feature_datasets_writes_matching_train_test_snapshots(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    Path(cfg.paths.intermediate_train).parent.mkdir(parents=True)
    _frame(1).to_csv(cfg.paths.intermediate_train, index=False)
    _frame(100).to_csv(cfg.paths.intermediate_test, index=False)

    build_feature_datasets(cfg)

    train_final = pd.read_csv(cfg.paths.final_train)
    test_final = pd.read_csv(cfg.paths.final_test)
    assert list(train_final.columns) == list(test_final.columns)
    assert Path(cfg.paths.preprocessor).exists()
