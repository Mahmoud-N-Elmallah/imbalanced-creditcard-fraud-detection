from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from credit_fraud.evaluation import evaluate_model
from credit_fraud.training import train_model


def _cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "stage": "all",
            "seed": 42,
            "target_column": "Class",
            "project": {"root": str(tmp_path)},
            "paths": {
                "intermediate_train": str(tmp_path / "Data" / "intermediate" / "train.csv"),
                "intermediate_test": str(tmp_path / "Data" / "intermediate" / "test.csv"),
                "model": str(tmp_path / "Models" / "model.pkl"),
                "reports_dir": str(tmp_path / "Reports"),
                "figures_dir": str(tmp_path / "Reports" / "figures"),
            },
            "features": {
                "log_time": True,
                "log_amount": True,
                "amount_per_unit_time": True,
                "epsilon": 1.0e-9,
                "scale": True,
                "rolling_amount_mean": {"enabled": False, "window": 10},
                "kmeans": {"enabled": False, "n_clusters": 2, "feature_name": "cluster"},
            },
            "sampler": {"name": "none", "target": None, "params": {}},
            "model": {
                "name": "logistic_regression",
                "target": "sklearn.linear_model.LogisticRegression",
                "params": {"max_iter": 500, "class_weight": "balanced", "random_state": 42},
            },
            "training": {
                "cv": {"n_splits": 2, "shuffle": True, "n_jobs": 1},
                "scoring": {"champion_metric": "test_pr_auc", "min_delta": 0.0},
            },
            "mlflow": {
                "enabled": False,
                "tracking_uri": str(tmp_path / "mlruns"),
                "experiment_name": "test",
                "run_name": "test",
                "model_artifact_path": "model",
                "registry": {
                    "enabled": False,
                    "model_name": "credit-card-fraud-detector",
                    "champion_alias": "champion",
                },
            },
        }
    )


def _dataset(rows: int, offset: int = 0) -> pd.DataFrame:
    records = []
    for idx in range(rows):
        label = int(idx % 4 == 0)
        amount = 100.0 + idx if label else 5.0 + idx
        records.append(
            {
                "Time": float(idx + 1 + offset),
                "V1": float(label) + idx * 0.01,
                "V2": float(idx % 3),
                "Amount": amount,
                "Class": label,
            }
        )
    return pd.DataFrame(records)


def test_train_evaluate_smoke_without_mlflow(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    Path(cfg.paths.intermediate_train).parent.mkdir(parents=True)
    _dataset(40).to_csv(cfg.paths.intermediate_train, index=False)
    _dataset(16, offset=100).to_csv(cfg.paths.intermediate_test, index=False)

    train_model(cfg)
    evaluate_model(cfg)

    assert Path(cfg.paths.model).exists()
    assert (Path(cfg.paths.reports_dir) / "cv_metrics.json").exists()
    assert (Path(cfg.paths.reports_dir) / "test_metrics.json").exists()
    assert (Path(cfg.paths.figures_dir) / "precision_recall_curve.png").exists()
