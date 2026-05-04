from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_fraud.factories import build_model_pipeline
from credit_fraud.metrics import evaluate_predictions, score_vector
from credit_fraud.mlflow_tracking import (
    configure_mlflow,
    log_common_params,
    log_metrics,
    mlflow_enabled,
)
from credit_fraud.utils import ensure_dir, resolve_path, set_seed, write_json, write_resolved_config


LOGGER = logging.getLogger(__name__)


def _split_xy(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[target_column]), frame[target_column]


def _cv_metrics(results: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, values in results.items():
        if key.startswith("test_"):
            name = key.removeprefix("test_")
            metrics[f"cv_{name}_mean"] = float(values.mean())
            metrics[f"cv_{name}_std"] = float(values.std())
    return metrics


def _run_cross_validation(cfg: DictConfig, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    minority_count = int(y.value_counts().min())
    n_splits = min(int(cfg.training.cv.n_splits), minority_count)
    if n_splits < 2:
        LOGGER.warning("Skipping CV because smallest class has fewer than two rows.")
        return {}

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "fraud_f1": make_scorer(f1_score, zero_division=0),
        "fraud_precision": make_scorer(precision_score, zero_division=0),
        "fraud_recall": make_scorer(recall_score, zero_division=0),
    }
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(cfg.training.cv.shuffle),
        random_state=int(cfg.seed),
    )
    results = cross_validate(
        build_model_pipeline(cfg),
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=int(cfg.training.cv.n_jobs),
        error_score="raise",
    )
    return _cv_metrics(results)


def train_model(cfg: DictConfig) -> Path:
    set_seed(int(cfg.seed))
    train_df = pd.read_csv(resolve_path(cfg.paths.intermediate_train))
    X_train, y_train = _split_xy(train_df, str(cfg.target_column))

    cv_metrics = _run_cross_validation(cfg, X_train, y_train)
    pipeline = build_model_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    train_score = score_vector(pipeline, X_train)
    train_results = evaluate_predictions(y_train, train_pred, train_score, prefix="train")

    model_path = resolve_path(cfg.paths.model)
    ensure_dir(model_path.parent)
    with model_path.open("wb") as file:
        pickle.dump(pipeline, file)

    reports_dir = ensure_dir(cfg.paths.reports_dir)
    write_json(reports_dir / "cv_metrics.json", cv_metrics)
    write_json(reports_dir / "train_metrics.json", train_results)
    write_resolved_config(cfg, reports_dir / "config_resolved.yaml")

    if mlflow_enabled(cfg):
        configure_mlflow(cfg)
        with mlflow.start_run(run_name=str(cfg.mlflow.run_name)) as run:
            log_common_params(cfg)
            log_metrics(cv_metrics)
            log_metrics(train_results["metrics"])
            mlflow.set_tag("pipeline_stage", "train")
            write_json(
                reports_dir / "mlflow_run.json",
                {
                    "run_id": run.info.run_id,
                    "tracking_uri": str(cfg.mlflow.tracking_uri),
                    "model_artifact_path": str(cfg.mlflow.model_artifact_path),
                },
            )
    else:
        write_json(reports_dir / "mlflow_run.json", {"run_id": None, "tracking_uri": None})

    LOGGER.info("Trained model and wrote artifact: %s", model_path)
    return model_path
