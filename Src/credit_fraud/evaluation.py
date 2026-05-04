from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from credit_fraud.metrics import evaluate_predictions, score_vector
from credit_fraud.mlflow_tracking import (
    configure_mlflow,
    log_common_params,
    log_metrics,
    log_model_artifact,
    log_report_artifacts,
    mlflow_enabled,
    promote_if_better,
)
from credit_fraud.utils import ensure_dir, read_json, resolve_path, write_json


LOGGER = logging.getLogger(__name__)


def _split_xy(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[target_column]), frame[target_column]


def _write_figures(cfg: DictConfig, y_true: pd.Series, y_pred, y_score) -> None:
    figures_dir = ensure_dir(cfg.paths.figures_dir)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _existing_run_id(cfg: DictConfig) -> str | None:
    run_file = resolve_path(cfg.paths.reports_dir) / "mlflow_run.json"
    if not run_file.exists():
        return None
    return read_json(run_file).get("run_id")


def evaluate_model(cfg: DictConfig) -> Path:
    model_path = resolve_path(cfg.paths.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")

    with model_path.open("rb") as file:
        model = pickle.load(file)
    test_df = pd.read_csv(resolve_path(cfg.paths.intermediate_test))
    X_test, y_test = _split_xy(test_df, str(cfg.target_column))
    y_pred = model.predict(X_test)
    y_score = score_vector(model, X_test)

    results = evaluate_predictions(y_test, y_pred, y_score, prefix="test")
    reports_dir = ensure_dir(cfg.paths.reports_dir)
    write_json(reports_dir / "test_metrics.json", results["metrics"])
    write_json(reports_dir / "classification_report.json", results["classification_report"])
    write_json(reports_dir / "confusion_matrix.json", {"confusion_matrix": results["confusion_matrix"]})
    _write_figures(cfg, y_test, y_pred, y_score)

    registry_result = {"promoted": False, "reason": "mlflow_disabled"}
    if mlflow_enabled(cfg):
        configure_mlflow(cfg)
        run_id = _existing_run_id(cfg)
        run_context = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(
            run_name=str(cfg.mlflow.run_name)
        )
        with run_context as run:
            if run_id is None:
                log_common_params(cfg)
            log_metrics(results["metrics"])
            mlflow.set_tag("pipeline_stage", "evaluate")
            log_model_artifact(cfg, model, X_test)
            log_report_artifacts(reports_dir)
            metric_name = str(cfg.training.scoring.champion_metric)
            candidate_metric = float(results["metrics"][metric_name])
            registry_result = promote_if_better(cfg, run.info.run_id, candidate_metric)

    write_json(reports_dir / "model_registry.json", registry_result)
    LOGGER.info("Evaluated model on untouched test split.")
    return reports_dir / "test_metrics.json"
