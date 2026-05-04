from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig

from credit_fraud.metrics import score_vector
from credit_fraud.utils import get_git_commit, resolve_path, safe_mlflow_params


LOGGER = logging.getLogger(__name__)


def mlflow_enabled(cfg: DictConfig) -> bool:
    return str(cfg.mlflow.enabled).lower() == "true"


def configure_mlflow(cfg: DictConfig) -> None:
    if not mlflow_enabled(cfg):
        return
    mlflow.set_tracking_uri(_tracking_uri(str(cfg.mlflow.tracking_uri)))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))


def _tracking_uri(uri: str) -> str:
    parsed = urlparse(uri)
    is_windows_drive = len(parsed.scheme) == 1 and len(uri) > 2 and uri[1] == ":"
    if parsed.scheme and not is_windows_drive:
        return uri
    return Path(uri).expanduser().resolve().as_uri()


def log_common_params(cfg: DictConfig) -> None:
    params = safe_mlflow_params(cfg)
    params["git_commit"] = get_git_commit(cfg.project.root)
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, Any]) -> None:
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and value == value
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def log_model_artifact(cfg: DictConfig, model: Any, input_example: pd.DataFrame) -> str:
    model_input = input_example.head(5).copy()
    predictions = score_vector(model, model_input)
    signature = infer_signature(model_input, predictions)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=str(cfg.mlflow.model_artifact_path),
        signature=signature,
        input_example=model_input,
    )
    return f"runs:/{mlflow.active_run().info.run_id}/{cfg.mlflow.model_artifact_path}"


def log_report_artifacts(reports_dir: str | Path) -> None:
    reports_path = resolve_path(reports_dir)
    if reports_path.exists():
        mlflow.log_artifacts(str(reports_path), artifact_path="reports")


def should_promote(
    candidate_metric: float,
    champion_metric: float | None,
    min_delta: float,
) -> bool:
    if champion_metric is None:
        return True
    return candidate_metric > champion_metric + min_delta


def _champion_metric(
    client: MlflowClient,
    model_name: str,
    alias: str,
    metric_name: str,
) -> float | None:
    try:
        champion = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(champion.run_id)
    except MlflowException:
        return None
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def promote_if_better(cfg: DictConfig, run_id: str, candidate_metric: float) -> dict[str, Any]:
    if not bool(cfg.mlflow.registry.enabled):
        return {"promoted": False, "reason": "registry_disabled"}

    client = MlflowClient()
    model_name = str(cfg.mlflow.registry.model_name)
    alias = str(cfg.mlflow.registry.champion_alias)
    metric_name = str(cfg.training.scoring.champion_metric)
    min_delta = float(cfg.training.scoring.min_delta)

    current_metric = _champion_metric(client, model_name, alias, metric_name)
    promote = should_promote(candidate_metric, current_metric, min_delta)
    result: dict[str, Any] = {
        "promoted": promote,
        "candidate_metric": candidate_metric,
        "champion_metric": current_metric,
        "metric_name": metric_name,
        "model_name": model_name,
        "champion_alias": alias,
    }
    if not promote:
        return result

    model_uri = f"runs:/{run_id}/{cfg.mlflow.model_artifact_path}"
    version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(model_name, alias, version.version)
    client.set_model_version_tag(model_name, version.version, metric_name, str(candidate_metric))
    result["model_version"] = version.version
    LOGGER.info("Promoted MLflow model %s version %s as %s.", model_name, version.version, alias)
    return result
