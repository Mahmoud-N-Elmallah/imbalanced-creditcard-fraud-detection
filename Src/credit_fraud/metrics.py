from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_vector(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores).ravel()
    return np.asarray(model.predict(X)).ravel()


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    prefix: str,
) -> dict[str, float]:
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(labels, y_pred)),
        f"{prefix}_fraud_precision": float(precision_score(labels, y_pred, zero_division=0)),
        f"{prefix}_fraud_recall": float(recall_score(labels, y_pred, zero_division=0)),
        f"{prefix}_fraud_f1": float(f1_score(labels, y_pred, zero_division=0)),
        f"{prefix}_macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
    }
    if len(np.unique(labels)) == 2:
        metrics[f"{prefix}_pr_auc"] = float(average_precision_score(labels, scores))
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics[f"{prefix}_pr_auc"] = float("nan")
        metrics[f"{prefix}_roc_auc"] = float("nan")
    return metrics


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    return {
        "metrics": binary_metrics(y_true, y_pred, y_score, prefix),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
