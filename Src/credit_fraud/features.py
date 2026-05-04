from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from credit_fraud.utils import ensure_dir, resolve_path, write_json


LOGGER = logging.getLogger(__name__)


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Feature transformer that learns scaler and cluster state from fit data only."""

    def __init__(
        self,
        log_time: bool = True,
        log_amount: bool = True,
        amount_per_unit_time: bool = True,
        epsilon: float = 1.0e-9,
        scale: bool = True,
        rolling_amount_mean_enabled: bool = False,
        rolling_amount_mean_window: int = 10,
        kmeans_enabled: bool = True,
        kmeans_n_clusters: int = 10,
        kmeans_feature_name: str = "cluster",
        random_state: int = 42,
    ) -> None:
        self.log_time = log_time
        self.log_amount = log_amount
        self.amount_per_unit_time = amount_per_unit_time
        self.epsilon = epsilon
        self.scale = scale
        self.rolling_amount_mean_enabled = rolling_amount_mean_enabled
        self.rolling_amount_mean_window = rolling_amount_mean_window
        self.kmeans_enabled = kmeans_enabled
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_feature_name = kmeans_feature_name
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureTransformer":
        features = self._make_base_features(X)
        self.feature_names_in_ = list(features.columns)

        if self.scale:
            self.scaler_ = RobustScaler()
            scaled = self.scaler_.fit_transform(features)
        else:
            self.scaler_ = None
            scaled = features.to_numpy(dtype=float)

        self.kmeans_ = None
        self.kmeans_n_clusters_ = 0
        if self.kmeans_enabled and len(features) >= 2:
            self.kmeans_n_clusters_ = min(int(self.kmeans_n_clusters), len(features))
            self.kmeans_ = KMeans(
                n_clusters=self.kmeans_n_clusters_,
                random_state=int(self.random_state),
                n_init="auto",
            )
            self.kmeans_.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self._make_base_features(X)
        features = features.reindex(columns=self.feature_names_in_, fill_value=0.0)

        if self.scaler_ is not None:
            values = self.scaler_.transform(features)
        else:
            values = features.to_numpy(dtype=float)

        transformed = pd.DataFrame(values, columns=self.feature_names_in_, index=features.index)
        if self.kmeans_enabled:
            if self.kmeans_ is None:
                labels = np.zeros(len(transformed), dtype=int)
            else:
                labels = self.kmeans_.predict(values)
            transformed[self.kmeans_feature_name] = labels.astype(float)
        return transformed

    def _make_base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=getattr(self, "raw_feature_names_", None))
        else:
            self.raw_feature_names_ = list(X.columns)

        missing = {"Time", "Amount"} - set(X.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")

        features = X.copy()
        if self.log_time:
            features["Time"] = np.log1p(features["Time"].clip(lower=0))
        if self.log_amount:
            features["Amount"] = np.log1p(features["Amount"].clip(lower=0))
        if self.amount_per_unit_time:
            denominator = features["Time"].abs().clip(lower=float(self.epsilon))
            features["amount_per_unit_time"] = features["Amount"] / denominator
        if self.rolling_amount_mean_enabled:
            features["amount_rolling_mean"] = (
                features["Amount"]
                .rolling(window=int(self.rolling_amount_mean_window), min_periods=1)
                .mean()
            )
        return features


def build_feature_transformer(cfg: DictConfig) -> FraudFeatureTransformer:
    return FraudFeatureTransformer(
        log_time=bool(cfg.features.log_time),
        log_amount=bool(cfg.features.log_amount),
        amount_per_unit_time=bool(cfg.features.amount_per_unit_time),
        epsilon=float(cfg.features.epsilon),
        scale=bool(cfg.features.scale),
        rolling_amount_mean_enabled=bool(cfg.features.rolling_amount_mean.enabled),
        rolling_amount_mean_window=int(cfg.features.rolling_amount_mean.window),
        kmeans_enabled=bool(cfg.features.kmeans.enabled),
        kmeans_n_clusters=int(cfg.features.kmeans.n_clusters),
        kmeans_feature_name=str(cfg.features.kmeans.feature_name),
        random_state=int(cfg.seed),
    )


def _split_xy(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[target_column]), frame[target_column]


def build_feature_datasets(cfg: DictConfig) -> tuple[Path, Path]:
    """Fit preprocessing on train only, then write transformed train/test snapshots."""
    train_df = pd.read_csv(resolve_path(cfg.paths.intermediate_train))
    test_df = pd.read_csv(resolve_path(cfg.paths.intermediate_test))
    target_column = str(cfg.target_column)

    X_train, y_train = _split_xy(train_df, target_column)
    X_test, y_test = _split_xy(test_df, target_column)

    transformer = build_feature_transformer(cfg)
    transformer.fit(X_train, y_train)
    train_features = transformer.transform(X_train)
    test_features = transformer.transform(X_test)

    train_final = train_features.copy()
    test_final = test_features.copy()
    train_final[target_column] = y_train.to_numpy()
    test_final[target_column] = y_test.to_numpy()

    train_path = resolve_path(cfg.paths.final_train)
    test_path = resolve_path(cfg.paths.final_test)
    ensure_dir(train_path.parent)
    ensure_dir(test_path.parent)
    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    preprocessor_path = resolve_path(cfg.paths.preprocessor)
    ensure_dir(preprocessor_path.parent)
    with preprocessor_path.open("wb") as file:
        pickle.dump(transformer, file)

    report = {
        "fit_on": str(resolve_path(cfg.paths.intermediate_train)),
        "transformed": [
            str(resolve_path(cfg.paths.intermediate_train)),
            str(resolve_path(cfg.paths.intermediate_test)),
        ],
        "feature_count": int(train_features.shape[1]),
        "features": list(train_features.columns),
    }
    write_json(resolve_path(cfg.paths.reports_dir) / "features.json", report)
    LOGGER.info("Wrote final feature datasets and preprocessor artifact.")
    return train_path, test_path
