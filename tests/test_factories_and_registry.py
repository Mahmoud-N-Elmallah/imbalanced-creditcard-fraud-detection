from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from credit_fraud.factories import build_model, build_sampler
from credit_fraud.mlflow_tracking import _tracking_uri, promote_if_better, should_promote


SAMPLERS = [
    "none",
    "smote",
    "borderline_smote",
    "adasyn",
    "smoteenn",
    "smotetomek",
    "kmeans_smote",
    "random_under",
    "allknn",
    "cluster_centroids",
    "smoten",
]
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / "Config")


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_hydra_can_instantiate_sampler_configs(sampler: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config", overrides=[f"sampler={sampler}", "mlflow.enabled=false"])
    sampler_instance = build_sampler(cfg)
    if sampler == "none":
        assert sampler_instance is None
    else:
        assert sampler_instance is not None


@pytest.mark.parametrize("model", MODELS)
def test_hydra_can_instantiate_model_configs(model: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(config_name="config", overrides=[f"model={model}", "mlflow.enabled=false"])
    assert build_model(cfg) is not None


def test_registry_promotion_decision_rules() -> None:
    assert should_promote(candidate_metric=0.8, champion_metric=None, min_delta=0.0)
    assert not should_promote(candidate_metric=0.79, champion_metric=0.8, min_delta=0.0)
    assert should_promote(candidate_metric=0.81, champion_metric=0.8, min_delta=0.0)


def test_mlflow_tracking_uri_normalizes_windows_paths() -> None:
    assert _tracking_uri("https://dagshub.com/example/repo.mlflow").startswith("https://")
    assert _tracking_uri(r"C:\tmp\mlruns").startswith("file:///")


def test_registry_disabled_does_not_need_model_uri(tmp_path) -> None:
    with initialize_config_dir(version_base=None, config_dir=_config_dir()):
        cfg = compose(
            config_name="config",
            overrides=[
                "mlflow.enabled=false",
                "mlflow.registry.enabled=false",
                f"project.root={tmp_path}",
            ],
        )

    result = promote_if_better(
        cfg=cfg,
        run_id="fake-run",
        candidate_metric=0.9,
        model_uri="models:/m-fake",
    )

    assert result == {"promoted": False, "reason": "registry_disabled"}
