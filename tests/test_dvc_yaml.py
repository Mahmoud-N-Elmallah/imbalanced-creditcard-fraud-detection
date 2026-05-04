from __future__ import annotations

from pathlib import Path

import yaml


VARIANTS = [
    "xgboost_smote",
    "xgboost_borderline_smote",
    "xgboost_none",
    "lightgbm_smote",
    "random_forest_smote",
    "logistic_regression_none",
]


def test_dvc_experiment_variants_use_isolated_outputs() -> None:
    dvc_yaml = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    stages = dvc_yaml["stages"]

    for variant in VARIANTS:
        train_stage = stages[f"train_{variant}"]
        evaluate_stage = stages[f"evaluate_{variant}"]
        model_path = f"Models/experiments/{variant}/model.pkl"
        reports_path = f"Reports/experiments/{variant}"
        mlflow_run_path = f"{reports_path}/mlflow_run.json"

        assert f"paths.model={model_path}" in train_stage["cmd"]
        assert f"paths.reports_dir={reports_path}" in train_stage["cmd"]
        assert f"paths.model={model_path}" in evaluate_stage["cmd"]
        assert f"paths.reports_dir={reports_path}" in evaluate_stage["cmd"]
        assert model_path in _out_paths(train_stage)
        assert mlflow_run_path in _out_paths(train_stage)
        assert model_path in evaluate_stage["deps"]
        assert mlflow_run_path in evaluate_stage["deps"]
        assert any(metric.startswith(f"{reports_path}/") for metric in _metric_paths(train_stage))
        assert any(metric.startswith(f"{reports_path}/") for metric in _metric_paths(evaluate_stage))


def test_dvc_experiment_variants_have_distinct_mlflow_run_names() -> None:
    dvc_yaml = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    stages = dvc_yaml["stages"]
    run_names = []

    for variant in VARIANTS:
        command = stages[f"train_{variant}"]["cmd"]
        token = next(part for part in command.split() if part.startswith("mlflow.run_name="))
        run_names.append(token.removeprefix("mlflow.run_name="))

    assert len(run_names) == len(set(run_names))


def _metric_paths(stage: dict) -> list[str]:
    paths = []
    for metric in stage["metrics"]:
        paths.extend(metric.keys())
    return paths


def _out_paths(stage: dict) -> list[str]:
    paths = []
    for output in stage["outs"]:
        if isinstance(output, str):
            paths.append(output)
        else:
            paths.extend(output.keys())
    return paths
