# Credit Card Fraud Detection

Leakage-safe, Hydra-configured machine learning pipeline for the Kaggle credit card fraud dataset.

The original work is preserved in `Notebooks/fraud.ipynb`. The project now moves the workflow into modular Python code with explicit data boundaries, DVC stages, and MLflow/DagsHub tracking.

## What This Project Does

- Downloads `mlg-ulb/creditcardfraud` from Kaggle only when raw data is missing.
- Splits data into train/test before feature engineering.
- Writes intermediate split data to `Data/intermediate`.
- Fits preprocessing on train only, then transforms train/test into `Data/final`.
- Trains configurable imbalanced-learning pipelines from Hydra config.
- Tracks train/CV/test metrics with MLflow.
- Promotes a new MLflow model version only when `test_pr_auc` beats current `champion`.
- Defines DVC stages for download, split, features, train, and evaluate.
- Serializes local Python objects as `.pkl`: `Artifacts/preprocessor.pkl` and `Models/model.pkl`.

## Project Layout

```text
Config/                 Hydra configs for paths, features, samplers, models, MLflow, DVC
Data/raw/               Raw Kaggle CSV
Data/intermediate/      Train/test split before feature engineering
Data/final/             Train/test snapshots after train-fitted feature engineering
Src/credit_fraud/       Modular pipeline package
Notebooks/              Original exploratory notebook
tests/                  Unit and smoke tests
dvc.yaml                Reproducible pipeline stages
main.py                 Hydra entrypoint
```

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Fill `.env` with Kaggle and DagsHub credentials. `.env` is ignored by Git.

The default DagsHub targets are:

- MLflow: `https://dagshub.com/mahmoudelmalah85/imbalanced-creditcard-fraud-detection.mlflow`
- DVC: `https://dagshub.com/mahmoudelmalah85/imbalanced-creditcard-fraud-detection.dvc`

Configure DVC auth locally, not in committed files:

```powershell
.\.venv\Scripts\dvc.exe remote modify --local dagshub auth basic
.\.venv\Scripts\dvc.exe remote modify --local dagshub user $env:DAGSHUB_USERNAME
.\.venv\Scripts\dvc.exe remote modify --local dagshub password $env:DAGSHUB_TOKEN
```

## Run Pipeline

Run everything:

```powershell
python main.py stage=all
```

Run one stage:

```powershell
python main.py stage=download
python main.py stage=split
python main.py stage=features
python main.py stage=train
python main.py stage=evaluate
```

Run without MLflow, useful before credentials are set:

```powershell
python main.py stage=train mlflow.enabled=false
python main.py stage=evaluate mlflow.enabled=false
```

Use another sampler or model:

```powershell
python main.py stage=train sampler=borderline_smote model=random_forest
python main.py stage=train sampler=none model=logistic_regression
```

Use DVC:

```powershell
.\.venv\Scripts\dvc.exe repro
.\.venv\Scripts\dvc.exe push
```

`dvc repro` trains these isolated experiment variants and logs each one as a separate MLflow run:

```text
xgboost_smote
xgboost_borderline_smote
xgboost_none
lightgbm_smote
random_forest_smote
logistic_regression_none
```

Each variant writes to its own local paths:

```text
Models/experiments/<variant>/model.pkl
Reports/experiments/<variant>/
```

## Leakage Controls

- Raw data is never transformed before split.
- `Data/intermediate/train.csv` and `Data/intermediate/test.csv` are created before feature engineering.
- Feature transformer fits `RobustScaler` and optional KMeans only on train data.
- Model training uses an imblearn pipeline with feature engineering inside cross-validation.
- Samplers are pipeline training steps, so they are not applied during test prediction.
- Test metrics are computed once in `stage=evaluate` from untouched intermediate test data.
- Local fitted objects are serialized with Python pickle using `.pkl` paths.

## Current Defaults

- Champion metric: `test_pr_auc`
- Default model: XGBoost
- Default sampler: SMOTE
- Test size: 30%
- Duplicate handling: exact duplicate rows removed before split
- Rolling amount mean: disabled by default because random split makes production semantics ambiguous

## Verification

```powershell
python -m pytest -q
python -m ruff check main.py Src tests
```
