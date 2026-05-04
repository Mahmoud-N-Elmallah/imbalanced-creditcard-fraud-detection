from __future__ import annotations

from typing import Any

from hydra.utils import get_class
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf

from credit_fraud.features import build_feature_transformer


def _params(section: DictConfig) -> dict[str, Any]:
    params = OmegaConf.to_container(section.get("params", {}), resolve=True)
    return params if isinstance(params, dict) else {}


def build_sampler(cfg: DictConfig) -> Any | None:
    if cfg.sampler.target is None:
        return None
    sampler_cls = get_class(str(cfg.sampler.target))
    return sampler_cls(**_params(cfg.sampler))


def build_model(cfg: DictConfig) -> Any:
    model_cls = get_class(str(cfg.model.target))
    return model_cls(**_params(cfg.model))


def build_model_pipeline(cfg: DictConfig) -> Pipeline:
    steps: list[tuple[str, Any]] = [("features", build_feature_transformer(cfg))]
    sampler = build_sampler(cfg)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", build_model(cfg)))
    return Pipeline(steps)
