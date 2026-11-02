"""Model construction: preprocessing pipeline, bounded regressor and candidate registry."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from leach.config import Config, categorical_features, numeric_features

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}
PARAM_PREFIX = "estimator__model__"


class ClippedRegressor(RegressorMixin, BaseEstimator):
    """Wrap a regressor so predictions always lie in ``[lower, upper]``.

    Recovery is a fraction, so a linear model's out-of-range predictions are clipped rather
    than reported as e.g. 1.07.
    """

    def __init__(self, estimator: BaseEstimator, lower: float = 0.0, upper: float = 1.0):
        self.estimator = estimator
        self.lower = lower
        self.upper = upper

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.clip(self.estimator_.predict(X), self.lower, self.upper)


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    """One-hot encode categoricals (fixed category order) and standardise numerics."""
    cats = categorical_features(cfg)
    levels = [cfg["schema"]["categories"][c] for c in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(categories=levels, handle_unknown="ignore"), cats),
            ("num", StandardScaler(), numeric_features(cfg)),
        ]
    )


def target_bounds(cfg: Config) -> tuple[float, float]:
    lo, hi = cfg["schema"]["ranges"][cfg["features"]["target"]]
    return float(lo), float(hi)


def build_candidate(name: str, cfg: Config, seed: int) -> tuple[ClippedRegressor, dict]:
    """Return an unfitted bounded pipeline and its (prefixed) hyperparameter grid."""
    spec = cfg["model"]["candidates"][name]
    est_cls = ESTIMATORS[spec["estimator"]]
    params = dict(spec.get("params") or {})
    if "random_state" in est_cls().get_params():
        params["random_state"] = seed
    pipe = Pipeline([("prep", build_preprocessor(cfg)), ("model", est_cls(**params))])
    grid = {PARAM_PREFIX + k: v for k, v in (spec.get("grid") or {}).items()}
    lo, hi = target_bounds(cfg)
    return ClippedRegressor(pipe, lower=lo, upper=hi), grid


def build_with_params(name: str, cfg: Config, seed: int, params: dict) -> ClippedRegressor:
    """Candidate with tuned hyperparameters (unprefixed names) already applied."""
    est, _ = build_candidate(name, cfg, seed)
    return est.set_params(**{PARAM_PREFIX + k: v for k, v in params.items()})


def strip_prefix(params: dict) -> dict:
    return {k.removeprefix(PARAM_PREFIX): v for k, v in params.items()}
