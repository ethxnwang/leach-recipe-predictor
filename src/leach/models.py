"""Model construction: preprocessing pipeline, bounded regressor and candidate registry.

A candidate spec in ``config.yaml -> model.candidates`` names an ``estimator`` and may add:

* ``engineered_features: true`` - prepend :class:`leach.features.FeatureEngineer`;
* ``target_transform: logit``   - fit on logit(recovery), invert on predict, which keeps
  predictions inside (0, 1) and respects the bounded target;
* ``gp`` (for ``gaussian_process``) - kernel settings;
* ``members`` (for ``voting``)     - inline member specs whose predictions are averaged.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from leach.config import Config, categorical_features, numeric_features
from leach.features import FeatureEngineer, engineered_names

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "dummy": DummyRegressor,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}
MODEL_STEP = "model__"


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


def logit(y: np.ndarray, eps: float) -> np.ndarray:
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _numeric_inputs(cfg: Config, engineered: bool) -> list[str]:
    extra = engineered_names(cfg["feature_engineering"], cfg["schema"]["categories"])
    return numeric_features(cfg) + (extra if engineered else [])


def build_preprocessor(cfg: Config, engineered: bool = False) -> ColumnTransformer:
    """One-hot encode categoricals (fixed category order) and standardise numerics."""
    cats = categorical_features(cfg)
    levels = [cfg["schema"]["categories"][c] for c in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(categories=levels, handle_unknown="ignore"), cats),
            ("num", StandardScaler(), _numeric_inputs(cfg, engineered)),
        ]
    )


def n_model_inputs(cfg: Config, engineered: bool) -> int:
    n_cat = sum(len(cfg["schema"]["categories"][c]) for c in categorical_features(cfg))
    return n_cat + len(_numeric_inputs(cfg, engineered))


def build_gp(gp_cfg: dict, n_inputs: int, seed: int) -> GaussianProcessRegressor:
    """Matern GP with learned signal and noise; ``kernel: ard`` learns one scale per input."""
    scale = np.ones(n_inputs) if gp_cfg["kernel"] == "ard" else 1.0
    kernel = ConstantKernel(1.0) * Matern(length_scale=scale, nu=gp_cfg["nu"]) + WhiteKernel(
        gp_cfg["noise_init"]
    )
    return GaussianProcessRegressor(
        kernel,
        normalize_y=True,
        n_restarts_optimizer=gp_cfg["n_restarts"],
        random_state=seed,
    )


def target_bounds(cfg: Config) -> tuple[float, float]:
    lo, hi = cfg["schema"]["ranges"][cfg["features"]["target"]]
    return float(lo), float(hi)


def _base_estimator(spec: dict, cfg: Config, seed: int) -> BaseEstimator:
    kind = spec["estimator"]
    if kind == "gaussian_process":
        return build_gp(
            spec["gp"], n_model_inputs(cfg, bool(spec.get("engineered_features"))), seed
        )
    est_cls = ESTIMATORS[kind]
    params = dict(spec.get("params") or {})
    if "random_state" in est_cls().get_params():
        params["random_state"] = seed
    return est_cls(**params)


def build_regressor(spec: dict, cfg: Config, seed: int) -> tuple[BaseEstimator, str]:
    """Unclipped regressor for a spec, plus the parameter path to its estimator step."""
    if spec["estimator"] == "voting":
        members = [(k, build_regressor(m, cfg, seed)[0]) for k, m in spec["members"].items()]
        return VotingRegressor(members), ""
    engineered = bool(spec.get("engineered_features"))
    steps = [("features", FeatureEngineer(cfg["feature_engineering"], cfg["schema"]["categories"]))]
    steps = (steps if engineered else []) + [
        ("prep", build_preprocessor(cfg, engineered)),
        ("model", _base_estimator(spec, cfg, seed)),
    ]
    model: BaseEstimator = Pipeline(steps)
    prefix = MODEL_STEP
    if spec.get("target_transform") == "logit":
        eps = cfg["model"]["logit_eps"]
        model = TransformedTargetRegressor(
            model, func=partial(logit, eps=eps), inverse_func=expit, check_inverse=False
        )
        prefix = "regressor__" + prefix
    return model, prefix


def build_candidate(name: str, cfg: Config, seed: int) -> tuple[ClippedRegressor, dict]:
    """Return an unfitted bounded model and its (fully prefixed) hyperparameter grid."""
    spec = cfg["model"]["candidates"][name]
    model, prefix = build_regressor(spec, cfg, seed)
    grid = {f"estimator__{prefix}{k}": v for k, v in (spec.get("grid") or {}).items()}
    lo, hi = target_bounds(cfg)
    return ClippedRegressor(model, lower=lo, upper=hi), grid


def build_with_params(name: str, cfg: Config, seed: int, params: dict) -> ClippedRegressor:
    """Candidate with tuned hyperparameters (unprefixed names) already applied."""
    est, _ = build_candidate(name, cfg, seed)
    _, prefix = build_regressor(cfg["model"]["candidates"][name], cfg, seed)
    return est.set_params(**{f"estimator__{prefix}{k}": v for k, v in params.items()})


def strip_prefix(params: dict) -> dict:
    """Drop the pipeline path, keeping the estimator's own parameter name."""
    return {k.rpartition(MODEL_STEP)[2]: v for k, v in params.items()}
