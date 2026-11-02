"""Shared fixtures: a small, fast configuration and a quickly-trained model bundle."""

from __future__ import annotations

import copy

import pytest

from leach.config import load_config
from leach.simulate import simulate_trials


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def fast_cfg(cfg):
    """Config with a tiny model grid and small DE/bootstrap budgets for fast tests."""
    c = copy.deepcopy(cfg)
    c["model"]["candidates"] = {
        "mean": cfg["model"]["candidates"]["mean"],
        "ridge": {"estimator": "ridge", "grid": {"alpha": [0.1, 1.0]}},
        "gradient_boosting": {
            "estimator": "gradient_boosting",
            "params": {"n_estimators": 50, "max_depth": 2},
        },
    }
    c["model"]["outer_folds"] = 3
    c["model"]["inner_folds"] = 2
    c["uncertainty"]["n_bootstrap"] = 5
    c["uncertainty"]["conformal_folds"] = 3
    c["recommend"]["de"].update({"maxiter": 10, "popsize": 6})
    return c


@pytest.fixture(scope="session")
def small_df(cfg):
    return simulate_trials(cfg, n_trials=72, n_batches=8, seed=123)


@pytest.fixture(scope="session")
def bundle(fast_cfg, small_df):
    from leach.train import fit_bundle

    fitted, _ = fit_bundle(small_df, fast_cfg, "gradient_boosting", seed=0)
    return fitted
