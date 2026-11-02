"""Metrics, batch-grouped splitting, nested cross-validation and the learning curve."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from leach.config import Config, feature_columns
from leach.models import ClippedRegressor, build_candidate, build_with_params, strip_prefix

logger = logging.getLogger(__name__)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, R^2 and mean bias (pred - actual) on the 0-1 recovery scale."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    defined_r2 = len(y_true) > 1 and np.ptp(y_true) > 0
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if defined_r2 else float("nan"),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def holdout_split(
    groups: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole batches: no batch appears in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    return next(gss.split(np.zeros(len(groups)), groups=groups))


def group_kfold(n_splits: int, groups: np.ndarray, seed: int) -> GroupKFold:
    n = min(n_splits, len(np.unique(groups)))
    if n < 2:
        raise ValueError("need at least two distinct batches for grouped CV")
    return GroupKFold(n_splits=n, shuffle=True, random_state=seed)


def group_splits(
    X: pd.DataFrame, groups: np.ndarray, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    yield from group_kfold(n_splits, groups, seed).split(X, groups=groups)


def tune(
    name: str, cfg: Config, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[ClippedRegressor, dict]:
    """Grid-search hyperparameters with GroupKFold on ``groups``; return refit model + params."""
    est, grid = build_candidate(name, cfg, seed)
    if not grid:
        return est.fit(X, y), {}
    search = GridSearchCV(
        est,
        grid,
        cv=group_kfold(cfg["model"]["inner_folds"], groups, seed),
        scoring=cfg["model"]["scoring"],
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, strip_prefix(search.best_params_)


def nested_cv(
    name: str, cfg: Config, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int
) -> list[dict]:
    """Outer GroupKFold for scoring, inner GroupKFold for tuning (no batch leaks across)."""
    folds = []
    for k, (tr, te) in enumerate(group_splits(X, groups, cfg["model"]["outer_folds"], seed)):
        model, params = tune(name, cfg, X.iloc[tr], y[tr], groups[tr], seed)
        metrics = regression_metrics(y[te], model.predict(X.iloc[te]))
        folds.append({"model": name, "fold": k, **metrics, "params": params})
    return folds


def compare_models(
    cfg: Config, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run nested CV for every candidate on identical outer folds.

    Returns:
        (fold-level results, per-model summary with mean/std of each metric)
    """
    rows = []
    for name in cfg["model"]["candidates"]:
        logger.info("nested CV: %s", name)
        rows.extend(nested_cv(name, cfg, X, y, groups, seed))
    folds = pd.DataFrame(rows)
    agg = folds.groupby("model", sort=False)[["rmse", "mae", "r2"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return folds, agg


def select_best(summary: pd.DataFrame, metric: str) -> str:
    """Lowest mean outer-CV error wins (R^2 would be maximised)."""
    col = f"{metric}_mean"
    return str(summary[col].idxmax() if metric == "r2" else summary[col].idxmin())


def learning_curve_study(
    cfg: Config,
    pool: pd.DataFrame,
    eval_df: pd.DataFrame,
    model_params: dict[str, tuple[str, dict]],
    seed: int,
) -> pd.DataFrame:
    """RMSE on fresh batches vs number of training trials sampled from ``pool``.

    Args:
        model_params: label -> (candidate name, fixed hyperparameters).
    """
    lc = cfg["model"]["learning_curve"]
    target, feats = cfg["features"]["target"], feature_columns(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for n in lc["sizes"]:
        n_eff = min(n, len(pool))
        for rep in range(lc["repeats"]):
            sample = pool.iloc[rng.choice(len(pool), n_eff, replace=False)]
            for label, (name, params) in model_params.items():
                est = build_with_params(name, cfg, seed + rep, params)
                est.fit(sample[feats], sample[target])
                m = regression_metrics(eval_df[target], est.predict(eval_df[feats]))
                rows.append({"model": label, "n_trials": n_eff, "repeat": rep, **m})
    raw = pd.DataFrame(rows)
    out = raw.groupby(["model", "n_trials"], sort=False)[["rmse", "mae", "r2"]].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()
