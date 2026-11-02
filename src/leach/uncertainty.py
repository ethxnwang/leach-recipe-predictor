"""Prediction uncertainty: grouped-CV conformal intervals and a batch bootstrap ensemble.

* ``GroupConformal``: symmetric intervals whose half-width is a finite-sample-corrected
  quantile of out-of-fold absolute residuals, where folds are whole batches. This targets
  coverage on NEW batches (approximate: the final model is refit on all folds).
* ``BootstrapEnsemble``: models refit on batch-level bootstrap resamples. Its spread
  measures model (epistemic) uncertainty only; it is NOT a prediction interval and is
  expected to under-cover. It is reported as a disagreement signal, useful off-distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from leach.evaluation import group_splits


@dataclass
class GroupConformal:
    half_width: float
    alpha: float
    lower_bound: float
    upper_bound: float

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, float)
        lo = np.clip(pred - self.half_width, self.lower_bound, self.upper_bound)
        hi = np.clip(pred + self.half_width, self.lower_bound, self.upper_bound)
        return lo, hi


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile used by split conformal."""
    n = len(abs_residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_residuals, level, method="higher"))


def oof_predictions(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold predictions where each fold holds out whole batches."""
    pred = np.empty(len(y))
    for tr, te in group_splits(X, groups, n_splits, seed):
        pred[te] = clone(estimator).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return pred


def fit_group_conformal(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    alpha: float,
    bounds: tuple[float, float],
    seed: int,
) -> GroupConformal:
    residuals = np.abs(y - oof_predictions(estimator, X, y, groups, n_splits, seed))
    return GroupConformal(conformal_quantile(residuals, alpha), alpha, *bounds)


@dataclass
class BootstrapEnsemble:
    models: list[BaseEstimator]
    alpha: float

    def predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.stack([m.predict(X) for m in self.models])

    def std(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions(X).std(axis=0)

    def interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.predictions(X)
        q = 100 * np.array([self.alpha / 2, 1 - self.alpha / 2])
        lo, hi = np.percentile(p, q, axis=0)
        return lo, hi


def fit_batch_bootstrap(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> BootstrapEnsemble:
    """Resample whole batches with replacement and refit ``n_boot`` clones."""
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    models = []
    for _ in range(n_boot):
        idx = np.concatenate([index_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        models.append(clone(estimator).fit(X.iloc[idx], y[idx]))
    return BootstrapEnsemble(models, alpha)


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Empirical coverage and mean width of intervals."""
    y = np.asarray(y, float)
    inside = (y >= lower) & (y <= upper)
    return {"coverage": float(inside.mean()), "mean_width": float(np.mean(upper - lower))}
