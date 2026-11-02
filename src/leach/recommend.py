"""Recipe recommender: maximise predicted recovery minus a reagent-cost penalty.

Search uses scipy's differential evolution over the recipe variables, bounded by the range
seen in training (so the optimiser never asks the model to extrapolate in recipe space).
Every candidate the optimiser evaluates is kept, and the top-k are chosen greedily subject
to a minimum normalised distance so the returned recipes are genuinely different.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from leach.bundle import ModelBundle
from leach.config import Config, recipe_features
from leach.schema import DataValidationError, build_feed_model

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    recipe: dict[str, float]
    predicted_recovery: float
    lower: float
    upper: float
    model_spread_std: float
    cost: float
    objective: float
    is_ood: bool
    ood_reasons: str


@dataclass
class RecommendResult:
    feed: dict
    cost_weight: float
    recommendations: list[Recommendation]
    feed_is_ood: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_feed(feed: Mapping, cfg: Config) -> dict:
    try:
        return build_feed_model(cfg).model_validate(dict(feed)).model_dump()
    except ValueError as exc:
        raise DataValidationError([str(exc)]) from exc


def normalise(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span


def recipe_cost(X: np.ndarray, lo: np.ndarray, hi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalised cost in [0, 1]: weighted position of each variable between its bounds."""
    total = weights.sum()
    if total == 0:
        return np.zeros(len(X))
    return normalise(X, lo, hi) @ weights / total


def candidate_frame(feed: Mapping, X: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(X, columns=cols).assign(**dict(feed))


def select_diverse(
    X: np.ndarray, score: np.ndarray, lo: np.ndarray, hi: np.ndarray, k: int, min_dist: float
) -> list[int]:
    """Greedy best-first pick of ``k`` rows at least ``min_dist`` apart (normalised, per-dim)."""
    Z = normalise(X, lo, hi)
    chosen: list[int] = []
    for i in np.argsort(-score):
        if len(chosen) == k:
            break
        if all(np.linalg.norm(Z[i] - Z[j]) / np.sqrt(Z.shape[1]) >= min_dist for j in chosen):
            chosen.append(int(i))
    return chosen


def _bound_warnings(recipe: Mapping, bounds: Mapping, tol: float) -> list[str]:
    out = []
    for col, (lo, hi) in bounds.items():
        eps = tol * (hi - lo)
        side = "lower" if recipe[col] <= lo + eps else "upper" if recipe[col] >= hi - eps else None
        if side:
            out.append(f"top recipe sits at the {side} search bound for {col}; the optimum may lie "
                       "outside the range covered by training data")
    return out


def _search(objective, bounds: list[tuple[float, float]], de_cfg: Mapping, seed: int) -> None:
    differential_evolution(
        objective,
        bounds,
        vectorized=True,
        updating="deferred",
        polish=False,
        seed=seed,
        maxiter=de_cfg["maxiter"],
        popsize=de_cfg["popsize"],
        tol=de_cfg["tol"],
        mutation=tuple(de_cfg["mutation"]),
        recombination=de_cfg["recombination"],
    )


def recommend(
    bundle: ModelBundle,
    feed: Mapping,
    cfg: Config,
    cost_weight: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> RecommendResult:
    """Search recipe space for ``feed``; return top-k recipes with intervals and OOD flags."""
    rc = cfg["recommend"]
    feed = validate_feed(feed, cfg)
    w = rc["cost_weight"] if cost_weight is None else float(cost_weight)
    k = min(rc["top_k"] if top_k is None else int(top_k), rc["max_top_k"])
    cols = recipe_features(cfg)
    lo = np.array([bundle.recipe_bounds[c][0] for c in cols])
    hi = np.array([bundle.recipe_bounds[c][1] for c in cols])
    weights = np.array([rc["cost_coefficients"].get(c, 0.0) for c in cols])
    seen_X: list[np.ndarray] = []
    seen_obj: list[np.ndarray] = []

    def objective(x: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(x.T) if x.ndim == 2 else x[None, :]
        obj = bundle.predict(candidate_frame(feed, X, cols)) - w * recipe_cost(X, lo, hi, weights)
        seen_X.append(X)
        seen_obj.append(obj)
        return -obj if x.ndim == 2 else -obj[0]

    _search(objective, list(zip(lo, hi, strict=True)), rc["de"], cfg["seed"] if seed is None else seed)
    X_all, obj_all = np.vstack(seen_X), np.concatenate(seen_obj)
    idx = select_diverse(X_all, obj_all, lo, hi, k, rc["diversity_min_distance"])
    frame = bundle.predict_frame(candidate_frame(feed, X_all[idx], cols))
    costs = recipe_cost(X_all[idx], lo, hi, weights)

    recs = [
        Recommendation(
            recipe={c: float(v) for c, v in zip(cols, X_all[i], strict=True)},
            predicted_recovery=float(row.predicted_recovery),
            lower=float(row.lower),
            upper=float(row.upper),
            model_spread_std=float(row.model_spread_std),
            cost=float(cost),
            objective=float(obj_all[i]),
            is_ood=bool(row.is_ood),
            ood_reasons=row.ood_reasons,
        )
        for i, row, cost in zip(idx, frame.itertuples(), costs, strict=True)
    ]
    feed_reasons = bundle.ood.feed_reasons(feed)
    warnings = [f"feed is out of distribution: {r}; predictions are extrapolations"
                for r in feed_reasons]
    if recs:
        warnings += _bound_warnings(recs[0].recipe, bundle.recipe_bounds, rc["bound_tolerance"])
    for msg in warnings:
        logger.warning(msg)
    return RecommendResult(feed, w, recs, bool(feed_reasons), warnings)
