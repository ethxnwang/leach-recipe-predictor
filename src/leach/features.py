"""Generic feature engineering driven by ``config.yaml -> feature_engineering``.

The transforms are generic modelling priors, not constants from the simulator: log scales
for time, size and concentration; inverse absolute temperature (Arrhenius-style rate
dependence); a squared term so a smooth optimum can be expressed; and category x numeric
interactions so, for example, the best pH may differ by feed type.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

KELVIN_OFFSET = 273.15  # unit conversion, not a tunable


def engineered_names(fe_cfg: Mapping, categories: Mapping[str, list[str]]) -> list[str]:
    """Names of the columns ``FeatureEngineer`` adds, in a fixed order."""
    names = [f"log_{c}" for c in fe_cfg.get("log", [])]
    names += [f"inv_kelvin_{c}" for c in fe_cfg.get("inverse_kelvin", [])]
    names += [f"sq_{c}" for c in fe_cfg.get("square", [])]
    for cat, cols in (fe_cfg.get("interact_with_category") or {}).items():
        names += [f"{c}_x_{cat}_{lvl}" for c in cols for lvl in categories[cat]]
    return names


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless transformer that appends engineered columns to a DataFrame."""

    def __init__(self, fe_cfg: dict, categories: dict):
        self.fe_cfg = fe_cfg
        self.categories = categories

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.fe_cfg.get("log", []):
            X[f"log_{c}"] = np.log1p(X[c].astype(float))
        for c in self.fe_cfg.get("inverse_kelvin", []):
            X[f"inv_kelvin_{c}"] = 1.0 / (X[c].astype(float) + KELVIN_OFFSET)
        for c in self.fe_cfg.get("square", []):
            X[f"sq_{c}"] = X[c].astype(float) ** 2
        for cat, cols in (self.fe_cfg.get("interact_with_category") or {}).items():
            for c in cols:
                for lvl in self.categories[cat]:
                    X[f"{c}_x_{cat}_{lvl}"] = X[c].astype(float) * (X[cat].astype(str) == lvl)
        return X
