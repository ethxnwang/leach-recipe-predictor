"""The deployable artifact: fitted model + uncertainty + OOD detector + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from leach.models import ClippedRegressor
from leach.ood import OODDetector
from leach.uncertainty import BootstrapEnsemble, GroupConformal


@dataclass
class ModelBundle:
    model_name: str
    params: dict
    estimator: ClippedRegressor
    conformal: GroupConformal
    bootstrap: BootstrapEnsemble
    ood: OODDetector
    feature_cols: list[str]
    recipe_bounds: dict[str, tuple[float, float]]
    metadata: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(df[self.feature_cols])

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point prediction, conformal interval, bootstrap spread and OOD flags per row."""
        X = df[self.feature_cols]
        pred = self.estimator.predict(X)
        lo, hi = self.conformal.interval(pred)
        reasons = self.ood.check(df)
        return pd.DataFrame(
            {
                "predicted_recovery": pred,
                "lower": lo,
                "upper": hi,
                "model_spread_std": self.bootstrap.std(X),
                "is_ood": [bool(r) for r in reasons],
                "ood_reasons": ["; ".join(r) for r in reasons],
            },
            index=df.index,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return obj
