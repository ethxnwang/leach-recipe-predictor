"""Out-of-distribution checks: per-feature training ranges, unseen categories, and a
Mahalanobis distance on feed composition (chi-square threshold, assumes rough normality)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from leach.config import Config, categorical_features, feed_numeric_features, numeric_features


@dataclass
class OODDetector:
    ranges: dict[str, tuple[float, float]]
    categories: dict[str, list[str]]
    tolerance: float
    mahal_cols: list[str]
    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, df: pd.DataFrame, cfg: Config) -> OODDetector:
        mcols = feed_numeric_features(cfg)
        Xm = df[mcols].to_numpy(float)
        cov = np.atleast_2d(np.cov(Xm, rowvar=False))
        return cls(
            ranges={c: (float(df[c].min()), float(df[c].max())) for c in numeric_features(cfg)},
            categories={c: sorted(df[c].astype(str).unique()) for c in categorical_features(cfg)},
            tolerance=cfg["ood"]["range_tolerance"],
            mahal_cols=mcols,
            mean=Xm.mean(axis=0),
            inv_cov=np.linalg.pinv(cov),
            threshold=float(np.sqrt(chi2.ppf(cfg["ood"]["mahalanobis_quantile"], df=len(mcols)))),
        )

    def mahalanobis(self, df: pd.DataFrame) -> np.ndarray:
        d = df[self.mahal_cols].to_numpy(float) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.inv_cov, d))

    def _range_reasons(self, row: Mapping) -> list[str]:
        out = []
        for col, (lo, hi) in self.ranges.items():
            if col not in row:
                continue
            slack = self.tolerance * (hi - lo)
            if not lo - slack <= row[col] <= hi + slack:
                out.append(f"{col}={row[col]:.4g} outside training range [{lo:.4g}, {hi:.4g}]")
        return out

    def _category_reasons(self, row: Mapping) -> list[str]:
        return [
            f"{col}='{row[col]}' not seen in training (seen: {levels})"
            for col, levels in self.categories.items()
            if col in row and str(row[col]) not in levels
        ]

    def check(self, df: pd.DataFrame) -> list[list[str]]:
        """Reasons each row is out-of-distribution (empty list = in-distribution)."""
        has_mahal = all(c in df.columns for c in self.mahal_cols)
        dist = self.mahalanobis(df) if has_mahal else np.zeros(len(df))
        reasons = []
        for row, d in zip(df.to_dict(orient="records"), dist, strict=True):
            r = self._category_reasons(row) + self._range_reasons(row)
            if d > self.threshold:
                r.append(f"feed Mahalanobis distance {d:.2f} > {self.threshold:.2f}")
            reasons.append(r)
        return reasons

    def feed_reasons(self, feed: Mapping) -> list[str]:
        return self.check(pd.DataFrame([dict(feed)]))[0]
