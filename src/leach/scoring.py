"""Score the model against measured wet-lab results (a CSV with features + measured recovery)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from leach.bundle import ModelBundle
from leach.config import Config
from leach.evaluation import regression_metrics
from leach.schema import validate_frame
from leach.uncertainty import coverage


@dataclass
class ScoreReport:
    """Aggregate scores; ``rows`` holds per-trial predictions and errors."""

    metrics: dict
    in_distribution_metrics: dict | None
    interval: dict
    nominal_coverage: float
    n_rows: int
    n_ood_rows: int
    model_name: str
    rows: pd.DataFrame

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("rows")
        return d


def score_lab_data(bundle: ModelBundle, df: pd.DataFrame, cfg: Config) -> ScoreReport:
    """Validate, predict, and compare to measured recovery (batch id optional)."""
    target = cfg["features"]["target"]
    df = validate_frame(df, cfg, require_target=True, require_group=False).reset_index(drop=True)
    frame = bundle.predict_frame(df)
    y = df[target].to_numpy(float)
    rows = pd.concat([df, frame], axis=1).assign(
        error=frame["predicted_recovery"] - y,
        abs_error=(frame["predicted_recovery"] - y).abs(),
        inside_interval=(y >= frame["lower"]) & (y <= frame["upper"]),
    )
    in_dist = ~frame["is_ood"].to_numpy()
    id_metrics = (
        regression_metrics(y[in_dist], frame["predicted_recovery"].to_numpy()[in_dist])
        if in_dist.any() and not in_dist.all()
        else None
    )
    return ScoreReport(
        metrics=regression_metrics(y, frame["predicted_recovery"]),
        in_distribution_metrics=id_metrics,
        interval=coverage(y, frame["lower"].to_numpy(), frame["upper"].to_numpy()),
        nominal_coverage=1 - cfg["uncertainty"]["alpha"],
        n_rows=len(df),
        n_ood_rows=int(frame["is_ood"].sum()),
        model_name=bundle.model_name,
        rows=rows,
    )
