"""Exploratory data analysis: summary table and figures saved to ``reports/figures``."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from leach import plots
from leach.config import Config, numeric_features, resolve_path

logger = logging.getLogger(__name__)


def summarize(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-column summary statistics for numeric features and the target."""
    cols = numeric_features(cfg) + [cfg["features"]["target"]]
    return df[cols].describe().T


def run_eda(df: pd.DataFrame, cfg: Config) -> list[Path]:
    """Write distribution, correlation and recovery-vs-feature figures; return their paths."""
    fig_dir = resolve_path(cfg, "figures_dir")
    target = cfg["features"]["target"]
    num = numeric_features(cfg)
    levels = cfg["schema"]["categories"]["feed_type"]
    paths = [
        plots.histograms(df, num + [target], fig_dir / "eda_distributions.png"),
        plots.correlation_heatmap(df, num + [target], fig_dir / "eda_correlations.png"),
        plots.recovery_vs_features(
            df, num, target, "feed_type", levels, fig_dir / "eda_recovery_vs_features.png"
        ),
    ]
    logger.info("EDA figures written: %s", [p.name for p in paths])
    return paths
