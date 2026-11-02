"""Matplotlib figures with one consistent, recessive style.

Colors are the first slots of a validated colorblind-safe categorical palette; text stays
in neutral ink, never series color.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def color_map(levels: Sequence[str]) -> dict[str, str]:
    """Fixed-order color per category (identity follows the entity, not its rank)."""
    return {lvl: SERIES[i] for i, lvl in enumerate(levels)}


def histograms(df: pd.DataFrame, columns: Sequence[str], path: Path) -> Path:
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.6 * n_rows))
    for ax, col in zip(axes.flat, columns, strict=False):
        ax.hist(df[col], bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.set_title(col, fontsize=9)
    for ax in list(axes.flat)[len(columns) :]:
        ax.set_visible(False)
    fig.suptitle("Feature and target distributions (synthetic)", color=TEXT_PRIMARY)
    return _save(fig, path)


def correlation_heatmap(df: pd.DataFrame, columns: Sequence[str], path: Path) -> Path:
    corr = df[list(columns)].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        if abs(v) >= 0.2 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=TEXT_PRIMARY)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation matrix (|r| >= 0.2 labelled)")
    return _save(fig, path)


def recovery_vs_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    hue: str,
    levels: Sequence[str],
    path: Path,
) -> Path:
    colors = color_map(levels)
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharey=True)
    for ax, col in zip(axes.flat, columns, strict=False):
        for lvl in levels:
            sub = df[df[hue] == lvl]
            ax.scatter(sub[col], sub[target], s=14, alpha=0.75, color=colors[lvl], label=lvl)
        ax.set_xlabel(col)
    for ax in list(axes.flat)[len(columns) :]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel(target)
    axes.flat[0].legend(title=hue, title_fontsize=8)
    fig.suptitle(f"{target} vs features, by {hue}", color=TEXT_PRIMARY)
    return _save(fig, path)


def model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of outer-CV RMSE (mean +/- sd across folds)."""
    s = summary.sort_values("rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(s) + 1.2))
    ax.barh(
        s.index,
        s["rmse_mean"],
        xerr=s["rmse_std"],
        color=SERIES[0],
        height=0.55,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1, "capsize": 3},
    )
    for y, v in enumerate(s["rmse_mean"]):
        ax.text(v, y, f"  {v:.3f}", va="center", ha="left", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlabel("RMSE, nested GroupKFold by batch (lower is better)")
    ax.set_title("Model comparison on unseen batches")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def predicted_vs_actual(
    y: np.ndarray, pred: np.ndarray, lower: np.ndarray, upper: np.ndarray, title: str, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.errorbar(y, pred, yerr=[pred - lower, upper - pred], fmt="none", ecolor=GRID, elinewidth=1)
    ax.scatter(y, pred, s=18, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle="--", label="y = x")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="actual recovery", ylabel="predicted recovery")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, path)


def residuals(y: np.ndarray, pred: np.ndarray, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.scatter(pred, y - pred, s=18, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    ax.set(xlabel="predicted recovery", ylabel="actual - predicted")
    ax.set_title(title)
    return _save(fig, path)


def learning_curve(curve: pd.DataFrame, path: Path) -> Path:
    """RMSE vs number of training trials, one line per model (mean +/- sd over repeats)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (model, sub) in enumerate(curve.groupby("model", sort=False)):
        sub = sub.sort_values("n_trials")
        c = SERIES[i]
        ax.fill_between(
            sub["n_trials"],
            sub["rmse_mean"] - sub["rmse_std"],
            sub["rmse_mean"] + sub["rmse_std"],
            color=c,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(sub["n_trials"], sub["rmse_mean"], color=c, marker="o", markersize=5, label=model)
    ax.set(xlabel="number of training trials", ylabel="RMSE on fresh simulated batches")
    ax.set_xticks(sorted(curve["n_trials"].unique()))
    ax.set_title("Learning curve")
    ax.legend()
    return _save(fig, path)


def cumulative_success(curves: Mapping[float, pd.DataFrame], budget: int, path: Path) -> Path:
    """Fraction of runs that reached each target by trial n, one panel per target."""
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (target, df) in zip(axes, curves.items(), strict=True):
        for i, strategy in enumerate(df.columns):
            ax.step(df.index, df[strategy], where="post", color=SERIES[i], label=strategy)
        ax.set(xlabel="trials run", xlim=(1, budget), ylim=(0, 1.02))
        ax.set_title(f"target recovery >= {target:.2f}")
    axes[0].set_ylabel("fraction of runs at target")
    axes[0].legend(loc="upper left")
    return _save(fig, path)
