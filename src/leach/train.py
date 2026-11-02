"""Training orchestration: compare models, tune, quantify uncertainty, save artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from leach import plots
from leach.bundle import ModelBundle
from leach.config import Config, feature_columns, recipe_features, resolve_path
from leach.evaluation import (
    compare_models,
    holdout_split,
    learning_curve_study,
    regression_metrics,
    select_best,
    tune,
)
from leach.models import target_bounds
from leach.ood import OODDetector
from leach.schema import validate_frame
from leach.simulate import simulate_trials
from leach.uncertainty import coverage, fit_batch_bootstrap, fit_group_conformal

logger = logging.getLogger(__name__)


def git_info(root: Path) -> dict:
    """Current commit and whether tracked files differ from it ("unknown" outside git)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "-uno"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _xyg(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    f = cfg["features"]
    return df[feature_columns(cfg)], df[f["target"]].to_numpy(float), df[f["group"]].to_numpy()


def fit_bundle(train_df: pd.DataFrame, cfg: Config, model_name: str, seed: int) -> tuple[ModelBundle, dict]:
    """Tune ``model_name`` on ``train_df`` and attach intervals + OOD detector."""
    X, y, g = _xyg(train_df, cfg)
    unc = cfg["uncertainty"]
    model, params = tune(model_name, cfg, X, y, g, seed)
    conformal = fit_group_conformal(
        model, X, y, g, unc["conformal_folds"], unc["alpha"], target_bounds(cfg), seed
    )
    boot = fit_batch_bootstrap(
        model, X, y, g, unc["n_bootstrap"], unc["alpha"], np.random.default_rng(seed)
    )
    bounds = {c: (float(train_df[c].min()), float(train_df[c].max())) for c in recipe_features(cfg)}
    bundle = ModelBundle(
        model_name=model_name,
        params=params,
        estimator=model,
        conformal=conformal,
        bootstrap=boot,
        ood=OODDetector.fit(train_df, cfg),
        feature_cols=feature_columns(cfg),
        recipe_bounds=bounds,
    )
    return bundle, params


def evaluate_holdout(bundle: ModelBundle, test_df: pd.DataFrame, cfg: Config) -> dict:
    """Point metrics and interval coverage on held-out batches."""
    X, y, _ = _xyg(test_df, cfg)
    frame = bundle.predict_frame(test_df)
    boot_lo, boot_hi = bundle.bootstrap.interval(X)
    return {
        "metrics": regression_metrics(y, frame["predicted_recovery"]),
        "conformal_interval": coverage(y, frame["lower"], frame["upper"]),
        "bootstrap_interval": coverage(y, boot_lo, boot_hi),
        "nominal_coverage": 1 - cfg["uncertainty"]["alpha"],
        "n_ood_rows": int(frame["is_ood"].sum()),
    }


def _learning_curve(cfg: Config, pool: pd.DataFrame, best: str, best_params: dict, cv_folds: pd.DataFrame, seed: int) -> pd.DataFrame:
    lc = cfg["model"]["learning_curve"]
    eval_df = simulate_trials(cfg, n_trials=lc["eval_n_trials"], n_batches=lc["eval_n_batches"],
                              seed=seed + lc["eval_seed_offset"], prefix=lc["eval_batch_prefix"])
    specs = {}
    for label in lc["models"]:
        name = best if label == "best" else label
        params = best_params if name == best else _modal_params(cv_folds, name)
        specs[f"{name} (selected)" if label == "best" else name] = (name, params)
    return learning_curve_study(cfg, pool, eval_df, specs, seed)


def _modal_params(cv_folds: pd.DataFrame, name: str) -> dict:
    """Most frequently selected hyperparameters across outer folds."""
    picks = cv_folds.loc[cv_folds["model"] == name, "params"].map(lambda p: json.dumps(p, sort_keys=True))
    return json.loads(picks.mode().iloc[0]) if len(picks) else {}


def _figures(cfg: Config, summary: pd.DataFrame, bundle: ModelBundle, test_df: pd.DataFrame,
             curve: pd.DataFrame) -> None:
    fig_dir = resolve_path(cfg, "figures_dir")
    y = test_df[cfg["features"]["target"]].to_numpy()
    frame = bundle.predict_frame(test_df)
    pred = frame["predicted_recovery"].to_numpy()
    pct = int(round(100 * (1 - cfg["uncertainty"]["alpha"])))
    plots.model_comparison(summary, fig_dir / "model_comparison.png")
    plots.predicted_vs_actual(y, pred, frame["lower"].to_numpy(), frame["upper"].to_numpy(),
                              f"Held-out batches: {bundle.model_name} ({pct}% intervals)",
                              fig_dir / "pred_vs_actual.png")
    plots.residuals(y, pred, f"Residuals on held-out batches: {bundle.model_name}",
                    fig_dir / "residuals.png")
    plots.learning_curve(curve, fig_dir / "learning_curve.png")


def run_training(cfg: Config) -> dict:
    """Full training run; writes model bundle, metrics.json, CV/learning-curve CSVs, figures."""
    seed = cfg["seed"]
    data_path = resolve_path(cfg, "data")
    df = validate_frame(pd.read_csv(data_path), cfg)
    _, _, groups = _xyg(df, cfg)

    tr, te = holdout_split(groups, cfg["model"]["test_batch_fraction"], seed)
    train_df, test_df = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    gcol = cfg["features"]["group"]
    logger.info("train: %d rows / %d batches; test: %d rows / %d batches", len(train_df),
                train_df[gcol].nunique(), len(test_df), test_df[gcol].nunique())

    folds, summary = compare_models(cfg, *_xyg(train_df, cfg), seed)
    best = select_best(summary, cfg["model"]["selection_metric"])
    logger.info("selected model: %s\n%s", best, summary.round(4))

    bundle, best_params = fit_bundle(train_df, cfg, best, seed)
    holdout = evaluate_holdout(bundle, test_df, cfg)
    curve = _learning_curve(cfg, df, best, best_params, folds, seed)
    _figures(cfg, summary, bundle, test_df, curve)

    metrics = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(Path(cfg["_root"])),
        "seed": seed,
        "data": {"path": cfg["paths"]["data"], "sha256": file_sha256(data_path), "n_rows": len(df),
                 "n_batches": int(df[gcol].nunique())},
        "split": {"train_batches": sorted(train_df[gcol].unique()),
                  "test_batches": sorted(test_df[gcol].unique())},
        "cv_summary": summary.round(5).to_dict(orient="index"),
        "selected_model": best,
        "selected_params": best_params,
        "holdout": holdout,
        "conformal_half_width": bundle.conformal.half_width,
        "learning_curve": curve.round(5).to_dict(orient="records"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    bundle.metadata = {k: metrics[k] for k in ("created_utc", "git", "seed", "data", "selected_model")}
    bundle.save(resolve_path(cfg, "model"))
    _write_outputs(cfg, metrics, folds, curve)
    return metrics


def _write_outputs(cfg: Config, metrics: dict, folds: pd.DataFrame, curve: pd.DataFrame) -> None:
    out = resolve_path(cfg, "metrics")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=str))
    folds.assign(params=folds["params"].map(json.dumps)).to_csv(resolve_path(cfg, "cv_results"), index=False)
    curve.to_csv(resolve_path(cfg, "learning_curve"), index=False)
    logger.info("wrote %s", out)
