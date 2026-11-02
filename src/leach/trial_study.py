"""Simulation study: trials needed to reach a target recovery, manual vs model-guided.

For each run a NEW feed batch is drawn from the simulator and every strategy gets the same
trial budget on its own virtual lab (same batch, independent measurement noise):

* ``random``              - uniformly random recipes within the search bounds.
* ``manual_greedy``       - one-factor-at-a-time hill climb from the mid-point recipe, accepting
                            a move when the *measured* (noisy) recovery improves; step halves
                            after a full sweep without improvement.
* ``model_top_k``         - run the recommender's diverse top-k recipes in order (open loop;
                            the model is never updated with new results).
* ``model_seeded_greedy`` - the manual hill climb, started from the model's top recipe.

A trial counts as reaching the target when its NOISE-FREE recovery meets it, so lucky
measurement noise is not rewarded. Strategies only ever see noisy measurements.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from leach import plots
from leach.bundle import ModelBundle
from leach.config import Config, recipe_features, resolve_path
from leach.recommend import recommend
from leach.simulate import LabSimulator, sample_batches

logger = logging.getLogger(__name__)

DISCLAIMER = "Synthetic data; demonstrates the methodology only; unrelated to any proprietary work."
STRATEGIES = ("random", "manual_greedy", "model_top_k", "model_seeded_greedy")


def trials_to_target(truths: Sequence[float], target: float) -> int | None:
    """1-based index of the first trial whose true recovery meets ``target``."""
    return next((i + 1 for i, t in enumerate(truths) if t >= target), None)


def run_random(lab: LabSimulator, budget: int, lo, hi, cols: list[str], rng) -> None:
    for _ in range(budget):
        lab.run(dict(zip(cols, rng.uniform(lo, hi), strict=True)))


def run_sequence(lab: LabSimulator, recipes: Sequence[Mapping], budget: int) -> None:
    for recipe in recipes[:budget]:
        lab.run(recipe)


def run_greedy(
    lab: LabSimulator,
    budget: int,
    lo,
    hi,
    cols: list[str],
    start: np.ndarray,
    step_fraction: float,
    min_step_fraction: float,
) -> None:
    """One-factor-at-a-time hill climb on noisy measurements."""
    x = np.clip(np.asarray(start, float), lo, hi)
    best = lab.run(dict(zip(cols, x, strict=True)))
    step, min_step = step_fraction * (hi - lo), min_step_fraction * (hi - lo)
    while len(lab.truth_history) < budget:
        improved = False
        for i in range(len(cols)):
            for direction in (1.0, -1.0):
                if len(lab.truth_history) >= budget:
                    return
                cand = x.copy()
                cand[i] = np.clip(cand[i] + direction * step[i], lo[i], hi[i])
                if cand[i] == x[i]:
                    continue
                y = lab.run(dict(zip(cols, cand, strict=True)))
                if y > best:
                    x, best, improved = cand, y, True
                    break
        if not improved:
            step = np.maximum(step / 2, min_step)


def oracle_max(lab: LabSimulator, lo, hi, cols: list[str], de_cfg: Mapping, seed: int) -> float:
    """Best achievable noise-free recovery for this batch within the bounds."""

    def neg_truth(x):
        return -lab.truth(pd.DataFrame(np.atleast_2d(x.T), columns=cols))

    res = differential_evolution(
        neg_truth,
        list(zip(lo, hi, strict=True)),
        vectorized=True,
        updating="deferred",
        polish=False,
        seed=seed,
        maxiter=de_cfg["maxiter"],
        popsize=de_cfg["popsize"],
        tol=de_cfg["tol"],
    )
    return float(-res.fun)


def run_one(run_idx: int, bundle: ModelBundle, cfg: Config) -> list[dict]:
    """Simulate every strategy on one freshly drawn batch."""
    ts = cfg["trial_study"]
    seed, budget = ts["seed"] + run_idx, ts["budget"]
    rng = np.random.default_rng(seed)
    batch = sample_batches(
        1, rng, cfg["simulation"], f"{ts['batch_prefix']}{run_idx}-", ensure_all_types=False
    )[0]
    cols = recipe_features(cfg)
    lo = np.array([bundle.recipe_bounds[c][0] for c in cols])
    hi = np.array([bundle.recipe_bounds[c][1] for c in cols])
    feed = {"feed_type": batch.feed_type, **batch.feed}
    rec = recommend(bundle, feed, cfg, cost_weight=ts["cost_weight"], top_k=budget, seed=seed)
    recipes = [r.recipe for r in rec.recommendations]
    top1 = rec.recommendations[0]
    labs = {
        s: LabSimulator(batch, cfg, np.random.default_rng([seed, i]))
        for i, s in enumerate(STRATEGIES)
    }

    greedy = (ts["step_fraction"], ts["min_step_fraction"])
    run_random(labs["random"], budget, lo, hi, cols, np.random.default_rng([seed, 99]))
    run_greedy(labs["manual_greedy"], budget, lo, hi, cols, (lo + hi) / 2, *greedy)
    run_sequence(labs["model_top_k"], recipes, budget)
    run_greedy(
        labs["model_seeded_greedy"],
        budget,
        lo,
        hi,
        cols,
        np.array([top1.recipe[c] for c in cols]),
        *greedy,
    )

    oracle = oracle_max(labs["random"], lo, hi, cols, ts["oracle_de"], seed)
    top1_truth = labs["model_top_k"].truth_history[0]
    rows = []
    for s, lab in labs.items():
        for target in ts["targets"]:
            rows.append(
                {
                    "run": run_idx,
                    "batch_id": batch.batch_id,
                    "feed_type": batch.feed_type,
                    "strategy": s,
                    "target": target,
                    "trials_to_target": trials_to_target(lab.truth_history, target),
                    "n_trials_run": len(lab.truth_history),
                    "best_true_recovery": max(lab.truth_history),
                    "oracle_max": oracle,
                    "achievable": oracle >= target,
                    "top1_predicted": top1.predicted_recovery,
                    "top1_lower": top1.lower,
                    "top1_upper": top1.upper,
                    "top1_true": top1_truth,
                }
            )
    return rows


def run_study(bundle: ModelBundle, cfg: Config) -> pd.DataFrame:
    rows = []
    for i in range(cfg["trial_study"]["n_runs"]):
        rows.extend(run_one(i, bundle, cfg))
        logger.info("trial study run %d/%d done", i + 1, cfg["trial_study"]["n_runs"])
    return pd.DataFrame(rows)


def _censored(runs: pd.DataFrame, budget: int) -> pd.Series:
    """Trials to target with failures counted as budget + 1 (a lower bound on true cost)."""
    return runs["trials_to_target"].fillna(budget + 1)


def summarize(runs: pd.DataFrame, budget: int) -> pd.DataFrame:
    runs = runs.assign(censored=_censored(runs, budget), success=runs["trials_to_target"].notna())
    g = runs.groupby(["target", "strategy"], sort=False)
    out = pd.DataFrame(
        {
            "success_rate": g["success"].mean(),
            "success_rate_if_achievable": runs[runs["achievable"]]
            .groupby(["target", "strategy"], sort=False)["success"]
            .mean(),
            "median_trials_censored": g["censored"].median(),
            "mean_trials_if_success": g["trials_to_target"].mean(),
        }
    )
    return out.reset_index()


def head_to_head(runs: pd.DataFrame, a: str, b: str, budget: int) -> pd.DataFrame:
    """Per target: how often strategy ``a`` needed fewer / equal / more trials than ``b``."""
    runs = runs.assign(censored=_censored(runs, budget))
    wide = runs.pivot_table(index=["target", "run"], columns="strategy", values="censored")
    diff = wide[a] - wide[b]
    res = pd.DataFrame({"a_fewer": diff < 0, "tie": diff == 0, "a_more": diff > 0})
    return res.groupby(level="target").mean().reset_index()


def cumulative_curves(runs: pd.DataFrame, budget: int) -> dict[float, pd.DataFrame]:
    curves = {}
    n = np.arange(1, budget + 1)
    for target, sub in runs.groupby("target"):
        curves[target] = pd.DataFrame(
            {
                s: [(sub.loc[sub["strategy"] == s, "trials_to_target"] <= k).mean() for k in n]
                for s in STRATEGIES
            },
            index=n,
        )
    return curves


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    def fmt(v):
        if isinstance(v, float | np.floating):
            return "-" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _no_help_findings(summary: pd.DataFrame, runs: pd.DataFrame, budget: int) -> list[str]:
    """Data-driven list of situations where model guidance did not beat the manual climb."""
    out = []
    model_strategies = ("model_top_k", "model_seeded_greedy")
    for target, sub in summary.groupby("target"):
        s = sub.set_index("strategy")
        manual = s.loc["manual_greedy"]
        for strat in model_strategies:
            if s.loc[strat, "success_rate"] < manual["success_rate"]:
                out.append(
                    f"target {target:.2f}: {strat} success rate {s.loc[strat, 'success_rate']:.2f} "
                    f"is below manual_greedy ({manual['success_rate']:.2f})."
                )
        medians = s.loc[[*model_strategies, "manual_greedy"], "median_trials_censored"]
        if (medians > budget).all():
            out.append(
                f"target {target:.2f}: the median run of every strategy failed within {budget} "
                "trials, so median trial counts cannot separate them."
            )
        elif medians[list(model_strategies)].min() >= manual["median_trials_censored"]:
            out.append(
                f"target {target:.2f}: no model-guided strategy beat manual_greedy on median "
                f"trials ({medians[list(model_strategies)].min():.1f} vs "
                f"{manual['median_trials_censored']:.1f})."
            )
    by_type = (
        runs.assign(success=runs["trials_to_target"].notna())
        .groupby(["target", "feed_type", "strategy"])["success"]
        .mean()
        .unstack("strategy")
    )
    for (target, ft), row in by_type.iterrows():
        if max(row["model_top_k"], row["model_seeded_greedy"]) < row["manual_greedy"]:
            out.append(
                f"target {target:.2f}, feed type {ft}: manual_greedy success rate "
                f"{row['manual_greedy']:.2f} exceeds both model strategies "
                f"({row['model_top_k']:.2f}, {row['model_seeded_greedy']:.2f})."
            )
    unachievable = runs.loc[~runs["achievable"]].groupby("target")["run"].nunique()
    for target, n in unachievable.items():
        out.append(
            f"target {target:.2f}: {n} run(s) had a feed where the target is unreachable "
            "within the bounds (oracle); no strategy can succeed there."
        )
    return out


def write_report(runs: pd.DataFrame, cfg: Config, figure: Path, path: Path) -> Path:
    ts = cfg["trial_study"]
    budget = ts["budget"]
    summary = summarize(runs, budget)
    one = runs.drop_duplicates("run")
    curse = one["top1_predicted"] - one["top1_true"]
    top1_cov = (
        (one["top1_true"] >= one["top1_lower"]) & (one["top1_true"] <= one["top1_upper"])
    ).mean()
    h2h = [
        head_to_head(runs, a, "manual_greedy", budget).assign(comparison=f"{a} vs manual_greedy")
        for a in ("model_seeded_greedy", "model_top_k")
    ]
    h2h_df = pd.concat(h2h)[["comparison", "target", "a_fewer", "tie", "a_more"]]
    achievable = one[["oracle_max"]].assign(
        **{f">= {t:.2f}": one["oracle_max"] >= t for t in ts["targets"]}
    )
    findings = _no_help_findings(summary, runs, budget)
    lines = [
        "# Trial-efficiency simulation study",
        "",
        f"> **{DISCLAIMER}**",
        "> Generated by `leach trial-study` from the invented simulator; regenerate to update.",
        "",
        "## Setup",
        "",
        f"- {ts['n_runs']} runs; each run draws a new, unseen feed batch from the simulator.",
        f"- Budget: {budget} trials per strategy per run. Targets: {ts['targets']}.",
        "- Success = first trial whose **noise-free** recovery meets the target; strategies only "
        "see noisy measurements.",
        "- Strategies: `random`, `manual_greedy` (one-factor-at-a-time hill climb from the "
        "mid-point recipe), `model_top_k` (recommender's diverse top-k, open loop, no retraining), "
        "`model_seeded_greedy` (hill climb started at the model's top recipe).",
        f"- Medians count failures as {budget + 1} trials (a lower bound on their true cost).",
        "",
        "## Results by target",
        "",
        md_table(summary),
        "",
        "## Head-to-head vs manual_greedy (fraction of runs)",
        "",
        md_table(h2h_df),
        "",
        "## How good is the model's first recommendation?",
        "",
        f"- Mean predicted recovery of top-1 recipe: {one['top1_predicted'].mean():.3f}; mean true "
        f"recovery of that recipe: {one['top1_true'].mean():.3f}.",
        f"- Mean optimism (predicted - true) at the recommended recipe: {curse.mean():+.3f} "
        f"(sd {curse.std():.3f}). Positive values are the expected optimiser's-curse bias.",
        f"- True recovery inside the reported interval for the top-1 recipe in {top1_cov:.0%} of "
        f"runs (nominal {1 - cfg['uncertainty']['alpha']:.0%}).",
        f"- Achievable (oracle) recovery across runs: mean {one['oracle_max'].mean():.3f}, "
        f"min {one['oracle_max'].min():.3f}; share of runs where each target is reachable: "
        + ", ".join(f"{c} {achievable[c].mean():.0%}" for c in achievable.columns[1:])
        + ".",
        "",
        "## Where the model does not help",
        "",
        *(
            [f"- {f}" for f in findings]
            or ["- No case found in this study where manual_greedy beat both model strategies."]
        ),
        "",
        f"![cumulative success](figures/{figure.name})",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def run_and_report(bundle: ModelBundle, cfg: Config) -> pd.DataFrame:
    runs = run_study(bundle, cfg)
    runs.to_csv(resolve_path(cfg, "trial_runs"), index=False)
    budget = cfg["trial_study"]["budget"]
    fig = plots.cumulative_success(
        cumulative_curves(runs, budget),
        budget,
        resolve_path(cfg, "figures_dir") / "trial_efficiency.png",
    )
    write_report(runs, cfg, fig, resolve_path(cfg, "trial_report"))
    return summarize(runs, budget)
