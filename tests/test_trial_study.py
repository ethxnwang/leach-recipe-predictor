import copy

import numpy as np
import pandas as pd

from leach.simulate import LabSimulator, sample_batches
from leach.trial_study import STRATEGIES, run_greedy, run_one, summarize, trials_to_target


def test_trials_to_target():
    assert trials_to_target([0.1, 0.5, 0.8, 0.9], 0.75) == 3
    assert trials_to_target([0.1, 0.2], 0.75) is None


def _lab(cfg, seed=0):
    rng = np.random.default_rng(seed)
    batch = sample_batches(1, rng, cfg["simulation"], "T", ensure_all_types=False)[0]
    return LabSimulator(batch, cfg, rng)


def test_greedy_uses_budget_and_respects_bounds(cfg):
    bounds = cfg["simulation"]["recipe_bounds"]
    cols = list(bounds)
    lo = np.array([bounds[c][0] for c in cols])
    hi = np.array([bounds[c][1] for c in cols])
    lab = _lab(cfg)
    seen = []
    original_run = lab.run

    def spy(recipe):
        seen.append(np.array([recipe[c] for c in cols]))
        return original_run(recipe)

    lab.run = spy
    run_greedy(lab, 12, lo, hi, cols, (lo + hi) / 2, 0.25, 0.03)
    assert len(seen) == 12
    assert all(np.all((x >= lo) & (x <= hi)) for x in seen)


def test_run_one_produces_row_per_strategy_and_target(bundle, fast_cfg):
    cfg = copy.deepcopy(fast_cfg)
    cfg["trial_study"].update({"budget": 4, "oracle_de": {"maxiter": 5, "popsize": 5, "tol": 1e-3}})
    rows = run_one(0, bundle, cfg)
    assert len(rows) == len(STRATEGIES) * len(cfg["trial_study"]["targets"])
    assert all(r["n_trials_run"] <= 4 for r in rows)
    summary = summarize(pd.DataFrame(rows), budget=4)
    assert summary["success_rate"].between(0, 1).all()
