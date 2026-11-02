import numpy as np
import pandas as pd
import pytest

from leach.config import feature_columns
from leach.simulate import (
    LabSimulator,
    allocate_trials,
    expected_recovery,
    sample_batches,
    simulate_trials,
)


def test_shape_columns_and_counts(cfg):
    df = simulate_trials(cfg)
    sim = cfg["simulation"]
    assert len(df) == sim["n_trials"]
    assert df[cfg["features"]["group"]].nunique() == sim["n_batches"]
    assert set(feature_columns(cfg)) <= set(df.columns)
    assert not df.isna().any().any()


def test_target_in_unit_interval(cfg):
    df = simulate_trials(cfg, seed=5)
    y = df[cfg["features"]["target"]]
    assert y.between(0.0, 1.0).all()
    assert y.std() > 0.05  # non-degenerate signal


def test_features_within_design_bounds(cfg):
    df = simulate_trials(cfg, seed=9)
    sim = cfg["simulation"]
    for col, (lo, hi) in {**sim["recipe_bounds"], **sim["feed_ranges"]}.items():
        assert df[col].between(lo, hi).all(), col


def test_reproducible_with_same_seed(cfg):
    pd.testing.assert_frame_equal(simulate_trials(cfg, seed=1), simulate_trials(cfg, seed=1))


def test_different_seeds_differ(cfg):
    a, b = simulate_trials(cfg, seed=1), simulate_trials(cfg, seed=2)
    assert not a.equals(b)


def test_every_feed_type_present(cfg):
    df = simulate_trials(cfg, n_trials=30, n_batches=3, seed=0)
    assert set(df["feed_type"]) == set(cfg["simulation"]["feed_type_probs"])


def test_feed_properties_constant_within_batch(cfg):
    df = simulate_trials(cfg)
    per_batch = df.groupby(cfg["features"]["group"])[["feed_type", "particle_size_um"]].nunique()
    assert (per_batch == 1).all().all()


def test_allocate_trials_sums_and_rejects_too_few():
    rng = np.random.default_rng(0)
    counts = allocate_trials(150, 12, rng)
    assert counts.sum() == 150 and counts.min() >= 12
    with pytest.raises(ValueError):
        allocate_trials(3, 5, rng)


def test_recovery_increases_with_time(cfg):
    row = simulate_trials(cfg, n_trials=12, n_batches=12, seed=3).iloc[[0]]
    times = [30.0, 120.0, 480.0]
    frame = pd.concat([row.assign(time_min=t) for t in times], ignore_index=True)
    r = expected_recovery(frame, cfg["simulation"]["kinetics"], np.zeros(len(frame)))
    assert np.all(np.diff(r) > 0)


def test_lab_simulator_records_truth(cfg):
    rng = np.random.default_rng(0)
    batch = sample_batches(1, rng, cfg["simulation"], "T", ensure_all_types=False)[0]
    lab = LabSimulator(batch, cfg, rng)
    recipe = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["simulation"]["recipe_bounds"].items()}
    measured = lab.run(recipe)
    assert 0.0 <= measured <= 1.0
    assert len(lab.truth_history) == 1
