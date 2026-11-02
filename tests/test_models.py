import numpy as np
import pandas as pd
import pytest

from leach.config import feature_columns
from leach.evaluation import (
    group_splits,
    holdout_split,
    nested_cv,
    regression_metrics,
    select_best,
)
from leach.models import build_candidate
from leach.uncertainty import conformal_quantile, coverage


def _xyg(df, cfg):
    return (
        df[feature_columns(cfg)],
        df[cfg["features"]["target"]].to_numpy(),
        df[cfg["features"]["group"]].to_numpy(),
    )


def test_holdout_split_has_no_batch_overlap(cfg, small_df):
    groups = small_df["batch_id"].to_numpy()
    tr, te = holdout_split(groups, 0.25, seed=0)
    assert set(groups[tr]).isdisjoint(groups[te])
    assert len(tr) + len(te) == len(groups)


def test_group_kfold_has_no_batch_leakage(cfg, small_df):
    X, _, g = _xyg(small_df, cfg)
    seen_test = []
    for tr, te in group_splits(X, g, 4, seed=0):
        assert set(g[tr]).isdisjoint(g[te])
        seen_test.extend(np.unique(g[te]))
    assert sorted(seen_test) == sorted(np.unique(g))  # every batch tested exactly once


def test_group_kfold_rejects_single_batch(cfg, small_df):
    one = small_df[small_df["batch_id"] == small_df["batch_id"].iloc[0]]
    X, _, g = _xyg(one, cfg)
    with pytest.raises(ValueError):
        list(group_splits(X, g, 3, seed=0))


def test_nested_cv_returns_fold_metrics(fast_cfg, small_df):
    folds = nested_cv("ridge", fast_cfg, *_xyg(small_df, fast_cfg), seed=0)
    assert len(folds) == fast_cfg["model"]["outer_folds"]
    assert all(f["rmse"] >= 0 and "alpha" in f["params"] for f in folds)


def test_select_best_minimises_rmse():
    summary = pd.DataFrame({"rmse_mean": [0.2, 0.1, 0.3]}, index=["a", "b", "c"])
    assert select_best(summary, "rmse") == "b"


def test_predictions_bounded_even_far_outside_training(fast_cfg, small_df):
    X, y, _ = _xyg(small_df, fast_cfg)
    model, _ = build_candidate("ridge", fast_cfg, seed=0)
    model.fit(X, y)
    extreme = X.copy()
    extreme["temperature_C"] = 1000.0
    extreme["time_min"] = -500.0
    pred = model.predict(extreme)
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_metrics_perfect_prediction():
    m = regression_metrics([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1


def test_conformal_quantile_and_coverage():
    res = np.arange(1, 101) / 100.0
    q = conformal_quantile(res, alpha=0.1)
    assert 0.9 <= q <= 0.92
    cov = coverage(np.array([0.5, 0.9]), np.array([0.4, 0.4]), np.array([0.6, 0.6]))
    assert cov["coverage"] == 0.5


def test_bundle_intervals_bracket_prediction(bundle, small_df):
    frame = bundle.predict_frame(small_df)
    assert (frame["lower"] <= frame["predicted_recovery"]).all()
    assert (frame["predicted_recovery"] <= frame["upper"]).all()
    assert frame["lower"].min() >= 0 and frame["upper"].max() <= 1


def test_bundle_roundtrip(tmp_path, bundle, small_df):
    path = bundle.save(tmp_path / "m.joblib")
    loaded = type(bundle).load(path)
    np.testing.assert_allclose(loaded.predict(small_df), bundle.predict(small_df))
