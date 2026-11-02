import numpy as np
import pytest

from leach.config import feature_columns
from leach.features import FeatureEngineer, engineered_names
from leach.models import build_candidate, build_with_params, n_model_inputs, strip_prefix


@pytest.fixture
def engineer(cfg):
    return FeatureEngineer(cfg["feature_engineering"], cfg["schema"]["categories"])


def test_engineered_columns_match_declared_names(cfg, small_df, engineer):
    out = engineer.fit_transform(small_df[feature_columns(cfg)])
    names = engineered_names(cfg["feature_engineering"], cfg["schema"]["categories"])
    assert list(out.columns[-len(names) :]) == names
    assert not out[names].isna().any().any()


def test_transform_values(cfg, small_df, engineer):
    out = engineer.transform(small_df)
    np.testing.assert_allclose(out["log_time_min"], np.log1p(small_df["time_min"]))
    np.testing.assert_allclose(
        out["inv_kelvin_temperature_C"], 1 / (small_df["temperature_C"] + 273.15)
    )


def test_category_interaction_is_zero_for_other_levels(small_df, engineer):
    out = engineer.transform(small_df)
    is_a = small_df["feed_type"] == "A"
    np.testing.assert_allclose(out.loc[is_a, "pH_x_feed_type_A"], small_df.loc[is_a, "pH"])
    assert (out.loc[~is_a, "pH_x_feed_type_A"] == 0).all()


def test_does_not_mutate_input(small_df, engineer):
    before = small_df.copy()
    engineer.transform(small_df)
    assert small_df.equals(before)


def test_model_input_count(cfg):
    names = engineered_names(cfg["feature_engineering"], cfg["schema"]["categories"])
    assert n_model_inputs(cfg, True) == n_model_inputs(cfg, False) + len(names)


@pytest.mark.parametrize("name", ["gaussian_process", "gp_ensemble", "gradient_boosting_fe"])
def test_new_candidates_fit_and_stay_bounded(cfg, small_df, name):
    model, _ = build_candidate(name, cfg, seed=0)
    X, y = small_df[feature_columns(cfg)], small_df["recovery_fraction"]
    model.fit(X, y)
    extreme = X.assign(temperature_C=1000.0, time_min=1e5)
    for pred in (model.predict(X), model.predict(extreme)):
        assert np.all((pred >= 0) & (pred <= 1))


def test_logit_candidate_grid_paths_resolve(cfg):
    model, grid = build_candidate("gradient_boosting_fe", cfg, seed=0)
    assert all(k.startswith("estimator__regressor__model__") for k in grid)
    model.set_params(**{k: v[0] for k, v in grid.items()})  # raises if a path is wrong
    tuned = build_with_params("gradient_boosting_fe", cfg, 0, strip_prefix(grid))
    assert (
        tuned.get_params()["estimator__regressor__model__max_depth"]
        == grid["estimator__regressor__model__max_depth"]
    )
