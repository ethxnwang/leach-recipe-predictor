import numpy as np
import pytest

from leach.schema import DataValidationError, build_feed_model, validate_frame


def test_valid_frame_passes(cfg, small_df):
    out = validate_frame(small_df, cfg)
    assert len(out) == len(small_df)


def test_missing_column_fails(cfg, small_df):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_frame(small_df.drop(columns=["pH"]), cfg)


def test_missing_values_fail(cfg, small_df):
    df = small_df.copy()
    df.loc[3, "temperature_C"] = np.nan
    with pytest.raises(DataValidationError, match="missing value"):
        validate_frame(df, cfg)


def test_out_of_range_fails(cfg, small_df):
    df = small_df.copy()
    df.loc[0, "pH"] = 20.0
    with pytest.raises(DataValidationError) as exc:
        validate_frame(df, cfg)
    assert any("row 0: pH" in e for e in exc.value.errors)


def test_target_outside_unit_interval_fails(cfg, small_df):
    df = small_df.copy()
    df.loc[1, "recovery_fraction"] = 1.2
    with pytest.raises(DataValidationError, match="recovery_fraction"):
        validate_frame(df, cfg)


def test_unknown_category_fails(cfg, small_df):
    df = small_df.copy()
    df.loc[2, "feed_type"] = "Z"
    with pytest.raises(DataValidationError, match="feed_type"):
        validate_frame(df, cfg)


def test_wrong_type_fails(cfg, small_df):
    df = small_df.astype({"ORP_mV": object})
    df.loc[0, "ORP_mV"] = "high"
    with pytest.raises(DataValidationError, match="ORP_mV"):
        validate_frame(df, cfg)


def test_target_and_group_optional_for_prediction(cfg, small_df):
    df = small_df.drop(columns=["recovery_fraction", "batch_id"])
    validate_frame(df, cfg, require_target=False, require_group=False)


def test_feed_model_rejects_negative_grade(cfg):
    model = build_feed_model(cfg)
    with pytest.raises(ValueError):
        model(feed_type="A", feed_metal_grade_pct=-1.0, particle_size_um=100.0)
