import pytest

from leach.schema import DataValidationError
from leach.scoring import score_lab_data
from leach.simulate import simulate_trials


def test_score_reports_metrics_and_rows(bundle, fast_cfg):
    lab = simulate_trials(fast_cfg, n_trials=20, n_batches=2, seed=77, prefix="LAB")
    report = score_lab_data(bundle, lab, fast_cfg)
    assert report.n_rows == 20
    assert report.metrics["rmse"] >= 0
    assert 0 <= report.interval["coverage"] <= 1
    assert {"predicted_recovery", "error", "inside_interval"} <= set(report.rows.columns)


def test_perfect_predictions_score_zero(bundle, fast_cfg, small_df):
    lab = small_df.drop(columns=["batch_id"]).assign(recovery_fraction=bundle.predict(small_df))
    report = score_lab_data(bundle, lab, fast_cfg)
    assert report.metrics["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert report.interval["coverage"] == 1.0


def test_ood_rows_counted_and_split_out(bundle, fast_cfg, small_df):
    lab = small_df.copy()
    lab.loc[:4, "temperature_C"] = 99.0
    report = score_lab_data(bundle, lab, fast_cfg)
    assert report.n_ood_rows == 5
    assert report.in_distribution_metrics["n"] == len(lab) - 5


def test_missing_target_rejected(bundle, fast_cfg, small_df):
    with pytest.raises(DataValidationError):
        score_lab_data(bundle, small_df.drop(columns=["recovery_fraction"]), fast_cfg)
