from leach.ood import OODDetector


def test_training_rows_are_in_distribution(cfg, small_df):
    det = OODDetector.fit(small_df, cfg)
    assert not any(det.check(small_df))


def test_out_of_range_feature_flagged(cfg, small_df):
    det = OODDetector.fit(small_df, cfg)
    row = small_df.iloc[[0]].assign(temperature_C=99.0)
    reasons = det.check(row)[0]
    assert any("temperature_C" in r for r in reasons)


def test_unseen_category_flagged(cfg, small_df):
    seen = small_df[small_df["feed_type"] != "C"]
    det = OODDetector.fit(seen, cfg)
    reasons = det.feed_reasons({"feed_type": "C", "feed_metal_grade_pct": 3.0, "particle_size_um": 120.0})
    assert any("feed_type" in r for r in reasons)


def test_extreme_feed_composition_flagged(cfg, small_df):
    det = OODDetector.fit(small_df, cfg)
    reasons = det.feed_reasons({"feed_type": "A", "feed_metal_grade_pct": 40.0, "particle_size_um": 4000.0})
    assert any("Mahalanobis" in r for r in reasons)
    assert any("particle_size_um" in r for r in reasons)
