import numpy as np
import pytest

from leach.recommend import recipe_cost, recommend, select_diverse
from leach.schema import DataValidationError

FEED = {"feed_type": "B", "feed_metal_grade_pct": 3.0, "particle_size_um": 120.0}


@pytest.fixture(scope="module")
def result(bundle, fast_cfg):
    return recommend(bundle, FEED, fast_cfg, top_k=3, seed=0)


def test_returns_top_k_sorted(result):
    assert len(result.recommendations) == 3
    obj = [r.objective for r in result.recommendations]
    assert obj == sorted(obj, reverse=True)


def test_recipes_respect_search_bounds(result, bundle):
    for rec in result.recommendations:
        for col, v in rec.recipe.items():
            lo, hi = bundle.recipe_bounds[col]
            assert lo <= v <= hi, col


def test_predictions_and_intervals_bounded(result):
    for rec in result.recommendations:
        assert 0 <= rec.lower <= rec.predicted_recovery <= rec.upper <= 1


def test_in_distribution_feed_not_flagged(result):
    assert not result.feed_is_ood


def test_ood_feed_is_flagged_with_warning(bundle, fast_cfg):
    feed = {**FEED, "particle_size_um": 2000.0}
    res = recommend(bundle, feed, fast_cfg, top_k=1, seed=0)
    assert res.feed_is_ood
    assert any("out of distribution" in w for w in res.warnings)
    assert res.recommendations[0].is_ood


def test_invalid_feed_rejected(bundle, fast_cfg):
    with pytest.raises(DataValidationError):
        recommend(bundle, {**FEED, "feed_type": "Z"}, fast_cfg)


def test_higher_cost_weight_does_not_raise_cost(bundle, fast_cfg):
    cheap = recommend(bundle, FEED, fast_cfg, cost_weight=2.0, top_k=1, seed=0)
    free = recommend(bundle, FEED, fast_cfg, cost_weight=0.0, top_k=1, seed=0)
    assert cheap.recommendations[0].cost <= free.recommendations[0].cost + 1e-9


def test_recipe_cost_range():
    lo, hi, w = np.zeros(2), np.ones(2), np.array([1.0, 3.0])
    assert recipe_cost(np.array([[0, 0], [1, 1]]), lo, hi, w).tolist() == [0.0, 1.0]


def test_select_diverse_enforces_distance():
    X = np.array([[0.0], [0.01], [0.5], [1.0]])
    idx = select_diverse(X, np.array([4, 3, 2, 1]), np.zeros(1), np.ones(1), k=3, min_dist=0.1)
    assert idx == [0, 2, 3]
