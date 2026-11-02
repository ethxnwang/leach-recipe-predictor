import copy

import pytest
from fastapi.testclient import TestClient

from leach.api import create_app

FEED = {"feed_type": "A", "feed_metal_grade_pct": 3.0, "particle_size_um": 120.0}
RECIPE = {
    "reagent_molarity": 1.0,
    "oxidant_molarity": 0.3,
    "solid_liquid_ratio": 0.15,
    "pH": 1.5,
    "ORP_mV": 600.0,
    "temperature_C": 60.0,
    "time_min": 240.0,
    "agitation_rpm": 400.0,
}


@pytest.fixture(scope="module")
def client(bundle, fast_cfg):
    return TestClient(create_app(fast_cfg, bundle))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_health_without_model(fast_cfg, tmp_path):
    cfg = copy.deepcopy(fast_cfg)
    cfg["paths"]["model"] = str(tmp_path / "missing.joblib")
    body = TestClient(create_app(cfg)).get("/health").json()
    assert body["model_loaded"] is False


def test_predict_ok_and_bounded(client):
    r = client.post("/predict", json={"trials": [{**FEED, **RECIPE}]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert 0 <= p["lower"] <= p["predicted_recovery"] <= p["upper"] <= 1
    assert p["is_ood"] is False


def test_predict_rejects_invalid_ph(client):
    r = client.post("/predict", json={"trials": [{**FEED, **RECIPE, "pH": 30}]})
    assert r.status_code == 422


def test_predict_rejects_missing_field_and_empty(client):
    bad = {**FEED, **RECIPE}
    del bad["time_min"]
    assert client.post("/predict", json={"trials": [bad]}).status_code == 422
    assert client.post("/predict", json={"trials": []}).status_code == 422


def test_predict_flags_ood_but_answers(client):
    r = client.post("/predict", json={"trials": [{**FEED, **RECIPE, "temperature_C": 99}]})
    assert r.status_code == 200
    assert r.json()["predictions"][0]["is_ood"] is True


def test_recommend(client, bundle):
    r = client.post("/recommend", json={"feed": FEED, "top_k": 2})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) == 2
    for rec in recs:
        for col, v in rec["recipe"].items():
            lo, hi = bundle.recipe_bounds[col]
            assert lo <= v <= hi


def test_recommend_rejects_bad_top_k(client):
    assert client.post("/recommend", json={"feed": FEED, "top_k": 0}).status_code == 422


def test_score(client):
    trials = [
        {**FEED, **RECIPE, "recovery_fraction": 0.6},
        {**FEED, **RECIPE, "time_min": 60, "recovery_fraction": 0.3},
    ]
    r = client.post("/score", json={"trials": trials})
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] == 2 and body["metrics"]["rmse"] >= 0
    assert len(body["rows"]) == 2
