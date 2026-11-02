"""Streamlit demo: predict, recommend, and score lab data. Synthetic data only.

Run with ``make app`` (requires a trained model: ``make pipeline``).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from leach.bundle import ModelBundle
from leach.config import load_config, recipe_features, resolve_path
from leach.recommend import recommend
from leach.schema import DataValidationError
from leach.scoring import score_lab_data

DISCLAIMER = "Synthetic data; demonstrates the methodology only; unrelated to any proprietary work."


@st.cache_resource
def load() -> tuple[dict, ModelBundle]:
    cfg = load_config()
    return cfg, ModelBundle.load(resolve_path(cfg, "model"))


def feed_inputs(cfg: dict, key: str) -> dict:
    ranges = cfg["simulation"]["feed_ranges"]
    c1, c2, c3 = st.columns(3)
    return {
        "feed_type": c1.selectbox(
            "feed_type", cfg["schema"]["categories"]["feed_type"], key=f"{key}t"
        ),
        "feed_metal_grade_pct": c2.number_input(
            "feed_metal_grade_pct",
            value=float(sum(ranges["feed_metal_grade_pct"]) / 2),
            key=f"{key}g",
        ),
        "particle_size_um": c3.number_input(
            "particle_size_um", value=float(sum(ranges["particle_size_um"]) / 2), key=f"{key}p"
        ),
    }


def predict_tab(cfg: dict, bundle: ModelBundle) -> None:
    feed = feed_inputs(cfg, "pred")
    recipe = {}
    cols = st.columns(4)
    for i, name in enumerate(recipe_features(cfg)):
        lo, hi = bundle.recipe_bounds[name]
        recipe[name] = cols[i % 4].slider(name, float(lo), float(hi), float((lo + hi) / 2))
    row = bundle.predict_frame(pd.DataFrame([{**feed, **recipe}])).iloc[0]
    st.metric(
        "Predicted recovery",
        f"{row.predicted_recovery:.3f}",
        help=f"{1 - cfg['uncertainty']['alpha']:.0%} interval shown below",
    )
    st.write(
        f"Interval: [{row.lower:.3f}, {row.upper:.3f}]  |  "
        f"model spread (sd): {row.model_spread_std:.3f}"
    )
    if row.is_ood:
        st.warning(f"Out of distribution: {row.ood_reasons}")


def recommend_tab(cfg: dict, bundle: ModelBundle) -> None:
    feed = feed_inputs(cfg, "rec")
    c1, c2 = st.columns(2)
    w = c1.number_input("cost_weight", min_value=0.0, value=float(cfg["recommend"]["cost_weight"]))
    k = c2.number_input("top_k", 1, cfg["recommend"]["max_top_k"], cfg["recommend"]["top_k"])
    if st.button("Recommend"):
        try:
            res = recommend(bundle, feed, cfg, cost_weight=w, top_k=int(k))
        except DataValidationError as exc:
            st.error(str(exc))
            return
        for msg in res.warnings:
            st.warning(msg)
        table = pd.DataFrame(
            [
                {
                    **r.recipe,
                    "predicted": r.predicted_recovery,
                    "lower": r.lower,
                    "upper": r.upper,
                    "cost": r.cost,
                }
                for r in res.recommendations
            ]
        )
        st.dataframe(table.round(3))


def score_tab(cfg: dict, bundle: ModelBundle) -> None:
    st.write("Upload a CSV with the feature columns and a measured `recovery_fraction`.")
    upload = st.file_uploader("lab results CSV", type="csv")
    if upload is None:
        return
    try:
        report = score_lab_data(bundle, pd.read_csv(upload), cfg)
    except DataValidationError as exc:
        st.error("\n".join(exc.errors))
        return
    m = report.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{m['rmse']:.3f}")
    c2.metric("MAE", f"{m['mae']:.3f}")
    c3.metric("R²", f"{m['r2']:.3f}")
    c4.metric(
        "Interval coverage",
        f"{report.interval['coverage']:.0%}",
        help=f"nominal {report.nominal_coverage:.0%}",
    )
    st.dataframe(report.rows.round(3))
    st.download_button(
        "Download summary JSON", json.dumps(report.summary(), indent=2), "lab_score.json"
    )


def main() -> None:
    st.set_page_config(page_title="Leach recipe predictor (synthetic)")
    st.title("Leach recipe predictor")
    st.info(DISCLAIMER)
    try:
        cfg, bundle = load()
    except FileNotFoundError:
        st.error("No trained model found. Run `make pipeline` first.")
        return
    t1, t2, t3 = st.tabs(["Predict", "Recommend", "Score lab data"])
    with t1:
        predict_tab(cfg, bundle)
    with t2:
        recommend_tab(cfg, bundle)
    with t3:
        score_tab(cfg, bundle)


main()
