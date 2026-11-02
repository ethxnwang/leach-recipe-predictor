"""FastAPI service: /health, /predict, /recommend, /score.

Request bodies are pydantic models generated from the config schema, so ranges and
categories are validated (HTTP 422) before reaching the model. Inputs that pass validation
but fall outside the training distribution are answered, with ``is_ood`` and reasons set.
"""

# No `from __future__ import annotations`: FastAPI must see the dynamic request models
# as real classes, not strings, to treat them as JSON bodies.
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from leach import __version__
from leach.bundle import ModelBundle
from leach.config import Config, load_config, resolve_path
from leach.recommend import recommend
from leach.schema import DataValidationError, build_feed_model, build_record_model
from leach.scoring import score_lab_data


class Prediction(BaseModel):
    predicted_recovery: float
    lower: float
    upper: float
    model_spread_std: float
    is_ood: bool
    ood_reasons: str


class PredictResponse(BaseModel):
    model_name: str
    nominal_coverage: float
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    version: str


def _request_models(cfg: Config) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
    max_n = cfg["api"]["max_trials_per_request"]
    trial, lab_trial, feed = (
        build_record_model(cfg),
        build_record_model(cfg, True),
        build_feed_model(cfg),
    )
    predict_req = create_model(
        "PredictRequest",
        trials=(Annotated[list[trial], Field(min_length=1, max_length=max_n)], ...),
    )
    score_req = create_model(
        "ScoreRequest",
        trials=(Annotated[list[lab_trial], Field(min_length=1, max_length=max_n)], ...),
    )
    recommend_req = create_model(
        "RecommendRequest",
        feed=(feed, ...),
        cost_weight=(float | None, Field(default=None, ge=0)),
        top_k=(int | None, Field(default=None, ge=1, le=cfg["recommend"]["max_top_k"])),
    )
    return predict_req, recommend_req, score_req


def create_app(cfg: Config | None = None, bundle: ModelBundle | None = None) -> FastAPI:
    """Build the app; the model bundle is loaded lazily from ``paths.model`` if not given."""
    cfg = cfg or load_config()
    state: dict = {"bundle": bundle}
    PredictRequest, RecommendRequest, ScoreRequest = _request_models(cfg)
    app = FastAPI(title="leach-recipe-predictor (synthetic demo)", version=__version__)

    def get_bundle() -> ModelBundle:
        if state["bundle"] is None:
            path = resolve_path(cfg, "model")
            if not path.exists():
                raise HTTPException(503, f"model not trained (expected {path})")
            state["bundle"] = ModelBundle.load(path)
        return state["bundle"]

    @app.get("/health", response_model=Health)
    def health() -> Health:
        try:
            b = get_bundle()
        except HTTPException:
            b = None
        return Health(
            status="ok",
            model_loaded=b is not None,
            model_name=b.model_name if b else None,
            version=__version__,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:  # type: ignore[valid-type]
        b = get_bundle()
        df = pd.DataFrame([t.model_dump() for t in req.trials])
        frame = b.predict_frame(df)
        return PredictResponse(
            model_name=b.model_name,
            nominal_coverage=1 - cfg["uncertainty"]["alpha"],
            predictions=[Prediction(**r) for r in frame.to_dict(orient="records")],
        )

    @app.post("/recommend")
    def recommend_endpoint(req: RecommendRequest) -> dict:  # type: ignore[valid-type]
        res = recommend(get_bundle(), req.feed.model_dump(), cfg, req.cost_weight, req.top_k)
        return res.to_dict()

    @app.post("/score")
    def score(req: ScoreRequest) -> dict:  # type: ignore[valid-type]
        df = pd.DataFrame([t.model_dump() for t in req.trials])
        try:
            report = score_lab_data(get_bundle(), df, cfg)
        except DataValidationError as exc:
            raise HTTPException(422, exc.errors) from exc
        rows = report.rows[
            ["predicted_recovery", "lower", "upper", "error", "inside_interval", "is_ood"]
        ]
        return {**report.summary(), "rows": rows.to_dict(orient="records")}

    return app


app = create_app()
