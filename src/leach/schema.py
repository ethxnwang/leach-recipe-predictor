"""Data validation: pydantic models built from the ``schema`` section of the config."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from leach.config import Config, categorical_features, feature_columns, feed_columns

_MODEL_CONFIG = ConfigDict(extra="ignore", allow_inf_nan=False)


class DataValidationError(ValueError):
    """Raised when a table fails schema checks; ``errors`` holds human-readable messages."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s): " + "; ".join(errors))


def _field_specs(cfg: Config, columns: list[str]) -> dict[str, tuple]:
    specs: dict[str, tuple] = {}
    for col in columns:
        if col in cfg["schema"]["categories"]:
            specs[col] = (Literal[tuple(cfg["schema"]["categories"][col])], ...)
        else:
            lo, hi = cfg["schema"]["ranges"][col]
            specs[col] = (float, Field(ge=lo, le=hi))
    return specs


def build_feed_model(cfg: Config) -> type[BaseModel]:
    """Pydantic model for a feedstock description."""
    return create_model(
        "FeedComposition", __config__=_MODEL_CONFIG, **_field_specs(cfg, feed_columns(cfg))
    )


def build_record_model(cfg: Config, require_target: bool = False) -> type[BaseModel]:
    """Pydantic model for one trial (feed + recipe, optionally the measured target)."""
    cols = feature_columns(cfg) + ([cfg["features"]["target"]] if require_target else [])
    name = "LabTrial" if require_target else "Trial"
    return create_model(name, __config__=_MODEL_CONFIG, **_field_specs(cfg, cols))


def _structural_errors(df: pd.DataFrame, required: list[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["table has no rows"]
    nulls = df[required].isna().sum()
    return [f"column '{c}' has {n} missing value(s)" for c, n in nulls.items() if n > 0]


def _row_errors(df: pd.DataFrame, model: type[BaseModel], limit: int) -> list[str]:
    errors: list[str] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        try:
            model.model_validate(row)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                errors.append(f"row {i}: {loc}: {err['msg']} (got {err.get('input')!r})")
        if len(errors) >= limit:
            errors.append("... further errors suppressed")
            break
    return errors


def validate_frame(
    df: pd.DataFrame, cfg: Config, require_target: bool = True, require_group: bool = True
) -> pd.DataFrame:
    """Check columns, missing values, types, categories and ranges; return a clean copy.

    Raises:
        DataValidationError: listing every problem found (up to the configured limit).
    """
    group, target = cfg["features"]["group"], cfg["features"]["target"]
    required = feature_columns(cfg) + [target] * require_target + [group] * require_group
    errors = _structural_errors(df, required)
    if not errors:
        model = build_record_model(cfg, require_target)
        errors = _row_errors(df, model, cfg["schema"]["max_errors_reported"])
    if errors:
        raise DataValidationError(errors)
    out = df.copy()
    for col in categorical_features(cfg):
        out[col] = out[col].astype(str)
    if require_group:
        out[group] = out[group].astype(str)
    return out
