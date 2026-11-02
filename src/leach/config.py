"""Configuration loading and small accessors shared across the package."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_SECTIONS = (
    "seed",
    "logging",
    "paths",
    "features",
    "feature_engineering",
    "schema",
    "simulation",
    "model",
    "uncertainty",
    "ood",
    "recommend",
    "trial_study",
    "api",
)

Config = dict[str, Any]


class ConfigError(ValueError):
    """Raised when the configuration file is missing required content."""


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config; ``$LEACH_CONFIG`` overrides the default path."""
    path = Path(path or os.environ.get("LEACH_CONFIG") or DEFAULT_CONFIG_PATH)
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config {path} is missing sections: {missing}")
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: Config, key: str) -> Path:
    """Return ``cfg['paths'][key]`` resolved relative to the config file's directory."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def categorical_features(cfg: Config) -> list[str]:
    return list(cfg["features"]["categorical"])


def feed_numeric_features(cfg: Config) -> list[str]:
    return list(cfg["features"]["feed_numeric"])


def recipe_features(cfg: Config) -> list[str]:
    return list(cfg["features"]["recipe_numeric"])


def numeric_features(cfg: Config) -> list[str]:
    return feed_numeric_features(cfg) + recipe_features(cfg)


def feature_columns(cfg: Config) -> list[str]:
    """All model input columns, in a fixed order (batch id is NOT a feature)."""
    return categorical_features(cfg) + numeric_features(cfg)


def feed_columns(cfg: Config) -> list[str]:
    """Columns describing the feedstock (as opposed to the recipe)."""
    return categorical_features(cfg) + feed_numeric_features(cfg)


def setup_logging(cfg: Config) -> None:
    logging.basicConfig(level=cfg["logging"]["level"], format=cfg["logging"]["format"])
