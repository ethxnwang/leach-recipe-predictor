"""Command-line interface: ``leach simulate | eda | train | evaluate | recommend | trial-study``."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import typer
from sklearn.exceptions import ConvergenceWarning

from leach.config import Config, load_config, resolve_path, setup_logging

# A GP length-scale reaching its upper bound just means that input was judged irrelevant.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process")

app = typer.Typer(
    help="Leaching recipe predictor (synthetic-data methodology demo).", no_args_is_help=True
)
logger = logging.getLogger("leach")


def _cfg(ctx: typer.Context) -> Config:
    return ctx.obj["cfg"]


def _bundle(cfg: Config):
    from leach.bundle import ModelBundle

    path = resolve_path(cfg, "model")
    if not path.exists():
        raise typer.BadParameter(f"no trained model at {path}; run `leach train` first")
    return ModelBundle.load(path)


@app.callback()
def main(ctx: typer.Context, config: Path = typer.Option(None, help="Path to config.yaml")) -> None:
    cfg = load_config(config)
    setup_logging(cfg)
    ctx.obj = {"cfg": cfg}


@app.command()
def simulate(
    ctx: typer.Context,
    n_trials: int = typer.Option(None, help="Number of trials (default from config)"),
    n_batches: int = typer.Option(None, help="Number of feed batches (default from config)"),
    seed: int = typer.Option(None, help="Random seed (default from config)"),
    prefix: str = typer.Option(None, help="Batch id prefix (default from config)"),
    out: Path = typer.Option(None, help="Output CSV (default: paths.data)"),
) -> None:
    """Generate synthetic wet-lab trials with the invented simulator."""
    from leach.simulate import simulate_trials

    cfg = _cfg(ctx)
    df = simulate_trials(cfg, n_trials, n_batches, seed, prefix)
    out = out or resolve_path(cfg, "data")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"wrote {len(df)} trials to {out}")


@app.command()
def eda(ctx: typer.Context) -> None:
    """Validate the training data and write EDA figures."""
    from leach.eda import run_eda, summarize
    from leach.schema import validate_frame

    cfg = _cfg(ctx)
    df = validate_frame(pd.read_csv(resolve_path(cfg, "data")), cfg)
    typer.echo(summarize(df, cfg).round(3).to_string())
    for p in run_eda(df, cfg):
        typer.echo(f"wrote {p}")


@app.command()
def train(ctx: typer.Context) -> None:
    """Compare models with nested grouped CV, fit the best, save bundle and metrics."""
    from leach.train import run_training

    cfg = _cfg(ctx)
    m = run_training(cfg)
    h = m["holdout"]
    typer.echo(f"selected: {m['selected_model']} {m['selected_params']}")
    typer.echo(f"holdout: {json.dumps(h['metrics'])}")
    typer.echo(
        f"conformal coverage {h['conformal_interval']['coverage']:.2f} "
        f"(nominal {h['nominal_coverage']:.2f})"
    )


@app.command()
def evaluate(
    ctx: typer.Context,
    data: Path = typer.Argument(..., help="CSV of measured trials: features + recovery_fraction"),
) -> None:
    """Score the trained model against measured (wet-lab) results."""
    from leach.scoring import score_lab_data

    cfg = _cfg(ctx)
    report = score_lab_data(_bundle(cfg), pd.read_csv(data), cfg)
    summary = report.summary()
    resolve_path(cfg, "lab_score_json").write_text(json.dumps(summary, indent=2))
    report.rows.to_csv(resolve_path(cfg, "lab_score_rows"), index=False)
    typer.echo(json.dumps(summary, indent=2))


@app.command("recommend")
def recommend_cmd(
    ctx: typer.Context,
    feed_type: str = typer.Option(..., help="Feed type category"),
    grade: float = typer.Option(..., help="feed_metal_grade_pct"),
    particle_size: float = typer.Option(..., help="particle_size_um"),
    cost_weight: float = typer.Option(None, help="Cost penalty weight (default from config)"),
    top_k: int = typer.Option(None, help="Number of recipes (default from config)"),
) -> None:
    """Recommend recipes for a new feed composition."""
    from leach.recommend import recommend

    cfg = _cfg(ctx)
    feed = {
        "feed_type": feed_type,
        "feed_metal_grade_pct": grade,
        "particle_size_um": particle_size,
    }
    res = recommend(_bundle(cfg), feed, cfg, cost_weight=cost_weight, top_k=top_k)
    typer.echo(json.dumps(res.to_dict(), indent=2))


@app.command("trial-study")
def trial_study(
    ctx: typer.Context, n_runs: int = typer.Option(None, help="Override run count")
) -> None:
    """Simulate manual vs model-guided experimentation; write reports/trial_efficiency.md."""
    from leach.trial_study import run_and_report

    cfg = _cfg(ctx)
    if n_runs:
        cfg["trial_study"]["n_runs"] = n_runs
    summary = run_and_report(_bundle(cfg), cfg)
    typer.echo(summary.round(3).to_string(index=False))
    typer.echo(f"wrote {resolve_path(cfg, 'trial_report')}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("leach.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
