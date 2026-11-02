"""An INVENTED, generic leaching simulator used to create synthetic "wet-lab trials".

DISCLAIMER: this simulator is made up for demonstration purposes. Its functional form and
constants are not representative of any real chemistry, reagent system, ore, or process, and
no output of it should be read as a statement about a real process.

Model (constants in ``config.yaml -> simulation.kinetics``)::

    R(t)      = R_max_eff * (1 - exp(-k * t))                       saturating kinetics
    R_max_eff = r_max[type] * f_reagent * f_pH * f_SL * f_grade * exp(b_batch)
      f_reagent = m / (m + K)                                     saturating in reagent
      f_pH      = exp(-((pH - pH_opt[type]) / w)^2)               feed-type-specific optimum
      f_SL      = 1 - p * max(0, SL - SL_threshold)               penalty for thick slurries
      f_grade   = 1 + g * (grade - grade_ref)
    k         = k_ref * f_T * f_ox * f_ORP * f_particle * f_agitation
      f_T       = exp(E * (1/T_ref - 1/T))                        Arrhenius-like (T in K)
      f_ox      = 1 + a * oxidant * exp(-max(0, T - T_decomp) / s) oxidant decays when hot
      f_ORP     = sigmoid((ORP - ORP_mid) / scale) * (1 - c * max(0, ORP - ORP_pass))
      f_particle= (d_ref / d) ^ n
      f_agit    = floor + (1 - floor) * min(1, rpm / rpm_sat)

Observed recovery = clip(R + eps, 0, 1) with heteroscedastic eps ~ N(0, base + peak*4R(1-R)).
``b_batch`` is a per-batch random effect the model cannot observe directly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from leach.config import Config, feed_numeric_features, recipe_features

logger = logging.getLogger(__name__)

KELVIN_OFFSET = 273.15  # unit conversion, not a tunable


@dataclass(frozen=True)
class Batch:
    """One feedstock batch: a feed type, its batch-level properties and a hidden effect."""

    batch_id: str
    feed_type: str
    feed: dict[str, float]
    effect: float


def sample_batches(
    n_batches: int,
    rng: np.random.Generator,
    sim_cfg: Mapping,
    prefix: str,
    ensure_all_types: bool = True,
) -> list[Batch]:
    """Draw feed batches; optionally guarantee each feed type appears at least once."""
    types = list(sim_cfg["feed_type_probs"])
    probs = np.array([sim_cfg["feed_type_probs"][t] for t in types], dtype=float)
    probs /= probs.sum()
    n_fixed = min(n_batches, len(types)) if ensure_all_types else 0
    chosen = types[:n_fixed] + [str(t) for t in rng.choice(types, n_batches - n_fixed, p=probs)]
    rng.shuffle(chosen)
    feeds = {
        name: rng.uniform(lo, hi, size=n_batches)
        for name, (lo, hi) in sim_cfg["feed_ranges"].items()
    }
    effects = rng.normal(0.0, sim_cfg["batch_effect_sd"], size=n_batches)
    width = len(str(n_batches))
    return [
        Batch(
            batch_id=f"{prefix}{str(i + 1).zfill(width)}",
            feed_type=chosen[i],
            feed={name: float(vals[i]) for name, vals in feeds.items()},
            effect=float(effects[i]),
        )
        for i in range(n_batches)
    ]


def sample_recipes(n: int, rng: np.random.Generator, bounds: Mapping) -> pd.DataFrame:
    """Uniformly sample ``n`` recipes inside ``bounds`` (a crude design of experiments)."""
    return pd.DataFrame({name: rng.uniform(lo, hi, size=n) for name, (lo, hi) in bounds.items()})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=float)


def max_recovery(df: pd.DataFrame, kin: Mapping) -> np.ndarray:
    """Plateau recovery R_max_eff before the batch effect."""
    feed_type = df["feed_type"]
    r_max = feed_type.map(kin["r_max"]).to_numpy(dtype=float)
    ph_opt = feed_type.map(kin["ph_opt"]).to_numpy(dtype=float)
    m = _col(df, "reagent_molarity")
    f_reagent = m / (m + kin["reagent_half_sat"])
    f_ph = np.exp(-(((_col(df, "pH") - ph_opt) / kin["ph_width"]) ** 2))
    excess_sl = np.clip(_col(df, "solid_liquid_ratio") - kin["sl_threshold"], 0.0, None)
    f_sl = 1.0 - kin["sl_penalty"] * excess_sl
    f_grade = 1.0 + kin["grade_effect"] * (_col(df, "feed_metal_grade_pct") - kin["grade_ref"])
    return r_max * f_reagent * f_ph * f_sl * f_grade


def rate_constant(df: pd.DataFrame, kin: Mapping) -> np.ndarray:
    """First-order rate constant k (per minute)."""
    temp_c = _col(df, "temperature_C")
    temp_k = temp_c + KELVIN_OFFSET
    t_ref_k = kin["temp_ref_C"] + KELVIN_OFFSET
    f_t = np.exp(kin["activation_K"] * (1.0 / t_ref_k - 1.0 / temp_k))
    decomp = np.exp(
        -np.clip(temp_c - kin["oxidant_decomp_C"], 0.0, None) / kin["oxidant_decomp_scale_C"]
    )
    f_ox = 1.0 + kin["oxidant_gain"] * _col(df, "oxidant_molarity") * decomp
    orp = _col(df, "ORP_mV")
    passivation = kin["orp_passivation_per_mV"] * np.clip(
        orp - kin["orp_passivation_mV"], 0.0, None
    )
    f_orp = _sigmoid((orp - kin["orp_mid_mV"]) / kin["orp_scale_mV"]) * np.clip(
        1.0 - passivation, 0.0, None
    )
    f_particle = (kin["particle_ref_um"] / _col(df, "particle_size_um")) ** kin["particle_exponent"]
    rpm_frac = np.clip(_col(df, "agitation_rpm") / kin["agitation_sat_rpm"], None, 1.0)
    f_agit = kin["agitation_floor"] + (1.0 - kin["agitation_floor"]) * rpm_frac
    return kin["k_ref_per_min"] * f_t * f_ox * f_orp * f_particle * f_agit


def expected_recovery(df: pd.DataFrame, kin: Mapping, batch_effect: np.ndarray) -> np.ndarray:
    """Noise-free recovery fraction in [0, 1] for each row."""
    r_inf = max_recovery(df, kin) * np.exp(batch_effect)
    r = r_inf * (1.0 - np.exp(-rate_constant(df, kin) * _col(df, "time_min")))
    return np.clip(r, 0.0, 1.0)


def noise_sd(recovery: np.ndarray, sim_cfg: Mapping) -> np.ndarray:
    """Heteroscedastic measurement noise, largest at mid-range recovery."""
    return sim_cfg["noise_sd_base"] + sim_cfg["noise_sd_peak"] * 4.0 * recovery * (1.0 - recovery)


def allocate_trials(n_trials: int, n_batches: int, rng: np.random.Generator) -> np.ndarray:
    """Spread trials as evenly as possible over batches (remainder assigned at random)."""
    if n_trials < n_batches:
        raise ValueError(f"need at least one trial per batch ({n_trials=} < {n_batches=})")
    counts = np.full(n_batches, n_trials // n_batches)
    counts[rng.choice(n_batches, n_trials % n_batches, replace=False)] += 1
    return counts


def _batch_frame(batches: list[Batch], group_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {group_col: b.batch_id, "feed_type": b.feed_type, **b.feed, "_effect": b.effect}
            for b in batches
        ]
    )


def simulate_trials(
    cfg: Config,
    n_trials: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Simulate a reproducible table of wet-lab trials (one row per trial)."""
    sim = cfg["simulation"]
    n_trials = n_trials or sim["n_trials"]
    n_batches = n_batches or sim["n_batches"]
    seed = cfg["seed"] if seed is None else seed
    prefix = prefix or sim["batch_prefix"]
    group, target = cfg["features"]["group"], cfg["features"]["target"]
    rng = np.random.default_rng(seed)

    batches = sample_batches(n_batches, rng, sim, prefix)
    counts = allocate_trials(n_trials, n_batches, rng)
    rows = _batch_frame(batches, group).loc[lambda d: d.index.repeat(counts)].reset_index(drop=True)
    df = pd.concat([rows, sample_recipes(n_trials, rng, sim["recipe_bounds"])], axis=1)

    truth = expected_recovery(df, sim["kinetics"], df["_effect"].to_numpy())
    df[target] = np.clip(truth + rng.normal(0.0, noise_sd(truth, sim)), 0.0, 1.0)
    cols = [group, "feed_type", *feed_numeric_features(cfg), *recipe_features(cfg), target]
    logger.info("simulated %d trials over %d batches (seed=%d)", n_trials, n_batches, seed)
    return df[cols].round(sim["round_decimals"])


@dataclass
class LabSimulator:
    """A virtual wet lab for one batch: run recipes, get noisy measurements.

    ``truth_history`` records the noise-free recovery of each recipe run, which strategies
    never see; it is used only to score how quickly a strategy reached a target.
    """

    batch: Batch
    cfg: Config
    rng: np.random.Generator
    truth_history: list[float] = field(default_factory=list)

    def frame(self, recipes: pd.DataFrame) -> pd.DataFrame:
        feed = {"feed_type": self.batch.feed_type, **self.batch.feed}
        return recipes.assign(**feed)

    def truth(self, recipes: pd.DataFrame) -> np.ndarray:
        effect = np.full(len(recipes), self.batch.effect)
        return expected_recovery(self.frame(recipes), self.cfg["simulation"]["kinetics"], effect)

    def run(self, recipe: Mapping[str, float]) -> float:
        """Run one trial and return the noisy measured recovery."""
        truth = float(self.truth(pd.DataFrame([dict(recipe)]))[0])
        self.truth_history.append(truth)
        noise = self.rng.normal(0.0, noise_sd(np.array(truth), self.cfg["simulation"]))
        return float(np.clip(truth + noise, 0.0, 1.0))
