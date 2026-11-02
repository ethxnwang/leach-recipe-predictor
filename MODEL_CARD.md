# Model card: leach recovery predictor (synthetic demo)

> **Synthetic data; demonstrates the methodology only; unrelated to any proprietary work.**

## Model details

- **Type:** gradient-boosted regression trees (scikit-learn `GradientBoostingRegressor`,
  `learning_rate=0.03, max_depth=2, n_estimators=400, subsample=0.8`). The model was selected
  from mean, ridge, elastic net, random forest and gradient boosting by nested,
  batch-grouped cross-validation.
- **Preprocessing:** one-hot `feed_type`, standardised numeric features, and predictions clipped
  to [0, 1].
- **Uncertainty:** grouped-CV conformal interval (90% nominal, half-width 0.217), plus a
  30-member batch-bootstrap spread reported as `model_spread_std`.
- **Artifact:** `models/best_model.joblib` (`ModelBundle`). Provenance (config, seed, data hash,
  git commit) is in `reports/metrics.json`.

## Intended use

- Demonstrating, on synthetic data, a workflow for predicting leaching recovery from a recipe
  and feed composition, recommending recipes and scoring the model against measured trials.
- Portfolio and educational use.

**Out of scope:** any real process decision. The model was trained only on an invented
simulator and knows nothing about real chemistry.

## Data

- 150 trials from 12 simulated feed batches (`data/synthetic_trials.csv`, seed 42).
- Features: `feed_type` (A/B/C), `feed_metal_grade_pct`, `particle_size_um`,
  `reagent_molarity`, `oxidant_molarity`, `solid_liquid_ratio`, `pH`, `ORP_mV`,
  `temperature_C`, `time_min`, `agitation_rpm`. `batch_id` is used only for grouping and is
  not a feature.
- Target: `recovery_fraction` in [0, 1].
- Split: 9 training batches and 3 held-out test batches (B01, B10, B11).

## Metrics (recovery on a 0–1 scale)

| Evaluation | n | RMSE | MAE | R² | 90% interval coverage |
|---|---|---|---|---|---|
| Nested CV, 5 outer batch folds (mean ± sd) | 114 | 0.125 ± 0.020 | 0.098 ± 0.019 | 0.60 ± 0.10 | – |
| Held-out test batches | 36 | 0.162 | 0.118 | 0.54 | 0.86 |
| Example lab CSV (4 new batches) | 40 | 0.134 | 0.100 | 0.68 | 0.93 |

The baselines in nested CV were the mean predictor (RMSE 0.212) and ridge (RMSE 0.142).

## Failure modes

- **New-batch shift.** A hidden per-batch effect means a whole new batch can be over- or
  under-predicted together. On the test batches the mean bias was −0.061.
- **Optimiser's curse.** At its own recommended recipe the model over-predicts by +0.053 on
  average (trial study, 100 feeds). Treat the top recommendation's point estimate as
  optimistic.
- **Interaction under-fit.** The model does not learn the feed-type-specific pH optimum. For
  grade 3% and particle size 120 µm it recommends pH 2.03, 2.08 and 2.10 for feed types A, B
  and C, whereas the simulator's optima are 1.2, 1.8 and 2.3.
- **Constant-width intervals.** Interval width does not adapt to input-dependent noise. They
  are too wide where recovery is near 0 or 1, and can be too narrow mid-range.
- **High targets.** For recovery targets ≥ 0.80, open-loop model recommendations succeed less
  often than a simple manual hill climb (see `reports/trial_efficiency.md`).

## Out-of-distribution behaviour

- A request is flagged (`is_ood=true`, with reasons) when it has:
  - a numeric feature more than 5% of its range outside the training min/max;
  - an unseen `feed_type`;
  - a feed-composition Mahalanobis distance above the chi-square 99% threshold (3.03 for 2
    features).
- Flagged requests are still answered, because tree models extrapolate as a flat
  continuation, but the API, CLI and UI surface the warning. Inputs outside the physical
  validation limits in `config.yaml -> schema` are rejected (HTTP 422 or `DataValidationError`).
- The recommender only searches within the training range of each recipe variable, and warns
  when the optimum sits on a bound.
- Do **not** rely on `model_spread_std` (bootstrap disagreement) to detect OOD. Tree ensembles
  extrapolate flat, so the spread barely changes off-distribution: in a check, the mean
  spread was 0.050 on training inputs and 0.052 on inputs with particle size 600 µm and grade
  15%. It is also not a calibrated interval: its 90% percentile band covered only 50% of test
  outcomes. The explicit OOD flags are the signal to use.

## Ethical and legal notes

No real, proprietary or personal data is used. The simulator, its constants and every
result are invented for demonstration.
