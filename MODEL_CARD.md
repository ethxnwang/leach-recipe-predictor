# Model card: leach recovery predictor (synthetic demo)

> **Synthetic data; demonstrates the methodology only; unrelated to any proprietary work.**

## Model details

- **Type:** `gp_ensemble`, the equal-weight average of two Gaussian-process regressors
  (scikit-learn `GaussianProcessRegressor`, Matern ν = 5/2 plus a white-noise kernel, with
  hyperparameters fitted by marginal likelihood):
  - one GP with a separate length-scale per input (ARD);
  - one GP with a single shared length-scale.
- **Selection:** chosen by nested, batch-grouped cross-validation over eight candidates: mean,
  ridge, elastic net, random forest, gradient boosting with raw and with engineered features,
  an ARD GP, and this ensemble.
- **Inputs:** one-hot `feed_type` and standardised numerics, plus engineered features:
  `log1p(time, particle size, reagent)`, 1/(T + 273.15), pH², and pH × feed type.
- **Target:** fitted on logit(recovery), clipped to [0.001, 0.999], then inverted; predictions
  are clipped to [0, 1].
- **Uncertainty:** grouped-CV conformal interval (90% nominal, half-width 0.171), plus a
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
- Split: 9 training batches and 3 held-out test batches (B01, B10, B11). The test batches were
  not used for any model, feature or hyperparameter choice.

## Metrics (recovery on a 0–1 scale)

| Evaluation | n | RMSE | MAE | R² | 90% interval coverage |
|---|---|---|---|---|---|
| Nested CV, 5 outer batch folds (mean ± sd) | 114 | 0.097 ± 0.041 | 0.072 ± 0.028 | 0.73 ± 0.23 | – |
| Held-out test batches | 36 | 0.120 | 0.079 | 0.75 | 0.83 |
| Example lab CSV (4 new batches) | 40 | 0.089 | 0.064 | 0.86 | 0.93 |
| Learning curve, 150 trials, 20 fresh batches | 400 | 0.075 | – | – | – |

Comparison models in nested CV:

| Model | RMSE |
|---|---|
| mean | 0.212 |
| ridge | 0.142 |
| gradient boosting | 0.125 |
| gradient boosting, engineered features + logit | 0.110 |

## Failure modes

- **Very small training sets.** At 30 trials the ensemble's fresh-batch RMSE (0.185) is
  slightly worse than gradient boosting's (0.176), because the GP kernel hyperparameters are
  poorly determined with so few points.
- **Fold-to-fold variability.** The CV RMSE standard deviation (0.041) is about twice
  gradient boosting's (0.020). On some batches the GP is much better, on others no better.
- **New-batch shift.** A hidden per-batch effect means a whole new batch can be over- or
  under-predicted together. On the test batches the mean bias was −0.055.
- **Interval under-coverage on test batches** (0.83 against 0.90 nominal). The intervals are
  constant-width and do not adapt to input-dependent noise.
- **Partly learned interaction.** For grade 3% and particle size 120 µm, the recommended pH is
  1.55, 1.99 and 2.37 for feed types A, B and C, against simulator optima of 1.2, 1.8 and 2.3.
- **High targets.** For recovery ≥ 0.90, model guidance barely beats a manual hill climb
  (success 0.10 against 0.08; see `reports/trial_efficiency.md`). This is mostly because the
  target is unreachable for 80% of feeds.

## Out-of-distribution behaviour

- A request is flagged (`is_ood=true`, with reasons) when it has:
  - a numeric feature more than 5% of its range outside the training min/max;
  - an unseen `feed_type`;
  - a feed-composition Mahalanobis distance above the chi-square 99% threshold (3.03 for 2
    features).
- Flagged requests are still answered, but the API, CLI and UI surface the warning. A GP
  reverts towards its mean far from the data, so predictions there become uninformative
  rather than wildly wrong. Inputs outside the physical validation limits in
  `config.yaml -> schema` are rejected (HTTP 422 or `DataValidationError`).
- The recommender only searches within the training range of each recipe variable, and warns
  when the optimum sits on a bound.
- Do **not** rely on `model_spread_std` to detect OOD. In a check, the mean spread was 0.050 on
  training inputs and only 0.060 on inputs with particle size 600 µm and grade 15%. Its 90%
  percentile band covered only 58% of test outcomes. Use the explicit OOD flags.

## Ethical and legal notes

No real, proprietary or personal data is used. The simulator, its constants and every
result are invented for demonstration.
