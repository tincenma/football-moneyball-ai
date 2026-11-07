# Data

External real-world datasets are **not bundled** with this starter. Download only datasets you are licensed to use, track original source, season and retrieval date, and avoid committing large or restricted data to GitHub.

The three supplied local ZIP archives were fully audited on 2026-11-07. See [the profiling report](../docs/data_profiling.md), [all actual source columns](../docs/data_dictionary.md), and [Feature Set v1](../config/feature_set_v1.json). Aggregate evidence is in `reports/data_profiling/`; extracted records, identity mappings and the 2,886-row valuation dataset are local only under gitignored `data/processed/profiling/`. The finalized training schema differs from the starter-app schema below.

## CSV schema for the starter app

The Step 2 notebook exports the unified historical table to `data/processed/preprocessing/moneyball_cm_2022-09-26_clean.csv` (2,886 CM, 47 columns). `dataset_manifest.json` specifies the 27 model inputs, metadata, baseline and target roles, source hashes, selection counts and validation results. Read IDs and categorical `league_id` as strings. Seven missing `league_level` values are intentional and flagged; impute them only after a train/test split. This dataset does not use the starter-app schema below.

Step 3 exports `moneyball_features_2022-09-26.csv`, similarity/valuation matrices, economic/performance layers, `player_split.csv`, `transform_parameters.json` and `feature_manifest.json` to `data/processed/features/`. Matrix ID columns are lookup keys, not model inputs. Scalers and encoders are fit on train only. Performance rates include all goals and retain missing values; publication time of the 21/22 aggregates is unverified, so they remain outside the core v1 model. Potential is economic analysis context, also excluded from style similarity and the fixed valuation v1 inputs.

Step 4 stores per-player neighbor pairs, affordable candidates, holdout predictions and cluster assignments locally in `data/processed/eda/`. Aggregate metrics, hypothesis summaries and PNG/SVG figures are in `reports/eda/`; interpreted findings are in `docs/eda_findings.md`. Research intervals are conditional on the selected cohort and fixed fitted models. The test split is now used for exploratory research, not reserved as an untouched final evaluation set.

Step 5 exports the rating-only `baseline_top5.csv`, complete eligible pool, eligibility audit and train-gallery control lists to `data/processed/baseline/`. The comparison protocol and three PNG/SVG figures are in `reports/baseline/`. Baseline and ML must use the same candidate IDs and eligibility parameters. TOP-5 contains alternatives for one transfer; its prices are not summed against the budget. Recorded TM values are screening proxies, with EUR explicitly assumed for the 15-million example.

Step 6 saves the fitted local bundle `player_models_2022-09-26.joblib`, KNN and same-pool baseline lists, pool hashes, cluster assignments and per-reference diagnostics in `data/processed/models/`. `reports/models/models_summary.json` records the bundle/source hashes and versions; the same directory contains aggregate quality tables and PNG/SVG figures. No price, age, overall or potential enters either model's geometry. Budget and age remain shared eligibility filters. Hidden-block and internal similarity metrics do not establish future sporting performance.

Model 3 saves `market_value_rf_2022-09-26.joblib`, `test_value_gaps.csv`, `train_nested_oof_value_gaps.csv`, `all_out_of_fit_predictions.csv`, separate TOP-10 lists, test sensitivity, outer-fold ID audit and a hashed prediction manifest in `data/processed/valuation/`. All 2,886 predictions exclude the player's own target from fitting and parameter selection: nested 5×3 CV for train, train-only model for test. Aggregate metrics and five PNG/SVG figures are in `reports/valuation/`. Preserve `prediction_origin`; OOF and test lists come from different fits and are ranked separately. Values are in unverified source currency units. Positive model gaps are screening signals, not objective fair prices.

Required columns:

- `short_name` (str) — player name
- `player_positions` (str) — positions separated by commas, e.g. `CM, CDM`
- `age` (number)
- `value_eur` (number) — recorded value, **not necessarily an actual transfer fee**
- `overall` (number) — rating used only for the baseline comparison
- `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` (numeric player attributes)

You can adapt the feature list in `src/scouting.py`. The demo excludes missing values for any required feature; full study preprocessing should audit the impact of that choice.

A tiny demonstration CSV (`synthetic_demo.csv`) is provided solely to verify the interface; names and numbers are fictional.
