# Proposed Methodology (Pre-Registration Draft)

This document retains the confirmatory evaluation plan. Completed exploratory work is linked below; it does not establish superiority over the scouting baseline.

**Data preparation update (2026-11-07):** the full [dataset profiling and join audit](data_profiling.md) is complete. The [v1 feature contract](feature_set.md) fixes the primary 2022-09-26 male CM snapshot, 18 similarity attributes, and 27 raw valuation inputs. It supersedes provisional feature choices in the starter prototype. Exploratory model fitting is documented below; independent confirmatory evaluation remains pending.

**Exploratory research update:** [Step 4](../notebooks/04_exploratory_data_analysis.ipynb) executes regression, clustering and nearest-neighbor analyses for H1–H4. [Computed findings](eda_findings.md) are exploratory observations. The existing holdout has been inspected during EDA and is no longer an untouched final test.

**Model implementation update:** [Step 6](../notebooks/06_ml_models.ipynb) implements the reusable K-Means/KNN engine and compares it with the frozen baseline on identical eligible candidate IDs. [Validation findings](ml_models_validation.md) distinguish computational correctness, cluster separation/stability and hidden-attribute diagnostics from real transfer outcomes. The 41 expensive test references are reported separately; conditional uncertainty and inconclusive findings are retained.

## 1. Study scope

Focus on central midfielders (`CM`). Keep the dataset season fixed. Filter to comparable positions before computing recommendations. Record how multi-position players are handled.

## 2. Data preparation

Inspect missingness, duplicates, outliers, schema consistency, and source definitions. Preserve IDs, names, age, position, season and listed value. Standardize numeric playing attributes within the eligible dataset. Do **not** use listed value in the similarity vector: it is a budget constraint and an external comparison target.

## 3. Candidate selection

For a selected reference player, filter to CM candidates within the user-set budget and age limit. Rank remaining candidates by cosine similarity of standardized attributes. Exclude the reference player itself.

## 4. Baseline

Independently shortlist eligible players by descending `overall` rating (tie break: lower cost). Compare top-k rankings under identical eligibility constraints.

## 5. Validation

Predefine an independent outcome or criterion for claims of superiority, if available. Internal similarity and cost indicators are **descriptive**, not independently validated measures of transfer quality. Perform sensitivity analysis: vary feature weights, missing-value rules, and maximum budget, and observe rank stability.

## 6. Limitations to discuss

1. FIFA-style data may not represent actual match performance, market value, wages or contract conditions.
2. Cross-position comparison is invalid without position-specific feature sets; even central midfield roles vary.
3. Selection bias and missing attributes can distort rankings.
4. Older data and seasonal drift reduce relevance to current transfer markets.

## 7. Decision

Deliver a defensible shortlist, not a promise of future sporting success. Discuss confidence, alternative players and what additional scouting evidence is needed.
