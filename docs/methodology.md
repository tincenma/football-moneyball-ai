# Proposed Methodology (Pre-Registration Draft)

This is a plan, **not evidence that a study has already been conducted**.

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
