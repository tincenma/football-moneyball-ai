# Moneyball 2.0 — AI-Powered Football Scouting

> **Status:** data profiling, preprocessing, feature engineering and exploratory hypothesis analysis are complete. Feature Set v1 is fixed. Exploratory regression and clustering have been executed; independent confirmatory evaluation remains pending. The runnable scouting prototype still uses its original six-attribute demo schema.

An academic research project exploring whether machine-learning-based player similarity can identify **potentially cost-effective alternatives** to expensive central midfielders (CM) for football clubs with limited transfer budgets.

## Research question

**To what extent can machine learning methods identify potentially undervalued central midfielders with playing characteristics comparable to more expensive players, and how do these recommendations compare with traditional rating-based selection methods?**

We use the term *potentially undervalued* cautiously: a lower recorded value paired with a similar feature profile is a **scouting lead**, not proof of mispricing or a guaranteed successful transfer.

## Team

- Ilyassov Temirlan
- Amanzholova Dilnaz
- Adylbekova Zhanel
- Zhunussov Nurbeket

Course: Laboratory Work No. 2 — **AI Football Intelligence**, Scenario 3: *Moneyball 2.0*.

## Research objectives

1. Audit and prepare football player data, preserving data provenance and season information.
2. Explore how playing attributes, positions, age, and recorded value are associated.
3. Build a comparable-player recommendation system using feature similarity (initially cosine similarity on standardized attributes).
4. Compare the approach against a simple rating-based selection baseline.
5. Test stability, identify at least two limitations, and present concrete budget-aware scouting recommendations.

## Repository structure

```text
app/                     Streamlit UI (starter demo)
src/                     Similarity and baseline ranking logic
tests/                   Unit tests with synthetic data
data/                    Instructions; external datasets are not committed
notebooks/               Executed data profiling notebook and future modeling notebooks
docs/                    Research question, hypotheses, methodological plan
requirements.txt         Python dependencies
```

## Quick start

The completed [data profiling report](docs/data_profiling.md) covers all 19 CSVs in the supplied FIFA and Transfermarkt archives, missingness, historical dates, CM counts, entity matching, and dated market-value joins. The [feature specification](docs/feature_set.md) and [machine-readable contract](config/feature_set_v1.json) fix 18 playing attributes for similarity and 27 raw inputs for valuation. The primary 2022-09-26 snapshot contains 4,184 CM players; 2,886 have a usable linked historical valuation. This contract has not yet been integrated into the demo below.

Open [Step 1: Data Profiling & Understanding](notebooks/01_data_profiling.ipynb) for executable checks, saved tables and 12 embedded figures. For rerunning, install `requirements-notebooks.txt` and use Run All. The default run validates the complete cached audit against archive SHA-256 hashes and independently recounts 18 CSVs; set `RUN_FULL_PROFILE=True` to repeat the full scan of all 19 CSVs. Archive paths are configurable with `MONEYBALL_DATA_DIR`.

[Step 2: Data Cleaning & Integration](notebooks/02_data_preprocessing.ipynb) rebuilds the cleaned 2022-09-26 CM dataset directly from the ZIPs, normalizes positions, quarantines conflicting keys, matches identities, and joins only historical market values. It exports 2,886 players with 27 model inputs plus targets and metadata to `data/processed/preprocessing/`, alongside a schema/manifest and selection audit. Seven unknown league levels remain explicitly marked for training-fold imputation. All notebook cells have saved outputs, including four figures.

[Step 3: Feature Engineering](notebooks/03_feature_engineering.ipynb) separates playing, performance and economic features. It derives total-goals/90 and assists/90 for the retrospective 21/22 layer, adds same-snapshot potential for economic analysis, and fits scaling/imputation/encoding on training players only. Standardized and block-weighted 18-attribute similarity matrices exclude price, age, potential and performance. Feature tables, player splits and transformation parameters are exported to `data/processed/features/`; five figures and all execution results are embedded in the notebook.

[Step 4: Exploratory Data Analysis](notebooks/04_exploratory_data_analysis.ipynb) examines H1–H4 through price-independent neighbors, train-only correlations and model selection, holdout regression, cluster separation/stability and budget-aware candidates. Nine figures are embedded and exported as PNG/SVG to `reports/eda/`. See [computed research findings](docs/eda_findings.md) for evidence and limitations. The existing test split has now been inspected for EDA and must not be described as an untouched final test in later work.

[Step 5: Baseline Model](notebooks/05_baseline_model.ipynb) implements the independent rating-only control: exact CM token, individual budget, maximum age, descending FIFA overall, TOP-5. The demonstration uses 15 million and age ≤30.0 years; EUR is explicitly a scenario assumption for the unverified TM currency. The reusable [baseline module](src/baseline.py) has no fitting or similarity dependency. Candidate lists and eligible IDs are local under `data/processed/baseline/`; `reports/baseline/baseline_protocol.json` fixes ranking, tie-breaks, source hashes and identical-pool requirements for later ML comparisons.

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload a CSV with the expected schema (see `data/README.md`), then choose a reference player and transfer budget. The app returns affordable CM candidates ranked by attribute similarity; a simple rating-based baseline is provided for comparison. A small **synthetic demonstration file** is included so you can test the interface; it must not be treated as research evidence.

Run tests:

```bash
pytest -q
```

## Proposed data source

The assignment recommends [FIFA 23 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset), alongside other open football datasets. Verify licensing and available columns before importing. **FIFA ratings and FIFA in-game values do not necessarily represent actual market performance or transfer fees.** We will record the season, measurement definitions, missing values, and sources before drawing conclusions.

## Independent evaluation plan (pending)

- Baseline: select affordable players in the same position using highest `overall` rating.
- Candidate system: feature-based similarity plus the same budget / position / age filters.
- Audit: compare similarity to the reference player, candidate price distribution, rank stability under feature perturbations, and case studies reviewed against external evidence where possible.
- Avoid circular validation: similarity computed from the features used to rank candidates cannot, on its own, demonstrate real-world superiority. A future independent evaluation criterion (e.g., outcomes, expert assessments, or held-out attributes) is required for stronger claims.
- Analyze at least two limitations: positional comparability; outdated or game-derived valuations; league-strength differences; small sample sizes; selection bias; potential leakage.

## AI usage disclosure

AI assistance may be used for drafting, code scaffolding, or debugging. Any such contribution should be documented and checked by the team. All empirical claims and final recommendations must be independently verified.

## Academic integrity

This is a **project starter**, not a completed research study. Do not present demonstration values or untested hypotheses as experimental results.
