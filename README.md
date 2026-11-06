# Moneyball 2.0 — AI-Powered Football Scouting

> **Status:** initial research and runnable scouting prototype. Data collection, empirical validation, and final conclusions are still pending.

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
notebooks/               Future EDA and modeling notebooks
docs/                    Research question, hypotheses, methodological plan
requirements.txt         Python dependencies
```

## Quick start

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

## Evaluation plan (not yet executed)

- Baseline: select affordable players in the same position using highest `overall` rating.
- Candidate system: feature-based similarity plus the same budget / position / age filters.
- Audit: compare similarity to the reference player, candidate price distribution, rank stability under feature perturbations, and case studies reviewed against external evidence where possible.
- Avoid circular validation: similarity computed from the features used to rank candidates cannot, on its own, demonstrate real-world superiority. A future independent evaluation criterion (e.g., outcomes, expert assessments, or held-out attributes) is required for stronger claims.
- Analyze at least two limitations: positional comparability; outdated or game-derived valuations; league-strength differences; small sample sizes; selection bias; potential leakage.

## AI usage disclosure

AI assistance may be used for drafting, code scaffolding, or debugging. Any such contribution should be documented and checked by the team. All empirical claims and final recommendations must be independently verified.

## Academic integrity

This is a **project starter**, not a completed research study. Do not present demonstration values or untested hypotheses as experimental results.
