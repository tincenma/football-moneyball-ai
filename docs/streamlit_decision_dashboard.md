# Step 9 — Historical Decision Intelligence Dashboard

The Streamlit interface in `app/streamlit_app.py` integrates the completed `RecommendationSystem` from Step 8. It uses the 2022-09-26 male CM snapshot, 18-feature similarity model, saved Random Forest model, and nested out-of-fit (OOF) training predictions. It does not fit any model at request time or silently replace historical data with synthetic demo data.

## Local startup

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

**Important:** `data/processed/` is intentionally `.gitignore`d. GitHub cloning alone will not include the feature table, fitted models, or nested OOF bank. Regenerate these *locally* with the project's Step 2, Step 3, Step 6, and Market Value Prediction notebooks/scripts. `RecommendationSystem.from_artifacts()` checks persisted manifests, SHA-256 hashes, and training / fold assignments. Do not download unknown `.joblib` artifacts from third parties.

## UI workflow

- Pick a style recipe, a reference player, or both. The Deep-Lying Playmaker recipe is a manually defined train-derived scouting prototype, **not** a validated football-role label.
- Set maximum historical recorded value, maximum age, minimum cosine similarity, TOP-k, evaluation-date freshness, optional same-cluster restriction, and ranking priority.
- Inspect the ordered shortlist and model explanations; inspect the rating-based baseline selected from the **same initial eligible pool**. The baseline deliberately does *not* receive the ML similarity/style threshold.
- Export the shortlist CSV and provenance-rich request/audit JSON.

## Research integrity

- Historical Transfermarkt value is a *recorded valuation*, **not a transfer fee** or live asking price.
- Supplied source files lack a currency field. Display source units by default; EUR is available only as an explicitly **unverified scenario assumption**.
- `similarity_percent` means `100 * cosine_similarity`. It is **not** a success probability.
- A positive `value_gap` is an out-of-fit model residual, **not proof of market mispricing**. Negative gaps must remain visible.
- The baseline uses exactly the pre-similarity eligibility filters (position, age, budget, vintage, excluded reference, optional cluster); UI checks the eligible-ID SHA-256 to avoid invalid comparisons. This is a **methodological control**, not a separate confirmatory experiment.
- Use `docs/model_evaluation.md` for external-outcome coverage and sensitivity limitations. The Streamlit view alone does not prove superior sporting performance.

## UI testing

Run `python -m pytest -q tests/test_dashboard_helpers.py` for formatting, reference disambiguation, stable baseline ordering, and negative-gap presentation. Full end-to-end Streamlit tests require the local persisted research artifacts and an environment with Streamlit installed.
