# ⚽ Moneyball 2.0 — AI-Powered Football Scouting

**Historical, evidence-aware decision support for finding affordable central midfielders.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-0B766E)](#research-integrity-and-limitations)

> **Research snapshot:** 26 September 2022 (FIFA 23) · **Position:** CM · **Currency:** source units, *not independently verified as EUR* · **Scope:** academic research, **not** live transfer advice.

## Why this project?

A club with a limited transfer budget cannot always afford the highest-rated players. **Moneyball 2.0** tests whether a player-profile recommender can identify central midfielders with similar game-derived characteristics at lower **recorded historical valuations** than a simple overall-rating baseline.

**Research question:** *To what extent can machine learning methods identify potentially undervalued central midfielders with playing characteristics comparable to more expensive players, and how do these recommendations compare with traditional rating-based selection methods?*

A *potentially undervalued player* here means a **scouting lead**, not proven market mispricing, a bargain transfer fee, or a prediction of future sporting success.

## What it does

- **Profile matching:** 18 detailed FIFA attributes, train-fitted standardization, group weighting and cosine-nearest-neighbor search.
- **Style exploration:** K-Means (three algorithmic clusters); optional same-cluster filter. Clusters are **not** externally validated football positions.
- **Recorded-value modeling:** Random Forest regression with 27 allowable features. Out-of-fit predictions are used for training-player residuals; current recorded values never enter similarity geometry.
- **Decision support:** filter by budget, age, playing style or reference player; compare rankings; inspect per-candidate explanations and research audits.
- **Streamlit UI:** historical shortlist, identical initial-eligibility rating baseline, export to CSV and JSON, and explicit uncertainty warnings.

```text
FIFA 23 player snapshot           Transfermarkt historical records
          │                                   │
          └────────── identity matching ───────┘
                              │
                    2022-09-26 CM cohort
                              │
                     cleaning + features
                              │
                   ┌──────────┴──────────┐
                   │                     │
               similarity             valuation
            18 attributes          27 raw features
           K-Means + KNN            RF + OOF gaps
                   │                     │
                   └──────────┬──────────┘
                              │
                    rules + explanations
                              │
                  Streamlit scouting UI
```

## Dataset and audit

| Data checkpoint | Players | Notes |
|---|---:|---|
| FIFA 23 male CM population | **4,184** | Exact `CM` position token, snapshot 2022-09-26 |
| Reliably matched to Transfermarkt | **3,057** | 73.06% cross-source coverage |
| Positive, historically usable valuations | **2,886** | Value timestamp ≤ cutoff, at most 365 days old |
| Training / diagnostic split | **2,308 / 578** | The 578-player split has been inspected during EDA; **not** an untouched confirmatory test |

**Sources:** [FIFA 23 Complete Player Dataset (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) · [Football Datasets / Transfermarkt (Kaggle)](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets/data).

Identity matching does **not** join the unrelated FIFA and Transfermarkt numeric player IDs: it checks normalized name aliases, birth dates and conflicts. Fuzzy candidates require manual review. Source ZIPs, individual joined records and saved model bundles are **not committed**; see [profiling](docs/data_profiling.md), the [feature contract](config/feature_set_v1.json) and [data handling notes](data/README.md).

## Research findings (descriptive)

| Experiment | Reported result | Interpretation |
|---|---|---|
| Similar profiles, different values | 132/221 (59.7%) high-similarity nearest pairs had ≥2× recorded-value difference | A screening opportunity in this sample; **not proof** of mispricing |
| K-Means (k=3) | Test silhouette **0.169**; median refit ARI **0.863** | Fairly stable assignments, **weak** cluster separation |
| Expensive reference cohort (41 players) | Mean cosine: **0.819** baseline vs **0.934** KNN | Model improves a *same-feature* metric by construction; not independent transfer quality |
| Market-value regression | RF raw-value MAE **1.334M**, R² **0.811**; Ridge raw-value MAE **1.316M** | Different models win on different error scales; no universal RF superiority |
| Dated valuation backtest, 15M budget | **40/41** complete pairs; average `(KNN − baseline)` log-value change **+0.1539** | Descriptive, dependent query slots; **10M budget reverses sign** (−0.0541) |

All amounts are **recorded source units**; the scenario's `€` label, when selected, is an **unverified assumption**. Follow-up market-value quotes are not transfer profits or sporting outcomes.

Read the **[full Research Report](docs/research_report.md)** for methods, hypotheses, results, limitations and a worked scouting decision. Supporting machine-readable evidence lives in [`reports/`](reports/), especially [`reports/eda/`](reports/eda/), [`reports/models/`](reports/models/), [`reports/valuation/`](reports/valuation/) and [`reports/evaluation/`](reports/evaluation/).

## Case study: 15M / ≤25 / Deep-Lying Playmaker

The saved Step 8 validation reports **986** initially eligible CM candidates, **32** qualifying the style/similarity gates and a TOP-5 shortlist. A local execution of the historical scorer listed **J. Veerman, Jean Lucas, Manu Morlanes, J. Moder and Y. Adli**. This is a **2022 scouting demonstration**, not a current transfer ranking. A positive estimated value gap should trigger **manual review**, not automatic purchase. See [`reports/recommendation/example_validation.json`](reports/recommendation/example_validation.json).

## Run locally

**Requirements:** Python 3.12 and the versions in [`requirements.txt`](requirements.txt); local **trusted** research artifacts produced from the historical source ZIPs. `joblib` files may fail to deserialize across pandas/NumPy/scikit-learn versions.

```powershell
# From the repository root (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m pytest -q
```

### First-time artifact preparation

Cloning GitHub **does not** fetch the source ZIPs or models. Place the following two ZIP archives in `data/raw/`, or set `MONEYBALL_DATA_DIR` to the folder containing them:

```text
stefanoleone992_fifa-23-complete-player-dataset_male.zip
xfkzujqjvx97n_football-datasets_data.zip
```

The pipeline checks the ZIPs against the hashes recorded in [`reports/data_profiling/`](reports/data_profiling/). **Do not silently replace the audited versions.** Open the following notebooks (for example, in JupyterLab) and run all cells **in this order**:

1. [`02_data_preprocessing.ipynb`](notebooks/02_data_preprocessing.ipynb) → cleaned historical cohort.
2. [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) → dated feature tables, split and manifest.
3. [`06_ml_models.ipynb`](notebooks/06_ml_models.ipynb) → fitted K-Means/KNN bundle.
4. [`07_market_value_prediction.ipynb`](notebooks/07_market_value_prediction.ipynb) → Random Forest, OOF predictions and manifest.

The optional [`01_data_profiling.ipynb`](notebooks/01_data_profiling.ipynb) repeats source audits. Other notebooks reproduce EDA and backtesting but are not required to load the dashboard. Models are stored under **`data/processed/` (gitignored)**; the app never downloads unknown model binaries or retrains silently.

Check artifact loading and launch:

```powershell
.\.venv\Scripts\python.exe -c "from src.recommendation import RecommendationSystem; RecommendationSystem.from_artifacts(); print('Artifacts OK')"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The UI will open at **http://localhost:8501**. If required files are missing or their hashes do not match, the app displays an error rather than substituting synthetic data.

## Repository guide

| Path | Purpose |
|---|---|
| [`src/models.py`](src/models.py) | Train-only K-Means, exact cosine-neighbor retrieval |
| [`src/valuation.py`](src/valuation.py) | Leak-resistant recorded-value regression |
| [`src/baseline.py`](src/baseline.py) | Rating-only control and shared eligibility policy |
| [`src/evaluation.py`](src/evaluation.py) | Dated follow-up market-value analysis |
| [`src/recommendation.py`](src/recommendation.py) | Decision pipeline, styles, economic screening and explanations |
| [`app/`](app/) | Streamlit dashboard and UI helpers |
| [`config/feature_set_v1.json`](config/feature_set_v1.json) | Fixed input-feature and temporal contract |
| [`notebooks/`](notebooks/) | End-to-end research notebooks with saved analysis |
| [`scripts/`](scripts/) | Notebook builders and data audits |
| [`reports/`](reports/) | Figures, summary metrics and reproducibility manifests |
| [`tests/`](tests/) | Unit/contract tests; not a substitute for live end-to-end validation |

`src/scouting.py` is the **legacy six-aggregate-feature starter prototype**; the research dashboard uses the newer 18-feature system.

## Research integrity and limitations

- **Historical** (2022): no live player availability, current contracts, real-time pricing or completed-deal prediction.
- FIFA ratings describe **game-derived assessments**, not measured progressive passes, pressures, xA or verified match-by-match actions.
- Recorded Transfermarkt valuations are **not** fees, wages or objective fair value. The source's currency denomination was not independently verified.
- Data matching is incomplete and rating-biased; the historical candidate gallery contains only **training players (2,308)**.
- K-Means separation is weak; the Deep-Lying Playmaker profile is an explicit training-data recipe, not a validated position label.
- The test split was explored during model development; independent temporal validation and measured sporting outcomes are still needed.
- Subsequent-value comparisons reuse candidate identities, are sensitive to budget and quote coverage, and do **not** prove superior transfer returns.

## Team and acknowledgement

**Temirlan Ilyassov · Dilnaz Amanzholova · Zhanel Adylbekova · Nurbeket Zhunussov**  
Laboratory Work No. 2 — *AI Football Intelligence*, Scenario 3, *Moneyball 2.0*.

AI tools assisted with portions of code/document drafting. The team is responsible for verifying primary data, reproducibility, source conditions and all scientific claims. The report separates reproducible repository results from assumptions and illustrative decision recommendations.

*External datasets retain their original terms and licenses; check them before redistribution. No license for this project's original code has been declared in this repository.*
