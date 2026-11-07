"""Moneyball 2.0 — historical CM decision-support dashboard (2022-09-26).

This app serves ONLY previously fitted and provenance-checked project artifacts.
It never retrains models or silently switches to synthetic demo values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dashboard_helpers import (  # noqa: E402
    format_recorded_value,
    presentation_table,
    rank_baseline_from_pool,
    reference_catalog,
)
from src.recommendation import RecommendationSystem, STYLE_NAME  # noqa: E402

st.set_page_config(page_title="Moneyball 2.0 | Decision Intelligence", page_icon="⚽", layout="wide")
st.title("⚽ Moneyball 2.0 — Decision Intelligence")
st.caption("Historical central midfield scouting • snapshot: 26 Sep 2022 • research prototype")
st.warning(
    "Recorded Transfermarkt values are NOT confirmed EUR-denominated transfer fees. "
    "Cosine similarity is not the probability of a successful transfer. "
    "Model-predicted value minus recorded value is a screening signal, not fair market value."
)


@st.cache_resource(show_spinner="Validating locally saved research models…")
def load_system(root: str) -> RecommendationSystem:
    # The loading function verifies manifests and their SHA-256 checksums.
    # joblib bundles must be generated locally by trusted project notebooks.
    return RecommendationSystem.from_artifacts(Path(root))


try:
    system = load_system(str(ROOT))
except (OSError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
    st.error("Validated saved artifacts were not found or could not be loaded. No synthetic fallback was used.")
    st.code(str(exc), language="text")
    st.info(
        "Run the project's Data Preprocessing → Feature Engineering → ML Models → "
        "Market Value Prediction notebooks locally to regenerate data/processed/. "
        "This directory is deliberately gitignored; cloning the repository alone is not enough. "
        "Use only locally produced, trusted joblib files."
    )
    st.stop()

if system.references is None or system.references.empty:
    st.error("The validated model did not include reference profiles.")
    st.stop()

catalog = reference_catalog(system.references)
ids = list(catalog)

with st.sidebar:
    st.header("Scouting request")
    mode = st.radio(
        "Find players by",
        ("Deep-Lying Playmaker profile", "Reference player", "Profile + reference"),
        help="The style is a hand-defined archetype derived from training data, not a validated real-world role.",
    )
    chosen_id = None
    if mode != "Deep-Lying Playmaker profile":
        chosen_id = st.selectbox("Reference central midfielder", options=ids, format_func=catalog.__getitem__)
    budget_m = st.number_input("Maximum recorded value (millions of source units)", min_value=0.0, value=15.0, step=1.0)
    max_age = st.slider("Maximum age at snapshot", min_value=15, max_value=50, value=25)
    similarity_pct = st.slider("Minimum cosine similarity (%)", min_value=0, max_value=100, value=85)
    top_k = st.slider("Number of candidates", min_value=1, max_value=20, value=5)
    priority = st.radio("Ranking priority", ("Playing similarity", "Positive model gap"))
    within_cluster = st.checkbox("Same K-Means cluster only", value=False,
                                 help="Experimental constraint; clusters are not confirmed football roles.")
    require_level = st.checkbox("Constrain absolute attribute-level distance", value=False)
    max_rms_z = st.slider("Maximum RMS standardized distance", 0.1, 3.0, 0.75, 0.05) if require_level else None
    max_staleness = st.slider("Maximum age of the recorded valuation (days)", 0, 365, 365)
    assume_eur = st.checkbox("Label source units as EUR (UNVERIFIED assumption)", value=False)
    submitted = st.button("Find candidates", type="primary", use_container_width=True)

st.markdown(
    "**Search scope:** CM players with a reliable FIFA ↔ Transfermarkt match and a "
    "positive historical valuation. The saved research model is fixed at 2022-09-26."
)

if not submitted:
    st.info("Select a scouting scenario and click **Find candidates**. No model is retrained by this dashboard.")
    st.stop()

reference = chosen_id
playing_style = None if mode == "Reference player" else STYLE_NAME
budget = float(budget_m) * 1_000_000
ranking = "similarity" if priority == "Playing similarity" else "value_gap"
request = {
    "position": "CM",
    "budget": budget,
    "max_age": max_age,
    "playing_style": playing_style,
    "similarity_threshold": similarity_pct / 100,
    "reference": reference,
    "top_k": top_k,
    "ranking": ranking,
    "within_cluster": within_cluster,
    "max_rms_z": max_rms_z,
    "max_value_age_days": max_staleness,
    "currency": "EUR" if assume_eur else "source_units",
}

try:
    recommendations = system.recommend(**request)
    query = system._reference(reference) if reference is not None else system._style()[0].iloc[0]
    # These filters reproduce the pre-similarity player universe used by KNN.
    eligible = system.player_models.eligible_pool(
        query,
        max_value=budget,
        max_age=max_age,
        max_value_age_days=max_staleness,
        within_cluster=within_cluster,
    )
    baseline = rank_baseline_from_pool(eligible, top_k=top_k)
except (ValueError, TypeError, KeyError, IndexError) as exc:
    st.error("Unable to build a recommendation for these parameters.")
    st.code(str(exc), language="text")
    st.stop()

audit = recommendations.attrs.get("audit", {})
pool_audit = audit.get("pool", {})
# A different ID hash means the baseline and AI did not start from one
# eligibility universe; do not display a misleading comparison in that case.
from src.models import eligible_ids_hash  # noqa: E402
same_pool = eligible_ids_hash(eligible.fifa_id) == pool_audit.get("eligible_ids_sha256")
if not same_pool:
    st.error("Baseline candidate universe does not match the model pool. Comparison disabled.")
    baseline = baseline.iloc[:0].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Eligible players", int(pool_audit.get("eligible_count", 0)))
m2.metric("Above similarity threshold", int(audit.get("after_similarity", 0)))
m3.metric("After style/level filters", int(audit.get("after_level_filter", 0)))
m4.metric("Returned", len(recommendations))

if recommendations.empty:
    st.warning(f"No candidates met every constraint. Reason: {audit.get('empty_reason', 'not specified')}.")
    st.caption("Adjust constraints to explore the scenario; the system did not silently relax them.")

results_tab, baseline_tab, audit_tab = st.tabs(["🎯 Recommendations", "⚖️ Rating baseline", "🔬 Research audit"])

with results_tab:
    if not recommendations.empty:
        st.subheader("Candidate shortlist")
        display = presentation_table(recommendations)
        st.dataframe(display, hide_index=True, use_container_width=True)
        bars = recommendations[["short_name", "similarity_percent"]].copy()
        bars = bars.rename(columns={"similarity_percent": "Cosine similarity (%)"}).set_index("short_name")
        st.bar_chart(bars)
        st.markdown("#### Why these candidates were recommended")
        for row in recommendations.itertuples(index=False):
            with st.expander(f"#{row.rank} {row.short_name} · {row.similarity_percent:.1f}% cosine", expanded=row.rank == 1):
                st.write(row.explanation)
                st.caption(f"Recorded value: {format_recorded_value(row.tm_market_value, currency_assumption=assume_eur)}")
                st.caption(f"Prediction origin: {row.prediction_origin}; economic status: {row.economic_status}")
        st.download_button(
            "⬇️ Download shortlist (CSV)",
            data=recommendations.to_csv(index=False).encode("utf-8-sig"),
            file_name="moneyball_historical_cm_shortlist.csv",
            mime="text/csv",
        )

with baseline_tab:
    st.subheader("Rating-only control")
    st.caption(
        "Baseline ranks the SAME pre-similarity eligible pool using FIFA overall, then lower recorded value. "
        "It does not apply the ML similarity/style thresholds. The result is a methodological comparison, "
        "not an independent proof that either list would produce better transfers."
    )
    if same_pool:
        st.success(f"Candidate-pool ID hash verified • {len(eligible)} eligible players")
        st.dataframe(
            baseline[["baseline_rank", "fifa_id", "short_name", "overall", "age_at_cutoff", "tm_market_value"]],
            hide_index=True, use_container_width=True,
        )
        candidate_ids = set(recommendations.fifa_id.astype(str))
        baseline_ids = set(baseline.fifa_id.astype(str))
        st.metric("TOP-k overlap", len(candidate_ids & baseline_ids))

with audit_tab:
    st.subheader("Reproducibility and caveats")
    st.write(f"**Snapshot:** {system.cutoff}")
    st.write(f"**Player target:** {audit.get('similarity_target', '')}")
    st.write(f"**Economic estimates available:** {audit.get('economic_available', 0)}")
    st.write("**Serving behavior:** pre-trained models, no fitting and no future-outcome columns")
    st.json(audit, expanded=False)
    if audit.get("artifact_hashes"):
        st.caption("Artifact SHA-256 hashes are recorded in the audit output.")
    download_audit = {"request": request, "audit": audit, "same_eligible_pool": same_pool}
    st.download_button(
        "⬇️ Download request + audit (JSON)",
        data=json.dumps(download_audit, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name="moneyball_historical_cm_audit.json",
        mime="application/json",
    )
    st.info(
        "The market-value model's out-of-fit predictions support research screening only. "
        "Transfer fees, live availability, contracts and future sporting outcomes are not predicted. "
        "See docs/model_evaluation.md for the retrospective evaluation limitations."
    )
