"""Streamlit demo for academic Moneyball 2.0 project."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scouting import prepare_players, recommend  # noqa: E402

st.set_page_config(page_title="Moneyball 2.0", page_icon="⚽", layout="wide")
st.title("⚽ Moneyball 2.0 — Football Scouting")
st.caption("Research prototype — profile similarity is not proof of market undervaluation.")

uploaded = st.file_uploader("Upload player CSV", type=["csv"])
use_demo = st.checkbox("Use synthetic demo data (not research evidence)", value=uploaded is None)
if uploaded is None and not use_demo:
    st.info("Upload a CSV to start. Required columns are documented in data/README.md.")
    st.stop()

try:
    data = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(ROOT / "data/synthetic_demo.csv")
    players = prepare_players(data)
except (ValueError, OSError, pd.errors.ParserError) as exc:
    st.error(str(exc))
    st.stop()

st.success(f"{len(players)} eligible CM players with complete features")
reference_index = st.selectbox(
    "Reference central midfielder",
    options=players.index.tolist(),
    format_func=lambda i: f"{players.loc[i, 'short_name']} — €{players.loc[i, 'value_eur']:,.0f}",
)
col1, col2, col3 = st.columns(3)
with col1:
    budget_m = st.number_input("Maximum recorded value (€ million)", min_value=0.0, value=15.0, step=1.0)
with col2:
    max_age = st.slider("Maximum age", 18, 45, 30)
with col3:
    top_k = st.slider("Number of candidates", 1, 20, 5)

if st.button("Find comparable players", type="primary"):
    found, baseline = recommend(players, reference_index, budget_m * 1e6, max_age, top_k)
    if found.empty:
        st.warning("No candidates match the current filters.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Feature-similarity shortlist")
            st.dataframe(found[["short_name", "age", "value_eur", "overall", "similarity"]], hide_index=True)
        with right:
            st.subheader("Rating-based baseline")
            st.dataframe(baseline[["short_name", "age", "value_eur", "overall"]], hide_index=True)
        st.info("Cosine similarity is a profile match score, not a probability of transfer success. No experimental superiority claim has been established.")
