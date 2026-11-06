"""Transparent, research-stage player similarity recommender.

This is a reproducible baseline prototype, not a validated transfer model.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

FEATURES = ("pace", "shooting", "passing", "dribbling", "defending", "physic")
REQUIRED = ("short_name", "player_positions", "age", "value_eur", "overall", *FEATURES)


def prepare_players(df: pd.DataFrame) -> pd.DataFrame:
    """Clean required columns and retain players who have CM among their positions."""
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")

    players = df.loc[:, list(REQUIRED)].copy()
    players["short_name"] = players["short_name"].astype("string").str.strip()
    players["player_positions"] = players["player_positions"].astype("string")
    players = players.loc[
        players["player_positions"].str.split(",").map(
            lambda parts: any(p.strip().upper() == "CM" for p in parts) if isinstance(parts, list) else False
        )
    ].copy()

    numeric = ("age", "value_eur", "overall", *FEATURES)
    for column in numeric:
        players[column] = pd.to_numeric(players[column], errors="coerce")
    players = players.dropna(subset=["short_name", *numeric])
    players = players.loc[(players["age"] >= 15) & (players["value_eur"] >= 0)]
    players = players.drop_duplicates().reset_index(drop=True)
    if players.empty:
        raise ValueError("No CM players with complete required attributes were found.")
    return players


def recommend(
    players: pd.DataFrame,
    reference_index: int,
    max_value_eur: float,
    max_age: int = 35,
    top_k: int = 5,
    features: Iterable[str] = FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ML similarity shortlist, overall-rating baseline) for the same pool.

    Similarity is descriptive and does not establish real-world transfer value.
    """
    selected_features = list(features)
    if not selected_features or any(f not in players.columns for f in selected_features):
        raise ValueError("Provide existing playing-attribute features.")
    if reference_index not in players.index:
        raise ValueError("Reference index does not exist in prepared players.")
    if max_value_eur < 0 or top_k < 1:
        raise ValueError("Budget must be nonnegative and top_k positive.")

    eligible = players.loc[
        (players["value_eur"] <= max_value_eur)
        & (players["age"] <= max_age)
        & (players.index != reference_index)
    ].copy()
    if eligible.empty:
        return eligible.assign(similarity=pd.Series(dtype=float)), eligible

    # Avoid data leakage from prices or overall ratings into profile similarity.
    scaler = StandardScaler()
    vectors = scaler.fit_transform(players[selected_features].astype(float))
    scores = cosine_similarity(vectors[reference_index:reference_index + 1], vectors[eligible.index])[0]
    eligible["similarity"] = scores
    # High cosine similarity is *not* a confidence/probability percentage.
    ranked = eligible.sort_values(["similarity", "value_eur"], ascending=[False, True]).head(top_k)
    baseline = eligible.sort_values(["overall", "value_eur"], ascending=[False, True]).head(top_k)
    return ranked.reset_index(drop=True), baseline.reset_index(drop=True)
