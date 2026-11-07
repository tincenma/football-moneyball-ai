"""Pure view/pool helpers for the historical Moneyball decision dashboard.

Keep the Streamlit page thin and preserve the research engine's cohort policy.
Neither a model gap nor similarity constitutes evidence of transfer success.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = (
    "rank", "fifa_id", "short_name", "age_at_cutoff", "overall",
    "tm_market_value", "similarity_percent", "style_similarity",
    "rms_z", "cluster", "predicted_value", "value_gap",
    "value_gap_fraction", "prediction_origin", "economic_status",
)


def format_recorded_value(value: object, *, currency_assumption: bool = False) -> str:
    """Present prices without silently claiming source units are euros."""
    if value is None or pd.isna(value):
        return "Unavailable"
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        return "Unavailable"
    suffix = "EUR (assumed)" if currency_assumption else "source units"
    return f"{float(value):,.0f} {suffix}"


def reference_catalog(reference_profiles: pd.DataFrame) -> dict[str, str]:
    """Map unique string IDs to clear labels; prevent ambiguous same-name picks."""
    required = {"fifa_id", "short_name"}
    if not isinstance(reference_profiles, pd.DataFrame) or not required.issubset(reference_profiles.columns):
        raise ValueError("Reference profiles must include fifa_id and short_name")
    frame = reference_profiles[["fifa_id", "short_name"]].copy()
    frame["fifa_id"] = frame.fifa_id.astype("string").str.strip()
    if frame.fifa_id.isna().any() or frame.fifa_id.eq("").any() or frame.fifa_id.duplicated().any():
        raise ValueError("Reference fifa_id must be present and unique")
    frame["short_name"] = frame.short_name.astype("string").fillna("Unknown player").str.strip()
    frame.loc[frame.short_name.eq(""), "short_name"] = "Unknown player"
    ordered = frame.sort_values(["short_name", "fifa_id"], kind="stable")
    return {str(row.fifa_id): f"{row.short_name} [FIFA {row.fifa_id}]" for row in ordered.itertuples(index=False)}


def rank_baseline_from_pool(pool: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    """Rank only the *already eligible* pool using baseline's documented sort.

    The eligible pool must come from player_models.eligible_pool using exactly the
    same budget, age, value-age, cluster and reference filters as ML. The ML
    similarity/style thresholds are deliberately NOT applied to this control.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    needed = {"fifa_id", "short_name", "overall", "tm_market_value", "age_at_cutoff"}
    if not isinstance(pool, pd.DataFrame) or not needed.issubset(pool.columns):
        raise ValueError("Missing columns in eligible pool")
    if pool.fifa_id.astype("string").duplicated().any():
        raise ValueError("Eligible player IDs must be unique")
    valid = pool.sort_values(["overall", "tm_market_value", "fifa_id"], ascending=[False, True, True], kind="stable")
    valid = valid.head(top_k).copy().reset_index(drop=True)
    valid.insert(0, "baseline_rank", range(1, len(valid) + 1))
    return valid


def presentation_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Select only known display fields; never reinterpret missing values as 0."""
    present = [col for col in CANDIDATE_COLUMNS if col in recommendations.columns]
    result = recommendations.loc[:, present].copy()
    if "similarity_percent" in result:
        result["similarity_percent"] = pd.to_numeric(result["similarity_percent"], errors="coerce").round(1)
    if "value_gap_fraction" in result:
        result["value_gap_fraction"] = (100 * pd.to_numeric(result["value_gap_fraction"], errors="coerce")).round(1)
        result = result.rename(columns={"value_gap_fraction": "model_gap_percent"})
    return result
