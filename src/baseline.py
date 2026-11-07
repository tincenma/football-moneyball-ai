"""Rating-only control system for the cleaned historical Moneyball dataset."""
from __future__ import annotations

from numbers import Integral, Real
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = ("fifa_id", "short_name", "player_positions_normalized", "age_at_cutoff", "overall", "tm_market_value",
            "as_of_date", "fifa_update_date", "market_value_date")
RANK_COLUMNS = ("overall", "tm_market_value", "fifa_id")


def _nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def build_eligible_pool(
    players: pd.DataFrame, *, max_value: float = 15_000_000, max_age: float = 30,
    cutoff: str = "2022-09-26", max_value_age_days: int = 365,
    excluded_fifa_ids: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eligible players, one reason per input row).

    Price units must be settled by the caller. A positive recorded market value
    is a screening proxy, not a transfer fee. All cutoffs are inclusive.
    No playing attributes, fitted estimators or similarity scores are accessed.
    """
    for value, name in [(max_value, "max_value"), (max_age, "max_age"), (max_value_age_days, "max_value_age_days")]:
        _nonnegative_number(value, name)
    if not isinstance(max_value_age_days, Integral):
        raise ValueError("max_value_age_days must be an integer")
    try:
        anchor = pd.to_datetime(cutoff, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("cutoff must be a valid ISO date") from exc
    if pd.isna(anchor) or anchor.tzinfo is not None:
        raise ValueError("cutoff must be a valid timezone-free ISO date")
    missing = sorted(set(REQUIRED) - set(players.columns))
    if missing:
        raise ValueError(f"Missing baseline columns: {', '.join(missing)}")
    pool = players.copy()
    pool["fifa_id"] = pool.fifa_id.astype("string").str.strip()
    if pool.fifa_id.isna().any() or pool.fifa_id.eq("").any() or pool.fifa_id.duplicated().any():
        raise ValueError("fifa_id must be present and unique in the input snapshot")
    if isinstance(excluded_fifa_ids, str):
        raise ValueError("excluded_fifa_ids must be a collection, not a single string")
    excluded = set() if excluded_fifa_ids is None else {str(x).strip() for x in excluded_fifa_ids}
    for column in ("age_at_cutoff", "overall", "tm_market_value"):
        pool[column] = pd.to_numeric(pool[column], errors="coerce")
    numeric = (np.isfinite(pool.age_at_cutoff) & pool.age_at_cutoff.between(15, 50)
               & np.isfinite(pool.overall) & pool.overall.between(1, 99) & pool.overall.mod(1).eq(0)
               & np.isfinite(pool.tm_market_value) & pool.tm_market_value.gt(0))
    named = pool.short_name.astype("string").str.strip().fillna("").ne("")
    observation = pd.to_datetime(pool.as_of_date, format="%Y-%m-%d", errors="coerce")
    fifa_date = pd.to_datetime(pool.fifa_update_date, format="%Y-%m-%d", errors="coerce")
    value_date = pd.to_datetime(pool.market_value_date, format="%Y-%m-%d", errors="coerce")
    value_age = (anchor - value_date).dt.days
    historical = observation.eq(anchor) & fifa_date.eq(anchor) & value_age.between(0, max_value_age_days)
    is_cm = pool.player_positions_normalized.astype("string").fillna("").str.upper().str.replace("|", ",", regex=False).str.split(",").map(
        lambda parts: "CM" in [part.strip() for part in parts])
    conditions = [~(numeric & named), ~historical, ~is_cm, pool.tm_market_value.gt(max_value),
                  pool.age_at_cutoff.gt(max_age), pool.fifa_id.isin(excluded)]
    labels = ["invalid_required_value", "invalid_historical_date", "not_cm", "over_budget", "over_age", "excluded_reference"]
    status = np.select([c.fillna(True).to_numpy(dtype=bool) for c in conditions], labels, default="eligible")
    audit = pd.DataFrame({"fifa_id": pool.fifa_id.to_numpy(), "status": status})
    return pool.loc[status == "eligible"].copy().reset_index(drop=True), audit


def select_baseline(
    players: pd.DataFrame, *, max_value: float = 15_000_000, max_age: float = 30, top_k: int = 5,
    cutoff: str = "2022-09-26", max_value_age_days: int = 365,
    excluded_fifa_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """CM -> individual budget -> maximum age -> overall descending -> TOP-k.

    Ties: lower market value, then lexicographically smaller string fifa_id.
    No sum-of-prices optimization: returned rows are alternative candidates.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    eligible, _ = build_eligible_pool(players, max_value=max_value, max_age=max_age, cutoff=cutoff,
                                    max_value_age_days=max_value_age_days, excluded_fifa_ids=excluded_fifa_ids)
    ranked = eligible.sort_values(list(RANK_COLUMNS), ascending=[False, True, True], kind="stable").head(top_k)
    ranked = ranked.drop(columns="baseline_rank", errors="ignore").reset_index(drop=True)
    ranked.insert(0, "baseline_rank", np.arange(1, len(ranked) + 1))
    return ranked
