"""Dated outcome joins for frozen historical scouting lists.

Future values are evaluation labels only. A recorded-value change is neither
a causal transfer return nor an independent observation for every query slot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dated_value_outcomes(players, history, *, cutoff="2022-09-26", horizon_days=365, max_lag_days=120):
    """Latest quote at/before horizon, strictly after selection and within lag.

    Return one row per player, including explicit missing statuses. Exact
    duplicates collapse. Conflicting quotes on any date in the outcome window
    quarantine that player's window. A latest zero quote stays unavailable;
    an older positive quote is never silently substituted for it.
    """
    required={"fifa_id","tm_id","as_of_date","market_value_date","tm_market_value"}
    if not required.issubset(players) or not {"player_id","date_unix","value"}.issubset(history):
        raise ValueError("Missing historical player or outcome columns")
    for value in [horizon_days,max_lag_days]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError("Horizon and maximum lag must be positive integer days")
    if max_lag_days>=horizon_days:
        raise ValueError("Outcome window must lie strictly after selection")
    anchor=pd.Timestamp(cutoff)
    if pd.isna(anchor) or anchor.tzinfo is not None:
        raise ValueError("Invalid cutoff")
    result=players.copy()
    for col in ["fifa_id","tm_id"]:
        result[col]=result[col].astype("string").str.strip()
        if result[col].isna().any() or result[col].eq("").any() or result[col].duplicated().any():
            raise ValueError("Player identities must be present and one-to-one")
    if not pd.to_datetime(result.as_of_date,errors="coerce").eq(anchor).all():
        raise ValueError("Selection profiles must match cutoff")
    base_age=(anchor-pd.to_datetime(result.market_value_date,errors="coerce")).dt.days
    base=pd.to_numeric(result.tm_market_value,errors="coerce").to_numpy(float)
    if not base_age.between(0,365).all() or not np.isfinite(base).all() or (base<=0).any():
        raise ValueError("Baseline quotes must be historical, positive and at most 365 days old")
    result["tm_market_value"]=base
    target=anchor+pd.Timedelta(days=int(horizon_days))
    start=target-pd.Timedelta(days=int(max_lag_days))
    records=history[["player_id","date_unix","value"]].copy()
    records["tm_id"]=records.player_id.astype("string").str.strip()
    records=records.loc[records.tm_id.isin(result.tm_id)].copy()
    records["future_value_date"]=pd.to_datetime(records.date_unix,format="%Y-%m-%d",errors="coerce")
    records["future_value"]=pd.to_numeric(records.value,errors="coerce")
    valid=records.future_value_date.between(start,target) & np.isfinite(records.future_value) & records.future_value.ge(0)
    window=records.loc[valid,["tm_id","future_value_date","future_value"]].drop_duplicates()
    counts=window.groupby(["tm_id","future_value_date"]).future_value.nunique()
    conflicted=set(counts.loc[counts.gt(1)].index.get_level_values("tm_id"))
    clean=window.loc[~window.tm_id.isin(conflicted)]
    latest=clean.sort_values("future_value_date").drop_duplicates("tm_id",keep="last")
    result=result.merge(latest,on="tm_id",how="left",validate="one_to_one")
    result["outcome_status"]="observed"
    result.loc[result.future_value.isna(),"outcome_status"]="no_valid_quote_in_window"
    result.loc[result.future_value.eq(0),"outcome_status"]="latest_quote_zero"
    result.loc[result.tm_id.isin(conflicted),"outcome_status"]="conflicting_quotes_in_window"
    result["outcome_observed"]=result.outcome_status.eq("observed")
    result["outcome_target_date"]=target
    result["future_quote_lag_days"]=(target-result.future_value_date).dt.days
    result["baseline_quote_age_days"]=(anchor-pd.to_datetime(result.market_value_date)).dt.days
    result["quote_interval_days"]=(result.future_value_date-pd.to_datetime(result.market_value_date)).dt.days
    result["value_change"]=np.where(result.outcome_observed,result.future_value-result.tm_market_value,np.nan)
    result["value_change_fraction"]=result.value_change/result.tm_market_value
    result["log_value_ratio"]=np.log1p(result.value_change_fraction)
    assert len(result)==len(players) and result.fifa_id.is_unique
    return result


def paired_list_outcomes(recommendations, outcomes, *, top_k=5, max_baseline_age_days=365):
    """Compare means only when both fixed lists have every outcome observed.

    No reranking or replacement using outcome availability. Lists must contain
    baseline and knn methods, unique candidate slots and exactly TOP-k each.
    Missing follow-up remains missing, never a zero return.
    """
    required={"reference_id","method","fifa_id"}
    if not required.issubset(recommendations) or not {"fifa_id","outcome_observed","baseline_quote_age_days","log_value_ratio","value_change_fraction"}.issubset(outcomes):
        raise ValueError("Missing list or outcome columns")
    if recommendations.duplicated(["reference_id","method","fifa_id"]).any() or not outcomes.fifa_id.is_unique:
        raise ValueError("Duplicate list slots or outcome identities")
    if set(recommendations.method)!={"baseline","knn"}:
        raise ValueError("Exactly baseline and knn methods are required")
    sizes=recommendations.groupby(["reference_id","method"]).size().unstack()
    if sizes.isna().any().any() or not sizes.eq(top_k).all().all():
        raise ValueError("Every reference needs both complete fixed TOP-k lists")
    columns=["fifa_id","outcome_observed","baseline_quote_age_days","log_value_ratio","value_change_fraction"]
    joined=recommendations.merge(outcomes[columns],on="fifa_id",how="left",validate="many_to_one")
    if joined.outcome_observed.isna().any():
        raise ValueError("Every recommended ID needs an explicit outcome status")
    joined["usable"]=joined.outcome_observed & joined.baseline_quote_age_days.le(max_baseline_age_days)
    parts=[]
    for (reference,method),part in joined.groupby(["reference_id","method"]):
        complete=bool(part.usable.all())
        parts.append({"reference_id":reference,"method":method,"observed_slots":int(part.usable.sum()),
                      "complete":complete,"mean_log_value_ratio":float(part.log_value_ratio.mean()) if complete else np.nan,
                      "mean_value_change_fraction":float(part.value_change_fraction.mean()) if complete else np.nan})
    table=pd.DataFrame(parts)
    paired=table.pivot(index="reference_id",columns="method")
    result=pd.DataFrame(index=paired.index)
    result["complete_both"]=paired["complete"].all(axis=1)
    for method in ["baseline","knn"]:
        result[f"{method}_observed_slots"]=paired["observed_slots"][method]
        for metric in ["mean_log_value_ratio","mean_value_change_fraction"]:
            result[f"{method}_{metric}"]=paired[metric][method].where(result.complete_both)
    result["knn_minus_baseline_log_ratio"]=(result.knn_mean_log_value_ratio-result.baseline_mean_log_value_ratio).where(result.complete_both)
    return result.reset_index()
