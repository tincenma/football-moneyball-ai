"""Conservative entity matching and dated joins for the supplied local data."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "profiling"
OUT = ROOT / "reports" / "data_profiling"
CORE = json.loads((ROOT / "config" / "feature_set_v1.json").read_text(encoding="utf-8"))["similarity_features"]


def read(name):
    return pd.read_csv(DATA / name, dtype=str, na_filter=False)


def normalize(value):
    value = re.sub(r"\s*\(\d+\)\s*$", "", value)
    value = value.casefold().translate(str.maketrans({"ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "æ": "ae", "œ": "oe"}))
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def fifa_aliases(row):
    slug = row.player_url.split("/")[3] if len(row.player_url.split("/")) > 3 else ""
    return {normalize(v) for v in [row.long_name, row.short_name, slug] if v}


def tm_aliases(row):
    return {normalize(v) for v in [row.player_name, row.name_in_home_country, row.player_slug] if v}


def match(fifa, tm):
    index = defaultdict(set)
    birthdays = defaultdict(list)
    aliases = {}
    for r in tm.itertuples():
        aliases[r.player_id] = tm_aliases(r)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.date_of_birth):
            birthdays[r.date_of_birth].append(r.player_id)
            for alias in aliases[r.player_id]:
                index[(r.date_of_birth, alias)].add(r.player_id)
    lookup = tm.set_index("player_id")
    results, review = [], []
    candidates = {}
    for r in fifa.itertuples():
        names = fifa_aliases(r)
        found = set().union(*(index[(r.dob, a)] for a in names))
        if len(found) == 1:
            tid = next(iter(found))
            h = float(lookup.loc[tid, "height"] or 0)
            delta = abs(float(r.height_cm) - h) if h > 0 else None
            candidates[r.player_id] = (tid, delta, names & aliases[tid])
        elif len(found) > 1:
            review.append({"fifa_id": r.player_id, "fifa_name": r.long_name, "dob": r.dob,
                           "reason": "ambiguous_exact", "tm_candidates": ";".join(sorted(found))})
        else:
            scored = []
            for tid in birthdays.get(r.dob, []):
                score = max((SequenceMatcher(None, a, b).ratio() for a in names for b in aliases[tid]), default=0)
                if score >= 0.70:
                    scored.append((score, tid))
            if scored:
                scored.sort(reverse=True)
                for score, tid in scored[:2]:
                    review.append({"fifa_id": r.player_id, "fifa_name": r.long_name, "dob": r.dob,
                                   "reason": "fuzzy_review_only", "tm_candidates": tid,
                                   "tm_name": lookup.loc[tid, "player_name"], "name_score": score})
    target_counts = Counter(t[0] for t in candidates.values())
    for fid, (tid, delta, common) in candidates.items():
        reason = "exact_alias_and_dob"
        if target_counts[tid] > 1:
            reason = "nonreciprocal_collision"
        elif delta is not None and delta > 10:
            reason = "height_conflict_over_10cm"
        if reason == "exact_alias_and_dob":
            results.append({"fifa_id": fid, "tm_id": tid, "match_rule": reason,
                            "height_difference_cm": delta, "common_alias": ";".join(sorted(common))})
        else:
            review.append({"fifa_id": fid, "tm_candidates": tid, "reason": reason})
    links = pd.DataFrame(results, columns=["fifa_id", "tm_id", "match_rule", "height_difference_cm", "common_alias"])
    assert not links.fifa_id.duplicated().any() and not links.tm_id.duplicated().any()
    return links, pd.DataFrame(review)


def dated_values(joined, values, cutoff):
    """One row per FIFA identity, use latest value on/before cutoff, never nearest."""
    valid = values.loc[values.date_unix.le(cutoff)].copy()
    assert not valid.duplicated(["player_id", "date_unix"]).any()
    valid = valid.sort_values(["player_id", "date_unix"]).drop_duplicates("player_id", keep="last")
    valid = valid.rename(columns={"player_id": "tm_id", "date_unix": "market_value_date", "value": "tm_market_value"})
    j = joined.merge(valid, how="left", on="tm_id", validate="many_to_one")
    j["tm_market_value"] = pd.to_numeric(j.tm_market_value, errors="coerce")
    j["market_value_age_days"] = (pd.Timestamp(cutoff) - pd.to_datetime(j.market_value_date)).dt.days
    j["valid_market_value"] = j.tm_market_value.gt(0) & j.market_value_age_days.between(0, 365)
    assert j.loc[j.market_value_date.notna(), "market_value_date"].le(cutoff).all()
    return j


def aggregate_performance(perf, season):
    p = perf.loc[perf.season_name.eq(season)].copy()
    keys = ["player_id", "season_name", "competition_id", "team_id"]
    assert not p.duplicated(keys).any()
    numeric = ["nb_in_group", "nb_on_pitch", "goals", "assists", "minutes_played", "yellow_cards", "second_yellow_cards", "direct_red_cards", "penalty_goals"]
    for c in numeric:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    # Missing a component makes the sum unknown; do not convert unknown minutes to zero.
    summed = p.groupby("player_id")[numeric].agg(lambda x: x.sum(min_count=len(x)))
    summed = summed.add_prefix("prev_").reset_index().rename(columns={"player_id": "tm_id"})
    summed["prev_non_penalty_goals"] = summed.prev_goals - summed.prev_penalty_goals
    for label, num in [("goals_p90", "prev_non_penalty_goals"), ("assists_p90", "prev_assists"), ("yellow_cards_p90", "prev_yellow_cards")]:
        summed["prev_" + label] = summed[num] * 90 / summed.prev_minutes_played.where(summed.prev_minutes_played.ge(450))
    return summed


def main():
    tm, values = read("player_profiles.csv"), read("player_market_value.csv")
    perf = read("player_performances.csv")
    assert not tm.player_id.duplicated().any()
    summaries = []
    for filename in ["male_legacy23.csv", "female_legacy23.csv", "male_players_fifa23_latest.csv", "female_players_fifa23_latest.csv"]:
        if not (DATA / filename).exists():
            continue
        fifa = read(filename)
        assert not fifa.player_id.duplicated().any()
        cutoff = fifa.fifa_update_date.iloc[0]
        assert fifa.fifa_update_date.nunique() == 1
        links, review = match(fifa, tm)
        stem = filename.removesuffix(".csv")
        links.to_csv(DATA / f"{stem}_links.csv", index=False)
        review.to_csv(DATA / f"{stem}_review.csv", index=False)
        joined = fifa.rename(columns={"player_id": "fifa_id"}).merge(links, how="left", on="fifa_id", validate="one_to_one")
        joined = dated_values(joined, values, cutoff)
        cm = joined.player_positions.str.contains(r"(?:^|,\s*)CM(?:,|$)")
        primary = joined.player_positions.str.split(",").str[0].eq("CM")
        item = {"snapshot": stem, "cutoff": cutoff, "fifa_players": len(fifa), "cm_players": int(cm.sum()),
                "primary_cm_players": int(primary.sum()), "matched": len(links), "matched_pct": len(links)/len(fifa)*100,
                "cm_matched": int(joined.loc[cm, "tm_id"].notna().sum()), "cm_matched_pct": joined.loc[cm, "tm_id"].notna().mean()*100,
                "valid_value": int(joined.valid_market_value.sum()), "cm_valid_value": int(joined.loc[cm, "valid_market_value"].sum()),
                "cm_value_missing": int(joined.loc[cm, "tm_market_value"].isna().sum()),
                "cm_value_stale": int(joined.loc[cm, "market_value_age_days"].gt(365).sum()),
                "cm_value_zero": int(joined.loc[cm, "tm_market_value"].eq(0).sum()),
                "cm_no_link": int(joined.loc[cm, "tm_id"].isna().sum()),
                "cm_linked_no_historical_value": int((cm & joined.tm_id.notna() & joined.tm_market_value.isna()).sum()),
                "cm_positive_value_stale": int((cm & joined.tm_market_value.gt(0) & joined.market_value_age_days.gt(365)).sum()),
                "review_rows": len(review), "review_reasons": review.reason.value_counts().to_dict() if len(review) else {},
                "matched_tm_cm": int(joined.loc[cm, "tm_id"].isin(tm.loc[tm.position.eq("Midfield - Central Midfield"), "player_id"]).sum())}
        raw_id = fifa.merge(tm, on="player_id", suffixes=("_fifa", "_tm"))
        item["false_shared_numeric_id_overlap"] = len(raw_id)
        item["same_id_same_dob"] = int(raw_id.dob.eq(raw_id.date_of_birth).sum())
        for label, mask in [("all_cm", cm), ("matched_cm", cm & joined.tm_id.notna()), ("valued_cm", cm & joined.valid_market_value)]:
            x = joined.loc[mask, CORE].apply(pd.to_numeric, errors="coerce")
            item[label + "_complete_core"] = int(x.notna().all(axis=1).sum())
            item[label + "_missing_core_cells"] = int(x.isna().sum().sum())
        if filename == "male_legacy23.csv":
            item["cm_value_staleness_sensitivity"] = {
                str(days): int((cm & joined.tm_market_value.gt(0) & joined.market_value_age_days.le(days)).sum())
                for days in [90, 180, 365, 730]
            }
            item["cm_match_bias"] = {
                "overall_mean_matched": float(pd.to_numeric(joined.loc[cm & joined.tm_id.notna(), "overall"]).mean()),
                "overall_mean_unmatched": float(pd.to_numeric(joined.loc[cm & joined.tm_id.isna(), "overall"]).mean()),
            }
            historical = aggregate_performance(perf, "21/22")
            joined = joined.merge(historical, how="left", on="tm_id", validate="many_to_one")
            for c in ["prev_minutes_played", "prev_nb_on_pitch", "prev_goals_p90", "prev_assists_p90"]:
                joined[c] = pd.to_numeric(joined[c], errors="coerce")
                item[c + "_cm_available"] = int(joined.loc[cm, c].notna().sum())
                item[c + "_valued_cm_available"] = int(joined.loc[cm & joined.valid_market_value, c].notna().sum())
            selected = joined.loc[cm].copy()
            x = selected[CORE].apply(pd.to_numeric, errors="coerce")
            x.describe().T.to_csv(OUT / "core_feature_descriptives.csv")
            corr = x.corr(method="spearman")
            corr.to_csv(OUT / "core_feature_spearman.csv")
            item["core_pairs_abs_spearman_ge_095"] = [
                {"a": a, "b": b, "rho": float(corr.loc[a, b])}
                for i, a in enumerate(CORE) for b in CORE[i+1:] if abs(corr.loc[a, b]) >= .95
            ]
            selected.to_csv(DATA / "moneyball_cm_2022-09-26_audit.csv", index=False)
            selected.groupby(["league_id", "league_name"]).agg(
                cm_players=("fifa_id", "size"), matched=("tm_id", "count"), valued=("valid_market_value", "sum")
            ).to_csv(OUT / "cm_coverage_by_league.csv")
            train = selected.loc[selected.valid_market_value].copy()
            train["target_log1p_tm_value"] = np.log1p(train.tm_market_value)
            train["age_at_cutoff"] = (pd.Timestamp(cutoff) - pd.to_datetime(train.dob)).dt.days / 365.2425
            work = train.work_rate.str.split("/", expand=True)
            train["work_rate_attack"], train["work_rate_defence"] = work[0], work[1]
            covariates = ["age_at_cutoff", "height_cm", "weight_kg", "weak_foot", "preferred_foot",
                          "work_rate_attack", "work_rate_defence", "league_id", "league_level"]
            item["training_covariates_missing"] = {c: int(train[c].eq("").sum()) for c in covariates}
            allowed = ["fifa_id", "tm_id", "fifa_update_date", "market_value_date", "market_value_age_days",
                       *CORE, *covariates, "tm_market_value", "target_log1p_tm_value"]
            train[allowed].to_csv(DATA / "moneyball_cm_2022-09-26_training.csv", index=False)
            # Strong exclusion conditions are intentional, not inferred zero values.
            assert train[CORE].replace("", np.nan).notna().all().all()
            assert train.tm_id.is_unique and train.fifa_id.is_unique
            assert (item["cm_no_link"] + item["cm_linked_no_historical_value"] + item["cm_value_zero"]
                    + item["cm_positive_value_stale"] + item["cm_valid_value"]) == int(cm.sum())
        summaries.append(item)
        print(json.dumps(item, ensure_ascii=True), flush=True)
    (OUT / "join_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
