"""Relational integrity, date anomalies, season coverage, and missing sentinels."""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from audit_moneyball_join import CORE, DATA, OUT, read


def date_stats(series, anchor="2025-09-13"):
    raw = series[series.ne("")]
    iso = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    # String bounds preserve invalid-but-ISO years such as 3026 for the anomaly count.
    good = raw[iso]
    parsed = pd.to_datetime(good, errors="coerce")
    return {"blank": int(series.eq("").sum()), "non_iso": int((~iso).sum()),
            "unparseable_iso": int(parsed.isna().sum()),
            "min_valid": str(parsed.min().date()) if parsed.notna().any() else None,
            "max_valid": str(parsed.max().date()) if parsed.notna().any() else None,
            "before_1950": int(good.lt("1950-01-01").sum()),
            "after_extract_anchor": int(good.gt(anchor).sum()),
            "beyond_2035": int(good.gt("2035-12-31").sum())}


def main():
    result = {}
    tm = read("player_profiles.csv")
    ids = set(tm.player_id)
    births = tm.set_index("player_id").date_of_birth
    integrity = []
    for name in ["player_injuries.csv", "player_market_value.csv", "player_latest_market_value.csv",
                 "player_national_performances.csv", "player_performances.csv",
                 "player_teammates_played_with.csv", "transfer_history.csv"]:
        d = read(name)
        present = set(d.player_id)
        integrity.append({"file": name, "unique_player_ids": len(present), "ids_in_profiles": len(present & ids),
                          "ids_without_profile": len(present - ids), "rows_without_profile": int((~d.player_id.isin(ids)).sum())})
        for col in ["transfer_date", "date_unix", "from_date", "end_date"]:
            if col in d:
                result[f"{name}:{col}"] = date_stats(d[col])
                dob = d.player_id.map(births)
                result[f"{name}:{col}"]["date_before_player_birth"] = int(
                    (d[col].ne("") & dob.notna() & dob.ne("") & d[col].lt(dob.fillna(""))).sum())
        if name in ["player_injuries.csv", "transfer_history.csv"]:
            col = "from_date" if name == "player_injuries.csv" else "transfer_date"
            def season_start(label):
                first = label.split("/")[0]
                if not first.isdigit():
                    return float("nan")
                year = int(first)
                return year if len(first) == 4 else (2000 + year if year <= 40 else 1900 + year)
            starts = d.season_name.map({x: season_start(x) for x in d.season_name.unique()})
            years = pd.to_numeric(d[col].str[:4], errors="coerce")
            mismatch = starts.notna() & years.notna() & ~(years.eq(starts) | years.eq(starts + 1))
            result[f"{name}:date_year_outside_season_years"] = int(mismatch.sum())
            d.loc[mismatch].to_csv(DATA / f"{name[:-4]}_season_date_conflicts.csv", index=False)
        if name == "player_injuries.csv":
            start, end = pd.to_datetime(d.from_date, errors="coerce"), pd.to_datetime(d.end_date, errors="coerce")
            result["injury_end_before_start"] = int((end < start).sum())
        if name == "transfer_history.csv":
            result["transfer_types"] = d.transfer_type.value_counts().to_dict()
            result["transfer_value_at_transfer_zero"] = int(pd.to_numeric(d.value_at_transfer).eq(0).sum())
            result["transfer_fee_zero"] = int(pd.to_numeric(d.transfer_fee).eq(0).sum())
            bad = d.loc[d.transfer_date.gt("2035-12-31")]
            bad.to_csv(DATA / "invalid_future_transfers.csv", index=False)
    for col in ["date_of_birth", "joined", "contract_expires", "date_of_last_contract_extension", "date_of_death"]:
        result[f"player_profiles.csv:{col}"] = date_stats(tm[col])
    result["tm_placeholder_counts"] = {
        c: tm.loc[tm[c].isin(["-", "?", "NA", "N/A", "NULL", "NaN", "nan", "None", "null", " ", "  "]), c].value_counts().to_dict()
        for c in ["place_of_birth", "citizenship", "foot"]
    }
    result["tm_height_zero"] = int(pd.to_numeric(tm.height, errors="coerce").eq(0).sum())
    result["tm_height_blank"] = int(tm.height.eq("").sum())
    # A simple static-ID merge demonstrably cannot be used across sources.
    latest = read("player_latest_market_value.csv")
    history = read("player_market_value.csv").sort_values(["player_id", "date_unix"]).drop_duplicates("player_id", keep="last")
    check = latest.merge(history, on="player_id", validate="one_to_one", suffixes=("_latest", "_hist"))
    result["latest_history_consistency"] = {
        "rows": len(check), "date_disagreements": int(check.date_unix_latest.ne(check.date_unix_hist).sum()),
        "value_disagreements": int(pd.to_numeric(check.value_latest).ne(pd.to_numeric(check.value_hist)).sum()),
        "latest_after_2022_cutoff": int(latest.date_unix.gt("2022-09-26").sum()),
    }
    legacy_rows = []
    for sex in ["male", "female"]:
        with zipfile.ZipFile(Path.home() / "Downloads" / f"stefanoleone992_fifa-23-complete-player-dataset_{sex}.zip") as z:
            d = pd.read_csv(z.open(f"{sex}_players (legacy).csv"), dtype=str, na_filter=False)
        for (version, cutoff), g in d.groupby(["fifa_version", "fifa_update_date"]):
            cm = g.player_positions.str.contains(r"(?:^|,\s*)CM(?:,|$)")
            x = g.loc[cm, CORE]
            legacy_rows.append({"sex": sex, "version": version, "cutoff": cutoff, "players": len(g),
                                "cm": int(cm.sum()), "primary_cm": int(g.player_positions.str.split(",").str[0].eq("CM").sum()),
                                "cm_complete_core": int(x.ne("").all(axis=1).sum()),
                                "missing_core_cells": int(x.eq("").sum().sum())})
    pd.DataFrame(legacy_rows).to_csv(OUT / "legacy_season_coverage.csv", index=False)
    pd.DataFrame(integrity).to_csv(OUT / "referential_integrity.csv", index=False)
    (OUT / "quality_findings.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
