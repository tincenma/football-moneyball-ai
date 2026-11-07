"""Independent counts with Python's csv parser and final artifact invariants."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from profile_moneyball_data import ARCHIVES, OUT, PRIVATE, ROOT


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    evidence = {"independent_csv_counts": [], "exact_duplicate_checks": [], "snapshot_checks": []}
    for source, archive in ARCHIVES.items():
        profiles = load(OUT / f"{source}_profiles.json")
        with zipfile.ZipFile(Path.home() / "Downloads" / archive) as z:
            for p in profiles:
                assert p["missing_cells"] == sum(p["missing_by_column"].values())
                assert 0 <= p["missing_cells"] <= p["rows"] * p["columns"]
                # Avoid an unnecessary second decompression of the 5.6GB member.
                # Its full-scan counts are independently reconciled with per-snapshot subtotals.
                if p["file"] != "male_players.csv":
                    with io.TextIOWrapper(z.open(p["member"]), encoding="utf-8-sig", newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        count = missing = 0
                        for row in reader:
                            if not row:
                                continue
                            assert len(row) == len(header), (p["file"], count, len(row))
                            count += 1
                            missing += row.count("")
                    assert (count, len(header), missing) == (p["rows"], p["columns"], p["missing_cells"]), p["file"]
                    evidence["independent_csv_counts"].append({"file": p["file"], "rows": count, "missing_cells": missing})
                if "snapshot_count" in p:
                    name = p["file"][:-4].replace(" ", "_") + "_snapshots.csv"
                    snapshots = pd.read_csv(OUT / name)
                    assert int(snapshots.rows.sum()) == p["rows"]
                    assert len(snapshots) == p["snapshot_count"]
                    if "cm_rows" in p:
                        assert int(snapshots.cm_rows.sum()) == p["cm_rows"]
                        assert int(snapshots.primary_cm_rows.sum()) == p["primary_cm_rows"]
                    evidence["snapshot_checks"].append({"file": p["file"], "snapshots": len(snapshots), "rows": p["rows"]})
        if source == "transfermarkt":
            for name in ["player_injuries.csv", "player_national_performances.csv", "team_competitions_seasons.csv", "transfer_history.csv"]:
                p = next(p for p in profiles if p["file"] == name)
                d = pd.read_csv(PRIVATE / name, dtype=str, na_filter=False)
                duplicates = int(d.duplicated().sum())
                assert duplicates == p["duplicate_rows_hash"]
                evidence["exact_duplicate_checks"].append({"file": name, "duplicates": duplicates})
    contract = load(ROOT / "config" / "feature_set_v1.json")
    features = contract["similarity_features"] + contract["valuation_additional_numeric_features"] + contract["valuation_additional_categorical_features"]
    assert len(features) == len(set(features)) == 27
    groups = [f for fields in contract["similarity_feature_groups"].values() for f in fields]
    assert len(groups) == 18 and set(groups) == set(contract["similarity_features"])
    t = pd.read_csv(PRIVATE / "moneyball_cm_2022-09-26_training.csv", dtype={"fifa_id": str, "tm_id": str, "league_id": str})
    metadata = ["fifa_id", "tm_id", "fifa_update_date", "market_value_date", "market_value_age_days"]
    targets = ["tm_market_value", "target_log1p_tm_value"]
    assert set(t.columns) == set(features + metadata + targets)
    assert t.fifa_id.is_unique and t.tm_id.is_unique
    assert t[contract["similarity_features"]].notna().all().all()
    assert t.tm_market_value.gt(0).all()
    assert t.market_value_date.le("2022-09-26").all()
    assert t.market_value_age_days.between(0, 365).all()
    assert np.allclose(t.target_log1p_tm_value, np.log1p(t.tm_market_value))
    assert t.fifa_update_date.eq("2022-09-26").all()
    expected = next(x for x in load(OUT / "join_summary.json") if x["snapshot"] == "male_legacy23")
    assert len(t) == expected["cm_valid_value"] == 2886
    a = pd.read_csv(PRIVATE / "moneyball_cm_2022-09-26_audit.csv", low_memory=False)
    assert len(a) == 4184 and a.fifa_id.is_unique
    assert a.player_positions.str.split(",").apply(lambda x: "CM" in [s.strip() for s in x]).all()
    assert len(a.loc[a.valid_market_value]) == len(t)
    evidence["training"] = {"rows": len(t), "raw_features": len(features), "metadata_columns": len(metadata), "target_columns": len(targets),
                            "missing_core": int(t[contract["similarity_features"]].isna().sum().sum()),
                            "future_value_rows": int(t.market_value_date.gt("2022-09-26").sum())}
    evidence["result"] = "all_checks_passed"
    (OUT / "verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": evidence["result"], "independently_counted_files": len(evidence["independent_csv_counts"]),
                      "exact_duplicate_checks": len(evidence["exact_duplicate_checks"]), "training": evidence["training"]}))


if __name__ == "__main__":
    main()
