"""Full, local-only audit of the three supplied archives; no network or sampling.

Run with the bundled Python documented in docs/data_profiling.md.
Raw data and per-player outputs go to gitignored data/processed/profiling/.
Small aggregate evidence goes to reports/data_profiling/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "data_profiling"
PRIVATE = ROOT / "data" / "processed" / "profiling"
ARCHIVES = {
    "fifa_female": "stefanoleone992_fifa-23-complete-player-dataset_female.zip",
    "fifa_male": "stefanoleone992_fifa-23-complete-player-dataset_male.zip",
    "transfermarkt": "xfkzujqjvx97n_football-datasets_data.zip",
}
KEYS = {
    "player_profiles.csv": ["player_id"],
    "player_market_value.csv": ["player_id", "date_unix"],
    "player_latest_market_value.csv": ["player_id"],
    "player_performances.csv": ["player_id", "season_name", "competition_id", "team_id"],
    "player_injuries.csv": ["player_id", "from_date", "end_date", "injury_reason"],
    "player_national_performances.csv": ["player_id", "team_id"],
    "player_teammates_played_with.csv": ["player_id", "teammate_player_id"],
    "team_children.csv": ["parent_team_id", "child_team_id"],
    "team_competitions_seasons.csv": ["club_id", "season_id", "competition_id"],
    "team_details.csv": ["club_id"],
}
DATE_COLUMNS = {
    "fifa_update_date", "dob", "club_joined_date", "date_of_birth", "joined",
    "contract_expires", "date_of_last_contract_extension", "contract_there_expires",
    "date_of_death", "date_unix", "from_date", "end_date", "debut",
    "transfer_date", "_last_modified_at",
}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def scan(archive: Path, source: str):
    start = time.monotonic()
    profiles, schemas, columns = [], {}, []
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not entry.filename.endswith(".csv"):
                continue
            name = Path(entry.filename).name
            print(f"START {source}/{name} ({entry.file_size:,} bytes)", flush=True)
            rows = 0
            empty = Counter()
            sentinels = Counter()
            dates = {}
            categories = defaultdict(Counter)
            ids = defaultdict(set)
            row_hashes, key_hashes = [], []
            snapshots = Counter()
            cm_by_snapshot = Counter()
            primary_by_snapshot = Counter()
            cm_ids = set()
            primary_ids = set()
            cm_by_version = defaultdict(set)
            version_ids = defaultdict(set)
            latest23, latest_date = [], ""
            numerical = {}
            key = None
            for d in pd.read_csv(z.open(entry), dtype=str, na_filter=False, chunksize=25000):
                if rows == 0:
                    schemas[name] = list(d.columns)
                    key = KEYS.get(name)
                    if "fifa_version" in d:
                        key = ["player_id" if "player_id" in d else "team_id", "fifa_version", "fifa_update"]
                    elif "coach_id" in d and name.endswith("coaches.csv"):
                        key = ["coach_id"]
                    # Only classify columns as numeric if every nonblank in the first chunk parses.
                    for c in d:
                        vals = d.loc[d[c].ne(""), c]
                        if len(vals) and pd.to_numeric(vals, errors="coerce").notna().all():
                            numerical[c] = {"n": 0, "invalid": 0, "min": None, "max": None, "zero": 0, "negative": 0}
                rows += len(d)
                blanks = d.eq("")
                empty.update(blanks.sum().to_dict())
                # Report these separately: they are not silently converted to null.
                sentinels.update(d.isin(["NA", "N/A", "NULL", "null", "None", "NaN", "nan", "-", "?", " ", "  ", "\t"]).sum().to_dict())
                row_hashes.append(pd.util.hash_pandas_object(d, index=False).to_numpy())
                if key:
                    key_hashes.append(pd.util.hash_pandas_object(d[key], index=False).to_numpy())
                for c in ("player_id", "team_id", "club_id", "coach_id"):
                    if c in d:
                        ids[c].update(d.loc[d[c].ne(""), c])
                for c in DATE_COLUMNS.intersection(d.columns):
                    vals = d.loc[d[c].ne(""), c]
                    if len(vals):
                        lo, hi = vals.min(), vals.max()
                        old = dates.get(c, {"min": lo, "max": hi})
                        dates[c] = {"min": min(lo, old["min"]), "max": max(hi, old["max"])}
                for c in ("fifa_version", "season_name", "season_id", "position", "main_position", "transfer_type"):
                    if c in d:
                        categories[c].update(d[c].value_counts().to_dict())
                for c, stats in numerical.items():
                    v = pd.to_numeric(d[c], errors="coerce")
                    stats["invalid"] += int((d[c].ne("") & v.isna()).sum())
                    stats["n"] += int(v.notna().sum())
                    stats["zero"] += int(v.eq(0).sum())
                    stats["negative"] += int(v.lt(0).sum())
                    if v.notna().any():
                        lo, hi = float(v.min()), float(v.max())
                        stats["min"] = lo if stats["min"] is None else min(lo, stats["min"])
                        stats["max"] = hi if stats["max"] is None else max(hi, stats["max"])
                if "fifa_version" in d:
                    group = ["fifa_version", "fifa_update", "fifa_update_date"]
                    snapshots.update({tuple(k): int(v) for k, v in d.groupby(group).size().items()})
                    if "player_positions" in d:
                        cm = d.player_positions.str.contains(r"(?:^|,\s*)CM(?:,|$)", regex=True)
                        primary = d.player_positions.str.split(",").str[0].eq("CM")
                        cm_ids.update(d.loc[cm, "player_id"])
                        primary_ids.update(d.loc[primary, "player_id"])
                        cm_by_snapshot.update({tuple(k): int(v) for k, v in d.loc[cm].groupby(group).size().items()})
                        primary_by_snapshot.update({tuple(k): int(v) for k, v in d.loc[primary].groupby(group).size().items()})
                        for version, g in d.groupby("fifa_version"):
                            version_ids[version].update(g.player_id)
                            cm_by_version[version].update(g.loc[cm.loc[g.index], "player_id"])
                        f23 = d.loc[d.fifa_version.eq("23")]
                        if len(f23):
                            date = f23.fifa_update_date.max()
                            if date > latest_date:
                                latest_date, latest23 = date, []
                            if date == latest_date:
                                latest23.append(f23.loc[f23.fifa_update_date.eq(date)].copy())
                if rows % 250000 == 0:
                    print(f"PROGRESS {name}: {rows:,} rows; {time.monotonic()-start:.0f}s", flush=True)
            full_hash = np.concatenate(row_hashes)
            key_duplicate = rows - len(np.unique(np.concatenate(key_hashes))) if key_hashes else None
            p = {
                "source": source, "file": name, "member": entry.filename,
                "uncompressed_bytes": entry.file_size, "zip_crc32": f"{entry.CRC:08x}",
                "rows": rows, "columns": len(schemas[name]), "missing_cells": sum(empty.values()),
                "missing_pct": sum(empty.values()) / (rows * len(schemas[name])) * 100,
                "missing_by_column": dict(empty), "literal_sentinel_counts": dict(sentinels),
                "duplicate_rows_hash": rows - len(np.unique(full_hash)),
                "key": key, "duplicate_key_rows_hash": key_duplicate,
                "unique_ids": {c: len(v) for c, v in ids.items()},
                "date_ranges_lexical": dates, "categories": dict(categories), "numeric": numerical,
            }
            if "player_positions" in schemas[name]:
                p.update(cm_rows=sum(cm_by_snapshot.values()), cm_unique_players=len(cm_ids),
                         primary_cm_rows=sum(primary_by_snapshot.values()), primary_cm_unique_players=len(primary_ids),
                         cm_unique_by_version={v: len(s) for v, s in cm_by_version.items()},
                         unique_players_by_version={v: len(s) for v, s in version_ids.items()})
            if snapshots:
                table = [{"fifa_version": k[0], "fifa_update": k[1], "fifa_update_date": k[2], "rows": n,
                          "cm_rows": cm_by_snapshot[k], "primary_cm_rows": primary_by_snapshot[k]}
                         for k, n in sorted(snapshots.items())]
                pd.DataFrame(table).to_csv(OUT / f"{name[:-4].replace(' ', '_')}_snapshots.csv", index=False)
                p["snapshot_count"] = len(snapshots)
            if latest23:
                latest = pd.concat(latest23, ignore_index=True)
                latest.to_csv(PRIVATE / f"{name[:-4].replace(' ', '_')}_fifa23_latest.csv", index=False)
                p["fifa23_latest_date"] = latest_date
                p["fifa23_latest_rows"] = len(latest)
            profiles.append(p)
            for c in schemas[name]:
                columns.append({"source": source, "file": name, "column": c, "missing": empty[c],
                                "missing_pct": 100 * empty[c] / rows, "literal_sentinels": sentinels[c],
                                "numeric_first_chunk": c in numerical, **numerical.get(c, {})})
            dump(OUT / f"{source}_profiles.json", profiles)
            dump(OUT / f"{source}_schemas.json", schemas)
            pd.DataFrame(columns).to_csv(OUT / f"{source}_columns.csv", index=False)
            print(f"DONE {name}: rows={rows:,}, missing={sum(empty.values()):,}, duplicates={p['duplicate_rows_hash']}", flush=True)
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--source", choices=list(ARCHIVES), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = args.downloads / ARCHIVES[args.source]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    dump(OUT / f"{args.source}_archive.json", {"name": path.name, "bytes": path.stat().st_size,
         "sha256": h.hexdigest(), "profile_date": "2026-11-07", "missing_definition": "CSV empty fields; literal sentinels reported separately"})
    scan(path, args.source)


if __name__ == "__main__":
    main()
