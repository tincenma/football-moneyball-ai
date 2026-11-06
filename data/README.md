# Data

External real-world datasets are **not bundled** with this starter. Download only datasets you are licensed to use, track original source, season and retrieval date, and avoid committing large or restricted data to GitHub.

## CSV schema for the starter app

Required columns:

- `short_name` (str) — player name
- `player_positions` (str) — positions separated by commas, e.g. `CM, CDM`
- `age` (number)
- `value_eur` (number) — recorded value, **not necessarily an actual transfer fee**
- `overall` (number) — rating used only for the baseline comparison
- `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` (numeric player attributes)

You can adapt the feature list in `src/scouting.py`. The demo excludes missing values for any required feature; full study preprocessing should audit the impact of that choice.

A tiny demonstration CSV (`synthetic_demo.csv`) is provided solely to verify the interface; names and numbers are fictional.
