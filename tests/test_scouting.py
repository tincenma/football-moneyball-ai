import pandas as pd
import pytest

from src.scouting import prepare_players, recommend


def sample():
    return pd.DataFrame([
        dict(short_name="Reference", player_positions="CM", age=25, value_eur=40_000_000, overall=90, pace=70, shooting=70, passing=90, dribbling=85, defending=65, physic=75),
        dict(short_name="Affordable", player_positions="CM, CDM", age=22, value_eur=8_000_000, overall=78, pace=68, shooting=68, passing=85, dribbling=84, defending=68, physic=75),
        dict(short_name="Other", player_positions="CM", age=31, value_eur=9_000_000, overall=82, pace=80, shooting=55, passing=60, dribbling=60, defending=90, physic=80),
        dict(short_name="Forward", player_positions="ST", age=21, value_eur=5_000_000, overall=92, pace=90, shooting=90, passing=50, dribbling=90, defending=20, physic=80),
    ])


def test_only_central_midfielders_and_name_matching():
    players = prepare_players(sample())
    assert set(players.short_name) == {"Reference", "Affordable", "Other"}


def test_budget_age_and_reference_exclusion():
    players = prepare_players(sample())
    recommended, baseline = recommend(players, reference_index=0, max_value_eur=8_000_000, max_age=25)
    assert recommended.short_name.tolist() == ["Affordable"]
    assert baseline.short_name.tolist() == ["Affordable"]


def test_invalid_schema():
    with pytest.raises(ValueError, match="missing required"):
        prepare_players(pd.DataFrame({"short_name": ["A"]}))


def test_empty_results():
    players = prepare_players(sample())
    rec, base = recommend(players, reference_index=0, max_value_eur=1)
    assert rec.empty and base.empty
