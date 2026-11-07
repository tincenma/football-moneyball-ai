"""Behavioral checks for the rating-only historical control system."""
import numpy as np
import pandas as pd
import pytest

from src.baseline import build_eligible_pool, select_baseline


def player(identifier, *, positions="CM", age=25, value=1_000_000, rating=80, date="2022-06-01"):
    return dict(fifa_id=identifier, short_name=identifier, player_positions_normalized=positions,
                age_at_cutoff=age, tm_market_value=value, overall=rating,
                as_of_date="2022-09-26", fifa_update_date="2022-09-26", market_value_date=date)


def test_inclusive_limits_and_exact_position_tokens():
    rows = [player("boundary", age=30, value=15_000_000), player("rich", value=15_000_001, rating=99),
            player("old", age=30.01, rating=99), player("cam", positions="CAM|CDM", rating=99),
            player("mixed", positions=" cdm, cm ", rating=79)]
    result = select_baseline(pd.DataFrame(rows))
    assert result.fifa_id.tolist() == ["boundary", "mixed"]


def test_ties_are_deterministic_and_shuffle_independent():
    frame = pd.DataFrame([player("b", value=2_000_000), player("c"), player("a")])
    assert select_baseline(frame).fifa_id.tolist() == ["a", "c", "b"]
    assert select_baseline(frame.sample(frac=1, random_state=3)).fifa_id.tolist() == ["a", "c", "b"]


def test_invalid_and_future_records_are_not_free_or_eligible():
    frame = pd.DataFrame([player("future", date="2023-01-01"), player("stale", date="2021-09-25"),
                          player("unknown", value=np.nan), player("zero", value=0), player("valid")])
    eligible, audit = build_eligible_pool(frame)
    assert eligible.fifa_id.tolist() == ["valid"]
    assert len(audit) == len(frame)


def test_exclude_reference_and_small_or_empty_pool():
    frame = pd.DataFrame([player("a"), player("b", rating=79)])
    assert select_baseline(frame, excluded_fifa_ids=["a"]).fifa_id.tolist() == ["b"]
    assert select_baseline(frame, max_value=0).empty
    assert select_baseline(frame.iloc[:0]).empty


def test_no_dependence_on_playing_features_or_price_prediction():
    frame = pd.DataFrame([player("a"), player("b", rating=79)])
    modified = frame.assign(similarity=[-100, 100], potential=[1, 99], passing=[1, 99], predicted_value=[1e12, 1])
    assert select_baseline(frame).fifa_id.tolist() == select_baseline(modified).fifa_id.tolist()


@pytest.mark.parametrize("kwargs", [{"max_value": -1}, {"max_value": np.inf}, {"max_age": np.nan},
                                    {"top_k": 0}, {"top_k": True}, {"top_k": 1.5}, {"max_value_age_days": -1},
                                    {"excluded_fifa_ids": "a"}, {"cutoff": "invalid"}])
def test_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        select_baseline(pd.DataFrame([player("a")]), **kwargs)


def test_ambiguous_ids_and_missing_schema_fail():
    with pytest.raises(ValueError, match="unique"):
        select_baseline(pd.DataFrame([player("a"), player("a")]))
    with pytest.raises(ValueError, match="Missing baseline columns"):
        select_baseline(pd.DataFrame({"fifa_id": ["a"]}))
