"""Pure display-helper smoke tests for Streamlit decision dashboard."""
import numpy as np
import pandas as pd
import pytest

from app.dashboard_helpers import (
    format_recorded_value,
    presentation_table,
    rank_baseline_from_pool,
    reference_catalog,
)


def test_value_format_does_not_invent_eur():
    assert format_recorded_value(10_000_000) == "10,000,000 source units"
    assert format_recorded_value(10_000_000, currency_assumption=True) == "10,000,000 EUR (assumed)"
    assert format_recorded_value(np.nan) == "Unavailable"
    assert format_recorded_value(True) == "Unavailable"


def test_reference_catalog_disambiguates_same_name_and_sorts():
    values = pd.DataFrame({"fifa_id": ["9", "2"], "short_name": ["Alex", "Alex"]})
    assert list(reference_catalog(values).values()) == ["Alex [FIFA 2]", "Alex [FIFA 9]"]
    with pytest.raises(ValueError, match="unique"):
        reference_catalog(pd.concat([values, values.iloc[[0]]], ignore_index=True))


def test_baseline_reuses_pool_and_tiebreaks():
    pool = pd.DataFrame({
        "fifa_id": ["3", "2", "1"],
        "short_name": ["C", "B", "A"],
        "overall": [80, 83, 83],
        "age_at_cutoff": [22., 24., 20.],
        "tm_market_value": [3_000_000., 5_000_000., 5_000_000.],
    })
    original = pool.copy(deep=True)
    selected = rank_baseline_from_pool(pool, top_k=3)
    assert selected.fifa_id.tolist() == ["1", "2", "3"]
    assert selected.baseline_rank.tolist() == [1, 2, 3]
    pd.testing.assert_frame_equal(original, pool)
    assert rank_baseline_from_pool(pool.iloc[:0]).empty
    with pytest.raises(ValueError):
        rank_baseline_from_pool(pool, top_k=0)


def test_presented_percent_and_negative_gap_preserved():
    rows = pd.DataFrame({
        "rank": [1], "fifa_id": ["x"], "similarity_percent": [94.812],
        "value_gap_fraction": [-0.2], "predicted_value": [8_000_000.],
        "value_gap": [-2_000_000.], "tm_market_value": [10_000_000.],
        "future_sporting_performance": [99],
    })
    display = presentation_table(rows)
    assert display.similarity_percent.tolist() == [94.8]
    assert display.model_gap_percent.tolist() == [-20.0]
    assert "future_sporting_performance" not in display.columns
