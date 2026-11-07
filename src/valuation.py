"""Market-value regression; model gaps are screening signals, not fair prices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

_CONTRACT = json.loads((Path(__file__).resolve().parents[1]/"config/feature_set_v1.json").read_text(encoding="utf-8"))
PLAYING_FEATURES = tuple(_CONTRACT["similarity_features"])
NUMERIC_FEATURES = PLAYING_FEATURES + tuple(_CONTRACT["valuation_additional_numeric_features"])
CATEGORICAL_FEATURES = tuple(_CONTRACT["valuation_additional_categorical_features"])
VALUE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def nonnegative_expm1(values):
    """Inverse log target; regression point estimates are not arithmetic means."""
    return np.maximum(np.expm1(values), 0)


def value_preprocessor(*, scale_numeric=False):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), list(NUMERIC_FEATURES)),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(CATEGORICAL_FEATURES)),
    ], remainder="drop")


def make_forest_estimator(*, n_estimators=160, min_samples_leaf=2, max_features=.7,
                          max_depth=None, random_state=42):
    """CV-safe pipeline: fold-local imputation/encoding + RF on log1p(y).

    Returns predictions in the original source units. Scaling is unnecessary
    for RF splits. Additional input columns are dropped by the allowlist.
    """
    pipeline = Pipeline([
        ("preprocess", value_preprocessor()),
        ("forest", RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                         max_features=max_features, max_depth=max_depth, random_state=random_state, n_jobs=1)),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=nonnegative_expm1,
                                       check_inverse=True)


def make_ridge_estimator(alpha=10.):
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", value_preprocessor(scale_numeric=True)), ("ridge", Ridge(alpha=alpha))]),
        func=np.log1p, inverse_func=nonnegative_expm1, check_inverse=True)


def value_gap_table(players, predictions, *, origins, cutoff="2022-09-26", fitted_ids=None):
    """Attach PredictedValue - MarketValue after proving a disjoint fit set.

    fitted_ids is mandatory for this single-model helper. Nested cross-fitting
    calls it per outer validation fold, with that fold's actual training IDs.
    """
    required = {"fifa_id", "tm_market_value", "market_value_date", "as_of_date", "fifa_update_date"}
    if not required.issubset(players):
        raise ValueError("Gap calculation needs IDs, historical dates and recorded market values")
    result = players.copy()
    ids = result.fifa_id.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Player IDs must be present and unique")
    if fitted_ids is None:
        raise ValueError("Provide the actual fitted IDs to guard against in-sample gaps")
    if set(ids) & set(map(str, fitted_ids)):
        raise ValueError("In-sample price gaps must not be ranked as scouting signals")
    anchor = pd.Timestamp(cutoff)
    for field in ["as_of_date", "fifa_update_date"]:
        if not pd.to_datetime(result[field], format="%Y-%m-%d", errors="coerce").eq(anchor).all():
            raise ValueError("Player characteristics must match the fixed cutoff")
    ages = (anchor-pd.to_datetime(result.market_value_date, format="%Y-%m-%d", errors="coerce")).dt.days
    if not ages.between(0, 365).all():
        raise ValueError("Recorded values must be historical and at most 365 days old")
    actual = pd.to_numeric(result.tm_market_value, errors="coerce").to_numpy(float)
    predicted = np.asarray(predictions, dtype=float)
    if predicted.ndim != 1 or len(predicted) != len(result):
        raise ValueError("One prediction per input row is required")
    if not np.isfinite(actual).all() or (actual <= 0).any() or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Values must be finite; actual positive and predicted nonnegative")
    result["fifa_id"] = ids
    result["predicted_value"] = predicted
    result["value_gap"] = predicted - actual
    result["value_gap_fraction"] = result.value_gap / actual
    result["predicted_to_recorded_ratio"] = predicted / actual
    result["positive_model_gap"] = result.value_gap.gt(0)
    result["prediction_origin"] = origins
    return result


class MarketValuePredictor(RegressorMixin, BaseEstimator):
    """Standalone third model using the 27-feature v1 contract.

    fit learns from positive historical recorded prices. predict does not read
    price, rating, potential or ID as features. gap_table rejects fitted players.
    """

    def __init__(self, n_estimators=160, min_samples_leaf=2, max_features=.7,
                 max_depth=None, random_state=42, cutoff="2022-09-26"):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.cutoff = cutoff

    def _profiles(self, frame):
        required = {"fifa_id", "as_of_date", "fifa_update_date", "player_positions_normalized", *VALUE_FEATURES}
        if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
            raise ValueError("Missing valuation profile columns")
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate columns are not allowed")
        result = frame.copy()
        result["fifa_id"] = result.fifa_id.astype("string").str.strip()
        if result.fifa_id.isna().any() or result.fifa_id.eq("").any() or result.fifa_id.duplicated().any():
            raise ValueError("fifa_id must be present and unique")
        anchor = pd.Timestamp(self.cutoff)
        if pd.isna(anchor) or anchor.tzinfo is not None:
            raise ValueError("cutoff must be a valid timezone-free date")
        for field in ["as_of_date", "fifa_update_date"]:
            if not pd.to_datetime(result[field], format="%Y-%m-%d", errors="coerce").eq(anchor).all():
                raise ValueError("Profiles must match the historical cutoff")
        cm = result.player_positions_normalized.astype("string").fillna("").str.upper().str.replace("|", ",", regex=False).str.split(",").map(
            lambda parts: "CM" in [p.strip() for p in parts])
        if not cm.all():
            raise ValueError("Only exact CM positions are supported")
        for column in NUMERIC_FEATURES:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        mandatory = [c for c in NUMERIC_FEATURES if c != "league_level"]
        if not np.isfinite(result[mandatory].to_numpy()).all():
            raise ValueError("Mandatory numeric features must be finite and complete")
        ranges = {**{c:(1,99) for c in PLAYING_FEATURES}, "age_at_cutoff":(15,50), "height_cm":(120,230),
                  "weight_kg":(40,140), "weak_foot":(1,5), "league_level":(1,10)}
        for column, (low, high) in ranges.items():
            if not (result[column].isna() | result[column].between(low, high)).all():
                raise ValueError(f"Invalid range for {column}")
        for column in CATEGORICAL_FEATURES:
            result[column] = result[column].astype("string").str.strip().replace("", pd.NA).fillna("UNKNOWN").astype(str)
        return result

    def fit(self, X, y=None):
        valid = self._profiles(X)
        if valid.empty:
            raise ValueError("Training data cannot be empty")
        if "market_value_date" not in valid:
            raise ValueError("Training requires the date of the recorded target")
        ages = (pd.Timestamp(self.cutoff)-pd.to_datetime(valid.market_value_date, format="%Y-%m-%d", errors="coerce")).dt.days
        if not ages.between(0,365).all():
            raise ValueError("Targets must be historical and no older than 365 days")
        if y is None:
            if "tm_market_value" not in valid:
                raise ValueError("Training requires recorded targets")
            y = valid.tm_market_value
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("Target index must align with profile rows")
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(valid) or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError("One finite positive target per training row is required")
        estimator = make_forest_estimator(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                          max_features=self.max_features, max_depth=self.max_depth, random_state=self.random_state)
        estimator.fit(valid[list(VALUE_FEATURES)], target)
        self.estimator_ = estimator
        self.train_ids_ = valid.fifa_id.tolist()
        self.feature_names_in_ = np.array(VALUE_FEATURES, dtype=object)
        self.n_features_in_ = len(VALUE_FEATURES)
        return self

    def predict(self, X):
        check_is_fitted(self, ["estimator_", "train_ids_"])
        valid = self._profiles(X)
        if valid.empty:
            return np.array([], dtype=float)
        predictions = np.asarray(self.estimator_.predict(valid[list(VALUE_FEATURES)]), dtype=float)
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Nonfinite or negative model predictions")
        return predictions

    def gap_table(self, players):
        return value_gap_table(players, self.predict(players), origins="held_out_from_this_model",
                               cutoff=self.cutoff, fitted_ids=self.train_ids_)
