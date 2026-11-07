import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.utils.validation import NotFittedError

from src.valuation import MarketValuePredictor, VALUE_FEATURES, PLAYING_FEATURES, value_gap_table


@pytest.fixture
def profiles():
    rng=np.random.default_rng(19)
    d=pd.DataFrame(rng.integers(35,90,size=(40,len(PLAYING_FEATURES))).astype(float),columns=PLAYING_FEATURES)
    d["fifa_id"]=[f"v{i:03}" for i in range(len(d))]
    d["short_name"]=d.fifa_id
    d["as_of_date"]=d["fifa_update_date"]="2022-09-26"
    d["market_value_date"]="2022-06-01"
    d["player_positions_normalized"]="CM|CDM"
    d["age_at_cutoff"]=rng.uniform(19,33,len(d))
    d["height_cm"],d["weight_kg"],d["weak_foot"]=180.,75.,3.
    d["league_level"]=1.
    d.loc[0,"league_level"]=np.nan
    d["preferred_foot"],d["work_rate_attack"],d["work_rate_defence"],d["league_id"]="Right","High","Medium","10"
    d["tm_market_value"]=100_000+20_000*d.attacking_short_passing+10_000*d.mentality_vision-2_000*d.age_at_cutoff
    return d


def fitted(frame):
    return MarketValuePredictor(n_estimators=12,min_samples_leaf=1).fit(frame)


def test_prediction_units_and_gap_sign(profiles):
    model=fitted(profiles.iloc[:30])
    query=profiles.iloc[30:]
    p=model.predict(query)
    assert np.isfinite(p).all() and (p>0).all()
    assert p.min()>=profiles.iloc[:30].tm_market_value.min()-1e-6
    assert p.max()<=profiles.iloc[:30].tm_market_value.max()+1e-6
    gaps=model.gap_table(query)
    assert np.allclose(gaps.value_gap,p-query.tm_market_value)
    assert np.allclose(gaps.value_gap_fraction,gaps.value_gap/query.tm_market_value)
    assert np.allclose(gaps.predicted_to_recorded_ratio,p/query.tm_market_value)


def test_predict_ignores_actual_price_and_all_excluded_fields(profiles):
    model=fitted(profiles.iloc[:30]);query=profiles.iloc[30:].copy()
    expected=model.predict(query)
    changed=query.assign(tm_market_value=query.tm_market_value*10,overall=99,potential=99,wage_eur=1e9,value_eur=1e12)
    assert np.array_equal(expected,model.predict(changed))
    assert np.array_equal(expected,model.predict(query.drop(columns=["tm_market_value","market_value_date"])))
    assert list(model.feature_names_in_)==list(VALUE_FEATURES)
    assert list(model.estimator_.regressor_.named_steps["preprocess"].feature_names_in_)==list(VALUE_FEATURES)


def test_no_in_sample_scouting_signal(profiles):
    model=fitted(profiles.iloc[:30])
    with pytest.raises(ValueError,match="In-sample"):
        model.gap_table(profiles.iloc[:1])
    with pytest.raises(ValueError,match="actual fitted IDs"):
        value_gap_table(profiles.iloc[30:],[1.]*10,origins="unknown")


def test_unknown_category_missing_league_and_train_only_imputation(profiles):
    model=fitted(profiles.iloc[:30]);query=profiles.iloc[30:].copy()
    imputer=model.estimator_.regressor_.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    before=imputer.statistics_.copy()
    query["league_id"]="new_unseen_league"
    query["league_level"]=np.nan
    assert np.isfinite(model.predict(query)).all()
    assert np.array_equal(before,imputer.statistics_)
    assert before[list(VALUE_FEATURES).index("league_level")]==1.


def test_out_of_fold_ids_and_persistence(profiles,tmp_path):
    parts=[]
    for fit_ids,held_ids in KFold(3,shuffle=True,random_state=42).split(profiles):
        model=fitted(profiles.iloc[fit_ids])
        part=model.gap_table(profiles.iloc[held_ids]);parts.append(part)
        assert set(part.fifa_id).isdisjoint(model.train_ids_)
    combined=pd.concat(parts)
    assert len(combined)==len(profiles) and combined.fifa_id.is_unique
    path=tmp_path/"valuation.joblib";joblib.dump(model,path);restored=joblib.load(path)
    assert np.array_equal(restored.predict(profiles),model.predict(profiles))


@pytest.mark.parametrize("case",["zero","negative","nan","inf","bad_length","two_dimensional","misaligned_index"])
def test_invalid_targets(profiles,case):
    y=profiles.tm_market_value.copy()
    if case=="zero":y.iloc[0]=0
    elif case=="negative":y.iloc[0]=-1
    elif case=="nan":y.iloc[0]=np.nan
    elif case=="inf":y.iloc[0]=np.inf
    elif case=="bad_length":y=y.to_numpy()[:-1]
    elif case=="two_dimensional":y=y.to_numpy()[None,:]
    elif case=="misaligned_index":y=y.iloc[::-1]
    with pytest.raises(ValueError):MarketValuePredictor(n_estimators=5).fit(profiles,y)


@pytest.mark.parametrize("case",["missing_feature","invalid_core","future_feature","future_target","stale_target","duplicate_id","not_cm"])
def test_invalid_profiles(profiles,case):
    d=profiles.copy()
    if case=="missing_feature":d=d.drop(columns=PLAYING_FEATURES[0])
    elif case=="invalid_core":d.loc[0,PLAYING_FEATURES[0]]=np.inf
    elif case=="future_feature":d.loc[0,"fifa_update_date"]="2023-01-01"
    elif case=="future_target":d.loc[0,"market_value_date"]="2023-01-01"
    elif case=="stale_target":d.loc[0,"market_value_date"]="2020-01-01"
    elif case=="duplicate_id":d.loc[1,"fifa_id"]=d.loc[0,"fifa_id"]
    elif case=="not_cm":d.loc[0,"player_positions_normalized"]="CAM|CDM"
    with pytest.raises(ValueError):fitted(d)


def test_empty_predict_and_unfitted(profiles):
    with pytest.raises(NotFittedError):MarketValuePredictor().predict(profiles)
    model=fitted(profiles)
    assert model.predict(profiles.iloc[:0]).shape==(0,)
    with pytest.raises(ValueError,match="empty"):fitted(profiles.iloc[:0])
