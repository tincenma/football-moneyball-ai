"""Correctness, leakage and retrieval edge cases for the real model engine."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import NotFittedError
from threadpoolctl import threadpool_limits

from src.models import PlayerModels, CORE_FEATURES, eligible_ids_hash


@pytest.fixture(autouse=True)
def serial_execution(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def profiles():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.integers(35, 85, size=(24, len(CORE_FEATURES))).astype(float), columns=CORE_FEATURES)
    df["fifa_id"] = [f"p{i:02}" for i in range(len(df))]
    df["short_name"] = df.fifa_id
    df["as_of_date"] = df["fifa_update_date"] = "2022-09-26"
    df["market_value_date"] = "2022-06-01"
    df["age_at_cutoff"], df["overall"], df["tm_market_value"] = 25., 80, 1_000_000.
    df["player_positions_normalized"] = "CM|CDM"
    return df


def test_exact_knn_matches_independent_full_gallery_calculation(profiles):
    model = PlayerModels().fit(profiles.iloc[:18], gallery=profiles.iloc[:18])
    query = profiles.iloc[18]
    result = model.recommend(query, top_k=5)
    z = (query[list(CORE_FEATURES)].to_numpy(float)-model.scaler_.mean_)/model.scaler_.scale_
    g = (model.gallery_[list(CORE_FEATURES)].to_numpy(float)-model.scaler_.mean_)/model.scaler_.scale_
    oracle = cosine_distances((z*model.weights_)[None,:], g*model.weights_)[0]
    expected = model.gallery_[["fifa_id", "tm_market_value"]].assign(distance=oracle).sort_values(["distance", "tm_market_value", "fifa_id"]).head(5)
    assert result.fifa_id.tolist() == expected.fifa_id.tolist()
    assert np.allclose(result.cosine_distance, expected.distance)
    assert np.allclose(result.similarity, 1-result.cosine_distance)


def test_budget_filter_after_complete_search_does_not_lose_affordable_neighbor(profiles):
    train = profiles.copy()
    query = train.iloc[0]
    gallery = train.iloc[:12].copy()
    gallery["fifa_id"] = ["p00"] + [f"g{i:02}" for i in range(1, 12)]
    for i in range(1, 8):
        gallery.loc[gallery.index[i], list(CORE_FEATURES)] = query[list(CORE_FEATURES)].to_numpy(float) + i/10
    gallery["tm_market_value"] = 50_000_000.
    gallery.loc[gallery.index[7], "tm_market_value"] = 5_000_000.
    model = PlayerModels().fit(train, gallery)
    result = model.recommend(query, max_value=15_000_000, top_k=1)
    assert result.fifa_id.tolist() == ["g07"]
    assert result.attrs["pool_audit"]["eligible_count"] == 1


def test_same_cluster_pool_hash_and_baseline_universe(profiles):
    model = PlayerModels().fit(profiles)
    query = profiles.iloc[2]
    pool = model.eligible_pool(query, within_cluster=True)
    result = model.recommend(query, within_cluster=True, top_k=50)
    assert set(result.fifa_id) == set(pool.fifa_id)
    assert result.same_cluster.all()
    assert result.attrs["pool_audit"]["eligible_ids_sha256"] == eligible_ids_hash(pool.fifa_id)
    assert query.fifa_id not in result.fifa_id.tolist()


def test_holdout_and_economics_never_change_fitted_geometry(profiles):
    model = PlayerModels().fit(profiles.iloc[:18])
    before = model.scaler_.mean_.copy()
    changed = profiles.copy()
    changed["tm_market_value"], changed["overall"], changed["age_at_cutoff"], changed["potential"] = 9_000_000., 99, 49, 1
    other = PlayerModels().fit(changed.iloc[:18])
    assert np.allclose(model.transform(profiles), other.transform(changed))
    assert np.array_equal(model.predict_clusters(profiles), other.predict_clusters(changed))
    model.recommend(profiles.iloc[18])
    assert np.array_equal(before, model.scaler_.mean_)
    assert int(model.scaler_.n_samples_seen_) == 18
    assert list(model.feature_names_in_) == list(CORE_FEATURES)


def test_self_excluded_even_when_unlimited_budget(profiles):
    model = PlayerModels().fit(profiles)
    result = model.recommend("p00", max_value=1e9, top_k=50)
    assert "p00" not in result.fifa_id.tolist()
    assert len(result) == len(profiles)-1


def test_empty_small_pool_and_future_price(profiles):
    gallery = profiles.iloc[:2].copy()
    gallery.loc[gallery.index[0], "market_value_date"] = "2023-01-01"
    model = PlayerModels().fit(profiles.iloc[2:], gallery)
    result = model.recommend(profiles.iloc[3])
    assert len(result) == 1 and result.fifa_id.iloc[0] == "p01"
    assert model.recommend(profiles.iloc[3], max_value=0).empty
    empty, audit = model.recommend_many(profiles.iloc[:0])
    assert empty.empty and audit.empty


def test_nondefault_indices_and_batch_parity(profiles):
    altered = profiles.copy()
    altered.index = np.arange(200, 200+len(altered))*3
    model = PlayerModels().fit(altered.iloc[:18])
    batch, audits = model.recommend_many(altered.iloc[18:21])
    assert len(audits) == 3
    for _, query in altered.iloc[18:21].iterrows():
        one = model.recommend(query)
        assert one.fifa_id.tolist() == batch.loc[batch.reference_id.eq(query.fifa_id), "fifa_id"].tolist()


def test_persistence_and_shuffle_determinism(profiles, tmp_path):
    model = PlayerModels().fit(profiles.iloc[:18])
    other = PlayerModels().fit(profiles.iloc[:18].sample(frac=1, random_state=99))
    expected = model.recommend(profiles.iloc[18]).fifa_id.tolist()
    assert other.recommend(profiles.iloc[18]).fifa_id.tolist() == expected
    file = tmp_path/"models.joblib"
    joblib.dump(model, file)
    restored = joblib.load(file)
    assert restored.recommend(profiles.iloc[18]).fifa_id.tolist() == expected


@pytest.mark.parametrize("kwargs", [{"n_clusters": 1}, {"n_clusters": True}, {"n_clusters": 100},
                                    {"n_init": 0}, {"cluster_mode": "unknown"}, {"cutoff": "bad"}])
def test_invalid_model_parameters(profiles, kwargs):
    with pytest.raises(ValueError):
        PlayerModels(**kwargs).fit(profiles)


@pytest.mark.parametrize("mutation", ["missing", "nan", "infinite", "duplicate_id", "future", "position"])
def test_invalid_profiles(profiles, mutation):
    frame = profiles.copy()
    if mutation == "missing": frame = frame.drop(columns=CORE_FEATURES[0])
    elif mutation == "nan": frame.loc[0, CORE_FEATURES[0]] = np.nan
    elif mutation == "infinite": frame.loc[0, CORE_FEATURES[0]] = np.inf
    elif mutation == "duplicate_id": frame.loc[1, "fifa_id"] = frame.loc[0, "fifa_id"]
    elif mutation == "future": frame.loc[0, "fifa_update_date"] = "2023-01-01"
    elif mutation == "position": frame.loc[0, "player_positions_normalized"] = "CAM|CDM"
    with pytest.raises(ValueError):
        PlayerModels().fit(frame)


def test_unfitted_and_degenerate_profiles(profiles):
    with pytest.raises(NotFittedError): PlayerModels().recommend(profiles.iloc[0])
    same = profiles.copy()
    same.loc[:, list(CORE_FEATURES)] = 50
    with pytest.raises(ValueError, match="norm|distinct"):
        PlayerModels().fit(same)
    model = PlayerModels().fit(profiles)
    zero = profiles.iloc[0].copy()
    for column, value in zip(CORE_FEATURES, model.scaler_.mean_): zero[column] = value
    with pytest.raises(ValueError, match="Zero-norm"):
        model.recommend(zero)


def test_equal_distance_tie_breaks_and_weighted_mode(profiles):
    gallery=pd.concat([profiles.iloc[[3]]]*3,ignore_index=True)
    gallery["fifa_id"]=["b","c","a"]
    gallery["tm_market_value"]=[2_000_000.,1_000_000.,1_000_000.]
    model=PlayerModels(cluster_mode="weighted",group_weighting=False).fit(profiles,gallery)
    assert np.allclose(model.weights_,1/np.sqrt(len(CORE_FEATURES)))
    assert model.recommend(profiles.iloc[0]).fifa_id.tolist()==["a","c","b"]


def test_conflicting_shared_id_and_unknown_reference(profiles):
    conflicting=profiles.copy()
    conflicting.loc[0,CORE_FEATURES[0]]+=1
    with pytest.raises(ValueError,match="Conflicting"):
        PlayerModels().fit(profiles,conflicting)
    model=PlayerModels().fit(profiles.iloc[:18])
    with pytest.raises(ValueError,match="Reference ID"):
        model.recommend("external-unknown")
    future=profiles.iloc[18].copy()
    future["fifa_update_date"]="2023-01-01"
    with pytest.raises(ValueError,match="historical"):
        model.recommend(future)


def test_strict_cluster_filter_never_falls_back(profiles):
    original=PlayerModels().fit(profiles)
    labels=original.predict_clusters(profiles)
    query=profiles.iloc[0]
    gallery=profiles.loc[labels!=labels[0]].copy()
    assert len(gallery)>0
    model=PlayerModels().fit(profiles,gallery)
    assert not model.recommend(query).empty
    result=model.recommend(query,within_cluster=True)
    assert result.empty and result.attrs["pool_audit"]["eligible_count"]==0


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": True}, {"max_value": -1},
                                    {"max_age": np.nan}, {"within_cluster": "yes"}])
def test_invalid_request(profiles, kwargs):
    model = PlayerModels().fit(profiles)
    with pytest.raises(ValueError): model.recommend(profiles.iloc[0], **kwargs)
