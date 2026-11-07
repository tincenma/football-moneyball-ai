"""Decision pipeline correctness, honest economics, and style/price isolation."""
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from src.models import CORE_FEATURES, PlayerModels
from src.recommendation import RecommendationSystem, STYLE_NAME
from src.valuation import MarketValuePredictor


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def components():
    rng = np.random.default_rng(19)
    d = pd.DataFrame(rng.integers(35, 90, (40, len(CORE_FEATURES))).astype(float), columns=CORE_FEATURES)
    d["fifa_id"] = [f"p{i:02}" for i in range(40)]
    d["tm_id"] = [f"tm{i:02}" for i in range(40)]
    d["short_name"] = d.fifa_id
    d["as_of_date"] = d["fifa_update_date"] = "2022-09-26"
    d["market_value_date"] = "2022-06-01"
    d["player_positions_normalized"] = "CM|CDM"
    d["age_at_cutoff"] = 18 + np.arange(40) % 15
    d["overall"] = 70
    d["tm_market_value"] = 100_000 + 500_000 * (np.arange(40) % 30)
    d["height_cm"], d["weight_kg"], d["weak_foot"], d["league_level"] = 180., 75., 3., 1.
    d["preferred_foot"], d["work_rate_attack"], d["work_rate_defence"], d["league_id"] = "Right", "High", "Medium", "10"
    d["split"] = ["train"] * 32 + ["test"] * 8
    models = PlayerModels(n_init=5).fit(d.iloc[:32], gallery=d)
    value = MarketValuePredictor(n_estimators=8).fit(d.iloc[:32])
    banks = []; folds = []
    # Fixed hyperparameters: no target-dependent model selection in this fixture.
    for fold, (fit, held) in enumerate(KFold(4, shuffle=True, random_state=42).split(d.iloc[:32]), 1):
        estimator = MarketValuePredictor(n_estimators=8).fit(d.iloc[fit])
        part = estimator.gap_table(d.iloc[held]); part["outer_fold"] = fold
        part["prediction_origin"] = "nested_oof_train"; banks.append(part)
        folds.append({"fold": fold, "fit_ids": d.iloc[fit].fifa_id.tolist(), "validation_ids": d.iloc[held].fifa_id.tolist()})
    return d, models, value, pd.concat(banks), folds


@pytest.fixture
def system(components):
    d, models, value, bank, folds = components
    return RecommendationSystem(models, value, reference_profiles=d, oof_predictions=bank, oof_fold_audit=folds)


def query(system, **kwargs):
    options = dict(reference="p39", playing_style=None, similarity_threshold=0, max_age=50, budget=20_000_000, top_k=100)
    options.update(kwargs)
    return system.recommend(**options)


def test_complete_pipeline_explanations_and_no_refitting(system, monkeypatch):
    before = system.player_models.scaler_.mean_.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Serving must never fit")
    monkeypatch.setattr(PlayerModels, "fit", forbidden)
    monkeypatch.setattr(MarketValuePredictor, "fit", forbidden)
    result = query(system, top_k=5, max_age=25, budget=10_000_000, currency="EUR")
    assert 0 < len(result) <= 5
    assert result.age_at_cutoff.le(25).all() and result.tm_market_value.le(10_000_000).all()
    assert not result.fifa_id.eq("p39").any()
    assert np.allclose(result.value_gap, result.predicted_value - result.tm_market_value)
    assert result.explanation.str.contains("cosine, не вероятность успеха").all()
    assert result.explanation.str.contains("справедливая цена").all()
    assert result.explanation.str.contains(r"EUR \(допущение\)").all()
    assert np.array_equal(before, system.player_models.scaler_.mean_)
    assert result.attrs["audit"]["returned"] == len(result)
    assert "future_value" not in result and "potential" not in result


def test_similarity_ranking_matches_direct_numpy_oracle(system):
    result = query(system, top_k=5)
    query_row = system.references.loc[system.references.fifa_id.eq("p39")]
    gallery = system.player_models.gallery_
    a = system.player_models.transform(query_row)[0]; b = system.player_models.transform(gallery)
    cosine = (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a))
    expected = gallery.assign(cosine=cosine)
    expected = expected.loc[expected.fifa_id.ne("p39") & expected.cosine.ge(0)].sort_values("cosine", ascending=False).head(5)
    assert result.fifa_id.tolist() == expected.fifa_id.tolist()
    assert np.allclose(result.similarity, expected.cosine)
    assert np.allclose(result.similarity_percent, 100 * result.similarity)


def test_oof_and_heldout_economics_are_separate(system):
    result = query(system)
    training = set(system.valuation_model.train_ids_)
    assert result.loc[result.fifa_id.isin(training), "prediction_origin"].eq("nested_oof_train").all()
    assert result.loc[~result.fifa_id.isin(training), "prediction_origin"].eq("held_out_from_valuation_model").all()
    assert result.economic_status.eq("available").all()


def test_missing_bank_never_uses_in_sample_predictions(components, monkeypatch):
    d, models, value, _, _ = components
    system = RecommendationSystem(models, value, reference_profiles=d)
    original = value.predict
    def guarded(frame):
        assert not set(frame.fifa_id) & set(value.train_ids_)
        return original(frame)
    monkeypatch.setattr(value, "predict", guarded)
    result = query(system)
    infit = result.fifa_id.isin(value.train_ids_)
    assert infit.any() and result.loc[infit, "predicted_value"].isna().all()
    assert result.loc[infit, "positive_model_gap"].isna().all()
    assert result.loc[infit, "explanation"].str.contains("недоступен").all()
    ranked = query(system, ranking="value_gap")
    assert not set(ranked.fifa_id) & set(value.train_ids_)


def test_economic_ranking_uses_whole_qualified_pool(components, system):
    d, models, value, bank, folds = components
    all_candidates = query(system)
    chosen = all_candidates.loc[all_candidates.fifa_id.isin(value.train_ids_)].iloc[-1].fifa_id
    modified = bank.copy(); modified.loc[modified.fifa_id.eq(chosen), "predicted_value"] = 1e12
    updated = RecommendationSystem(models, value, reference_profiles=d, oof_predictions=modified, oof_fold_audit=folds)
    result = query(updated, ranking="value_gap", top_k=1)
    assert result.fifa_id.tolist() == [chosen]


def test_style_is_train_only_and_both_thresholds_apply(system):
    style = system.recommend(playing_style=STYLE_NAME, similarity_threshold=.2, max_age=50, top_k=100, budget=20_000_000)
    details = style.attrs["audit"]["style"]
    assert details["anchor_count"] > 0 and details["recipe"]["passing_mean_z_min_quantile"] == .75
    assert style.similarity.ge(.2).all() and style.style_similarity.ge(.2).all()
    q = query(system, similarity_threshold=.2)
    combined = query(system, playing_style=STYLE_NAME, similarity_threshold=.2)
    assert set(combined.fifa_id) == set(q.fifa_id) & set(style.fifa_id)
    assert combined.style_similarity.ge(.2).all()
    prototype_before = details["raw_prototype"]
    system.references.loc[system.references.split.eq("test"), list(CORE_FEATURES)] = 99.
    fresh = RecommendationSystem(system.player_models, system.valuation_model, reference_profiles=system.references)
    assert fresh._style()[1]["raw_prototype"] == prototype_before


def test_empty_short_lists_level_and_cluster_constraints(system):
    empty = query(system, budget=0, top_k=5)
    assert empty.empty and empty.attrs["audit"]["status"] == "no_candidates"
    assert empty.attrs["audit"]["empty_reason"] == "no_eligible_candidates"
    assert {"explanation", "rank", "predicted_value"}.issubset(empty)
    assert query(system, similarity_threshold=1).empty
    limited = query(system, within_cluster=True)
    assert limited.same_cluster.all()
    bounded = query(system, max_rms_z=.8)
    assert bounded.rms_z.le(.8).all()
    short = query(system, top_k=1000)
    assert short.attrs["audit"]["status"] == "fewer_than_requested"


def test_audit_style_metadata_cannot_mutate_archetype(system):
    result = system.recommend(similarity_threshold=0, max_age=50)
    prototype = result.attrs['audit']['style']['raw_prototype'].copy()
    result.attrs['audit']['style']['raw_prototype']['mentality_vision'] = 1
    result.attrs['audit']['style']['recipe']['passing_features'][0] = 'tm_market_value'
    again = system.recommend(similarity_threshold=0, max_age=50)
    assert again.fifa_id.tolist() == result.fifa_id.tolist()
    assert again.attrs['audit']['style']['raw_prototype'] == prototype
    assert again.attrs['audit']['style']['recipe']['passing_features'][0] == 'attacking_short_passing'


@pytest.mark.parametrize("kwargs", [
    {"position": "CDM"}, {"budget": -1}, {"budget": np.nan}, {"max_age": 14},
    {"similarity_threshold": 85}, {"similarity_threshold": True}, {"top_k": 0}, {"top_k": True},
    {"ranking": "profit"}, {"currency": "USD"}, {"within_cluster": 1}, {"max_rms_z": -1},
    {"playing_style": "Invented role"}, {"max_value_age_days": 366}, {"max_value_age_days": True},
    {"playing_style": None, "reference": None},
])
def test_invalid_requests_raise_clear_errors(system, kwargs):
    with pytest.raises(ValueError): query(system, **kwargs)


@pytest.mark.parametrize("case", ["missing_audit", "overlapping_fit", "wrong_fold", "duplicate_id", "bad_origin", "nan_prediction", "changed_target", "future_target"])
def test_invalid_or_unbound_oof_bank(components, case):
    d, models, value, source, original_folds = components
    bank = source.copy(); folds = copy.deepcopy(original_folds)
    if case == "missing_audit": folds = None
    elif case == "overlapping_fit": folds[0]["fit_ids"].append(folds[0]["validation_ids"][0])
    elif case == "wrong_fold": bank.iloc[0, bank.columns.get_loc("outer_fold")] = 99
    elif case == "duplicate_id": bank = pd.concat([bank, bank.iloc[:1]])
    elif case == "bad_origin": bank["prediction_origin"] = "in_sample"
    elif case == "nan_prediction": bank.iloc[0, bank.columns.get_loc("predicted_value")] = np.nan
    elif case == "changed_target": bank.iloc[0, bank.columns.get_loc("tm_market_value")] *= 2
    elif case == "future_target": bank["market_value_date"] = "2023-01-01"
    with pytest.raises(ValueError): RecommendationSystem(models, value, oof_predictions=bank, oof_fold_audit=folds)


def test_future_quotes_excluded_and_mismatched_cutoffs_rejected(components):
    d, original, value, _, _ = components
    models = copy.deepcopy(original)
    models.gallery_.loc[models.gallery_.fifa_id.eq("p38"), "market_value_date"] = "2023-01-01"
    system = RecommendationSystem(models, value, reference_profiles=d)
    assert "p38" not in set(query(system).fifa_id)
    wrong = copy.deepcopy(value); wrong.cutoff = "2023-01-01"
    with pytest.raises(ValueError, match="cutoff"): RecommendationSystem(models, wrong)


def test_artifact_loader_hashes_and_external_reference_lookup(components, tmp_path):
    d, _, value, bank, folds = components
    fd=tmp_path/"data/processed/features";vd=tmp_path/"data/processed/valuation";md=tmp_path/"data/processed/models"
    for p in [fd,vd,md,tmp_path/"reports/models",tmp_path/"reports/valuation"]:p.mkdir(parents=True)
    feature=fd/"moneyball_features_2022-09-26.csv";d.to_csv(feature,index=False)
    model_path=md/"player_models_2022-09-26.joblib";value_path=vd/"market_value_rf_2022-09-26.joblib"
    joblib.dump(PlayerModels(n_init=5).fit(d.iloc[:32]),model_path);joblib.dump(value,value_path)
    bank_path=vd/"train_nested_oof_value_gaps.csv";bank.to_csv(bank_path,index=False)
    audit_path=vd/"nested_fold_audit.json";audit_path.write_text(json.dumps(folds))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def write(p,obj):p.write_text(json.dumps(obj))
    write(fd/"feature_manifest.json",{"outputs":{feature.name:{"sha256":sha(feature)}}})
    write(tmp_path/"reports/models/models_summary.json",{"model_bundle":model_path.name,"model_sha256":sha(model_path),"input_sha256":sha(feature)})
    write(tmp_path/"reports/valuation/valuation_summary.json",{"model_sha256":sha(value_path),"source_sha256":sha(feature)})
    write(vd/"prediction_manifest.json",{"source_sha256":sha(feature),"outputs":{bank_path.name:sha(bank_path),audit_path.name:sha(audit_path)}})
    loaded=RecommendationSystem.from_artifacts(tmp_path)
    assert not query(loaded).empty and loaded.artifact_hashes
    bank_path.write_text(bank_path.read_text()+"\n")
    with pytest.raises(ValueError,match="hash mismatch"):RecommendationSystem.from_artifacts(tmp_path)
