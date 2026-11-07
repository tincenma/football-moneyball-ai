"""Historical CM decision support: filter -> profile -> economics -> explain.

Similarity is cosine, not a success probability. Style recipes are explicit
scouting hypotheses, not learned ground-truth roles. Economic gaps use nested
OOF predictions for RF training players and held-out predictions for new IDs.
No fitting, future outcomes, or in-sample price-gap ranking occurs at serving.
"""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .models import CORE_FEATURES, PlayerModels, eligible_ids_hash
from .valuation import MarketValuePredictor, VALUE_FEATURES

STYLE_NAME = "Deep-Lying Playmaker"
STYLE_RECIPE = {
    "passing_features": ["attacking_short_passing", "skill_long_passing", "mentality_vision"],
    "passing_mean_z_min_quantile": .75,
    "mentality_composure_min_quantile": .50,
    "mentality_interceptions_min_quantile": .50,
    "prototype": "mean of all 18 raw attributes of qualifying training players",
    "interpretation": "explicit train-derived scouting archetype, not a validated football role",
}
FEATURE_LABELS = dict(zip(CORE_FEATURES, [
    "короткий пас", "длинный пас", "видение поля", "контроль мяча", "дриблинг", "ловкость",
    "баланс", "реакция", "хладнокровие", "выносливость", "сила", "агрессия", "перехваты",
    "позиционная защита", "отбор", "ускорение", "скорость", "дальние удары",
]))
PLAYER_COLUMNS = tuple(dict.fromkeys([
    "fifa_id", "tm_id", "short_name", "as_of_date", "fifa_update_date", "market_value_date",
    "player_positions_normalized", "overall", *VALUE_FEATURES,
]))
SCORE_COLUMNS = ["similarity", "similarity_percent", "style_similarity", "rms_z", "cluster", "same_cluster",
                 "strong_profile_match", "predicted_value", "value_gap", "value_gap_fraction", "positive_model_gap",
                 "economic_status", "prediction_origin", "closest_attributes", "profile_caveat", "explanation", "rank"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name, low, high=np.inf):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


class RecommendationSystem:
    """Compose fitted PlayerModels and MarketValuePredictor without refitting.

    recommend returns a DataFrame with row explanations and attrs['audit'].
    similarity_threshold is in [0,1]; .85 means 85% cosine, never a probability.
    Default ranking: similarity desc, out-of-fit relative value gap desc, price
    asc, ID asc. ranking='value_gap' instead ranks economic leads *after* all
    profile filters and excludes candidates without out-of-fit economics.
    Neither policy is claimed to be independently validated as superior.
    """

    def __init__(self, player_models: PlayerModels, valuation_model: MarketValuePredictor, *,
                 reference_profiles=None, oof_predictions=None, oof_fold_audit=None):
        check_is_fitted(player_models, ["scaler_", "kmeans_", "neighbors_", "train_ids_", "gallery_"])
        check_is_fitted(valuation_model, ["estimator_", "train_ids_"])
        if player_models.cutoff != valuation_model.cutoff:
            raise ValueError("Similarity and valuation models must share the historical cutoff")
        if list(player_models.feature_names_in_) != list(CORE_FEATURES) or list(valuation_model.feature_names_in_) != list(VALUE_FEATURES):
            raise ValueError("Models must follow Feature Set v1")
        self.player_models = player_models
        self.valuation_model = valuation_model
        self.cutoff = player_models.cutoff
        # Validate all valuation inputs early; future/invalid price rows are
        # still handled by the shared eligibility filter at recommendation time.
        valuation_model._profiles(player_models.gallery_)
        self.references = None if reference_profiles is None else player_models._validate(reference_profiles)
        self._oof = self._validate_oof(oof_predictions, oof_fold_audit)
        self._prototype = None
        self._style_details = None
        self.artifact_hashes = {}

    @classmethod
    def from_artifacts(cls, root=None):
        """Load own trusted local bundles, checking hashes, profiles and OOF IDs.

        Requires outputs of Steps 3, 6 and Model 3. Does not read evaluation
        outcomes or refit models. Never load untrusted joblib/pickle files.
        """
        root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        fd = root / "data/processed/features"
        vd = root / "data/processed/valuation"
        fm = json.loads((fd / "feature_manifest.json").read_text(encoding="utf-8"))
        ms = json.loads((root / "reports/models/models_summary.json").read_text(encoding="utf-8"))
        vs = json.loads((root / "reports/valuation/valuation_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((vd / "prediction_manifest.json").read_text(encoding="utf-8"))
        feature_path = fd / "moneyball_features_2022-09-26.csv"
        model_path = root / "data/processed/models" / ms["model_bundle"]
        value_path = vd / "market_value_rf_2022-09-26.joblib"
        oof_path = vd / "train_nested_oof_value_gaps.csv"
        audit_path = vd / "nested_fold_audit.json"
        feature_hash = _sha(feature_path)
        if not feature_hash == fm["outputs"][feature_path.name]["sha256"] == ms["input_sha256"] == vs["source_sha256"] == manifest["source_sha256"]:
            raise ValueError("Feature-table source hashes do not agree")
        expected = {model_path: ms["model_sha256"], value_path: vs["model_sha256"],
                    oof_path: manifest["outputs"][oof_path.name], audit_path: manifest["outputs"][audit_path.name]}
        if any(_sha(path) != digest for path, digest in expected.items()):
            raise ValueError("Model or OOF artifact hash mismatch")
        profiles = pd.read_csv(feature_path, dtype={"fifa_id": "string", "tm_id": "string", "league_id": "string"})
        models, valuation = joblib.load(model_path), joblib.load(value_path)
        training = set(profiles.loc[profiles.split.eq("train"), "fifa_id"])
        if set(models.train_ids_) != training or set(valuation.train_ids_) != training or set(models.gallery_.fifa_id) != training:
            raise ValueError("Saved models do not match the fixed training split/gallery")
        oof = pd.read_csv(oof_path, dtype={"fifa_id": "string", "tm_id": "string", "league_id": "string"})
        folds = json.loads(audit_path.read_text(encoding="utf-8"))
        system = cls(models, valuation, reference_profiles=profiles, oof_predictions=oof, oof_fold_audit=folds)
        system.artifact_hashes = {"features": feature_hash, **{path.name: digest for path, digest in expected.items()}}
        return system

    def _validate_oof(self, frame, folds):
        if frame is None:
            if folds is not None:
                raise ValueError("Fold audit supplied without OOF predictions")
            return None
        required = {"fifa_id", "predicted_value", "prediction_origin", "outer_fold", "tm_market_value", "market_value_date", *VALUE_FEATURES}
        if not required.issubset(frame) or not folds:
            raise ValueError("OOF predictions require complete historical profiles and actual fold-ID provenance")
        bank = self.valuation_model._profiles(frame)
        target = pd.to_numeric(bank.tm_market_value, errors="coerce").to_numpy(float)
        target_age = (pd.Timestamp(self.cutoff) - pd.to_datetime(bank.market_value_date, errors="coerce")).dt.days
        if not np.isfinite(target).all() or (target <= 0).any() or not target_age.between(0, 365).all():
            raise ValueError("OOF bank requires positive historical target values no older than 365 days")
        bank["tm_market_value"] = target
        training = set(map(str, self.valuation_model.train_ids_))
        if set(bank.fifa_id) != training or not bank.prediction_origin.eq("nested_oof_train").all():
            raise ValueError("OOF bank must cover exactly the valuation training IDs with nested OOF origin")
        ownership = {}; fold_numbers = set()
        for fold in folds:
            fit_ids = set(map(str, fold["fit_ids"]))
            val_ids = set(map(str, fold["validation_ids"]))
            number = fold["fold"]
            if (number in fold_numbers or not fit_ids or not val_ids or fit_ids & val_ids or fit_ids | val_ids != training
                    or len(fit_ids) != len(fold["fit_ids"]) or len(val_ids) != len(fold["validation_ids"])):
                raise ValueError("Invalid/disjoint OOF training and validation fold IDs")
            if ownership.keys() & val_ids:
                raise ValueError("Each training player needs exactly one OOF validation fold")
            for field, ids in [("fit_ids_sha256", fit_ids), ("validation_ids_sha256", val_ids)]:
                if field in fold and fold[field] != eligible_ids_hash(ids):
                    raise ValueError("Fold-ID hash mismatch")
            ownership.update({pid: number for pid in val_ids}); fold_numbers.add(number)
        if len(fold_numbers) < 2 or set(ownership) != training or not bank.outer_fold.eq(bank.fifa_id.map(ownership)).all():
            raise ValueError("OOF fold assignment does not match every bank row")
        predictions = pd.to_numeric(bank.predicted_value, errors="coerce").to_numpy(float)
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("OOF predictions must be finite and nonnegative")
        bank["predicted_value"] = predictions
        # Binding the bank to actual serving profiles prevents stale gaps when
        # a caller changes a recorded target, date or input feature.
        self._check_bank_binding(bank, self.player_models.gallery_)
        return bank.set_index("fifa_id", drop=False)

    @staticmethod
    def _check_bank_binding(bank, profiles):
        bank = bank.set_index("fifa_id", drop=False)
        actual = profiles.set_index("fifa_id", drop=False)
        shared = bank.index.intersection(actual.index)
        a, b = bank.loc[shared], actual.loc[shared]
        for col in [*VALUE_FEATURES, "tm_market_value", "market_value_date", "as_of_date", "fifa_update_date"]:
            if col in ["preferred_foot", "work_rate_attack", "work_rate_defence", "league_id", "market_value_date", "as_of_date", "fifa_update_date"]:
                same = a[col].astype("string").fillna("UNKNOWN").to_numpy() == b[col].astype("string").fillna("UNKNOWN").to_numpy()
                valid = bool(np.all(same))
            else:
                valid = np.allclose(pd.to_numeric(a[col]).to_numpy(float), pd.to_numeric(b[col]).to_numpy(float), equal_nan=True, rtol=0, atol=1e-8)
            if not valid:
                raise ValueError(f"OOF bank does not match serving profiles: {col}")

    def _style(self):
        if self._prototype is None:
            training = self.player_models.catalog_.loc[self.player_models.train_ids_].copy()
            z = self.player_models.scaler_.transform(training[list(CORE_FEATURES)])
            passing = z[:, [list(CORE_FEATURES).index(f) for f in STYLE_RECIPE["passing_features"]]].mean(axis=1)
            passing_cutoff = float(np.quantile(passing, STYLE_RECIPE["passing_mean_z_min_quantile"]))
            composure_cutoff = float(training.mentality_composure.quantile(STYLE_RECIPE["mentality_composure_min_quantile"]))
            interceptions_cutoff = float(training.mentality_interceptions.quantile(STYLE_RECIPE["mentality_interceptions_min_quantile"]))
            mask = ((passing >= passing_cutoff) & training.mentality_composure.ge(composure_cutoff).to_numpy() &
                    training.mentality_interceptions.ge(interceptions_cutoff).to_numpy())
            anchors = training.loc[mask]
            if anchors.empty:
                raise ValueError("No training players satisfy the explicit style recipe; provide a reference profile")
            prototype = anchors[list(CORE_FEATURES)].mean().to_frame().T
            prototype["fifa_id"] = "__style_deep_lying_playmaker__"
            prototype["as_of_date"] = prototype["fifa_update_date"] = self.cutoff
            prototype["player_positions_normalized"] = "CM"
            # Verify nonzero direction using the same fitted transformation.
            vector = self.player_models.transform(prototype)[0]
            if np.linalg.norm(vector) <= 1e-12:
                raise ValueError("Style prototype has undefined cosine direction; provide a reference")
            self._prototype = prototype
            self._style_details = {"name": STYLE_NAME, "recipe": copy.deepcopy(STYLE_RECIPE), "anchor_count": len(anchors),
                                   "anchor_ids_sha256": eligible_ids_hash(anchors.fifa_id),
                                   "passing_mean_z_cutoff": passing_cutoff,
                                   "composure_cutoff": composure_cutoff,
                                   "interceptions_cutoff": interceptions_cutoff,
                                   "raw_prototype": {f: float(prototype.iloc[0][f]) for f in CORE_FEATURES}}
        return self._prototype.copy(), copy.deepcopy(self._style_details)

    def _reference(self, reference):
        if isinstance(reference, str) and self.references is not None:
            found = self.references.loc[self.references.fifa_id.eq(reference)]
            if not found.empty:
                return found.iloc[0]
        return self.player_models._reference(reference).iloc[0]

    def _economics(self, candidates):
        result = candidates.copy()
        result["predicted_value"] = np.nan
        result["prediction_origin"] = "unavailable"
        result["economic_status"] = "unavailable_no_out_of_fit_prediction"
        training = set(map(str, self.valuation_model.train_ids_))
        in_fit = result.fifa_id.isin(training)
        if self._oof is not None:
            self._check_bank_binding(self._oof.reset_index(drop=True), result)
            available = in_fit & result.fifa_id.isin(self._oof.index)
            result.loc[available, "predicted_value"] = result.loc[available, "fifa_id"].map(self._oof.predicted_value)
            result.loc[available, "prediction_origin"] = "nested_oof_train"
            result.loc[available, "economic_status"] = "available"
        external = ~in_fit
        if external.any():
            held = self.valuation_model.gap_table(result.loc[external])
            result.loc[external, "predicted_value"] = held.predicted_value.to_numpy()
            result.loc[external, "prediction_origin"] = "held_out_from_valuation_model"
            result.loc[external, "economic_status"] = "available"
        result["value_gap"] = result.predicted_value - result.tm_market_value
        result["value_gap_fraction"] = result.value_gap / result.tm_market_value
        result["positive_model_gap"] = pd.array(
            [pd.NA if pd.isna(v) else bool(v > 0) for v in result.value_gap], dtype="boolean")
        return result

    def recommend(self, *, position="CM", budget=15_000_000, max_age=25,
                  playing_style=STYLE_NAME, similarity_threshold=.85, reference=None,
                  top_k=5, ranking="similarity", within_cluster=False, max_rms_z=None,
                  max_value_age_days=365, currency="source_units"):
        """Return up to TOP-k qualified candidates with explanations and audit.

        Supply reference as ID, Series or mapping. With both style and reference,
        the same threshold applies separately to both cosine scores; primary
        ranking uses reference similarity. Set playing_style=None for reference
        only. max_rms_z optionally bounds differences of absolute rating level.
        EUR explicitly labels a scenario assumption; source units are unverified.
        Empty/short lists never relax the request or silently change clusters.
        """
        if not isinstance(position, str) or position.strip().upper() != "CM":
            raise ValueError("Only CM is supported by the trained Feature Set v1 models")
        _number(budget, "budget", 0); _number(max_age, "max_age", 15, 50)
        _number(similarity_threshold, "similarity_threshold (use .85 for 85%)", 0, 1)
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if isinstance(max_value_age_days, bool) or not isinstance(max_value_age_days, Integral) or not 0 <= max_value_age_days <= 365:
            raise ValueError("max_value_age_days must be an integer in [0,365]")
        if max_rms_z is not None:
            _number(max_rms_z, "max_rms_z", 0)
        if ranking not in {"similarity", "value_gap"} or currency not in {"source_units", "EUR"} or not isinstance(within_cluster, bool):
            raise ValueError("Invalid ranking, currency or within_cluster parameter")
        if playing_style is not None and playing_style != STYLE_NAME:
            raise ValueError(f"Unknown playing style; supported: {STYLE_NAME}")
        if reference is None and playing_style is None:
            raise ValueError("Choose a playing style or a reference player")
        style, details = self._style() if playing_style is not None else (None, None)
        query = self._reference(reference) if reference is not None else style.iloc[0]
        name = str(query.get("short_name", query.fifa_id)) if reference is not None else STYLE_NAME
        scores = self.player_models.recommend(query, max_value=budget, max_age=max_age,
                    max_value_age_days=max_value_age_days, within_cluster=within_cluster,
                    top_k=max(1, len(self.player_models.gallery_)))
        audit = {"cutoff": self.cutoff, "position": "CM", "budget": float(budget), "max_age": float(max_age),
                 "similarity_threshold": float(similarity_threshold), "max_rms_z": max_rms_z,
                 "currency": "source units" if currency == "source_units" else "EUR assumed; source currency unverified",
                 "similarity_target": name, "target_kind": "reference" if reference is not None else "style_prototype",
                 "style": details, "ranking": ranking, "within_cluster": within_cluster,
                 "max_value_age_days": int(max_value_age_days), "requested_top_k": int(top_k),
                 "pool": scores.attrs["pool_audit"], "artifact_hashes": self.artifact_hashes.copy(),
                 "ranking_definition": "similarity desc, relative value gap desc, recorded value asc, ID asc" if ranking == "similarity" else "relative value gap desc, similarity desc, recorded value asc, ID asc",
                 "limitations": ["Cosine is not a probability of transfer success", "Style is an explicit archetype, not a validated role",
                                  "Cluster separation is weak; same-cluster restriction is optional", "Model gap is not objective fair price",
                                  "Historical snapshot; no claim of current availability or causal transfer success"]}
        scores = scores.loc[scores.similarity.ge(similarity_threshold)].copy()
        audit["after_similarity"] = len(scores)
        scores["style_similarity"] = np.nan
        if style is not None and not scores.empty:
            a = self.player_models.transform(style)[0]; b = self.player_models.transform(scores)
            scores["style_similarity"] = np.clip((b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a)), -1, 1)
            scores = scores.loc[scores.style_similarity.ge(similarity_threshold)].copy()
        audit["after_style"] = len(scores)
        if max_rms_z is not None:
            scores = scores.loc[scores.rms_z.le(max_rms_z)].copy()
        audit["after_level_filter"] = len(scores)
        scores = self._economics(scores)
        audit["economic_available"] = int(scores.economic_status.eq("available").sum())
        audit["economic_unavailable"] = int(scores.economic_status.ne("available").sum())
        if ranking == "value_gap":
            scores = scores.loc[scores.economic_status.eq("available")].copy()
        order = ["similarity", "value_gap_fraction", "tm_market_value", "fifa_id"] if ranking == "similarity" else ["value_gap_fraction", "similarity", "tm_market_value", "fifa_id"]
        scores = scores.sort_values(order, ascending=[False, False, True, True], na_position="last").head(top_k).copy()
        scores["similarity_percent"] = 100 * scores.similarity
        scores["closest_attributes"] = ""
        scores["profile_caveat"] = ""
        scores["explanation"] = ""
        if not scores.empty:
            query_frame = pd.DataFrame([query])
            qz = self.player_models.scaler_.transform(query_frame[list(CORE_FEATURES)].astype(float))[0]
            cz = self.player_models.scaler_.transform(scores[list(CORE_FEATURES)].astype(float))
            for index, delta in zip(scores.index, np.abs(cz - qz)):
                closest = [f"{FEATURE_LABELS[CORE_FEATURES[i]]} (Δ {delta[i]:.2f}σ)" for i in np.argsort(delta, kind="stable")[:3]]
                scores.at[index, "closest_attributes"] = "; ".join(closest)
                row = scores.loc[index]
                caveat = "Рейтинговый уровень заметно отличается от целевого профиля." if row.rms_z > .75 else ""
                scores.at[index, "profile_caveat"] = caveat
                unit = "ед. источника" if currency == "source_units" else "EUR (допущение)"
                text = (f"{row.short_name} (срез {self.cutoff}): CM, {row.age_at_cutoff:.1f} года при лимите {max_age:g}; "
                        f"историческая оценка {row.tm_market_value:,.0f} ≤ {budget:,.0f} {unit}. "
                        f"Сходство с {'эталоном' if reference is not None else 'заданным профилем'} «{name}» — {row.similarity:.1%} "
                        f"(cosine, не вероятность успеха), порог {similarity_threshold:.0%}. ")
                if style is not None and reference is not None:
                    text += f"Сходство с профилем {STYLE_NAME}: {row.style_similarity:.1%}. "
                text += f"Алгоритмический кластер {int(row.cluster)}; наименьшие расхождения: {'; '.join(closest)}. "
                if row.economic_status == "available":
                    text += (f"Прогноз {row.predicted_value:,.0f}; прогноз минус оценка {row.value_gap:+,.0f} {unit} "
                             f"({row.prediction_origin}; вне собственного обучения). Это модельный сигнал, не справедливая цена. ")
                else:
                    text += "Экономический прогноз вне собственного обучения недоступен; вывод о недооценённости не делается. "
                scores.at[index, "explanation"] = text + caveat
        scores["rank"] = np.arange(1, len(scores) + 1)
        columns = [c for c in PLAYER_COLUMNS if c in scores] + ["tm_market_value"] + SCORE_COLUMNS
        result = scores[list(dict.fromkeys(columns))].reset_index(drop=True)
        audit["returned"] = len(result)
        audit["shortfall"] = max(0, int(top_k) - len(result))
        audit["status"] = "ok" if len(result) == top_k else "no_candidates" if result.empty else "fewer_than_requested"
        if result.empty:
            audit["empty_reason"] = ("no_eligible_candidates" if not audit["pool"]["eligible_count"] else
                                     "similarity_threshold" if not audit["after_similarity"] else
                                     "style_threshold" if not audit["after_style"] else
                                     "rating_level_filter" if not audit["after_level_filter"] else
                                     "no_out_of_fit_economic_predictions")
        result.attrs["audit"] = audit
        return result
