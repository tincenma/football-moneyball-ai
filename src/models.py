"""Train-only CM clustering and exact nearest-neighbor recommendations.

Economic fields are used by the shared eligibility policy, never by geometry.
KNN retrieves the full gallery before budget/age filtering, then takes TOP-k.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.validation import check_is_fitted

from .baseline import REQUIRED, build_eligible_pool

_CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "config/feature_set_v1.json").read_text(encoding="utf-8"))
CORE_FEATURES = tuple(_CONTRACT["similarity_features"])
FEATURE_GROUPS = _CONTRACT["similarity_feature_groups"]


def eligible_ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


class PlayerModels(BaseEstimator):
    """KMeans + NearestNeighbors on the fixed 18-feature playing contract.

    Fit accepts train and an optional separate candidate gallery. Query players
    may be external to both. Clusters are algorithmic profiles, not proven roles.
    """

    def __init__(self, n_clusters=3, random_state=42, n_init=20,
                 cluster_mode="relative", cutoff="2022-09-26", group_weighting=True):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_mode = cluster_mode
        self.cutoff = cutoff
        self.group_weighting = group_weighting

    def _validate(self, frame, *, gallery=False):
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Player profiles must be a DataFrame")
        needed = {"fifa_id", "as_of_date", "fifa_update_date", "player_positions_normalized", *CORE_FEATURES}
        if gallery:
            needed.update(REQUIRED)
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate column names are not allowed")
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise ValueError(f"Missing profile columns: {', '.join(missing)}")
        result = frame.copy()
        result["fifa_id"] = result.fifa_id.astype("string").str.strip()
        if result.fifa_id.isna().any() or result.fifa_id.eq("").any() or result.fifa_id.duplicated().any():
            raise ValueError("fifa_id must be present and unique")
        for col in ("as_of_date", "fifa_update_date"):
            dates = pd.to_datetime(result[col], format="%Y-%m-%d", errors="coerce")
            if not dates.eq(pd.Timestamp(self.cutoff)).all():
                raise ValueError("Profiles must belong to the fixed historical cutoff")
        cm = result.player_positions_normalized.astype("string").fillna("").str.upper().str.replace("|", ",", regex=False).str.split(",").map(
            lambda parts: "CM" in [p.strip() for p in parts])
        if not cm.all():
            raise ValueError("Only profiles with an exact CM position token are supported")
        try:
            values = result.loc[:, list(CORE_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Playing features must be complete finite ratings in [1, 99]") from exc
        if not np.isfinite(values).all() or (values < 1).any() or (values > 99).any():
            raise ValueError("Playing features must be complete finite ratings in [1, 99]")
        return result

    def _standardized(self, frame):
        return self.scaler_.transform(frame.loc[:, list(CORE_FEATURES)].astype(float))

    def _cluster_vectors(self, standardized):
        vectors = standardized.copy()
        if self.cluster_mode == "relative":
            vectors -= vectors.mean(axis=1, keepdims=True)
        vectors *= self.weights_
        if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
            raise ValueError("Zero-norm cluster profile; no meaningful relative direction")
        return normalize(vectors) if self.cluster_mode == "relative" else vectors

    def fit(self, train, gallery=None):
        for value, name, minimum in [(self.n_clusters, "n_clusters", 2), (self.n_init, "n_init", 1), (self.random_state, "random_state", 0)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.cluster_mode not in ("relative", "weighted"):
            raise ValueError("cluster_mode must be relative or weighted")
        if not isinstance(self.group_weighting, bool):
            raise ValueError("group_weighting must be bool")
        try:
            cutoff = pd.to_datetime(self.cutoff, format="%Y-%m-%d", errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("cutoff must be a valid ISO date") from exc
        if pd.isna(cutoff) or cutoff.tzinfo is not None:
            raise ValueError("cutoff must be a valid timezone-free ISO date")
        training = self._validate(train).sort_values("fifa_id").reset_index(drop=True)
        candidate_gallery = self._validate(train if gallery is None else gallery, gallery=True).sort_values("fifa_id").reset_index(drop=True)
        if len(training) < self.n_clusters or len(candidate_gallery) == 0:
            raise ValueError("Insufficient training profiles or empty gallery")
        shared = training.fifa_id.isin(candidate_gallery.fifa_id)
        if shared.any():
            left = training.loc[shared].set_index("fifa_id").sort_index()
            right = candidate_gallery.set_index("fifa_id").loc[left.index]
            if not np.allclose(left[list(CORE_FEATURES)].to_numpy(float), right[list(CORE_FEATURES)].to_numpy(float)):
                raise ValueError("Conflicting feature records for a shared player ID")
        self.scaler_ = StandardScaler().fit(training.loc[:, list(CORE_FEATURES)].astype(float))
        group_for = {f: group for group, fs in FEATURE_GROUPS.items() for f in fs}
        self.weights_ = np.array([np.sqrt(1 / (len(FEATURE_GROUPS) * len(FEATURE_GROUPS[group_for[f]]))) for f in CORE_FEATURES])
        if not self.group_weighting:
            self.weights_ = np.full(len(CORE_FEATURES), 1 / np.sqrt(len(CORE_FEATURES)))
        z = self._standardized(training)
        clustering = self._cluster_vectors(z)
        if len(np.unique(np.round(clustering, 12), axis=0)) < self.n_clusters:
            raise ValueError("Too few distinct playing profiles for n_clusters")
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state).fit(clustering)
        self.train_ids_ = training.fifa_id.tolist()
        self.gallery_ = candidate_gallery
        self.gallery_z_ = self._standardized(candidate_gallery)
        self.gallery_vectors_ = self.gallery_z_ * self.weights_
        if np.any(np.linalg.norm(self.gallery_vectors_, axis=1) <= 1e-12):
            raise ValueError("Zero-norm gallery profile cannot be indexed with cosine")
        self.gallery_clusters_ = self.kmeans_.predict(self._cluster_vectors(self.gallery_z_))
        self.neighbors_ = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(self.gallery_vectors_)
        self.catalog_ = pd.concat([training, candidate_gallery], ignore_index=True).drop_duplicates("fifa_id", keep="last").set_index("fifa_id", drop=False)
        self.feature_names_in_ = np.array(CORE_FEATURES, dtype=object)
        return self

    def transform(self, profiles):
        check_is_fitted(self, ["scaler_", "kmeans_", "neighbors_"])
        valid = self._validate(profiles)
        if valid.empty:
            return np.empty((0, len(CORE_FEATURES)))
        return self._standardized(valid) * self.weights_

    def predict_clusters(self, profiles):
        check_is_fitted(self, ["kmeans_", "neighbors_"])
        valid = self._validate(profiles)
        if valid.empty:
            return np.array([], dtype=int)
        return self.kmeans_.predict(self._cluster_vectors(self._standardized(valid)))

    def cluster_vectors(self, profiles):
        check_is_fitted(self, ["kmeans_", "neighbors_"])
        valid = self._validate(profiles)
        if valid.empty:
            return np.empty((0, len(CORE_FEATURES)))
        return self._cluster_vectors(self._standardized(valid))

    def _reference(self, reference):
        check_is_fitted(self, ["kmeans_", "neighbors_"])
        if isinstance(reference, str):
            if reference not in self.catalog_.index:
                raise ValueError("Reference ID is not in the catalog; pass its complete profile")
            return self.catalog_.loc[[reference]].reset_index(drop=True)
        if isinstance(reference, pd.Series):
            return reference.to_frame().T.reset_index(drop=True)
        if isinstance(reference, dict):
            return pd.DataFrame([reference])
        raise ValueError("Reference must be a fifa_id string, Series or mapping")

    def eligible_pool(self, reference, *, max_value=15_000_000, max_age=30, max_value_age_days=365, within_cluster=False):
        """Exact candidate universe for both KNN and its corresponding baseline."""
        if not isinstance(within_cluster, bool):
            raise ValueError("within_cluster must be bool")
        query = self._validate(self._reference(reference))
        pool, _ = build_eligible_pool(self.gallery_, max_value=max_value, max_age=max_age, cutoff=self.cutoff,
                                     max_value_age_days=max_value_age_days, excluded_fifa_ids=query.fifa_id.tolist())
        if within_cluster:
            cluster = int(self.predict_clusters(query)[0])
            labels = dict(zip(self.gallery_.fifa_id, self.gallery_clusters_))
            pool = pool.loc[pool.fifa_id.map(labels).eq(cluster)].copy()
        return pool.reset_index(drop=True)

    def recommend_many(self, references, *, max_value=15_000_000, max_age=30, top_k=5,
                       max_value_age_days=365, within_cluster=False):
        """Return (ranked rows, per-reference pool audit).

        Query all indexed players first; filter budget/age/self/optional cluster;
        then TOP-k. Empty and small candidate pools never trigger silent fallback.
        """
        check_is_fitted(self, ["kmeans_", "neighbors_"])
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(within_cluster, bool):
            raise ValueError("within_cluster must be bool")
        queries = self._validate(references).reset_index(drop=True)
        base_pool, _ = build_eligible_pool(self.gallery_, max_value=max_value, max_age=max_age, cutoff=self.cutoff,
                                          max_value_age_days=max_value_age_days)
        score_columns = ["reference_id", "model_rank", "cosine_distance", "similarity", "rms_z", "cluster", "same_cluster", "strong_profile_match"]
        empty = self.gallery_.iloc[:0].copy()
        for column in score_columns:
            empty[column] = pd.Series(dtype="object")
        pool_columns = ["reference_id", "reference_cluster", "eligible_count", "eligible_ids_sha256", "returned", "within_cluster"]
        if queries.empty:
            return empty, pd.DataFrame(columns=pool_columns)
        qz = self._standardized(queries)
        qvectors = qz * self.weights_
        if np.any(np.linalg.norm(qvectors, axis=1) <= 1e-12):
            raise ValueError("Zero-norm reference profile has undefined cosine direction")
        qclusters = self.kmeans_.predict(self._cluster_vectors(qz))
        distances, indices = self.neighbors_.kneighbors(qvectors, n_neighbors=len(self.gallery_))
        ids = self.gallery_.fifa_id.to_numpy(dtype=str)
        costs = pd.to_numeric(self.gallery_.tm_market_value, errors="coerce").to_numpy(dtype=float)
        base_allowed = np.isin(ids, base_pool.fifa_id.to_numpy(dtype=str))
        result_parts, audits = [], []
        for i, reference_id in enumerate(queries.fifa_id):
            distance = np.empty(len(self.gallery_), dtype=float)
            distance[indices[i]] = distances[i]
            allowed = base_allowed & (ids != reference_id)
            if within_cluster:
                allowed &= self.gallery_clusters_ == qclusters[i]
            eligible = np.flatnonzero(allowed)
            # Costs are only a deterministic tie-break, never part of distance.
            order = np.lexsort((ids[eligible], costs[eligible], distance[eligible]))
            chosen = eligible[order[:top_k]]
            audits.append({"reference_id": reference_id, "reference_cluster": int(qclusters[i]),
                           "eligible_count": len(eligible), "eligible_ids_sha256": eligible_ids_hash(ids[eligible]),
                           "returned": len(chosen), "within_cluster": within_cluster})
            if len(chosen):
                part = self.gallery_.iloc[chosen].drop(columns=score_columns, errors="ignore").copy()
                part["reference_id"], part["model_rank"] = reference_id, np.arange(1, len(chosen)+1)
                part["cosine_distance"] = distance[chosen]
                part["similarity"] = np.clip(1-distance[chosen], -1, 1)
                part["rms_z"] = np.sqrt(np.mean((self.gallery_z_[chosen]-qz[i])**2, axis=1))
                part["cluster"] = self.gallery_clusters_[chosen]
                part["same_cluster"] = self.gallery_clusters_[chosen] == qclusters[i]
                part["strong_profile_match"] = part.similarity.ge(.90) & part.rms_z.le(.75)
                result_parts.append(part)
        return (pd.concat(result_parts, ignore_index=True) if result_parts else empty), pd.DataFrame(audits, columns=pool_columns)

    def recommend(self, reference, **kwargs):
        """Single selected player; pass an ID or a complete external profile."""
        ranked, audit = self.recommend_many(self._reference(reference), **kwargs)
        ranked.attrs["pool_audit"] = audit.iloc[0].to_dict()
        return ranked
