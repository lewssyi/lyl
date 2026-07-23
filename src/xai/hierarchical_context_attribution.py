"""Pure data types and protocol helpers for the HC-SeqLIME feasibility prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Mapping, Sequence, Tuple

import numpy as np


PROTOCOL_VERSION = "hc_seqlime_feasibility_v1"
TOTAL_QUERY_BUDGET = 1530
CONTEXT_QUERIES = 306
WITHIN_EPOCH_QUERIES = 1200
FAITHFULNESS_QUERIES = 24


def _frozen_vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).ravel().copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("attribution vectors must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ContextAttributionResult:
    sample_id: int
    group_scheme: str
    context_group_importance: np.ndarray
    target_group_index: int
    selected_group_indices: Tuple[int, ...]
    target_epoch_drop: float
    target_contribution: float
    neighbor_context_contribution: float
    distant_context_contribution: float
    context_dependence_ratio: float
    query_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_group_importance", _frozen_vector(self.context_group_importance))
        object.__setattr__(self, "selected_group_indices", tuple(int(x) for x in self.selected_group_indices))
        if not 0.0 <= float(self.context_dependence_ratio) <= 1.0:
            raise ValueError("context dependence ratio must be in [0,1]")


@dataclass(frozen=True)
class WithinEpochAttributionResult:
    sample_id: int
    explained_epoch_indices: Tuple[int, ...]
    feature_count: int
    within_epoch_importance: Tuple[np.ndarray, ...]
    local_surrogate_r2: float
    query_count: int
    target_class: int
    target_protocol: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vectors = tuple(_frozen_vector(value) for value in self.within_epoch_importance)
        if len(vectors) != len(self.explained_epoch_indices):
            raise ValueError("one attribution vector is required per explained epoch")
        if any(len(value) != int(self.feature_count) for value in vectors):
            raise ValueError("within-epoch feature count mismatch")
        object.__setattr__(self, "within_epoch_importance", vectors)
        object.__setattr__(self, "explained_epoch_indices", tuple(int(x) for x in self.explained_epoch_indices))


@dataclass(frozen=True)
class HierarchicalExplanationResult:
    sample_id: int
    candidate: str
    context: ContextAttributionResult
    within_epoch: WithinEpochAttributionResult
    context_importance: np.ndarray
    within_epoch_importance: Tuple[np.ndarray, ...]
    total_query_count: int
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_importance", _frozen_vector(self.context_importance))
        object.__setattr__(self, "within_epoch_importance", tuple(_frozen_vector(x) for x in self.within_epoch_importance))
        if int(self.total_query_count) != TOTAL_QUERY_BUDGET:
            raise ValueError("hierarchical candidate must use the frozen equal budget")


@dataclass(frozen=True)
class ExplanationReliabilityResult:
    sample_id: int
    status: str
    local_surrogate_r2: float
    context_surrogate_r2: float
    faithfulness: float
    mask_seed_stability: float
    r7_similarity: float
    target_epoch_drop: float
    context_dependence_ratio: float
    response_variance: float
    effective_sample_size: float
    included_in_coverage: bool = True

    def __post_init__(self) -> None:
        allowed = {"LOCALLY_EXPLAINABLE", "CONTEXT_DOMINATED", "LOW_MODEL_RESPONSE", "SURROGATE_UNRELIABLE", "RANDOMIZATION_UNRELIABLE"}
        if self.status not in allowed:
            raise ValueError("unknown reliability status")
        if not self.included_in_coverage:
            raise ValueError("unreliable samples cannot be silently excluded")


def query_budget(candidate: str) -> Mapping[str, int]:
    if candidate not in {"H0", "H1", "H2", "H3"}:
        raise ValueError("unknown hierarchical candidate")
    payload = {"context": CONTEXT_QUERIES, "within_epoch": WITHIN_EPOCH_QUERIES, "faithfulness": FAITHFULNESS_QUERIES}
    payload["total"] = sum(payload.values())
    if payload["total"] != TOTAL_QUERY_BUDGET:
        raise RuntimeError("query budget drift")
    return payload


def deterministic_masks(feature_count: int, count: int, seed: int) -> np.ndarray:
    """Unique Bernoulli masks with explicit original/replacement endpoints."""
    feature_count, count = int(feature_count), int(count)
    if feature_count < 2 or count < 3 or count > 2 ** feature_count:
        raise ValueError("invalid mask space")
    rng = np.random.default_rng(int(seed))
    rows = [np.ones(feature_count, dtype=np.int8), np.zeros(feature_count, dtype=np.int8)]
    seen = {row.tobytes() for row in rows}
    while len(rows) < count:
        row = rng.integers(0, 2, size=feature_count, dtype=np.int8)
        if row.tobytes() not in seen:
            seen.add(row.tobytes())
            rows.append(row)
    return np.asarray(rows, dtype=np.int32)


def selected_epoch_offsets(candidate: str, ranked_non_target_groups: Sequence[int], group_centers: Sequence[int]) -> Tuple[int, ...]:
    if candidate == "H0":
        return (0,)
    if not ranked_non_target_groups:
        raise ValueError("context candidate requires ranked non-target groups")
    representatives = [int(group_centers[int(group)]) - 37 for group in ranked_non_target_groups]
    if candidate == "H1":
        return (representatives[0],)
    if candidate == "H2":
        if len(representatives) < 2:
            raise ValueError("H2 requires two context groups")
        return tuple(representatives[:2])
    if candidate == "H3":
        return (0, representatives[0])
    raise ValueError("unknown candidate")


def cache_identity(model_hash: str, sample_id: int, candidate: str, level: str, seed: int, target_protocol: str) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_hash": str(model_hash),
        "sample_id": int(sample_id),
        "candidate": str(candidate),
        "randomization_level": str(level),
        "randomization_seed": int(seed),
        "target_protocol": str(target_protocol),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
