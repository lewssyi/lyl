"""Strongly separated result types for attribution and clinical annotation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import numpy as np


def _immutable_vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).ravel().copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("result vectors must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ModelAttributionResult:
    sample_id: int
    method_id: str
    model_attribution_score: np.ndarray
    target_class: int
    target_protocol: str
    query_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_attribution_score",
            _immutable_vector(self.model_attribution_score),
        )
        if str(self.method_id) in {"M5", "M6"}:
            raise ValueError("concept-only scores cannot be represented as model attribution")


@dataclass(frozen=True)
class ClinicalAnnotationResult:
    sample_id: int
    source_attribution_method: str
    selected_feature_indices: Tuple[int, ...]
    clinical_concept_activation: np.ndarray
    clinical_annotation_score: np.ndarray
    concept_labels: Tuple[str, ...]
    annotation_kind: str = "derived clinical annotation"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_feature_indices",
            tuple(int(value) for value in self.selected_feature_indices),
        )
        object.__setattr__(
            self,
            "clinical_concept_activation",
            _immutable_vector(self.clinical_concept_activation),
        )
        object.__setattr__(
            self,
            "clinical_annotation_score",
            _immutable_vector(self.clinical_annotation_score),
        )
        object.__setattr__(self, "concept_labels", tuple(self.concept_labels))
        if self.annotation_kind != "derived clinical annotation":
            raise ValueError("clinical result must retain the derived-annotation label")
