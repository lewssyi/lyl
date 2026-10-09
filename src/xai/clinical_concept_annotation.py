"""Post-attribution clinical concept annotation with no ranking feedback path."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .attribution_types import ClinicalAnnotationResult, ModelAttributionResult


def annotate_clinical_concepts(
    attribution: ModelAttributionResult,
    concept_activation,
    concept_labels: Sequence[str],
    feature_to_concept: np.ndarray,
    top_k: int = 10,
    metadata: Mapping[str, object] | None = None,
) -> ClinicalAnnotationResult:
    """Annotate selected regions after attribution, never changing its scores."""
    before = attribution.model_attribution_score.copy()
    selected = tuple(
        int(value)
        for value in np.argsort(-np.abs(before), kind="stable")[: max(1, int(top_k))]
    )
    activations = np.asarray(concept_activation, dtype=np.float32).ravel()
    mapping = np.asarray(feature_to_concept, dtype=np.float32)
    if mapping.shape != (before.size, activations.size):
        raise ValueError("feature_to_concept must have [features, concepts] shape")
    region_weights = np.abs(before[list(selected)])
    selected_mapping = np.abs(mapping[list(selected)])
    evidence = np.sum(region_weights[:, None] * selected_mapping, axis=0) * activations
    if not np.array_equal(before, attribution.model_attribution_score):
        raise RuntimeError("clinical annotation modified model attribution")
    return ClinicalAnnotationResult(
        sample_id=attribution.sample_id,
        source_attribution_method=attribution.method_id,
        selected_feature_indices=selected,
        clinical_concept_activation=activations,
        clinical_annotation_score=evidence,
        concept_labels=tuple(concept_labels),
        metadata={
            "predicted_stage_usage": "display_context_only",
            "true_stage_used": False,
            "ranking_modified": False,
            **dict(metadata or {}),
        },
    )
