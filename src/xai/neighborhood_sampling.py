"""Auditable local-neighborhood samplers for MF-SeqLIME.

Every sampler receives an explicit seed and budget.  No class label, stage
prior, clinical annotation, or global NumPy RNG is available to this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_RETENTION_LEVELS = (0.90, 0.75, 0.50, 0.25, 0.10)
DEFAULT_BLOCK_LENGTHS = (1, 2, 3, 5, 8, 10)


@dataclass(frozen=True)
class NeighborhoodMasks:
    masks: np.ndarray
    mask_types: tuple[str, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        masks = np.asarray(self.masks, dtype=np.int8).copy()
        if masks.ndim != 2 or masks.shape[0] != len(self.mask_types):
            raise ValueError("masks must be [queries, features] and match mask_types")
        if not np.all((masks == 0) | (masks == 1)):
            raise ValueError("masks must be binary")
        masks.setflags(write=False)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "mask_types", tuple(self.mask_types))


def _validate(feature_count: int, sample_count: int) -> tuple[int, int]:
    feature_count = int(feature_count)
    sample_count = int(sample_count)
    if feature_count < 2 or sample_count < 1:
        raise ValueError("feature_count >= 2 and sample_count >= 1 are required")
    return feature_count, sample_count


def _retention_label(level: float) -> str:
    return f"retention_{int(round(float(level) * 100)):02d}"


def generate_stratified_masks(
    feature_count: int,
    sample_count: int,
    seed: int,
    retention_levels: Sequence[float] = DEFAULT_RETENTION_LEVELS,
    include_endpoints: bool = True,
) -> NeighborhoodMasks:
    """Generate balanced masks across explicit retained-feature fractions."""
    feature_count, sample_count = _validate(feature_count, sample_count)
    levels = tuple(float(value) for value in retention_levels)
    if not levels or any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError("retention levels must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    rows: list[np.ndarray] = []
    types: list[str] = []
    seen: set[bytes] = set()

    def add(candidate: np.ndarray, kind: str) -> bool:
        key = np.asarray(candidate, dtype=np.int8).tobytes()
        if key in seen:
            return False
        seen.add(key)
        rows.append(np.asarray(candidate, dtype=np.int8))
        types.append(kind)
        return True

    if include_endpoints:
        add(np.ones(feature_count, dtype=np.int8), "original_all_retained")
        if len(rows) < sample_count:
            add(np.zeros(feature_count, dtype=np.int8), "all_deleted_endpoint")
    cursor = 0
    attempts = 0
    while len(rows) < sample_count:
        level = levels[cursor % len(levels)]
        cursor += 1
        retained = int(np.clip(round(level * feature_count), 1, feature_count - 1))
        candidate = np.zeros(feature_count, dtype=np.int8)
        candidate[rng.choice(feature_count, size=retained, replace=False)] = 1
        add(candidate, _retention_label(level))
        attempts += 1
        if attempts > sample_count * 100:
            raise RuntimeError("unable to generate enough unique stratified masks")
    masks = np.asarray(rows, dtype=np.int8)
    counts = {kind: int(types.count(kind)) for kind in sorted(set(types))}
    return NeighborhoodMasks(
        masks=masks,
        mask_types=tuple(types),
        provenance={
            "schema_version": "stratified_retention_masks_v1",
            "strategy": "S1_STRATIFIED_RETENTION",
            "seed": int(seed),
            "query_budget": sample_count,
            "feature_count": feature_count,
            "retention_levels": list(levels),
            "mask_type_counts": counts,
            "global_rng_used": False,
            "true_label_used": False,
            "stage_prior_used": False,
        },
    )


def generate_contiguous_block_masks(
    channel_count: int,
    segment_count: int,
    sample_count: int,
    seed: int,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
) -> NeighborhoodMasks:
    """Delete short/medium/long contiguous blocks on one or more channels."""
    channel_count = int(channel_count)
    segment_count = int(segment_count)
    _, sample_count = _validate(channel_count * segment_count, sample_count)
    lengths = tuple(sorted({int(value) for value in block_lengths if 0 < int(value) <= segment_count}))
    if not lengths:
        raise ValueError("at least one valid block length is required")
    rng = np.random.default_rng(int(seed))
    pool_rows: list[np.ndarray] = []
    pool_types: list[str] = []
    for length in lengths:
        for start in range(segment_count - length + 1):
            for bits in range(1, 1 << channel_count):
                channels = [channel for channel in range(channel_count) if bits & (1 << channel)]
                scope = "single_channel" if len(channels) == 1 else "multi_channel"
                matrix = np.ones((channel_count, segment_count), dtype=np.int8)
                matrix[channels, start:start + length] = 0
                pool_rows.append(matrix.ravel())
                pool_types.append(f"contiguous_{scope}_length_{length}")
    if sample_count > len(pool_rows):
        raise ValueError(
            f"requested {sample_count} unique contiguous masks but only {len(pool_rows)} exist"
        )
    selected = rng.choice(len(pool_rows), size=sample_count, replace=False)
    rows = [pool_rows[int(index)] for index in selected]
    types = [pool_types[int(index)] for index in selected]
    counts = {kind: int(types.count(kind)) for kind in sorted(set(types))}
    return NeighborhoodMasks(
        masks=np.asarray(rows, dtype=np.int8),
        mask_types=tuple(types),
        provenance={
            "schema_version": "contiguous_block_masks_v1",
            "strategy": "S2_CONTIGUOUS_BLOCK",
            "seed": int(seed),
            "query_budget": sample_count,
            "channel_count": channel_count,
            "segment_count": segment_count,
            "block_lengths": list(lengths),
            "mask_type_counts": counts,
            "temporal_contiguity_enforced": True,
            "true_label_used": False,
            "stage_prior_used": False,
        },
    )


def generate_structured_masks(
    channel_count: int,
    segment_count: int,
    sample_count: int,
    seed: int,
    contiguous_fraction: float = 0.50,
    include_endpoints: bool = True,
) -> NeighborhoodMasks:
    """Mix S1 independent masks with S2 contiguous masks at a fixed budget."""
    feature_count = int(channel_count) * int(segment_count)
    _, sample_count = _validate(feature_count, sample_count)
    fraction = float(contiguous_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("contiguous_fraction must be in [0, 1]")
    contiguous_n = int(round(sample_count * fraction))
    independent_n = sample_count - contiguous_n
    minimum_independent = 2 if include_endpoints and sample_count >= 2 else 1
    independent_n = max(minimum_independent, independent_n)
    contiguous_n = sample_count - independent_n
    independent = generate_stratified_masks(
        feature_count, independent_n, seed=int(seed), include_endpoints=include_endpoints
    )
    if contiguous_n:
        contiguous = generate_contiguous_block_masks(
            channel_count, segment_count, contiguous_n, seed=int(seed) + 100003
        )
        combined_rows = [row.copy() for row in independent.masks]
        combined_types = list(independent.mask_types)
        seen = {row.tobytes() for row in combined_rows}
        for row, kind in zip(contiguous.masks, contiguous.mask_types):
            if row.tobytes() not in seen:
                seen.add(row.tobytes())
                combined_rows.append(row.copy())
                combined_types.append(kind)
        replacement_round = 0
        while len(combined_rows) < sample_count:
            needed = sample_count - len(combined_rows)
            replacement = generate_stratified_masks(
                feature_count,
                max(needed + 2, 3),
                seed=int(seed) + 200003 + replacement_round,
                include_endpoints=False,
            )
            for row, kind in zip(replacement.masks, replacement.mask_types):
                if row.tobytes() not in seen:
                    seen.add(row.tobytes())
                    combined_rows.append(row.copy())
                    combined_types.append(f"deduplicated_{kind}")
                    if len(combined_rows) == sample_count:
                        break
            replacement_round += 1
            if replacement_round > 100:
                raise RuntimeError("unable to fill structured mask budget uniquely")
        masks = np.asarray(combined_rows, dtype=np.int8)
        types = tuple(combined_types)
    else:
        masks = independent.masks.copy()
        types = independent.mask_types
    counts = {kind: int(types.count(kind)) for kind in sorted(set(types))}
    return NeighborhoodMasks(
        masks=masks,
        mask_types=types,
        provenance={
            "schema_version": "structured_neighborhood_masks_v1",
            "strategy": "S1_STRATIFIED_RETENTION_PLUS_S2_CONTIGUOUS_BLOCK",
            "seed": int(seed),
            "query_budget": sample_count,
            "requested_contiguous_fraction": fraction,
            "actual_contiguous_fraction": float(contiguous_n / sample_count),
            "independent_count": independent_n,
            "contiguous_count": contiguous_n,
            "mask_type_counts": counts,
            "true_label_used": False,
            "stage_prior_used": False,
        },
    )


def refine_boundary_masks(
    base_masks: np.ndarray,
    logit_margins: np.ndarray,
    additional_count: int,
    seed: int,
    channel_count: int = 3,
    segment_count: int = 30,
    boundary_fraction: float = 0.10,
    maximum_total_queries: int | None = None,
) -> NeighborhoodMasks:
    """Generate local neighbors around low-|margin| masks without labels."""
    base = np.asarray(base_masks, dtype=np.int8)
    margins = np.asarray(logit_margins, dtype=np.float64).ravel()
    additional_count = int(additional_count)
    if base.ndim != 2 or len(base) != len(margins):
        raise ValueError("base_masks and logit_margins must have matching rows")
    if base.shape[1] != int(channel_count) * int(segment_count):
        raise ValueError("feature layout does not match channel_count × segment_count")
    if additional_count < 0:
        raise ValueError("additional_count must be nonnegative")
    if maximum_total_queries is not None and len(base) + additional_count > int(maximum_total_queries):
        raise ValueError("boundary refinement exceeds explicit query budget")
    if additional_count == 0:
        return NeighborhoodMasks(
            masks=np.empty((0, base.shape[1]), dtype=np.int8),
            mask_types=(),
            provenance={
                "schema_version": "boundary_adaptive_masks_v1",
                "strategy": "S3_BOUNDARY_ADAPTIVE",
                "seed": int(seed), "query_budget": 0, "true_label_used": False,
            },
        )
    fraction = float(boundary_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("boundary_fraction must be in (0, 1]")
    boundary_n = max(1, int(np.ceil(len(base) * fraction)))
    parent_indices = np.argsort(np.abs(margins), kind="stable")[:boundary_n]
    rng = np.random.default_rng(int(seed))
    rows: list[np.ndarray] = []
    types: list[str] = []
    seen = {row.tobytes() for row in base}
    attempts = 0
    while len(rows) < additional_count:
        parent = base[int(rng.choice(parent_indices))].reshape(channel_count, segment_count).copy()
        operation = len(rows) % 3
        start = int(rng.integers(0, segment_count))
        length = int(rng.choice([1, 2, 3, 5]))
        end = min(segment_count, start + length)
        if operation == 0:
            channel = int(rng.integers(0, channel_count))
            parent[channel, start:end] = 1 - parent[channel, start:end]
            kind = "boundary_adjacent_flip"
        elif operation == 1:
            channel = int(rng.integers(0, channel_count))
            value = int(rng.integers(0, 2))
            parent[channel, max(0, start - 1):min(segment_count, end + 1)] = value
            kind = "boundary_block_expand_or_shrink"
        else:
            width = int(rng.integers(1, channel_count + 1))
            channels = rng.choice(channel_count, size=width, replace=False)
            parent[channels, start:end] = 1 - parent[channels, start:end]
            kind = "boundary_channel_combination"
        candidate = parent.ravel().astype(np.int8)
        key = candidate.tobytes()
        if key not in seen:
            seen.add(key)
            rows.append(candidate)
            types.append(kind)
        attempts += 1
        if attempts > max(1000, additional_count * 500):
            raise RuntimeError("unable to generate unique boundary-adaptive masks")
    counts = {kind: int(types.count(kind)) for kind in sorted(set(types))}
    threshold = float(np.max(np.abs(margins[parent_indices])))
    return NeighborhoodMasks(
        masks=np.asarray(rows, dtype=np.int8),
        mask_types=tuple(types),
        provenance={
            "schema_version": "boundary_adaptive_masks_v1",
            "strategy": "S3_BOUNDARY_ADAPTIVE",
            "seed": int(seed),
            "query_budget": additional_count,
            "boundary_fraction": fraction,
            "boundary_parent_count": boundary_n,
            "absolute_margin_threshold": threshold,
            "mask_type_counts": counts,
            "true_label_used": False,
            "stage_prior_used": False,
        },
    )


def calibration_masks(feature_count: int) -> NeighborhoodMasks:
    """Return original/full and all one-feature deletion/insertion masks."""
    feature_count, _ = _validate(feature_count, 1)
    identity = np.eye(feature_count, dtype=np.int8)
    masks = np.vstack([
        np.ones((1, feature_count), dtype=np.int8),
        np.zeros((1, feature_count), dtype=np.int8),
        np.ones((feature_count, feature_count), dtype=np.int8) - identity,
        identity,
    ])
    types = (
        "calibration_original", "calibration_all_deleted",
        *(("calibration_leave_one_out",) * feature_count),
        *(("calibration_one_only",) * feature_count),
    )
    return NeighborhoodMasks(
        masks=masks,
        mask_types=types,
        provenance={
            "schema_version": "zero_calibration_masks_v1",
            "query_budget": int(len(masks)),
            "feature_count": feature_count,
            "current_evaluated_model_required": True,
            "original_model_cache_allowed": False,
            "true_label_used": False,
            "stage_prior_used": False,
        },
    )


def candidate_budget(
    total_query_budget: int = 1530,
    faithfulness_query_budget: int = 24,
    zero_calibration_queries: int = 0,
) -> Mapping[str, int]:
    """Allocate one equal per-method budget and reject hidden extra queries."""
    total = int(total_query_budget)
    faith = int(faithfulness_query_budget)
    zero = int(zero_calibration_queries)
    neighborhood = total - faith
    model_driven = neighborhood - zero
    if min(total, faith, neighborhood, model_driven) < 0:
        raise ValueError("invalid query budget allocation")
    return {
        "total_query_budget": total,
        "neighborhood_query_budget": neighborhood,
        "model_driven_neighborhood_queries": model_driven,
        "zero_calibration_queries": zero,
        "faithfulness_queries": faith,
    }


def compute_weak_zero_calibration(
    model_driven_lime_score: np.ndarray,
    zero_marginal_score: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse only model-driven LIME and current-model zero marginal."""
    alpha = float(alpha)
    if alpha not in {0.0, 0.10, 0.20, 0.30}:
        raise ValueError("alpha must be one of 0.00, 0.10, 0.20, 0.30")

    def normalize(values):
        values = np.asarray(values, dtype=np.float32).ravel()
        positive = np.clip(values, 0.0, None)
        if float(positive.max(initial=0.0)) <= 0.0:
            positive = np.abs(values)
        maximum = float(positive.max(initial=0.0))
        return positive / maximum if maximum > 1e-8 else np.zeros_like(positive)

    lime = normalize(model_driven_lime_score)
    if alpha == 0.0:
        return lime
    zero = normalize(zero_marginal_score)
    return normalize((1.0 - alpha) * lime + alpha * zero)
