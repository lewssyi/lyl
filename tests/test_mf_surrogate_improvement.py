from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.xai.neighborhood_sampling import (
    calibration_masks,
    candidate_budget,
    compute_weak_zero_calibration,
    generate_contiguous_block_masks,
    generate_stratified_masks,
    generate_structured_masks,
    refine_boundary_masks,
)
from src.xai.surrogate_selection import (
    permutation_similarity_null,
    score_and_gate_candidates,
)


def test_stratified_mask_budget_and_binary_values():
    result = generate_stratified_masks(90, 503, seed=11)
    assert result.masks.shape == (503, 90)
    assert set(np.unique(result.masks)) == {0, 1}


def test_stratified_contains_both_explicit_endpoints():
    result = generate_stratified_masks(90, 10, seed=12)
    assert np.all(result.masks[0] == 1)
    assert np.all(result.masks[1] == 0)


def test_stratified_repeats_exactly_for_same_seed():
    first = generate_stratified_masks(90, 100, seed=13)
    second = generate_stratified_masks(90, 100, seed=13)
    np.testing.assert_array_equal(first.masks, second.masks)


def test_stratified_changes_for_different_seed():
    first = generate_stratified_masks(90, 100, seed=13)
    second = generate_stratified_masks(90, 100, seed=14)
    assert not np.array_equal(first.masks, second.masks)


def test_stratified_rejects_invalid_retention_level():
    with pytest.raises(ValueError, match="retention"):
        generate_stratified_masks(90, 10, seed=1, retention_levels=(0.0,))


def test_contiguous_masks_delete_only_contiguous_time_blocks():
    result = generate_contiguous_block_masks(3, 30, 120, seed=15)
    for row in result.masks.reshape(-1, 3, 30):
        for channel in row:
            deleted = np.flatnonzero(channel == 0)
            if deleted.size:
                np.testing.assert_array_equal(deleted, np.arange(deleted[0], deleted[-1] + 1))


def test_structured_masks_are_unique_and_exact_budget():
    result = generate_structured_masks(3, 30, 1000, seed=16, contiguous_fraction=0.5)
    assert result.masks.shape == (1000, 90)
    assert len({row.tobytes() for row in result.masks}) == 1000


def test_structured_mask_mix_records_requested_fraction():
    result = generate_structured_masks(3, 30, 101, seed=17, contiguous_fraction=0.3)
    assert result.provenance["requested_contiguous_fraction"] == pytest.approx(0.3)
    assert result.provenance["contiguous_count"] == 30


def test_boundary_refinement_uses_low_absolute_margin_pool():
    base = generate_structured_masks(3, 30, 100, seed=18).masks
    margins = np.linspace(-2.0, 2.0, len(base))
    result = refine_boundary_masks(base, margins, 30, seed=19)
    assert result.masks.shape == (30, 90)
    expected = np.sort(np.abs(margins))[:10].max()
    assert result.provenance["absolute_margin_threshold"] == pytest.approx(expected)


def test_boundary_refinement_repeats_for_same_seed():
    base = generate_structured_masks(3, 30, 50, seed=20).masks
    margins = np.arange(50, dtype=float) - 25
    first = refine_boundary_masks(base, margins, 20, seed=21)
    second = refine_boundary_masks(base, margins, 20, seed=21)
    np.testing.assert_array_equal(first.masks, second.masks)


def test_boundary_refinement_rejects_budget_overrun():
    base = generate_structured_masks(3, 30, 50, seed=22).masks
    with pytest.raises(ValueError, match="budget"):
        refine_boundary_masks(base, np.ones(50), 11, seed=23, maximum_total_queries=60)


def test_calibration_masks_have_required_182_queries():
    result = calibration_masks(90)
    assert result.masks.shape == (182, 90)
    assert np.all(result.masks[0] == 1) and np.all(result.masks[1] == 0)


def test_classic_budget_is_exactly_1530():
    budget = candidate_budget()
    assert budget == {
        "total_query_budget": 1530,
        "neighborhood_query_budget": 1506,
        "model_driven_neighborhood_queries": 1506,
        "zero_calibration_queries": 0,
        "faithfulness_queries": 24,
    }


def test_weak_zero_budget_deducts_calibration_from_neighborhood():
    budget = candidate_budget(zero_calibration_queries=182)
    assert budget["model_driven_neighborhood_queries"] == 1324
    assert sum((budget["model_driven_neighborhood_queries"], 182, 24)) == 1530


def test_budget_rejects_hidden_extra_queries():
    with pytest.raises(ValueError, match="invalid"):
        candidate_budget(total_query_budget=100, faithfulness_query_budget=24, zero_calibration_queries=90)


def test_alpha_zero_is_exact_normalized_model_driven_endpoint():
    lime = np.asarray([0.0, 1.0, 2.0])
    zero = np.asarray([2.0, 0.0, 1.0])
    np.testing.assert_allclose(compute_weak_zero_calibration(lime, zero, 0.0), [0.0, 0.5, 1.0])


def test_weak_zero_alpha_changes_score_without_extra_inputs():
    lime = np.asarray([0.0, 1.0, 2.0])
    zero = np.asarray([2.0, 0.0, 1.0])
    assert not np.array_equal(
        compute_weak_zero_calibration(lime, zero, 0.1),
        compute_weak_zero_calibration(lime, zero, 0.3),
    )


def test_weak_zero_rejects_unregistered_alpha():
    with pytest.raises(ValueError, match="alpha"):
        compute_weak_zero_calibration(np.ones(3), np.ones(3), 0.25)


def _summary_rows():
    return pd.DataFrame([
        {
            "candidate": "C0_CLASSIC_Y2", "validation_weighted_r2": 0.40,
            "validation_mae": 0.10, "validation_rmse": 0.12,
            "deletion_auc": 0.50, "insertion_auc": 0.60,
            "faithfulness_advantage_r0": 0.08, "faithfulness_advantage_r7": 0.01,
            "stability_spearman": 0.60, "r7_similarity": 0.10, "r7_null_q95": 0.20,
            "r7_t0_seed_pass_count": 3, "r7_t1_seed_pass_count": 3, "query_count": 1530,
        },
        {
            "candidate": "C2_BOUNDARY_M50", "validation_weighted_r2": 0.41,
            "validation_mae": 0.09, "validation_rmse": 0.11,
            "deletion_auc": 0.49, "insertion_auc": 0.60,
            "faithfulness_advantage_r0": 0.09, "faithfulness_advantage_r7": 0.02,
            "stability_spearman": 0.65, "r7_similarity": 0.08, "r7_null_q95": 0.20,
            "r7_t0_seed_pass_count": 2, "r7_t1_seed_pass_count": 2, "query_count": 1530,
        },
    ])


def test_validation_gate_stops_before_test_when_none_qualify():
    frame = _summary_rows()
    frame.loc[frame.candidate == "C2_BOUNDARY_M50", "validation_mae"] = 0.11
    table, decision = score_and_gate_candidates(frame)
    assert not table.loc[table.candidate == "C2_BOUNDARY_M50", "hard_gate_pass"].item()
    assert decision["status"] == "VALIDATION_NO_QUALIFIED_CANDIDATE"
    assert decision["test_evaluation_allowed"] is False


def test_validation_gate_freezes_qualified_candidate_without_test_data():
    table, decision = score_and_gate_candidates(_summary_rows())
    assert table.loc[table.candidate == "C2_BOUNDARY_M50", "selected"].item()
    assert decision["selected_candidate"] == "C2_BOUNDARY_M50"
    assert decision["test_data_used"] is False


def test_permutation_null_is_1000_draw_reproducible():
    first = np.arange(20, dtype=float)
    second = first[::-1]
    one = permutation_similarity_null(first, second, seed=24)
    two = permutation_similarity_null(first, second, seed=24)
    assert one["permutations"] == 1000
    np.testing.assert_array_equal(one["null_values"], two["null_values"])
