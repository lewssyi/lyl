"""Pure, auditable components for the Sequence-LIME A/B/C diagnosis."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


EPS = 1e-8
MODEL_DRIVEN_METHODS = ("M0", "M1", "M2", "M3", "M4")
CONCEPT_ONLY_METHODS = ("M5", "M6")


def positive_rank_score(values) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).ravel()
    values = np.clip(raw, 0.0, None)
    if float(values.max(initial=0.0)) <= 0.0:
        values = np.abs(raw)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > EPS else np.zeros_like(values)


def transform_surrogate_targets(
    probability_vectors: np.ndarray,
    target_class: int,
) -> Dict[str, np.ndarray]:
    """Return probability, centered-logit, and logit-margin targets.

    A softmax output determines logits only up to an additive constant. Y1 uses
    centered log probabilities, while Y2 is the exactly identifiable margin.
    """
    probabilities = np.clip(np.asarray(probability_vectors, dtype=np.float64), EPS, 1.0)
    log_probabilities = np.log(probabilities)
    centered_logits = log_probabilities - np.mean(log_probabilities, axis=1, keepdims=True)
    other = np.delete(centered_logits, int(target_class), axis=1)
    return {
        "Y0": probabilities[:, int(target_class)].astype(np.float32),
        "Y1": centered_logits[:, int(target_class)].astype(np.float32),
        "Y2": (
            centered_logits[:, int(target_class)] - np.max(other, axis=1)
        ).astype(np.float32),
    }


def resolve_target_protocols(
    original_probability: np.ndarray,
    randomized_probability: np.ndarray,
) -> Dict[str, int]:
    return {
        "T0": int(np.argmax(np.asarray(original_probability))),
        "T1": int(np.argmax(np.asarray(randomized_probability))),
    }


def compose_diagnostic_methods(
    classic_score,
    zero_marginal_score,
    physiological_marginal_score,
    predicted_stage_prior_score,
    stage_independent_concept_score,
    signal_energy_score,
    random_score,
) -> Dict[str, np.ndarray]:
    """Construct M0--M8 without silently sharing a concept-modified ranking."""
    classic = positive_rank_score(classic_score)
    zero = positive_rank_score(zero_marginal_score)
    physiological = positive_rank_score(physiological_marginal_score)
    predicted_prior = positive_rank_score(predicted_stage_prior_score)
    independent = positive_rank_score(stage_independent_concept_score)
    energy = positive_rank_score(signal_energy_score)
    random = positive_rank_score(random_score)
    return {
        "M0": classic.copy(),
        "M1": positive_rank_score(0.25 * classic + 0.75 * zero),
        "M2": positive_rank_score(0.25 * classic + 0.75 * physiological),
        "M3": positive_rank_score(0.20 * classic + 0.40 * zero + 0.40 * physiological),
        "M4": positive_rank_score(
            0.15 * classic + 0.35 * zero + 0.30 * physiological + 0.20 * predicted_prior
        ),
        "M5": predicted_prior.copy(),
        "M6": independent.copy(),
        "M7": energy.copy(),
        "M8": random.copy(),
    }


def grouped_temporal_masks(
    feature_count: int,
    sample_count: int,
    seed: int,
    group_size: int = 5,
) -> np.ndarray:
    """B3 candidate: contiguous grouped masks with explicit endpoints."""
    feature_count = int(feature_count)
    group_size = max(1, int(group_size))
    group_count = int(np.ceil(feature_count / group_size))
    rng = np.random.default_rng(int(seed))
    group_masks = rng.integers(0, 2, size=(max(int(sample_count) - 2, 0), group_count))
    masks = np.repeat(group_masks, group_size, axis=1)[:, :feature_count]
    return np.vstack([
        np.ones((1, feature_count), dtype=np.int32),
        np.zeros((1, feature_count), dtype=np.int32),
        masks.astype(np.int32),
    ])


def assert_validation_only_selection(records: Sequence[Mapping[str, object]]) -> None:
    forbidden = [row for row in records if str(row.get("selection_split", "")).lower() == "test"]
    if forbidden:
        raise RuntimeError("test set must not participate in method or weight selection")


def select_validation_candidate(
    records: Sequence[Mapping[str, object]],
    required_seeds: int = 2,
) -> Tuple[str, Dict[str, float]]:
    """Select only candidates meeting all non-inferiority gates on validation."""
    assert_validation_only_selection(records)
    grouped: Dict[str, list] = {}
    for row in records:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    eligible = []
    for candidate, rows in grouped.items():
        passing_seeds = {
            int(row["randomization_seed"])
            for row in rows
            if bool(row.get("fidelity_better"))
            and bool(row.get("faithfulness_noninferior"))
            and bool(row.get("stability_noninferior"))
            and bool(row.get("randomization_pass"))
            and bool(row.get("equal_query_budget"))
        }
        if len(passing_seeds) >= int(required_seeds):
            score = float(np.mean([float(row.get("validation_score", 0.0)) for row in rows]))
            eligible.append((score, candidate, {"passing_seed_count": float(len(passing_seeds))}))
    if not eligible:
        return "NO_MF_SEQLIME_CANDIDATE", {"passing_seed_count": 0.0}
    _, candidate, summary = max(eligible, key=lambda item: (item[0], item[1]))
    return candidate, summary
