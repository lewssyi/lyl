"""Authoritative guardrails for the frozen post-hoc XAI release.

The canonical method is intentionally limited to exploratory local
perturbation sensitivity.  It is not a training target and must never be
connected to the classifier loss.
"""
from __future__ import annotations

from typing import Optional

CANONICAL_POSTHOC_METHOD = "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME"
CANONICAL_DISPLAY_NAME = "Context-Preserving Classic Sequence-LIME"
METHOD_ROLE = "EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY"
SCIENTIFIC_STATUS = "MODEL_FAITHFULNESS_NOT_ESTABLISHED"
FEEDBACK_TRAINING_STATUS = "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD"

FEEDBACK_BLOCK_MESSAGE = (
    "Feedback training is not permitted for the current post-hoc method: "
    "authoritative model faithfulness has not been established."
)

DUMMY_CONTROL_TARGETS = frozenset(
    {"random_target_control", "shuffled_target_control"}
)


def assert_feedback_training_disabled(
    *,
    enabled: bool = False,
    loss_weight: float = 0.0,
    target_path: Optional[str] = None,
) -> None:
    """Fail before data/model loading if a real feedback path is requested."""
    if bool(enabled) or float(loss_weight) != 0.0 or bool(str(target_path or "").strip()):
        raise RuntimeError(FEEDBACK_BLOCK_MESSAGE)


def assert_offline_target_is_dummy_control(target_type: str) -> None:
    """Permit only random/shuffled artifacts for isolated unit controls."""
    if str(target_type).strip().lower() not in DUMMY_CONTROL_TARGETS:
        raise RuntimeError(FEEDBACK_BLOCK_MESSAGE)


def assert_model_xai_loss_is_zero(loss_weight: float) -> None:
    """Allow a forward-only head at lambda=0, but prohibit optimization."""
    if float(loss_weight) != 0.0:
        raise RuntimeError(FEEDBACK_BLOCK_MESSAGE)
