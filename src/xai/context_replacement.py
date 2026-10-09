"""Deterministic, label-free replacements for hierarchical context attribution."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


ContextGroup = Tuple[int, int]


def contiguous_context_groups(length: int = 75, group_count: int = 15) -> Tuple[ContextGroup, ...]:
    """Partition every position into contiguous, nonempty near-equal groups."""
    length, group_count = int(length), int(group_count)
    if length < 1 or not 1 <= group_count <= length:
        raise ValueError("group_count must be between one and length")
    indices = np.array_split(np.arange(length, dtype=np.int32), group_count)
    groups = tuple((int(part[0]), int(part[-1]) + 1) for part in indices)
    covered = [index for start, stop in groups for index in range(start, stop)]
    if covered != list(range(length)):
        raise RuntimeError("context groups must cover each position exactly once")
    return groups


def target_group_index(groups: Sequence[ContextGroup], target_position: int = 37) -> int:
    matches = [index for index, (start, stop) in enumerate(groups) if start <= int(target_position) < stop]
    if len(matches) != 1:
        raise ValueError("target position must belong to exactly one context group")
    return int(matches[0])


def group_offsets(groups: Sequence[ContextGroup], group_index: int, center: int = 37) -> Tuple[int, ...]:
    start, stop = groups[int(group_index)]
    return tuple(int(position - center) for position in range(start, stop))


def neutral_epoch_index(raw_epochs: np.ndarray) -> int:
    """Choose a recording-neutral epoch without labels or predictions."""
    raw = np.asarray(raw_epochs, dtype=np.float32)
    if raw.ndim != 3 or len(raw) == 0:
        raise ValueError("raw_epochs must be [epochs,time,channels]")
    energy = np.mean(np.square(raw, dtype=np.float64), axis=(1, 2))
    median = float(np.median(energy))
    return int(np.argmin(np.abs(energy - median)))


def replacement_epoch(
    raw_epochs: np.ndarray,
    segment_epochs: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    raw = np.asarray(raw_epochs, dtype=np.float32)
    segment = np.asarray(segment_epochs, dtype=np.float32)
    if len(raw) != len(segment) or len(raw) == 0:
        raise ValueError("raw and segment recordings must align")
    if mode == "zero":
        return np.zeros_like(raw[0]), np.zeros_like(segment[0]), None
    if mode == "local_temporal_mean":
        return np.mean(raw, axis=0, dtype=np.float64).astype(np.float32), np.mean(segment, axis=0, dtype=np.float64).astype(np.float32), None
    if mode == "recording_neutral_epoch":
        index = neutral_epoch_index(raw)
        return raw[index].copy(), segment[index].copy(), index
    raise ValueError(f"unknown replacement mode: {mode}")


def apply_context_mask_to_window(
    raw_window: np.ndarray,
    segment_window: np.ndarray,
    global_indices: Sequence[int],
    target_epoch: int,
    group_mask: Sequence[int],
    groups: Sequence[ContextGroup],
    replacement_raw: np.ndarray,
    replacement_segment: np.ndarray,
    keep_target: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace masked relative-offset groups while leaving all other epochs exact."""
    raw = np.asarray(raw_window, dtype=np.float32).copy()
    segment = np.asarray(segment_window, dtype=np.float32).copy()
    original_raw, original_segment = raw.copy(), segment.copy()
    mask = np.asarray(group_mask, dtype=np.int8).ravel()
    if len(mask) != len(groups):
        raise ValueError("group mask length mismatch")
    offset_to_group = {}
    for group_index, (start, stop) in enumerate(groups):
        for position in range(start, stop):
            offset_to_group[int(position - 37)] = int(group_index)
    for row, global_index in enumerate(global_indices):
        offset = int(global_index) - int(target_epoch)
        group_index = offset_to_group.get(offset)
        if group_index is None or mask[group_index] == 1:
            continue
        if keep_target and int(global_index) == int(target_epoch):
            continue
        raw[row] = replacement_raw
        segment[row] = replacement_segment
    changed = np.any(raw != original_raw, axis=tuple(range(1, raw.ndim)))
    for row, was_changed in enumerate(changed):
        if not was_changed and not np.array_equal(segment[row], original_segment[row]):
            raise RuntimeError("raw/segment replacement mismatch")
    return raw, segment
