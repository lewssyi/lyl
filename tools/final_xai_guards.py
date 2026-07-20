#!/usr/bin/env python
"""Run isolated canonical parity and 1,200-forward fixed-graph memory guard."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ROOT = PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42/final_validation"
PARITY = ROOT / "parity"
FORMAL_CACHE = PROJECT / "outputs/sequence_lime_paper_evaluation/isruc_seed42/cache/test_inputs_segments.npz"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parity_args() -> argparse.Namespace:
    return argparse.Namespace(
        mode="parity-diagnostic",
        dataset="isruc",
        seed=42,
        config=None,
        run_id="two_version_final_parity_batch16",
        evaluation_shard_count=1,
        evaluation_shard_index=0,
        window_inference_batch_size=16,
        lime_train_masks=None,
        lime_validation_masks=None,
        lime_target_segments=None,
        lime_mask_sampling=None,
        lime_response_transform=None,
        lime_surrogate_perturbation_mode=None,
        resume=False,
    )


def prepare_isolated_cache() -> None:
    cache = PARITY / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "test_inputs_segments.npz"
    if not target.exists():
        target.symlink_to(FORMAL_CACHE)


def memory_guard() -> dict[str, object]:
    z = np.load(evaluator.PRED_PATH)
    artifact = evaluator.load_artifact_split(z, "test")
    split_files = evaluator.normalize_split_files(evaluator.load_split_spec())
    x_raw, x_seg, _ = evaluator.load_or_build_test_inputs(PARITY, split_files, artifact)
    model = evaluator.build_model(x_raw.shape[1:], x_seg.shape[1:])
    weights_before = evaluator.model_weight_fingerprint(model)
    checkpoint_before = evaluator.file_sha256(evaluator.CKPT_PATH)
    raw_batch = np.asarray(x_raw[:75][None], dtype=np.float32)
    segment_batch = np.asarray(x_seg[:75][None], dtype=np.float32)
    for _ in range(20):
        evaluator.compiled_perturbation_output(model, raw_batch, segment_batch)
    device = "GPU:0"
    tf.config.experimental.reset_memory_stats(device)
    samples = []
    for index in range(1, 1201):
        output = evaluator.compiled_perturbation_output(model, raw_batch, segment_batch)
        if not np.all(np.isfinite(output)):
            raise RuntimeError("non-finite fixed-graph output")
        if index == 1 or index % 100 == 0:
            info = tf.config.experimental.get_memory_info(device)
            samples.append({
                "forward": index,
                "current_bytes": int(info["current"]),
                "peak_bytes": int(info["peak"]),
            })
    current = np.asarray([row["current_bytes"] for row in samples], dtype=np.int64)
    growth = int(current[-1] - current[0])
    late_span = int(np.max(current[len(current) // 2:]) - np.min(current[len(current) // 2:]))
    checkpoint_after = evaluator.file_sha256(evaluator.CKPT_PATH)
    weights_after = evaluator.model_weight_fingerprint(model)
    passed = bool(
        growth <= 1024 * 1024
        and late_span <= 1024 * 1024
        and checkpoint_before == checkpoint_after
        and weights_before == weights_after
    )
    return {
        "schema_version": "fixed_graph_memory_guard_v1",
        "status": "PASS" if passed else "FAIL",
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "forward_count": 1200,
        "warmup_forward_count": 20,
        "fixed_signature": True,
        "memory_samples": samples,
        "current_memory_growth_bytes": growth,
        "late_half_current_memory_span_bytes": late_span,
        "allowed_growth_bytes": 1024 * 1024,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "model_weights_unchanged": weights_before == weights_after,
    }


def main() -> None:
    prepare_isolated_cache()
    original_root = evaluator.OUT_ROOT
    evaluator.OUT_ROOT = PARITY
    try:
        parity = evaluator.run_parity_diagnostic(parity_args())
        memory = memory_guard()
    finally:
        evaluator.OUT_ROOT = original_root
    write_json(ROOT / "fixed_graph_memory_1200.json", memory)
    overall = {
        "schema_version": "two_version_final_guards_v1",
        "status": "PASS" if parity.get("xai_evaluation_allowed") and memory["status"] == "PASS" else "FAIL",
        "canonical_parity": parity,
        "fixed_graph_memory": memory,
        "formal_v113_output_root_written": False,
    }
    write_json(ROOT / "final_guard_summary.json", overall)
    print(json.dumps(overall, indent=2))
    if overall["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
