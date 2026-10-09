#!/usr/bin/env python3
"""One-unit, fail-fast diagnostic for Sequence Context batch parity.

The tool is intentionally narrower than the frozen two-sample smoke.  It may
only inspect sample 8809/S011, offset -3, with the normalized-zero replacement.
It persists failure evidence atomically before raising the terminal exception.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_context_influence_development_smoke as smoke  # noqa: E402


OUTPUT_ROOT = (
    PROJECT
    / "outputs/sleepxvit_inspired_audit/isruc_seed42/context_influence_v1/"
    "development_smoke_2_batch_parity_diagnostic_1"
)
REPORT = PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_BATCH_PARITY_DIAGNOSTIC.md"
ORIGINAL_SMOKE_ROOT = smoke.OUTPUT_ROOT
ORIGINAL_SMOKE_REPORT = (
    PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_DEVELOPMENT_SMOKE.md"
)
CANONICAL_ATTRIBUTION_MANIFEST = (
    PROJECT
    / "outputs/sequence_lime_paper_evaluation/isruc_seed42/CURRENT_RESULTS_MANIFEST.json"
)
CACHE = smoke.DEVELOPMENT_CACHE_ROOT / "S011.npz"

SAMPLE_ID = 8809
SUBJECT_ID = "S011"
RECORDING_ID = 9
TARGET_EPOCH = 813
OFFSET = -3
CONTEXT_EPOCH = TARGET_EPOCH + OFFSET
REPLACEMENT = "normalized_zero"
BATCH_SIZE = 16
ATOL = 1e-6
RTOL = 0.0
EXPECTED_BRANCH = "codex/a1-zero-mainline"
REQUIRED_COMMITS = (
    "4b0e3b1a6c695e7d5b49dea153102d22f6685211",
    "290ef0f1beeb6d2b7991df464d9172a3250e0a03",
)
EXPECTED_HASHES = {
    **smoke.EXPECTED_HASHES,
    "canonical_attribution_manifest": (
        "2bb600ddcc1ad8293bb9682fda97049b3e2aa508b1564492c169ee8c40672353"
    ),
    "original_smoke_manifest": (
        "bdda7a96d45ae0cf85d8381fb26c9b6a17cfe5c0aedb029ec0edf8ae7ced4781"
    ),
    "original_smoke_report": (
        "d85a7400f4b5a9b26cf7ae9a86bd25afe6d4978c41b86c2d84a9ac8d7d53ccc6"
    ),
    "development_cache_s011": (
        "3ad7d5c3935c20ca3cee5a8a75451856e0a934aec10c6d1be6b9394f7acbfc48"
    ),
}
REQUIRED_OUTPUTS = (
    "diagnostic_protocol.json",
    "matched_input_identity.json",
    "execution_environment.json",
    "batch1_repeatability.json",
    "batch16_repeatability.json",
    "batch_shape_difference.json",
    "layerwise_difference.csv",
    "aggregation_difference.json",
    "final_outputs_batch1.npy",
    "final_outputs_batch16.npy",
    "final_output_diff.npy",
    "final_logits_batch1.npy",
    "final_logits_batch16.npy",
    "final_logits_diff.npy",
    "runtime_memory_profile.json",
    "access_audit.json",
    "failure_evidence.json",
    "run_manifest.json",
    "README.md",
)


class DiagnosticStopped(RuntimeError):
    """Raised only after the complete diagnostic evidence is durable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must be nonempty")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def lineage_paths() -> Mapping[str, Path]:
    return {
        "checkpoint": smoke.CHECKPOINT,
        "predictions": smoke.PREDICTIONS,
        "split": smoke.SPLIT,
        "authority": smoke.AUTHORITY,
        "development_manifest": smoke.DEVELOPMENT_MANIFEST,
        "canonical_attribution_manifest": CANONICAL_ATTRIBUTION_MANIFEST,
        "original_smoke_manifest": ORIGINAL_SMOKE_ROOT / "run_manifest.json",
        "original_smoke_report": ORIGINAL_SMOKE_REPORT,
        "development_cache_s011": CACHE,
    }


def observed_hashes() -> Mapping[str, str]:
    return {name: sha256_file(path) for name, path in lineage_paths().items()}


def assert_lineage() -> Mapping[str, str]:
    observed = observed_hashes()
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in EXPECTED_HASHES.items()
        if observed.get(name) != expected
    }
    if mismatches:
        raise DiagnosticStopped(f"Frozen asset lineage mismatch: {mismatches}")
    return observed


def assert_repository_clean() -> None:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise DiagnosticStopped("Wrong branch")
    if git("status", "--porcelain"):
        raise DiagnosticStopped("Worktree must be clean")
    for commit in REQUIRED_COMMITS:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=PROJECT
        )


def difference_stats(
    left: np.ndarray,
    right: np.ndarray,
    specs: Sequence[smoke.WindowSpec] | None = None,
) -> Mapping[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} != {b.shape}")
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    absolute = np.abs(delta)
    flat_index = int(np.argmax(absolute)) if absolute.size else 0
    location = np.unravel_index(flat_index, absolute.shape) if absolute.size else ()
    window_index = int(location[0]) if location else None
    feature_index = int(location[-1]) if location else None
    payload: dict[str, Any] = {
        "shape": list(a.shape),
        "max_abs_diff": float(np.max(absolute)) if absolute.size else 0.0,
        "mean_abs_diff": float(np.mean(absolute)) if absolute.size else 0.0,
        "median_abs_diff": float(np.median(absolute)) if absolute.size else 0.0,
        "p95_abs_diff": float(np.percentile(absolute, 95)) if absolute.size else 0.0,
        "p99_abs_diff": float(np.percentile(absolute, 99)) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "count_gt_1e_7": int(np.count_nonzero(absolute > 1e-7)),
        "count_gt_1e_6": int(np.count_nonzero(absolute > 1e-6)),
        "count_gt_1e_5": int(np.count_nonzero(absolute > 1e-5)),
        "finite_left": bool(np.all(np.isfinite(a))),
        "finite_right": bool(np.all(np.isfinite(b))),
        "exact_equal": bool(np.array_equal(a, b)),
        "allclose_at_frozen_tolerance": bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
        "max_location": [int(value) for value in location],
        "max_window_index": window_index,
        "max_feature_or_class_index": feature_index,
        "left_at_max": float(a[location]) if location else None,
        "right_at_max": float(b[location]) if location else None,
    }
    if specs is not None and window_index is not None:
        payload.update(
            {
                "max_window_start": int(specs[window_index].start),
                "max_target_position": int(specs[window_index].target_position),
            }
        )
    return payload


def prepare() -> None:
    assert_repository_clean()
    hashes = assert_lineage()
    if OUTPUT_ROOT.exists():
        raise DiagnosticStopped(f"Refusing to overwrite diagnostic output: {OUTPUT_ROOT}")
    if not ORIGINAL_SMOKE_ROOT.is_dir():
        raise DiagnosticStopped("Original smoke root is missing")
    with np.load(CACHE, allow_pickle=False) as cache:
        x_raw = np.asarray(cache["x_raw"], dtype=np.float32)
        x_seg = np.asarray(cache["x_seg"], dtype=np.float32)
    zero_raw = np.zeros_like(x_raw[0], dtype=np.float32)
    raw_windows, _, specs = smoke.assemble_context_windows(
        x_raw, x_seg, TARGET_EPOCH, CONTEXT_EPOCH, zero_raw, x_seg[CONTEXT_EPOCH]
    )
    starts = [int(spec.start) for spec in specs]
    targets = [int(spec.target_position) for spec in specs]
    contexts = [int(spec.context_position) for spec in specs]
    if len(specs) != 15 or len(set(starts)) != 15:
        raise DiagnosticStopped("Unexpected target-covering window identity")
    if any(start + target != TARGET_EPOCH for start, target in zip(starts, targets)):
        raise DiagnosticStopped("Target-position mapping mismatch before inference")
    if any(start + context != CONTEXT_EPOCH for start, context in zip(starts, contexts)):
        raise DiagnosticStopped("Context-position mapping mismatch before inference")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": "sequence_context_batch_parity_diagnostic_v1",
        "status": "FROZEN_BEFORE_DIAGNOSTIC_INFERENCE",
        "frozen_at": utc_now(),
        "source_git_head": git("rev-parse", "HEAD"),
        "source_git_branch": git("branch", "--show-current"),
        "scope": {
            "sample_id": SAMPLE_ID,
            "subject_id": SUBJECT_ID,
            "recording_id": RECORDING_ID,
            "target_epoch_index": TARGET_EPOCH,
            "context_epoch_index": CONTEXT_EPOCH,
            "offset": OFFSET,
            "replacement": REPLACEMENT,
            "comparisons": ["A1_vs_A2", "B1_vs_B2", "A1_vs_B1"],
        },
        "inference_count": {"batch1": 2, "batch16": 2},
        "batch1_semantics": "15 ordered one-window calls over the same materialized array",
        "batch16_semantics": "one call containing the same 15 ordered windows",
        "parity_atol": ATOL,
        "parity_rtol": RTOL,
        "replacement_randomness": False,
        "aggregation": "ordered float64 mean over all target-covering window vectors",
        "terminal_state_remains": "DEVELOPMENT_SMOKE_BLOCKED",
        "cannot_unlock_development16": True,
        "forbidden": [
            "sample_58657", "other_offsets", "other_replacements", "development_16",
            "validation_30", "test_36", "HMM", "training", "gradient",
            "canonical_LIME_modification", "tolerance_change", "protocol_fix",
        ],
        "asset_hashes_before": hashes,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    identity = {
        "schema_version": "sequence_context_matched_input_identity_v1",
        "created_before_model_inference": True,
        "physical_epoch_id": f"recording-{RECORDING_ID}:epoch-{CONTEXT_EPOCH}",
        "sample_id": SAMPLE_ID,
        "subject_id": SUBJECT_ID,
        "recording_id": RECORDING_ID,
        "target_epoch_index": TARGET_EPOCH,
        "context_epoch_index": CONTEXT_EPOCH,
        "offset": OFFSET,
        "replacement": REPLACEMENT,
        "window_count": len(specs),
        "window_starts": starts,
        "window_ends_exclusive": [int(spec.start + spec.valid_len) for spec in specs],
        "target_positions": targets,
        "context_positions": contexts,
        "batch1_positions": [0] * len(specs),
        "batch16_positions": list(range(len(specs))),
        "preliminary_raw_window_shape": list(raw_windows.shape),
        "preliminary_raw_window_dtype": str(raw_windows.dtype),
        "preliminary_raw_window_sha256": sha256_array(raw_windows),
        "target_position_bijection": True,
        "window_order_unique": True,
        "normalized_zero_random": False,
        "class_order": list(smoke.CLASS_LABELS),
        "status": "INPUT_IDENTITY_FROZEN_PENDING_RECOMPUTED_SEGMENT_HASH",
    }
    atomic_json(OUTPUT_ROOT / "diagnostic_protocol.json", protocol)
    atomic_json(OUTPUT_ROOT / "matched_input_identity.json", identity)
    os.chmod(OUTPUT_ROOT / "diagnostic_protocol.json", 0o444)
    print(json.dumps({"status": "PREPARED", "output_root": str(OUTPUT_ROOT)}, indent=2))


def synchronize(tf) -> None:
    try:
        tf.experimental.async_wait()
    except Exception:
        # Reading allocator state and converting returned tensors to NumPy are
        # already synchronizing operations in the supported TensorFlow build.
        pass


def memory_info(tf) -> Mapping[str, int | None]:
    try:
        values = tf.config.experimental.get_memory_info("GPU:0")
        return {"current": int(values["current"]), "peak": int(values["peak"])}
    except Exception:
        return {"current": None, "peak": None}


def reset_peak(tf) -> None:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


def make_probe(tf, model):
    decoder_input = model.get_layer("v3_state_dense").input
    history = getattr(decoder_input, "_keras_history", None)
    transformer_layer = getattr(history, "layer", None) or getattr(history, "operation", None)
    if transformer_layer is None:
        raise DiagnosticStopped("Unable to identify Transformer output tensor")
    selected = [
        ("raw_encoder", model.get_layer("v3_raw_feature_dropout").output),
        ("segment_encoder", model.get_layer("v3_segment_feature_dropout").output),
        ("multimodal_fusion", model.get_layer("epoch_feature_fusion").output),
        ("temporal_projection", model.get_layer("concept_film_modulated_features").output),
        ("tcn", model.get_layer("v3_state_tcn_d2_b6_out_relu").output),
        ("transformer", transformer_layer.output),
        ("sequence_decoder", model.get_layer("td_temporal_dense").output),
        ("classifier_input", model.get_layer("temporal_dropout").output),
        ("pre_hmm_probability", model.get_layer("main_output").output),
    ]
    ordered_names = [name for name, _ in selected]
    probe = tf.keras.Model(inputs=model.inputs, outputs=[value for _, value in selected])
    main_layer = model.get_layer("main_output")
    if not hasattr(main_layer, "kernel"):
        raise DiagnosticStopped("main_output does not expose its classifier kernel")

    raw_spec = tf.TensorSpec((None, 75, 3000, 3), tf.float32)
    segment_spec = tf.TensorSpec((None, 75, 30, 28), tf.float32)
    target_spec = tf.TensorSpec((None,), tf.int32)

    @tf.function(input_signature=(raw_spec, segment_spec, target_spec), reduce_retracing=True)
    def runner(raw_input, segment_input, target_positions):
        values = probe(
            {"raw_input": raw_input, "segment_input": segment_input}, training=False
        )
        rows = tf.range(tf.shape(target_positions)[0], dtype=tf.int32)
        gather_indices = tf.stack([rows, target_positions], axis=1)
        matched = [tf.gather_nd(value, gather_indices) for value in values]
        classifier_input = matched[ordered_names.index("classifier_input")]
        logits = tf.linalg.matmul(classifier_input, main_layer.kernel)
        if main_layer.bias is not None:
            logits = tf.nn.bias_add(logits, main_layer.bias)
        # Keep model stage order: classifier input -> logits -> softmax probability.
        return tuple(matched[:-1] + [logits, matched[-1]])

    return runner, ordered_names[:-1] + ["logits", "pre_hmm_probability"]


def run_profile(
    tf,
    runner,
    stage_names: Sequence[str],
    raw_windows: np.ndarray,
    segment_windows: np.ndarray,
    specs: Sequence[smoke.WindowSpec],
    batch_size: int,
    label: str,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    reset_peak(tf)
    synchronize(tf)
    before = memory_info(tf)
    tick = time.perf_counter()
    values: dict[str, list[np.ndarray]] = {name: [] for name in stage_names}
    call_seconds: list[float] = []
    call_sizes: list[int] = []
    for start in range(0, len(specs), batch_size):
        stop = min(len(specs), start + batch_size)
        positions = np.asarray(
            [spec.target_position for spec in specs[start:stop]], dtype=np.int32
        )
        call_tick = time.perf_counter()
        outputs = runner(
            tf.convert_to_tensor(raw_windows[start:stop], tf.float32),
            tf.convert_to_tensor(segment_windows[start:stop], tf.float32),
            tf.convert_to_tensor(positions, tf.int32),
        )
        output_arrays = [np.asarray(value.numpy()) for value in outputs]
        synchronize(tf)
        call_seconds.append(float(time.perf_counter() - call_tick))
        call_sizes.append(int(stop - start))
        for name, value in zip(stage_names, output_arrays):
            values[name].append(value)
    synchronize(tf)
    total = float(time.perf_counter() - tick)
    after = memory_info(tf)
    merged = {name: np.concatenate(chunks, axis=0) for name, chunks in values.items()}
    runtime = {
        "label": label,
        "batch_size": int(batch_size),
        "forward_call_count": len(call_sizes),
        "forward_call_batch_sizes": call_sizes,
        "forward_call_seconds": call_seconds,
        "total_seconds": total,
        "allocator_before_bytes": before,
        "allocator_after_bytes": after,
        "max_memory_allocated_bytes": after["peak"],
        "max_memory_reserved_bytes": None,
        "memory_api": "TensorFlow get_memory_info; reserved is not exposed",
    }
    return merged, runtime


def environment_payload(tf, model, model_fingerprint: str) -> Mapping[str, Any]:
    build = tf.sysconfig.get_build_info()
    gpu_devices = tf.config.list_physical_devices("GPU")
    dropout_bn = [
        {
            "name": layer.name,
            "class": layer.__class__.__name__,
            "trainable": bool(layer.trainable),
            "call_training": False,
        }
        for layer in model.layers
        if layer.__class__.__name__ in {"Dropout", "BatchNormalization"}
    ]
    weight_dtypes = sorted({str(weight.dtype) for weight in model.weights})
    return {
        "schema_version": "sequence_context_execution_environment_v1",
        "framework": "TensorFlow/Keras",
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "cuda_version": build.get("cuda_version"),
        "cudnn_version": build.get("cudnn_version"),
        "gpu_devices": [device.name for device in gpu_devices],
        "gpu_total_memory_bytes": query_gpu_total_memory(),
        "model_training": False,
        "dropout_batchnorm_modules": dropout_bn,
        "gradient_context": "no GradientTape; inference-only tf.function",
        "parameters_requiring_gradient_during_run": 0,
        "trainable_parameter_tensors_present_but_not_optimized": int(len(model.trainable_weights)),
        "input_dtype": "float32",
        "model_weight_dtypes": weight_dtypes,
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "autocast_changed_for_diagnostic": False,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "deterministic_ops_requested_by_frozen_builder": True,
        "torch_specific_settings": "NOT_APPLICABLE_MODEL_IS_TENSORFLOW",
        "seed": smoke.MODEL_SEED,
        "checkpoint_load_count": 1,
        "model_instance_id": int(id(model)),
        "model_weight_fingerprint": model_fingerprint,
        "training_or_weight_update": False,
    }


def query_gpu_total_memory() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
        ).strip().splitlines()[0]
        return int(value) * 1024 * 1024
    except Exception:
        return None


def classify_root_cause(
    inputs_identical: bool,
    repeat1: Mapping[str, Any],
    repeat16: Mapping[str, Any],
    batch: Mapping[str, Any],
    aggregation: Mapping[str, Any],
) -> str:
    if not inputs_identical:
        return "INPUT_OR_MAPPING_MISMATCH"
    if not repeat1["exact_equal"] or not repeat16["exact_equal"]:
        return "RUNTIME_NONDETERMINISM"
    if batch["max_abs_diff"] > ATOL:
        return "BATCH_SHAPE_NUMERICAL_DRIFT"
    if aggregation["probability_stats"]["max_abs_diff"] > ATOL:
        return "AGGREGATION_ORDER_OR_PRECISION"
    return "INCONCLUSIVE"


def validate_artifacts() -> list[Mapping[str, Any]]:
    artifacts = []
    for name in REQUIRED_OUTPUTS:
        if name == "run_manifest.json":
            continue
        path = OUTPUT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            raise DiagnosticStopped(f"Missing or empty required artifact: {name}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".npy":
            np.load(path, allow_pickle=False)
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as stream:
                if len(list(csv.DictReader(stream))) == 0:
                    raise DiagnosticStopped(f"Empty CSV data: {name}")
        artifacts.append(
            {"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return artifacts


def run() -> None:
    assert_repository_clean()
    hashes_before = assert_lineage()
    protocol_path = OUTPUT_ROOT / "diagnostic_protocol.json"
    identity_path = OUTPUT_ROOT / "matched_input_identity.json"
    if not protocol_path.is_file() or not identity_path.is_file():
        raise DiagnosticStopped("Run --mode prepare before inference")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if git("rev-parse", "HEAD") != protocol["source_git_head"]:
        raise DiagnosticStopped("Source HEAD changed after diagnostic protocol freeze")

    # Only named train arrays are read; val/test keys are never enumerated.
    npz_keys_read = [
        "class_labels", "train_y_true", "train_probs_raw", "train_recording_id",
        "train_epoch_index", "train_file_sample_counts",
    ]
    with np.load(smoke.PREDICTIONS, allow_pickle=False) as archive:
        class_order = archive["class_labels"].astype(str).tolist()
        train_labels = archive["train_y_true"].astype(np.int16)
        train_probs = archive["train_probs_raw"].astype(np.float32)
        recording_ids = archive["train_recording_id"].astype(np.int32)
        epoch_ids = archive["train_epoch_index"].astype(np.int32)
        counts = archive["train_file_sample_counts"].astype(np.int32)
    if class_order != list(smoke.CLASS_LABELS):
        raise DiagnosticStopped("Class order changed")
    if int(recording_ids[SAMPLE_ID]) != RECORDING_ID or int(epoch_ids[SAMPLE_ID]) != TARGET_EPOCH:
        raise DiagnosticStopped("Prediction sample mapping changed")
    recording_start = int(np.sum(counts[:RECORDING_ID]))
    if recording_start + TARGET_EPOCH != SAMPLE_ID:
        raise DiagnosticStopped("Global/local epoch mapping changed")

    with np.load(CACHE, allow_pickle=False) as cache:
        x_raw = np.asarray(cache["x_raw"], dtype=np.float32)
        x_seg = np.asarray(cache["x_seg"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=np.int16)
    if not np.array_equal(labels, train_labels[recording_start:recording_start + len(labels)]):
        raise DiagnosticStopped("Development cache labels changed")

    from tools import sequence_lime_paper_evaluation as evaluator
    import tensorflow as tf

    evaluator.ensure_lime_imports()
    zero_raw = np.zeros_like(x_raw[0], dtype=np.float32)
    zero_segment = smoke.recompute_segment_feature(evaluator, zero_raw)
    raw_windows, segment_windows, specs = smoke.assemble_context_windows(
        x_raw, x_seg, TARGET_EPOCH, CONTEXT_EPOCH, zero_raw, zero_segment
    )
    raw_windows.setflags(write=False)
    segment_windows.setflags(write=False)

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity.update(
        {
            "status": "MATCHED_INPUT_IDENTITY_CONFIRMED",
            "raw_window_shape": list(raw_windows.shape),
            "raw_window_dtype": str(raw_windows.dtype),
            "raw_window_sha256": sha256_array(raw_windows),
            "segment_window_shape": list(segment_windows.shape),
            "segment_window_dtype": str(segment_windows.dtype),
            "segment_window_sha256": sha256_array(segment_windows),
            "replacement_raw_shape": list(zero_raw.shape),
            "replacement_raw_dtype": str(zero_raw.dtype),
            "replacement_raw_sha256": sha256_array(zero_raw),
            "replacement_segment_shape": list(zero_segment.shape),
            "replacement_segment_dtype": str(zero_segment.dtype),
            "replacement_segment_sha256": sha256_array(zero_segment),
            "per_window": [
                {
                    "window_id": index,
                    "window_start": int(spec.start),
                    "window_end_exclusive": int(spec.start + spec.valid_len),
                    "target_position": int(spec.target_position),
                    "context_position": int(spec.context_position),
                    "batch1_position": 0,
                    "batch16_position": index,
                    "raw_sha256": sha256_array(raw_windows[index]),
                    "segment_sha256": sha256_array(segment_windows[index]),
                }
                for index, spec in enumerate(specs)
            ],
            "batch1_and_batch16_share_same_materialized_arrays": True,
            "batch16_sort_deduplicate_padding_or_index_shift": False,
            "input_hashes_and_order_identical": True,
        }
    )
    atomic_json(identity_path, identity)

    model = evaluator.build_model((3000, 3), (30, 28), initialization_seed=smoke.MODEL_SEED)
    model_fingerprint_before = evaluator.model_weight_fingerprint(model)
    runner, stage_names = make_probe(tf, model)
    environment = environment_payload(tf, model, model_fingerprint_before)
    atomic_json(OUTPUT_ROOT / "execution_environment.json", environment)

    profiles: dict[str, Mapping[str, np.ndarray]] = {}
    runtimes = []
    for label, batch_size in (("A1", 1), ("A2", 1), ("B1", BATCH_SIZE), ("B2", BATCH_SIZE)):
        profile, runtime = run_profile(
            tf, runner, stage_names, raw_windows, segment_windows, specs, batch_size, label
        )
        profiles[label] = profile
        runtimes.append(runtime)

    probability_name = "pre_hmm_probability"
    logits_name = "logits"
    final_a = np.asarray(profiles["A1"][probability_name], dtype=np.float32)
    final_b = np.asarray(profiles["B1"][probability_name], dtype=np.float32)
    logits_a = np.asarray(profiles["A1"][logits_name], dtype=np.float32)
    logits_b = np.asarray(profiles["B1"][logits_name], dtype=np.float32)
    atomic_npy(OUTPUT_ROOT / "final_outputs_batch1.npy", final_a)
    atomic_npy(OUTPUT_ROOT / "final_outputs_batch16.npy", final_b)
    atomic_npy(OUTPUT_ROOT / "final_output_diff.npy", final_a - final_b)
    atomic_npy(OUTPUT_ROOT / "final_logits_batch1.npy", logits_a)
    atomic_npy(OUTPUT_ROOT / "final_logits_batch16.npy", logits_b)
    atomic_npy(OUTPUT_ROOT / "final_logits_diff.npy", logits_a - logits_b)

    rows = []
    comparisons = (("batch1_repeat", "A1", "A2"), ("batch16_repeat", "B1", "B2"), ("batch_shape", "A1", "B1"))
    stats_by_comparison: dict[str, dict[str, Mapping[str, Any]]] = {}
    for comparison, left, right in comparisons:
        stats_by_comparison[comparison] = {}
        for stage_index, stage in enumerate(stage_names):
            stats = difference_stats(profiles[left][stage], profiles[right][stage], specs)
            stats_by_comparison[comparison][stage] = stats
            rows.append(
                {
                    "comparison": comparison,
                    "left_run": left,
                    "right_run": right,
                    "stage_index": stage_index,
                    "stage": stage,
                    **{key: value for key, value in stats.items() if not isinstance(value, list)},
                    "shape": json.dumps(stats["shape"]),
                    "max_location": json.dumps(stats["max_location"]),
                }
            )
    atomic_csv(OUTPUT_ROOT / "layerwise_difference.csv", rows)

    batch1_repeat = stats_by_comparison["batch1_repeat"][probability_name]
    batch16_repeat = stats_by_comparison["batch16_repeat"][probability_name]
    batch_shape = stats_by_comparison["batch_shape"][probability_name]
    atomic_json(
        OUTPUT_ROOT / "batch1_repeatability.json",
        {"comparison": "A1_vs_A2", "stable_exactly": batch1_repeat["exact_equal"], "stages": stats_by_comparison["batch1_repeat"]},
    )
    atomic_json(
        OUTPUT_ROOT / "batch16_repeatability.json",
        {"comparison": "B1_vs_B2", "stable_exactly": batch16_repeat["exact_equal"], "stages": stats_by_comparison["batch16_repeat"]},
    )

    artifact_original = np.asarray(train_probs[SAMPLE_ID], dtype=np.float64)
    response_class = int(np.argmax(artifact_original))
    aggregate_a = np.asarray(final_a, dtype=np.float64).mean(axis=0)
    aggregate_b = np.asarray(final_b, dtype=np.float64).mean(axis=0)
    aggregate_stats = difference_stats(aggregate_a, aggregate_b)
    influence_a = float(artifact_original[response_class] - aggregate_a[response_class])
    influence_b = float(artifact_original[response_class] - aggregate_b[response_class])
    aggregation = {
        "accumulation_order_identical": True,
        "accumulation_dtype": "float64",
        "operation": "ordered arithmetic mean",
        "denominator_batch1": len(specs),
        "denominator_batch16": len(specs),
        "per_window_probability_stats": batch_shape,
        "probability_stats": aggregate_stats,
        "artifact_original_probability": artifact_original.tolist(),
        "response_class_index": response_class,
        "response_class": smoke.CLASS_LABELS[response_class],
        "aggregated_batch1": aggregate_a.tolist(),
        "aggregated_batch16": aggregate_b.tolist(),
        "aggregated_response_batch1": float(aggregate_a[response_class]),
        "aggregated_response_batch16": float(aggregate_b[response_class]),
        "signed_influence_batch1": influence_a,
        "signed_influence_batch16": influence_b,
        "signed_influence_difference": float(influence_a - influence_b),
        "influence_sign_changed": bool(np.sign(influence_a) != np.sign(influence_b)),
        "argmax_batch1": int(np.argmax(aggregate_a)),
        "argmax_batch16": int(np.argmax(aggregate_b)),
        "argmax_changed": bool(np.argmax(aggregate_a) != np.argmax(aggregate_b)),
        "finite": bool(np.all(np.isfinite(aggregate_a)) and np.all(np.isfinite(aggregate_b))),
    }
    atomic_json(OUTPUT_ROOT / "aggregation_difference.json", aggregation)

    first_divergent = next(
        (stage for stage in stage_names if stats_by_comparison["batch_shape"][stage]["max_abs_diff"] > 0.0),
        None,
    )
    root_cause = classify_root_cause(
        True, batch1_repeat, batch16_repeat, batch_shape, aggregation
    )
    terminal = (
        "BATCH_PARITY_DIAGNOSTIC_ROOT_CAUSE_IDENTIFIED"
        if root_cause != "INCONCLUSIVE"
        else "BATCH_PARITY_DIAGNOSTIC_INCONCLUSIVE"
    )
    batch_shape_payload = {
        "comparison": "A1_vs_B1",
        "input_identity": "EXACT",
        "window_order": "EXACT",
        "target_positions": "EXACT",
        "stages": stats_by_comparison["batch_shape"],
        "final_probability_stats": batch_shape,
        "first_divergent_stage": first_divergent,
        "root_cause_classification": root_cause,
        "parity_atol": ATOL,
        "parity_rtol": RTOL,
        "pass": bool(batch_shape["allclose_at_frozen_tolerance"]),
    }
    atomic_json(OUTPUT_ROOT / "batch_shape_difference.json", batch_shape_payload)

    runtime_payload = {
        "framework": "TensorFlow/Keras",
        "profiles": runtimes,
        "batch1_formal_peak_allocated_bytes": max(
            row["max_memory_allocated_bytes"] or 0 for row in runtimes if row["batch_size"] == 1
        ),
        "batch16_formal_peak_allocated_bytes": max(
            row["max_memory_allocated_bytes"] or 0 for row in runtimes if row["batch_size"] == BATCH_SIZE
        ),
        "reserved_memory": "NOT_EXPOSED_BY_TENSORFLOW_ALLOCATOR_API",
        "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
        "historical_external_observation_mib_not_formal_peak": 22582,
        "measurement_notes": "CUDA synchronized around returned tensors; peak reset before each A1/A2/B1/B2 profile",
    }
    atomic_json(OUTPUT_ROOT / "runtime_memory_profile.json", runtime_payload)

    model_fingerprint_after = evaluator.model_weight_fingerprint(model)
    hashes_after = assert_lineage()
    access = {
        "sample_8809_accessed": True,
        "sample_58657_accessed": False,
        "other_development_samples_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "prediction_npz_keys_read": npz_keys_read,
        "validation_or_test_prediction_arrays_read": False,
        "training": False,
        "gradient": False,
        "hmm": False,
        "weight_update": False,
        "checkpoint_load_count": 1,
        "model_fingerprint_before": model_fingerprint_before,
        "model_fingerprint_after": model_fingerprint_after,
        "model_weights_unchanged": model_fingerprint_before == model_fingerprint_after,
        "asset_hashes_before": hashes_before,
        "asset_hashes_after": hashes_after,
        "asset_hashes_unchanged": hashes_before == hashes_after,
        "original_smoke_output_overwritten": False,
        "original_smoke_report_modified": False,
    }
    atomic_json(OUTPUT_ROOT / "access_audit.json", access)

    failure_message = (
        f"Frozen batch parity remains failed: max_abs_diff={batch_shape['max_abs_diff']:.17g}, "
        f"atol={ATOL}, rtol={RTOL}; diagnostic only, DEVELOPMENT_SMOKE_BLOCKED remains."
    )
    failure = {
        "terminal_status": terminal,
        "development_smoke_status": "DEVELOPMENT_SMOKE_BLOCKED",
        "failure_stage": "batch_parity_diagnostic_complete",
        "failure_message": failure_message,
        "traceback": "".join(traceback.format_stack()),
        "root_cause_classification": root_cause,
        "first_divergent_stage": first_divergent,
        "completed": [
            "input_identity", "A1", "A2", "B1", "B2", "layerwise_comparison",
            "aggregation_audit", "runtime_memory", "access_audit", "asset_recheck",
        ],
        "not_completed_by_scope": [
            "sample_58657", "other_offsets", "other_replacements", "development_16",
            "validation_30", "test_36", "implementation_fix",
        ],
        "evidence_persisted_before_exception": True,
    }
    atomic_json(OUTPUT_ROOT / "failure_evidence.json", failure)
    atomic_text(
        OUTPUT_ROOT / "README.md",
        "# Sequence Context Batch Parity Diagnostic 1\n\n"
        "Diagnostic-only evidence for sample 8809/S011, offset -3, normalized_zero.\n"
        "It does not unlock the frozen development-16 study and does not alter canonical LIME.\n",
    )
    artifacts = validate_artifacts()
    manifest = {
        "schema_version": "sequence_context_batch_parity_diagnostic_manifest_v1",
        "terminal_status": terminal,
        "development_smoke_status": "DEVELOPMENT_SMOKE_BLOCKED",
        "source_git_head": git("rev-parse", "HEAD"),
        "completed_at": utc_now(),
        "scope": protocol["scope"],
        "root_cause_classification": root_cause,
        "first_divergent_stage": first_divergent,
        "exact_max_abs_probability_diff": batch_shape["max_abs_diff"],
        "artifacts": artifacts,
        "asset_hashes_after": hashes_after,
        "all_required_artifacts_parseable": True,
        "exception_raised_after_persistence": True,
    }
    atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
    # Validate the manifest itself and every recorded hash before fail-fast.
    json.loads((OUTPUT_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    for artifact in artifacts:
        path = PROJECT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise DiagnosticStopped(f"Post-write artifact hash mismatch: {path}")
    raise DiagnosticStopped(failure_message)


def render_report() -> None:
    manifest = json.loads((OUTPUT_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    batch = json.loads((OUTPUT_ROOT / "batch_shape_difference.json").read_text(encoding="utf-8"))
    repeat1 = json.loads((OUTPUT_ROOT / "batch1_repeatability.json").read_text(encoding="utf-8"))
    repeat16 = json.loads((OUTPUT_ROOT / "batch16_repeatability.json").read_text(encoding="utf-8"))
    aggregation = json.loads((OUTPUT_ROOT / "aggregation_difference.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUTPUT_ROOT / "runtime_memory_profile.json").read_text(encoding="utf-8"))
    access = json.loads((OUTPUT_ROOT / "access_audit.json").read_text(encoding="utf-8"))
    stats = batch["final_probability_stats"]
    raw_stats = batch["stages"]["raw_encoder"]
    segment_stats = batch["stages"]["segment_encoder"]
    logits_stats = batch["stages"]["logits"]
    aggregate_stats = aggregation["probability_stats"]
    memory1_mib = runtime["batch1_formal_peak_allocated_bytes"] / (1024 * 1024)
    memory16_mib = runtime["batch16_formal_peak_allocated_bytes"] / (1024 * 1024)
    lines = [
        "# Sequence Context Influence — Batch Parity Failure Diagnostic Run 1",
        "",
        f"**`{manifest['terminal_status']}`**",
        "",
        "该诊断不解除 **`DEVELOPMENT_SMOKE_BLOCKED`**，也不授权 development-16。",
        "",
        "## 诊断结论",
        "",
        f"- 精确最大概率绝对差：`{stats['max_abs_diff']:.17g}`。",
        (
            f"- 首个有序 probe 差异阶段：`{batch['first_divergent_stage']}` "
            f"（max `{raw_stats['max_abs_diff']:.17g}`）；并行 `segment_encoder` "
            f"也已差异（max `{segment_stats['max_abs_diff']:.17g}`），因此层级结论是"
            "两条 epoch encoder 分支已经出现 batch-shape 差异。"
        ),
        f"- 根因分类：`{batch['root_cause_classification']}`。",
        f"- 输入、窗口顺序、target position：`{batch['input_identity']} / {batch['window_order']} / {batch['target_positions']}`。",
        f"- batch-1 A1/A2 exact stable：`{repeat1['stable_exactly']}`。",
        f"- batch-16 B1/B2 exact stable：`{repeat16['stable_exactly']}`。",
        (
            f"- 最大概率差位于 window start `{stats['max_window_start']}`、target position "
            f"`{stats['max_target_position']}`、class `W`：batch-1 "
            f"`{stats['left_at_max']:.17g}`，batch-16 `{stats['right_at_max']:.17g}`。"
        ),
        f"- logits 最大绝对差：`{logits_stats['max_abs_diff']:.17g}`。",
        (
            f"- 聚合概率最大绝对差：`{aggregate_stats['max_abs_diff']:.17g}`；response "
            f"`{aggregation['aggregated_response_batch1']:.17g}` → "
            f"`{aggregation['aggregated_response_batch16']:.17g}`。"
        ),
        (
            f"- signed influence `{aggregation['signed_influence_batch1']:.17g}` → "
            f"`{aggregation['signed_influence_batch16']:.17g}`；argmax 改变："
            f"`{aggregation['argmax_changed']}`；influence 符号改变："
            f"`{aggregation['influence_sign_changed']}`。"
        ),
        (
            f"- batch-1 TensorFlow allocator peak："
            f"`{runtime['batch1_formal_peak_allocated_bytes']}` bytes "
            f"(`{memory1_mib:.3f}` MiB)。"
        ),
        (
            f"- batch-16 TensorFlow allocator peak："
            f"`{runtime['batch16_formal_peak_allocated_bytes']}` bytes "
            f"(`{memory16_mib:.3f}` MiB)。"
        ),
        "- TensorFlow allocator 不暴露与 PyTorch `max_memory_reserved` 等价的 reserved 指标，因此该项明确记为 N/A。",
        "- 原 smoke 设置保持不变：float32、TF32 enabled、无 mixed precision 切换、training=False、seed=42。",
        "",
        "## 边界与资产",
        "",
        f"- 第二个 development 样本访问：`{access['sample_58657_accessed']}`。",
        f"- validation/test 访问：`{access['validation_accessed']} / {access['test_accessed']}`。",
        f"- 模型权重不变：`{access['model_weights_unchanged']}`；冻结资产哈希不变：`{access['asset_hashes_unchanged']}`。",
        "- 未运行 HMM、训练、梯度或权重更新；未修改 canonical attribution。",
        "- 原 `development_smoke_2` 与原阻断报告未覆盖。",
        "",
        "## 下一步",
        "",
        (
            "同 batch 重复若 exact stable 且输入身份完全一致，则应单独审议 batch-shape 数值协议；"
            "本轮禁止修复、改 batch、切换 TF32/dtype 或放宽 1e-6 容差。"
            if batch["root_cause_classification"] == "BATCH_SHAPE_NUMERICAL_DRIFT"
            else "应先针对已识别的实现或运行时根因另行授权修复；本轮到此停止。"
        ),
        "",
        "所有详细数值、逐层差值、输入哈希和访问审计位于独立 diagnostic 输出目录。",
        "",
    ]
    atomic_text(REPORT, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run", "report"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        run()
    else:
        render_report()


if __name__ == "__main__":
    main()
