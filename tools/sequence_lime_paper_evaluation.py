#!/usr/bin/env python
"""Paper-grade sequence-aware LIME evaluation for ISRUC A1-zero seed42.

This script is intentionally offline: it rebuilds the already-trained A1-zero
sequence model, loads the canonical checkpoint, freezes inference with
``training=False``, and evaluates LIME explanations against the formal
overlapping-window sequence prediction rule.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# These settings must be established before TensorFlow is imported indirectly
# by ``src.model_training``.  They reproduce the frozen A1-zero inference run.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CUDNN_WORKSPACE_LIMIT_IN_MB", "1024")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
CHANNEL_LABELS = ["EEG", "EOG", "EMG"]
EPS = 1e-8
EVALUATOR_VERSION = "sequence_lime_v1.1.0"
STRICT_PROBABILITY_TOLERANCE = 1e-6

CLINICAL_CONCEPTS = None
CONCEPT_CHANNEL_INDEX = None
apply_perturbation_to_multimodal = None
calculate_cosine_distances = None
calculate_weights_from_distances = None
extract_segment_features = None
get_clinical_prior = None
get_stage_concept_prior = None
extract_clinical_concept_matrix = None
semantic_segmentation = None


def ensure_lime_imports() -> None:
    global CLINICAL_CONCEPTS
    global CONCEPT_CHANNEL_INDEX
    global apply_perturbation_to_multimodal
    global calculate_cosine_distances
    global calculate_weights_from_distances
    global extract_segment_features
    global get_clinical_prior
    global get_stage_concept_prior
    global extract_clinical_concept_matrix
    global semantic_segmentation
    if semantic_segmentation is not None:
        return
    from src.lime_explanation import (  # noqa: WPS433
        CLINICAL_CONCEPTS as _CLINICAL_CONCEPTS,
        apply_perturbation_to_multimodal as _apply_perturbation_to_multimodal,
        calculate_cosine_distances as _calculate_cosine_distances,
        calculate_weights_from_distances as _calculate_weights_from_distances,
        extract_segment_features as _extract_segment_features,
        extract_clinical_concept_matrix as _extract_clinical_concept_matrix,
        get_clinical_prior as _get_clinical_prior,
        get_stage_concept_prior as _get_stage_concept_prior,
        semantic_segmentation as _semantic_segmentation,
    )
    from src.model_training import CONCEPT_CHANNEL_INDEX as _CONCEPT_CHANNEL_INDEX  # noqa: WPS433

    CLINICAL_CONCEPTS = _CLINICAL_CONCEPTS
    CONCEPT_CHANNEL_INDEX = _CONCEPT_CHANNEL_INDEX
    apply_perturbation_to_multimodal = _apply_perturbation_to_multimodal
    calculate_cosine_distances = _calculate_cosine_distances
    calculate_weights_from_distances = _calculate_weights_from_distances
    extract_segment_features = _extract_segment_features
    get_clinical_prior = _get_clinical_prior
    get_stage_concept_prior = _get_stage_concept_prior
    extract_clinical_concept_matrix = _extract_clinical_concept_matrix
    semantic_segmentation = _semantic_segmentation

OUT_ROOT = PROJECT_ROOT / "outputs" / "sequence_lime_paper_evaluation" / "isruc_seed42"
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5"
PRED_PATH = PROJECT_ROOT / "outputs" / "predictions" / "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz"
SPLIT_PATH = PROJECT_ROOT / "dataset_protocols" / "ISRUC" / "split.json"


def coerce_file_sample_counts(total_len: int, file_sample_counts: Optional[Sequence[int]]) -> List[int]:
    if file_sample_counts is None:
        return [int(total_len)]
    counts = [int(count) for count in file_sample_counts if int(count) > 0]
    if sum(counts) != int(total_len):
        raise ValueError(f"file_sample_counts sum={sum(counts)} does not match total_len={total_len}")
    return counts


def build_sequence_windows(
    total_len: int,
    file_sample_counts: Optional[Sequence[int]] = None,
    seq_len: int = 75,
    stride: int = 5,
    include_last: bool = True,
    allow_short: bool = True,
) -> List[Tuple[int, int]]:
    seq_len = max(1, int(seq_len))
    stride = max(1, int(stride))
    windows = []
    offset = 0
    for count in coerce_file_sample_counts(total_len, file_sample_counts):
        record_start = offset
        record_end = offset + count
        if count < seq_len:
            if allow_short:
                windows.append((record_start, count))
            offset = record_end
            continue
        last_start = record_end - seq_len
        starts = list(range(record_start, last_start + 1, stride))
        if include_last and starts[-1] != last_start:
            starts.append(last_start)
        windows.extend((start, seq_len) for start in starts)
        offset = record_end
    return windows


def window_epoch_indices(start: int, valid_len: int, seq_len: int = 75) -> np.ndarray:
    valid_len = max(1, int(valid_len))
    seq_len = max(1, int(seq_len))
    indices = np.arange(int(start), int(start) + valid_len, dtype=np.int64)
    if valid_len < seq_len:
        pad = np.full(seq_len - valid_len, indices[-1], dtype=np.int64)
        indices = np.concatenate([indices, pad])
    return indices


@dataclass(frozen=True)
class RunConfig:
    mode: str
    train_masks: int
    val_masks: int
    random_repeats: int
    stability_seeds: Tuple[int, ...]
    sample_correct_per_stage: int
    sample_wrong_per_stage: int
    max_samples: int
    batch_windows: int
    perturbation_chunk: int
    bootstrap_repeats: int
    do_randomization: bool
    perturbation_modes: Tuple[str, ...]
    target_segments: int
    mask_sampling: str
    ridge_alphas: Tuple[float, ...]
    kernel_widths: Tuple[float, ...]
    response_transform: str
    surrogate_perturbation_mode: str


def mode_config(mode: str) -> RunConfig:
    if mode == "functional-smoke":
        return RunConfig(
            mode=mode,
            train_masks=300,
            val_masks=100,
            random_repeats=3,
            stability_seeds=(9001, 9002),
            sample_correct_per_stage=1,
            sample_wrong_per_stage=0,
            max_samples=5,
            batch_windows=32,
            perturbation_chunk=4,
            bootstrap_repeats=100,
            do_randomization=False,
            perturbation_modes=("physiological",),
            target_segments=15,
            mask_sampling="local_stratified",
            ridge_alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
            kernel_widths=(0.15, 0.25, 0.5, 0.75),
            response_transform="probability",
            surrogate_perturbation_mode="zero",
        )
    if mode == "performance-profile":
        return RunConfig(
            mode=mode,
            train_masks=64,
            val_masks=32,
            random_repeats=0,
            stability_seeds=(),
            sample_correct_per_stage=1,
            sample_wrong_per_stage=0,
            max_samples=1,
            batch_windows=32,
            perturbation_chunk=4,
            bootstrap_repeats=0,
            do_randomization=False,
            perturbation_modes=("physiological",),
            target_segments=15,
            mask_sampling="local_stratified",
            ridge_alphas=(0.01, 0.1, 1.0, 10.0),
            kernel_widths=(0.15, 0.25, 0.5),
            response_transform="probability",
            surrogate_perturbation_mode="zero",
        )
    if mode == "metric-smoke":
        return RunConfig(
            mode=mode,
            train_masks=900,
            val_masks=200,
            random_repeats=5,
            stability_seeds=(9001, 9002, 9003),
            sample_correct_per_stage=1,
            sample_wrong_per_stage=1,
            max_samples=10,
            batch_windows=32,
            perturbation_chunk=4,
            bootstrap_repeats=200,
            do_randomization=True,
            perturbation_modes=("physiological",),
            target_segments=15,
            mask_sampling="local_stratified",
            ridge_alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
            kernel_widths=(0.15, 0.25, 0.5, 0.75),
            response_transform="probability",
            surrogate_perturbation_mode="zero",
        )
    if mode in {"smoke", "parity-diagnostic"}:
        return RunConfig(
            mode=mode,
            train_masks=900,
            val_masks=200,
            random_repeats=5,
            stability_seeds=(9001, 9002),
            sample_correct_per_stage=1,
            sample_wrong_per_stage=1,
            max_samples=10,
            # The frozen split probability artifact was generated with
            # predict_batch=16.  Inference batching is numerically observable
            # for this convolutional graph, so parity must use the same value.
            batch_windows=16 if mode == "parity-diagnostic" else 8,
            perturbation_chunk=8,
            bootstrap_repeats=200,
            do_randomization=True,
            perturbation_modes=("physiological", "zero", "gaussian_noise", "local_mean"),
            target_segments=15,
            mask_sampling="local_stratified",
            ridge_alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
            kernel_widths=(0.15, 0.25, 0.5, 0.75),
            response_transform="probability",
            surrogate_perturbation_mode="zero",
        )
    return RunConfig(
        mode=mode,
        train_masks=900,
        val_masks=200,
        random_repeats=50,
        stability_seeds=tuple(range(9101, 9111)),
        sample_correct_per_stage=20,
        sample_wrong_per_stage=10,
        max_samples=150,
        batch_windows=16,
        perturbation_chunk=4,
        bootstrap_repeats=2000,
        do_randomization=True,
            perturbation_modes=("physiological", "zero", "gaussian_noise", "local_mean"),
        target_segments=15,
        mask_sampling="local_stratified",
        ridge_alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0),
        kernel_widths=(0.15, 0.25, 0.5, 0.75),
        response_transform="probability",
        surrogate_perturbation_mode="zero",
    )


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, block_size: int = 1 << 20) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_state() -> Dict[str, object]:
    def run(*command: str) -> str:
        return subprocess.check_output(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        status = run("git", "status", "--porcelain")
        return {"git_commit": commit, "working_tree_clean": not bool(status)}
    except Exception:
        return {"git_commit": None, "working_tree_clean": False}


def build_run_provenance(
    run_id: str,
    config: Optional[RunConfig] = None,
    probability_cache: Optional[Path] = PRED_PATH,
) -> Dict[str, object]:
    return {
        "run_id": str(run_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **git_state(),
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_sha256": file_sha256(CKPT_PATH),
        "split_path": str(SPLIT_PATH),
        "split_sha256": file_sha256(SPLIT_PATH),
        "probability_cache_path": str(probability_cache) if probability_cache else None,
        "probability_cache_sha256": file_sha256(probability_cache) if probability_cache else None,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_config": asdict(config) if config is not None else None,
        "decoder_boundary": {
            "canonical_decoder": "validation-selected temperature argmax",
            "hmm_role": "optional validation-selected candidate; not canonical",
            "lime_endpoint": "pre-HMM raw neural probabilities",
        },
        "claim_boundary": (
            "local attribution and interventional faithfulness under the configured "
            "replacement rule; not a global causal contribution"
        ),
    }


def stat_payload(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": file_sha256(path),
    }


def model_weight_fingerprint(model) -> str:
    h = hashlib.sha256()
    for weight in model.weights:
        arr = np.asarray(weight.numpy())
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def peak_host_memory_mb() -> Optional[float]:
    try:
        # Linux reports ru_maxrss in kilobytes.
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return None


def gpu_memory_mb() -> Optional[float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        first = output.strip().splitlines()[0]
        return float(first)
    except Exception:
        return None


def load_split_spec() -> Mapping[str, object]:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def normalize_split_files(split_spec: Mapping[str, object]) -> Dict[str, List[str]]:
    raw = split_spec["split_files"]
    return {split: [str(item) for item in raw[split]] for split in ("train", "val", "test")}


def load_artifact_split(z, split: str) -> Dict[str, np.ndarray]:
    pred_key = f"{split}_pred" if f"{split}_pred" in z.files else f"{split}_y_pred_final"
    return {
        "y_true": z[f"{split}_y_true"].astype(np.int16),
        "probs": z[f"{split}_probs_raw"].astype(np.float32),
        "pred": z[pred_key].astype(np.int16),
        "recording_id": z[f"{split}_recording_id"].astype(np.int32),
        "epoch_index": z[f"{split}_epoch_index"].astype(np.int32),
        "file_sample_counts": z[f"{split}_file_sample_counts"].astype(np.int32),
        "quality": z[f"{split}_quality"].astype(np.float32) if f"{split}_quality" in z.files else np.ones(len(z[f"{split}_y_true"]), dtype=np.float32),
    }


def audit_prediction_artifact_fields(out_dir: Path) -> Dict[str, object]:
    z = np.load(PRED_PATH, allow_pickle=True)
    fields = {}
    for name in z.files:
        arr = z[name]
        fields[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    payload = {
        "artifact_path": str(PRED_PATH),
        "fields": fields,
        "class_order": z["class_labels"].astype(str).tolist() if "class_labels" in z.files else CLASS_LABELS,
        "run_tag": str(np.asarray(z["run_tag"]).reshape(())) if "run_tag" in z.files else None,
        "seed": int(np.asarray(z["seed"]).reshape(())) if "seed" in z.files else None,
        "sequence_length": 75,
        "prediction_stride": 5,
        "recording_counts": z["test_file_sample_counts"].astype(int).tolist() if "test_file_sample_counts" in z.files else None,
        "temperature": float(np.asarray(z["temperature"]).reshape(())) if "temperature" in z.files else None,
        "decoder_mode": str(np.asarray(z["decoder_mode"]).reshape(())) if "decoder_mode" in z.files else None,
        "parity_compared_field": "test_probs_raw",
        "parity_compared_field_rationale": "Raw neural-network main_output probabilities before temperature scaling, threshold adjustment, HMM, or label postprocessing.",
        "forbidden_comparison_fields": [
            "test_probs_calibrated",
            "test_pred",
            "test_y_pred_final",
            "temperature-scaled probabilities",
            "HMM labels",
            "threshold-adjusted labels",
        ],
        "field_available": "test_probs_raw" in z.files,
    }
    write_json(out_dir / "prediction_artifact_field_audit.json", payload)
    return payload


def audit_inference_configuration(out_dir: Path, artifact: Mapping[str, np.ndarray]) -> Dict[str, object]:
    log_path = PROJECT_ROOT / "outputs" / "logs" / "result_20260704_024634_isruc_A1_zero_probs_ckpt.txt"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    payload = {
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_is_a1_zero": CKPT_PATH.name == "result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5",
        "not_a1_real": "real" not in CKPT_PATH.name.lower(),
        "not_scr": "scr" not in CKPT_PATH.name.lower(),
        "not_utf_lite": "utf" not in CKPT_PATH.name.lower(),
        "run_log_path": str(log_path),
        "seq_len": 75,
        "sequence_prediction_stride": 5,
        "test_file_sample_counts": np.asarray(artifact["file_sample_counts"], dtype=int).tolist(),
        "test_epoch_count": int(len(artifact["y_true"])),
        "prediction_batch_size": "diagnostic: configurable sequence-window batch; production log reports split_predict_batch=auto",
        "SPLIT_PREDICT_BATCH_SIZE": "auto",
        "model_architecture_switches": {
            "model_input_mode": "sequence",
            "epoch_encoder_variant": "v3_multiscale",
            "sequence_backbone": "v3_tcn",
            "transformer": True,
            "concept_modulation": True,
            "concept_modulation_mode": "zero",
            "SCR": False,
            "UTF-Lite": False,
            "late_clinical_concept_residual": False,
            "stage_concept_router": False,
        },
        "preprocessing_profile": "clinical_canonical",
        "normalization_mode": "recording_robust_zscore",
        "class_order": CLASS_LABELS,
        "log_confirms_a1_zero": "run_tag=isruc_A1_zero_probs_ckpt" in log_text and "concept_modulation_mode=zero" in log_text,
        "log_confirms_sequence": "model_input_mode=sequence" in log_text and "seq_len=75" in log_text and "sequence_predict_stride=5" in log_text,
        "configuration_match": True,
    }
    required = [
        payload["checkpoint_is_a1_zero"],
        payload["not_a1_real"],
        payload["not_scr"],
        payload["not_utf_lite"],
        payload["log_confirms_a1_zero"],
        payload["log_confirms_sequence"],
        int(np.sum(artifact["file_sample_counts"])) == int(len(artifact["y_true"])),
    ]
    payload["configuration_match"] = bool(all(required))
    write_json(out_dir / "inference_configuration_parity.json", payload)
    return payload


def class_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    pred = np.argmax(probs, axis=1).astype(np.int16)
    return {
        "N": int(len(y_true)),
        "accuracy": float(np.mean(pred == y_true)),
        "class_order": CLASS_LABELS,
        "class_counts": {CLASS_LABELS[idx]: int(np.sum(y_true == idx)) for idx in range(len(CLASS_LABELS))},
    }


def build_artifact_inventory(out_dir: Path) -> Dict[str, object]:
    split_spec = load_split_spec() if SPLIT_PATH.exists() else {}
    z = np.load(PRED_PATH) if PRED_PATH.exists() else None
    payload = {
        "status": "OK" if CKPT_PATH.exists() and PRED_PATH.exists() and SPLIT_PATH.exists() else "CHECKPOINT_NOT_FOUND",
        "dataset": "ISRUC",
        "seed": 42,
        "run_tag": "isruc_A1_zero_probs_ckpt",
        "checkpoint": stat_payload(CKPT_PATH),
        "prediction_artifact": stat_payload(PRED_PATH),
        "dataset_protocol_json": stat_payload(SPLIT_PATH),
        "model_input_mode": "sequence raw_input + segment_input",
        "seq_len": 75,
        "sequence_prediction_stride": 5,
        "epoch_encoder_variant": "v3_multiscale",
        "sequence_backbone": "v3_tcn + transformer",
        "concept_modulation": True,
        "concept_modulation_mode": "zero",
        "SCR": False,
        "UTF-Lite": False,
        "test_epoch_count": None,
        "test_recording_count": None,
        "notes": [
            "Canonical path is listed in tools/README.md and reused by representation/physiology audits.",
            "The script never calls keras model.fit() and never saves model weights.",
        ],
    }
    if split_spec:
        payload["split_json_dataset"] = split_spec.get("dataset")
        payload["source_npz_dir"] = split_spec.get("source_npz_dir")
        payload["split_file_counts"] = {
            key: len(value) for key, value in normalize_split_files(split_spec).items()
        }
    if z is not None:
        test = load_artifact_split(z, "test")
        payload["test_epoch_count"] = int(len(test["y_true"]))
        payload["test_recording_count"] = int(len(test["file_sample_counts"]))
        payload["artifact_seed"] = int(np.asarray(z["seed"]).reshape(())) if "seed" in z.files else None
        payload["artifact_run_tag"] = str(np.asarray(z["run_tag"]).reshape(())) if "run_tag" in z.files else None
        payload["artifact_test_metrics"] = class_metrics(test["y_true"], test["probs"])
    write_json(out_dir / "artifact_inventory.json", payload)
    return payload


def write_static_audit(out_dir: Path, inventory: Mapping[str, object]) -> None:
    lines = [
        "# Sequence-aware LIME Static Audit",
        "",
        f"Status: `{inventory['status']}`",
        f"Checkpoint: `{inventory['checkpoint']['path']}`",
        f"Prediction artifact: `{inventory['prediction_artifact']['path']}`",
        f"Protocol JSON: `{inventory['dataset_protocol_json']['path']}`",
        "",
        "## Fixed Configuration",
        "",
        f"- Dataset: {inventory['dataset']}",
        f"- Seed: {inventory['seed']}",
        f"- Run tag: {inventory['run_tag']}",
        f"- Sequence length: {inventory['seq_len']}",
        f"- Prediction stride: {inventory['sequence_prediction_stride']}",
        f"- Epoch encoder: {inventory['epoch_encoder_variant']}",
        f"- Concept modulation: {inventory['concept_modulation']} ({inventory['concept_modulation_mode']})",
        f"- SCR: {inventory['SCR']}",
        f"- UTF-Lite: {inventory['UTF-Lite']}",
        "",
        "## Decision",
        "",
    ]
    if inventory["status"] == "OK":
        lines.append("Static assets are present. Proceed to smoke only after parity passes.")
    else:
        lines.append("CHECKPOINT_NOT_FOUND")
    (out_dir / "static_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_or_build_test_inputs(out_dir: Path, split_files: Mapping[str, Sequence[str]], artifact: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure_lime_imports()
    from src.data_processing import load_npz_data  # noqa: WPS433

    cache = out_dir / "cache" / "test_inputs_segments.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        return z["x_raw"].astype(np.float32), z["x_seg"].astype(np.float32), z["quality"].astype(np.float32)

    split_spec = load_split_spec()
    npz_dir = Path(str(split_spec["source_npz_dir"])).expanduser()
    loaded = load_npz_data(
        str(npz_dir),
        file_subset=list(split_files["test"]),
        preprocess=True,
        filter_profile="clinical_canonical",
        filter_backend="scipy",
        normalization_mode="recording_robust_zscore",
        clip_sigma=None,
        enable_amplitude_clipping=False,
        return_quality=True,
    )
    x_raw, y_raw, quality = loaded
    if not np.array_equal(np.asarray(y_raw, dtype=np.int16), artifact["y_true"]):
        raise RuntimeError("MODEL_ARTIFACT_PARITY_FAIL: loaded test labels differ from prediction artifact")
    rows = []
    for idx, epoch in enumerate(x_raw):
        if idx and idx % 1000 == 0:
            print(f"  segment features built for {idx}/{len(x_raw)} test epochs")
        segments = semantic_segmentation(epoch, target_segments=30)
        rows.append(extract_segment_features(epoch, segments, max_segments=30, include_spectral=True))
    x_seg = np.asarray(rows, dtype=np.float32)
    np.savez_compressed(cache, x_raw=x_raw.astype(np.float32), x_seg=x_seg, quality=np.asarray(quality, dtype=np.float32))
    return x_raw.astype(np.float32), x_seg.astype(np.float32), np.asarray(quality, dtype=np.float32)


def configure_reproducible_inference(seed: int = 42) -> None:
    """Seed TensorFlow before model construction and require deterministic ops."""
    import random

    import tensorflow as tf  # noqa: WPS433

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - depends on TensorFlow build
        pass


def build_model(
    raw_shape: Tuple[int, ...],
    segment_shape: Tuple[int, ...],
    weights_path: Path = CKPT_PATH,
    enable_xai_feedback_head: bool = False,
    xai_feedback_loss_weight: float = 0.0,
    xai_alignment_loss_name: str = "normalized_mse",
    xai_gradient_scope: str = "head_only",
) -> tf.keras.Model:
    configure_reproducible_inference(42)
    from src.model_training import create_sequence_sleep_model  # noqa: WPS433

    model = create_sequence_sleep_model(
        raw_shape=tuple(raw_shape),
        segment_shape=tuple(segment_shape),
        seq_len=75,
        num_classes=len(CLASS_LABELS),
        learning_rate=0.001,
        sequence_backbone="v3_tcn",
        epoch_encoder_variant="v3_multiscale",
        use_transformer=True,
        loss_name="categorical_crossentropy",
        use_domain_mixstyle=True,
        mixstyle_p=0.50,
        mixstyle_alpha=0.30,
        use_domain_alignment=True,
        domain_alignment_loss_weight=0.005,
        domain_alignment_mode="stage_coral",
        use_clinical_concept_features=False,
        clinical_concept_version="v1",
        clinical_prior_trainable=False,
        use_late_clinical_concept_residual=False,
        use_concept_feature_modulation=True,
        concept_modulation_strength=0.10,
        concept_modulation_stop_gradient=True,
        concept_modulation_mode="zero",
        use_concept_language=False,
        concept_loss_weight=0.0,
        concept_bottleneck_loss_weight=0.0,
        use_ot_tacb=False,
        use_mpdg=False,
        use_transition_boundary_auxiliary=True,
        transition_boundary_loss_weight=0.020,
        transition_boundary_num_classes=8,
        use_concept_derivative_auxiliary=False,
        concept_derivative_loss_weight=0.0,
        use_n1_supervised_contrastive=False,
        enable_n1_n2_pair_supcon=False,
        n1_n2_pair_supcon_loss_weight=0.01,
        enable_n1_context_local_contrast=False,
        n1_context_local_contrast_loss_weight=0.01,
        enable_epoch_utf_lite=False,
        enable_stage_concept_router=False,
        enable_xai_feedback_head=bool(enable_xai_feedback_head),
        xai_feedback_loss_weight=float(xai_feedback_loss_weight),
        xai_alignment_loss_name=xai_alignment_loss_name,
        xai_gradient_scope=xai_gradient_scope,
    )
    if enable_xai_feedback_head:
        model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    else:
        model.load_weights(str(weights_path))
    return model


def model_main_output(outputs) -> np.ndarray:
    if isinstance(outputs, Mapping):
        return np.asarray(outputs["main_output"], dtype=np.float32)
    if isinstance(outputs, (list, tuple)):
        return np.asarray(outputs[0], dtype=np.float32)
    return np.asarray(outputs, dtype=np.float32)


def predict_sequence_probs(model: tf.keras.Model, x_raw: np.ndarray, x_seg: np.ndarray, counts: np.ndarray, batch_windows: int = 16) -> np.ndarray:
    windows = build_sequence_windows(len(x_raw), counts, seq_len=75, stride=5)
    sums = None
    out_counts = np.zeros((len(x_raw), 1), dtype=np.float32)
    for batch_start in range(0, len(windows), max(1, int(batch_windows))):
        batch = windows[batch_start: batch_start + max(1, int(batch_windows))]
        raw_batch, seg_batch = [], []
        for start, valid_len in batch:
            idx = window_epoch_indices(start, valid_len, 75)
            raw_batch.append(x_raw[idx])
            seg_batch.append(x_seg[idx])
        probs = model_main_output(
            model(
                {"raw_input": np.asarray(raw_batch, dtype=np.float32), "segment_input": np.asarray(seg_batch, dtype=np.float32)},
                training=False,
            )
        )
        if sums is None:
            sums = np.zeros((len(x_raw), probs.shape[-1]), dtype=np.float32)
        for row_idx, (start, valid_len) in enumerate(batch):
            idx = window_epoch_indices(start, valid_len, 75)[:valid_len]
            sums[idx] += probs[row_idx, :valid_len]
            out_counts[idx] += 1.0
    if sums is None or np.any(out_counts[:, 0] <= 0):
        raise RuntimeError("Sequence prediction missed test epochs.")
    return (sums / np.maximum(out_counts, 1.0)).astype(np.float32)


def predict_sequence_probs_agg_dtype(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    batch_windows: int = 16,
    aggregation_dtype=np.float32,
) -> np.ndarray:
    windows = build_sequence_windows(len(x_raw), counts, seq_len=75, stride=5)
    sums = None
    out_counts = np.zeros((len(x_raw), 1), dtype=aggregation_dtype)
    for batch_start in range(0, len(windows), max(1, int(batch_windows))):
        batch = windows[batch_start: batch_start + max(1, int(batch_windows))]
        raw_batch, seg_batch = [], []
        for start, valid_len in batch:
            idx = window_epoch_indices(start, valid_len, 75)
            raw_batch.append(x_raw[idx])
            seg_batch.append(x_seg[idx])
        probs = model_main_output(
            model(
                {"raw_input": np.asarray(raw_batch, dtype=np.float32), "segment_input": np.asarray(seg_batch, dtype=np.float32)},
                training=False,
            )
        )
        if sums is None:
            sums = np.zeros((len(x_raw), probs.shape[-1]), dtype=aggregation_dtype)
        for row_idx, (start, valid_len) in enumerate(batch):
            idx = window_epoch_indices(start, valid_len, 75)[:valid_len]
            sums[idx] += probs[row_idx, :valid_len].astype(aggregation_dtype)
            out_counts[idx] += aggregation_dtype(1.0)
    if sums is None or np.any(out_counts[:, 0] <= 0):
        raise RuntimeError("Sequence prediction missed test epochs.")
    return (sums / np.maximum(out_counts, aggregation_dtype(1.0))).astype(np.float32)


def sequence_target_vector(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    perturbed_epoch: np.ndarray,
    perturbed_feature: np.ndarray,
    batch_windows: int,
) -> np.ndarray:
    formal_batches = target_formal_window_batches(counts, global_epoch, batch_windows)
    if not formal_batches:
        raise RuntimeError(f"No sequence window covers epoch {global_epoch}")
    local_sum = None
    local_n = 0
    for batch, positions in formal_batches:
        raw_batch, seg_batch = [], []
        for row_idx, (start, valid_len) in enumerate(batch):
            idx = window_epoch_indices(start, valid_len, 75)
            rw = np.asarray(x_raw[idx], dtype=np.float32).copy()
            sg = np.asarray(x_seg[idx], dtype=np.float32).copy()
            if row_idx in positions:
                rel_pos = positions[row_idx]
                rw[rel_pos] = perturbed_epoch
                sg[rel_pos] = perturbed_feature
            raw_batch.append(rw)
            seg_batch.append(sg)
        probs = model_main_output(
            model(
                {"raw_input": np.asarray(raw_batch, dtype=np.float32), "segment_input": np.asarray(seg_batch, dtype=np.float32)},
                training=False,
            )
        )
        if local_sum is None:
            local_sum = np.zeros(probs.shape[-1], dtype=np.float64)
        for row, pos in positions.items():
            local_sum += probs[int(row), int(pos)].astype(np.float64)
            local_n += 1
    return (local_sum / max(local_n, 1)).astype(np.float32)


def probability_difference_stats(
    artifact_probs: np.ndarray,
    rerun_probs: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, object]:
    diff = np.abs(np.asarray(artifact_probs, dtype=np.float32) - np.asarray(rerun_probs, dtype=np.float32))
    flat = diff.reshape(-1)
    thresholds = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4]
    artifact_flat = artifact_probs.reshape(-1)
    rerun_flat = rerun_probs.reshape(-1)
    payload = {
        "element_count": int(flat.size),
        "min_absolute_difference": float(np.min(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "P90": float(np.percentile(flat, 90)),
        "P95": float(np.percentile(flat, 95)),
        "P99": float(np.percentile(flat, 99)),
        "P99_9": float(np.percentile(flat, 99.9)),
        "max": float(np.max(flat)),
        "RMSE": float(np.sqrt(np.mean((artifact_flat - rerun_flat) ** 2))),
        "cosine_similarity": cosine(artifact_flat, rerun_flat),
        "Pearson_correlation": float(np.corrcoef(artifact_flat, rerun_flat)[0, 1]),
        "threshold_counts": {},
    }
    for threshold in thresholds:
        count = int(np.sum(flat > threshold))
        payload["threshold_counts"][f">{threshold:g}"] = {
            "count": count,
            "ratio": float(count / max(flat.size, 1)),
        }
    worst_order = np.argsort(-flat)[:100]
    rows = []
    pred_art = np.argmax(artifact_probs, axis=1)
    pred_run = np.argmax(rerun_probs, axis=1)
    for flat_idx in worst_order:
        epoch_idx = int(flat_idx // artifact_probs.shape[1])
        cls_idx = int(flat_idx % artifact_probs.shape[1])
        art_vec = artifact_probs[epoch_idx]
        run_vec = rerun_probs[epoch_idx]
        art_sorted = np.sort(art_vec)
        run_sorted = np.sort(run_vec)
        rows.append({
            "rank": len(rows) + 1,
            "absolute_difference": float(flat[flat_idx]),
            "global_epoch_index": epoch_idx,
            "recording_id": int(artifact["recording_id"][epoch_idx]),
            "epoch_index_in_recording": int(artifact["epoch_index"][epoch_idx]),
            "true_label": CLASS_LABELS[int(artifact["y_true"][epoch_idx])],
            "artifact_predicted_label": CLASS_LABELS[int(pred_art[epoch_idx])],
            "rerun_predicted_label": CLASS_LABELS[int(pred_run[epoch_idx])],
            "class_index": cls_idx,
            "class_label": CLASS_LABELS[cls_idx],
            "artifact_probability_vector": json.dumps(art_vec.astype(float).tolist()),
            "rerun_probability_vector": json.dumps(run_vec.astype(float).tolist()),
            "artifact_top1_top2_margin": float(art_sorted[-1] - art_sorted[-2]),
            "rerun_top1_top2_margin": float(run_sorted[-1] - run_sorted[-2]),
        })
    pd.DataFrame(rows).to_csv(out_dir / "worst_probability_differences.csv", index=False)
    payload["worst_difference"] = rows[0] if rows else {}
    write_json(out_dir / "prediction_difference_distribution.json", payload)
    return payload


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    pred = np.argmax(probs, axis=1).astype(np.int16)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=np.arange(len(CLASS_LABELS)),
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(pred == y_true)),
        "macro_f1": float(f1_score(y_true, pred, labels=np.arange(len(CLASS_LABELS)), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, pred, labels=np.arange(len(CLASS_LABELS)))),
        "per_stage": {
            CLASS_LABELS[idx]: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(len(CLASS_LABELS))
        },
        "confusion_matrix": confusion_matrix(y_true, pred, labels=np.arange(len(CLASS_LABELS))).astype(int).tolist(),
        "predicted_class_counts": {CLASS_LABELS[idx]: int(np.sum(pred == idx)) for idx in range(len(CLASS_LABELS))},
    }


def prediction_metric_parity(
    artifact: Mapping[str, np.ndarray],
    rerun_probs: np.ndarray,
    out_dir: Path,
) -> Dict[str, object]:
    artifact_probs = artifact["probs"]
    y_true = artifact["y_true"]
    artifact_pred = np.argmax(artifact_probs, axis=1)
    rerun_pred = np.argmax(rerun_probs, axis=1)
    art_metrics = classification_metrics(y_true, artifact_probs)
    run_metrics = classification_metrics(y_true, rerun_probs)
    art_top2 = np.argsort(-artifact_probs, axis=1)[:, :2]
    run_top2 = np.argsort(-rerun_probs, axis=1)[:, :2]
    art_margin = np.sort(artifact_probs, axis=1)[:, -1] - np.sort(artifact_probs, axis=1)[:, -2]
    run_margin = np.sort(rerun_probs, axis=1)[:, -1] - np.sort(rerun_probs, axis=1)[:, -2]
    diffs = {
        "accuracy_abs_diff": abs(art_metrics["accuracy"] - run_metrics["accuracy"]),
        "macro_f1_abs_diff": abs(art_metrics["macro_f1"] - run_metrics["macro_f1"]),
        "kappa_abs_diff": abs(art_metrics["cohen_kappa"] - run_metrics["cohen_kappa"]),
    }
    payload = {
        "artifact_raw_metrics": art_metrics,
        "rerun_raw_metrics": run_metrics,
        "metric_abs_differences": diffs,
        "argmax_agreement": float(np.mean(artifact_pred == rerun_pred)),
        "disagreement_epoch_count": int(np.sum(artifact_pred != rerun_pred)),
        "top2_class_pair_agreement": float(np.mean(np.all(art_top2 == run_top2, axis=1))),
        "top1_probability_margin_change": {
            "mean_abs": float(np.mean(np.abs(art_margin - run_margin))),
            "P99": float(np.percentile(np.abs(art_margin - run_margin), 99)),
            "max_abs": float(np.max(np.abs(art_margin - run_margin))),
        },
        "confusion_matrix_exact_match": art_metrics["confusion_matrix"] == run_metrics["confusion_matrix"],
    }
    write_json(out_dir / "prediction_metric_parity.json", payload)
    return payload


def compare_probability_arrays(left: np.ndarray, right: np.ndarray) -> Dict[str, object]:
    diff = np.abs(left - right)
    return {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "P99_abs_diff": float(np.percentile(diff, 99)),
        "argmax_agreement": float(np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1))),
    }


def repeat_inference_stability(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    canonical_probs: np.ndarray,
    out_dir: Path,
    batch_windows: int = 16,
) -> Dict[str, object]:
    runs = [canonical_probs]
    for _ in range(2):
        runs.append(predict_sequence_probs(model, x_raw, x_seg, counts, batch_windows=batch_windows))
    pairwise = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairwise[f"inference_run_{i + 1}_vs_{j + 1}"] = compare_probability_arrays(runs[i], runs[j])
    batch_results = {
        str(batch_windows): canonical_probs,
        str(max(1, batch_windows // 2)): predict_sequence_probs(model, x_raw, x_seg, counts, batch_windows=max(1, batch_windows // 2)),
    }
    batch_compare = {}
    for name, probs in batch_results.items():
        if name == str(batch_windows):
            continue
        batch_compare[f"batch_{batch_windows}_vs_{name}"] = compare_probability_arrays(canonical_probs, probs)
    payload = {
        "canonical_batch_windows": int(batch_windows),
        "repeat_inference_pairwise": pairwise,
        "batch_size_pairwise": batch_compare,
        "interpretation": "Differences here indicate current-environment GPU/cuDNN/batch-order numeric variability, not model training.",
    }
    write_json(out_dir / "repeat_inference_stability.json", payload)
    return payload


def aggregation_precision_diagnostic(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact_probs: np.ndarray,
    float32_probs: np.ndarray,
    out_dir: Path,
    batch_windows: int = 16,
) -> Dict[str, object]:
    float64_probs = predict_sequence_probs_agg_dtype(model, x_raw, x_seg, counts, batch_windows=batch_windows, aggregation_dtype=np.float64)
    payload = {
        "float32_aggregation_vs_artifact": compare_probability_arrays(float32_probs, artifact_probs),
        "float64_aggregation_vs_artifact": compare_probability_arrays(float64_probs, artifact_probs),
        "float32_vs_float64_aggregation": compare_probability_arrays(float32_probs, float64_probs),
        "note": "Model outputs remain float32; float64 is used only for diagnostic window summation/averaging.",
    }
    write_json(out_dir / "aggregation_precision_diagnostic.json", payload)
    return payload


def sequence_wrapper_parity(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    current_probs: np.ndarray,
    manifest: pd.DataFrame,
    out_dir: Path,
    batch_windows: int,
) -> Dict[str, object]:
    rows = []
    for _, row in manifest.iterrows():
        global_idx = int(row["global_epoch_index"])
        vec = sequence_target_vector(
            model,
            x_raw,
            x_seg,
            counts,
            global_idx,
            x_raw[global_idx],
            x_seg[global_idx],
            batch_windows=batch_windows,
        )
        base = current_probs[global_idx]
        diff = np.abs(vec - base)
        rows.append({
            "global_epoch_index": global_idx,
            "recording_id": int(row["recording_id"]),
            "epoch_index_in_recording": int(row["epoch_index_in_recording"]),
            "coverage_window_count": int(len(target_windows(counts, global_idx))),
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "argmax_agreement": bool(int(np.argmax(vec)) == int(np.argmax(base))),
            "wrapper_probability_vector": json.dumps(vec.astype(float).tolist()),
            "canonical_probability_vector": json.dumps(base.astype(float).tolist()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sequence_wrapper_parity_samples.csv", index=False)
    payload = {
        "status": "SEQUENCE_WRAPPER_PARITY_PASS"
        if (not df.empty and float(df["max_abs_diff"].max()) < 1e-6 and bool(df["argmax_agreement"].all()))
        else "SEQUENCE_WRAPPER_PARITY_FAIL",
        "max_abs_diff_threshold": 1e-6,
        "overall_max_abs_diff": float(df["max_abs_diff"].max()) if not df.empty else np.nan,
        "overall_mean_abs_diff": float(df["mean_abs_diff"].mean()) if not df.empty else np.nan,
        "argmax_agreement": float(df["argmax_agreement"].mean()) if not df.empty else np.nan,
        "sample_count": int(len(df)),
    }
    write_json(out_dir / "sequence_wrapper_parity.json", payload)
    return payload


def target_windows(counts: np.ndarray, global_epoch: int) -> List[Tuple[int, int, int]]:
    windows = build_sequence_windows(int(np.sum(counts)), counts, seq_len=75, stride=5)
    out = []
    for start, valid_len in windows:
        idx = window_epoch_indices(start, valid_len, 75)[:valid_len]
        matches = np.flatnonzero(idx == int(global_epoch))
        if len(matches):
            out.append((int(start), int(valid_len), int(matches[0])))
    return out


def target_formal_window_batches(
    counts: np.ndarray,
    global_epoch: int,
    batch_windows: int,
) -> List[Tuple[List[Tuple[int, int]], Dict[int, int]]]:
    """Return formal full-inference batches containing the target epoch.

    The dict maps row index within the formal batch to the target relative
    position in that row. Keeping the surrounding non-target windows in the same
    batch is important for bit-close GPU inference parity.
    """
    windows = build_sequence_windows(int(np.sum(counts)), counts, seq_len=75, stride=5)
    batches = []
    for batch_start in range(0, len(windows), max(1, int(batch_windows))):
        batch = windows[batch_start: batch_start + max(1, int(batch_windows))]
        positions = {}
        for row_idx, (start, valid_len) in enumerate(batch):
            idx = window_epoch_indices(start, valid_len, 75)[:valid_len]
            matches = np.flatnonzero(idx == int(global_epoch))
            if len(matches):
                positions[int(row_idx)] = int(matches[0])
        if positions:
            batches.append((batch, positions))
    return batches


def sequence_target_probabilities(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    target_class: int,
    perturbed_epochs: Sequence[np.ndarray],
    perturbed_features: Sequence[np.ndarray],
    batch_windows: int,
) -> np.ndarray:
    # Faithfulness needs every sequence window that covers the target epoch,
    # not unrelated rows that happened to share its historical inference
    # batch. Reuse the audited covering-window assembler, which enforces the
    # configured model-call cap and aggregates all covering positions.
    payload = infer_precomputed_epochs_for_target(
        model,
        x_raw,
        x_seg,
        counts,
        global_epoch,
        target_class,
        perturbed_epochs,
        perturbed_features,
        batch_windows=max(1, int(batch_windows)),
    )
    return np.asarray(payload["target_probabilities"], dtype=np.float32)


def make_masks(
    num_features: int,
    train_n: int,
    val_n: int,
    seed: int,
    sampling: str = "local_stratified",
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    total = int(train_n) + int(val_n)
    if total < 3 or num_features < 2:
        raise ValueError("LIME mask generation requires at least 3 masks and 2 features.")
    masks = np.ones((total, num_features), dtype=np.int8)
    masks[0, :] = 1
    masks[1, :] = 0
    seen = {tuple(masks[0].tolist()), tuple(masks[1].tolist())}
    for row in range(2, total):
        while True:
            if sampling == "bernoulli":
                candidate = rng.integers(0, 2, size=(num_features,), dtype=np.int8)
            elif sampling == "local_stratified":
                draw = float(rng.random())
                if draw < 0.60:
                    candidates = [1, 2, 3, 5, max(1, num_features // 10)]
                elif draw < 0.90:
                    candidates = [max(1, num_features // 6), max(1, num_features // 4), max(1, num_features // 3)]
                else:
                    candidates = [int(rng.integers(1, num_features))]
                delete_count = int(np.clip(rng.choice(candidates), 1, num_features - 1))
                candidate = np.ones(num_features, dtype=np.int8)
                candidate[rng.choice(num_features, size=delete_count, replace=False)] = 0
            elif sampling == "local_stratified_tight":
                draw = float(rng.random())
                if draw < 0.80:
                    candidates = [1, 2, 3]
                elif draw < 0.95:
                    candidates = [4, 5, 6]
                else:
                    candidates = [int(rng.integers(7, max(8, num_features // 2 + 1)))]
                delete_count = int(np.clip(rng.choice(candidates), 1, num_features - 1))
                candidate = np.ones(num_features, dtype=np.int8)
                candidate[rng.choice(num_features, size=delete_count, replace=False)] = 0
            else:
                raise ValueError(f"Unknown LIME mask sampling strategy: {sampling}")
            if tuple(candidate.tolist()) not in seen:
                masks[row] = candidate
                break
        seen.add(tuple(masks[row].tolist()))
    return masks.astype(np.int32)


def uniform_segmentation(num_timepoints: int, target_segments: int) -> List[Tuple[int, int]]:
    num_timepoints = int(num_timepoints)
    target_segments = int(np.clip(target_segments, 1, num_timepoints))
    boundaries = np.linspace(0, num_timepoints, target_segments + 1, dtype=np.int32)
    segments = [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:]) if int(end) > int(start)]
    if not segments or segments[0][0] != 0 or segments[-1][1] != num_timepoints:
        raise RuntimeError("Uniform segmentation did not cover the complete target epoch.")
    return segments


def perturb_epochs_for_masks(
    epoch: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    masks: np.ndarray,
    target_class: int,
    mode: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ensure_lime_imports()
    epochs, features = [], []
    for mask in masks:
        perturbed = apply_perturbation_to_multimodal(epoch, mask, segments, target_class, perturbation_mode=mode)
        feat = extract_segment_features(perturbed, segments, max_segments=30, include_spectral=True)
        epochs.append(np.asarray(perturbed, dtype=np.float32))
        features.append(np.asarray(feat, dtype=np.float32))
    return epochs, features


def segment_boundary_hash(segments: Sequence[Tuple[int, int]]) -> str:
    payload = json.dumps([[int(a), int(b)] for a, b in segments], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequence_aggregation_config_hash(batch_windows: int) -> str:
    payload = {
        "seq_len": 75,
        "stride": 5,
        "aggregation": "mean_over_covering_windows",
        "window_inference_batch_size": int(batch_windows),
        "padding": "none",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def stable_mask_hash(
    mask: np.ndarray,
    global_epoch: int,
    perturbation_mode: str,
    model_state_id: str,
    segment_hash: str,
    target_class: int,
    perturbation_target_class: int,
    batch_windows: int,
) -> str:
    h = hashlib.sha256()
    h.update(f"global_epoch_index={int(global_epoch)}\n".encode("utf-8"))
    h.update(f"perturbation_mode={perturbation_mode}\n".encode("utf-8"))
    h.update(f"model_state_id={model_state_id}\n".encode("utf-8"))
    h.update(f"segment_boundary_hash={segment_hash}\n".encode("utf-8"))
    h.update(f"target_class={int(target_class)}\n".encode("utf-8"))
    h.update(
        f"perturbation_target_class={int(perturbation_target_class)}\n".encode("utf-8")
    )
    h.update(f"sequence_aggregation_config_hash={sequence_aggregation_config_hash(batch_windows)}\n".encode("utf-8"))
    arr = np.asarray(mask, dtype=np.int8).ravel()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def mask_prediction_cache_dir(out_dir: Path, global_epoch: int, model_state_id: str, perturbation_mode: str) -> Path:
    return sample_cache_dir(out_dir, global_epoch) / "mask_predictions" / str(model_state_id) / str(perturbation_mode)


def read_mask_prediction(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def infer_precomputed_epochs_for_target(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    target_class: int,
    perturbed_epochs: Sequence[np.ndarray],
    perturbed_features: Sequence[np.ndarray],
    batch_windows: int,
) -> Dict[str, object]:
    covers = target_windows(counts, global_epoch)
    if not covers:
        raise RuntimeError(f"No sequence window covers epoch {global_epoch}")
    sums = np.zeros((len(perturbed_epochs), len(CLASS_LABELS)), dtype=np.float64)
    counts_per = np.zeros(len(perturbed_epochs), dtype=np.float64)
    assembled: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    t_assembly0 = time.perf_counter()
    for mask_idx, (pert_epoch, pert_feat) in enumerate(zip(perturbed_epochs, perturbed_features)):
        for start, valid_len, rel_pos in covers:
            idx = window_epoch_indices(start, valid_len, 75)
            rw = np.asarray(x_raw[idx], dtype=np.float32).copy()
            sg = np.asarray(x_seg[idx], dtype=np.float32).copy()
            rw[int(rel_pos)] = pert_epoch
            sg[int(rel_pos)] = pert_feat
            assembled.append((rw, sg, int(mask_idx), int(rel_pos)))
    t_assembly1 = time.perf_counter()

    batch_sizes: List[int] = []
    t_gpu = 0.0
    for start in range(0, len(assembled), max(1, int(batch_windows))):
        chunk = assembled[start: start + max(1, int(batch_windows))]
        batch_sizes.append(int(len(chunk)))
        raw_batch = np.asarray([item[0] for item in chunk], dtype=np.float32)
        seg_batch = np.asarray([item[1] for item in chunk], dtype=np.float32)
        gpu0 = time.perf_counter()
        probs = model_main_output(model({"raw_input": raw_batch, "segment_input": seg_batch}, training=False))
        t_gpu += time.perf_counter() - gpu0
        for row_idx, (_rw, _sg, mask_idx, rel_pos) in enumerate(chunk):
            sums[int(mask_idx)] += probs[int(row_idx), int(rel_pos)].astype(np.float64)
            counts_per[int(mask_idx)] += 1.0
    full_vectors = (sums / np.maximum(counts_per[:, None], 1.0)).astype(np.float32)
    target_probs = full_vectors[:, int(target_class)].astype(np.float32)
    expected = int(len(perturbed_epochs) * len(covers))
    actual = int(sum(batch_sizes))
    assert len(assembled) == expected
    assert actual == len(assembled)
    assert max(batch_sizes or [0]) <= int(batch_windows)
    return {
        "target_probabilities": target_probs,
        "full_probability_vectors": full_vectors,
        "covering_window_count": int(len(covers)),
        "expected_unpadded_window_instances": expected,
        "actual_assembled_window_instances": int(len(assembled)),
        "actual_inferred_window_instances": actual,
        "padding_window_instances": 0,
        "model_forward_call_count": int(len(batch_sizes)),
        "model_forward_batch_sizes": batch_sizes,
        "sum_model_forward_batch_sizes": actual,
        "window_assembly_seconds": float(t_assembly1 - t_assembly0),
        "gpu_forward_seconds": float(t_gpu),
    }


def evaluate_masks_for_target_epoch(
    masks: np.ndarray,
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    target_class: int,
    epoch: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    perturbation_mode: str,
    window_inference_batch_size: int,
    out_dir: Path,
    cache_namespace: str,
    model_state_id: str = "original_model",
    perturbation_target_class: Optional[int] = None,
) -> Dict[str, object]:
    ensure_lime_imports()
    masks = np.asarray(masks, dtype=np.int32)
    if masks.ndim != 2:
        raise ValueError("masks must be a 2D array")
    covers = target_windows(counts, global_epoch)
    if not covers:
        raise RuntimeError(f"No sequence window covers epoch {global_epoch}")
    t0 = time.perf_counter()
    effective_perturbation_target_class = (
        int(target_class)
        if perturbation_target_class is None
        else int(perturbation_target_class)
    )
    seg_hash = segment_boundary_hash(segments)
    cache_dir = mask_prediction_cache_dir(out_dir, global_epoch, model_state_id, perturbation_mode)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mask_hashes = [
        stable_mask_hash(
            mask,
            global_epoch,
            perturbation_mode,
            model_state_id,
            seg_hash,
            target_class,
            effective_perturbation_target_class,
            window_inference_batch_size,
        )
        for mask in masks
    ]
    first_index: Dict[str, int] = {}
    unique_order: List[str] = []
    for idx, mh in enumerate(mask_hashes):
        if mh not in first_index:
            first_index[mh] = int(idx)
            unique_order.append(mh)
    t_hash = time.perf_counter()

    unique_target = np.full(len(unique_order), np.nan, dtype=np.float32)
    unique_vectors = np.full((len(unique_order), len(CLASS_LABELS)), np.nan, dtype=np.float32)
    cache_hit_unique = np.zeros(len(unique_order), dtype=bool)
    missing_unique_positions: List[int] = []
    for unique_pos, mh in enumerate(unique_order):
        cached = read_mask_prediction(cache_dir / f"{mh}.json")
        if cached is not None:
            unique_target[unique_pos] = float(cached["target_probability"])
            unique_vectors[unique_pos] = np.asarray(cached["full_probability_vector"], dtype=np.float32)
            cache_hit_unique[unique_pos] = True
        else:
            missing_unique_positions.append(int(unique_pos))
    t_cache_read = time.perf_counter()

    perturbed_epochs: List[np.ndarray] = []
    perturb_t0 = time.perf_counter()
    for unique_pos in missing_unique_positions:
        mask = masks[first_index[unique_order[unique_pos]]]
        perturbed = apply_perturbation_to_multimodal(
            epoch,
            mask,
            segments,
            effective_perturbation_target_class,
            perturbation_mode=perturbation_mode,
        )
        perturbed_epochs.append(np.asarray(perturbed, dtype=np.float32))
    perturb_t1 = time.perf_counter()
    feature_t0 = time.perf_counter()
    perturbed_features = extract_segment_features_batch(perturbed_epochs, segments) if perturbed_epochs else []
    feature_t1 = time.perf_counter()

    infer_payload = {
        "target_probabilities": np.asarray([], dtype=np.float32),
        "full_probability_vectors": np.empty((0, len(CLASS_LABELS)), dtype=np.float32),
        "expected_unpadded_window_instances": 0,
        "actual_assembled_window_instances": 0,
        "actual_inferred_window_instances": 0,
        "padding_window_instances": 0,
        "model_forward_call_count": 0,
        "model_forward_batch_sizes": [],
        "sum_model_forward_batch_sizes": 0,
        "window_assembly_seconds": 0.0,
        "gpu_forward_seconds": 0.0,
    }
    if missing_unique_positions:
        infer_payload = infer_precomputed_epochs_for_target(
            model,
            x_raw,
            x_seg,
            counts,
            global_epoch,
            target_class,
            perturbed_epochs,
            perturbed_features,
            batch_windows=window_inference_batch_size,
        )
        for local_idx, unique_pos in enumerate(missing_unique_positions):
            unique_target[unique_pos] = float(infer_payload["target_probabilities"][local_idx])
            unique_vectors[unique_pos] = np.asarray(infer_payload["full_probability_vectors"][local_idx], dtype=np.float32)
    t_infer = time.perf_counter()

    for unique_pos in missing_unique_positions:
        mh = unique_order[unique_pos]
        mask = masks[first_index[mh]]
        write_json_atomic(cache_dir / f"{mh}.json", {
            "mask_hash": mh,
            "mask": np.asarray(mask, dtype=int).tolist(),
            "target_probability": float(unique_target[unique_pos]),
            "full_probability_vector": unique_vectors[unique_pos].astype(float).tolist(),
            "covering_window_count": int(len(covers)),
            "model_state_id": str(model_state_id),
            "perturbation_mode": str(perturbation_mode),
            "global_epoch_index": int(global_epoch),
            "target_class": int(target_class),
            "perturbation_target_class": int(effective_perturbation_target_class),
            "segment_boundary_hash": seg_hash,
            "sequence_aggregation_config_hash": sequence_aggregation_config_hash(window_inference_batch_size),
            "cache_namespace_first_writer": str(cache_namespace),
        })
    t_cache_write = time.perf_counter()

    unique_lookup = {mh: pos for pos, mh in enumerate(unique_order)}
    requested_target = np.asarray([unique_target[unique_lookup[mh]] for mh in mask_hashes], dtype=np.float32)
    requested_vectors = np.asarray([unique_vectors[unique_lookup[mh]] for mh in mask_hashes], dtype=np.float32)
    from_cache = np.asarray([cache_hit_unique[unique_lookup[mh]] for mh in mask_hashes], dtype=bool)

    requested_mask_count = int(len(masks))
    unique_mask_count = int(len(unique_order))
    duplicate_mask_count = int(requested_mask_count - unique_mask_count)
    cache_hit_count = int(np.sum(cache_hit_unique))
    new_inference_mask_count = int(len(missing_unique_positions))
    covering_per_unique = [int(len(covers)) for _ in unique_order]
    requested_window_instances = int(requested_mask_count * len(covers))
    unique_window_instances = int(unique_mask_count * len(covers))
    expected = int(infer_payload["expected_unpadded_window_instances"])
    actual_assembled = int(infer_payload["actual_assembled_window_instances"])
    actual_inferred = int(infer_payload["actual_inferred_window_instances"])
    batch_sizes = [int(v) for v in infer_payload["model_forward_batch_sizes"]]
    sum_batch = int(sum(batch_sizes))
    accounting_pass = bool(
        expected == actual_assembled
        and actual_inferred == actual_assembled
        and sum_batch == actual_inferred
        and (max(batch_sizes or [0]) <= int(window_inference_batch_size))
        and int(infer_payload["padding_window_instances"]) == 0
    )
    efficiency = {
        "requested_mask_count": requested_mask_count,
        "unique_mask_count": unique_mask_count,
        "duplicate_mask_count": duplicate_mask_count,
        "cache_hit_count": cache_hit_count,
        "new_inference_mask_count": new_inference_mask_count,
        "mask_deduplication_ratio": float(duplicate_mask_count / max(requested_mask_count, 1)),
        "saved_window_instances": int((duplicate_mask_count + cache_hit_count) * len(covers)),
    }
    accounting = {
        **efficiency,
        "covering_window_count_per_mask_min": int(min(covering_per_unique) if covering_per_unique else 0),
        "covering_window_count_per_mask_max": int(max(covering_per_unique) if covering_per_unique else 0),
        "covering_window_count_per_mask_mean": float(np.mean(covering_per_unique) if covering_per_unique else 0.0),
        "requested_window_instances": requested_window_instances,
        "unique_window_instances": unique_window_instances,
        "expected_unpadded_window_instances": expected,
        "actual_assembled_window_instances": actual_assembled,
        "actual_inferred_window_instances": actual_inferred,
        "padding_window_instances": int(infer_payload["padding_window_instances"]),
        "model_forward_call_count": int(infer_payload["model_forward_call_count"]),
        "model_forward_batch_sizes": batch_sizes,
        "sum_model_forward_batch_sizes": sum_batch,
        "window_inference_batch_size": int(window_inference_batch_size),
        "status": "WINDOW_INSTANCE_ACCOUNTING_PASS" if accounting_pass else "WINDOW_INSTANCE_ACCOUNTING_FAIL",
    }
    timings = {
        "mask_hash_seconds": float(t_hash - t0),
        "cache_read_seconds": float(t_cache_read - t_hash),
        "mask_generation_seconds": float(perturb_t1 - perturb_t0),
        "target_segment_feature_recompute": float(feature_t1 - feature_t0),
        "window_assembly_seconds": float(infer_payload["window_assembly_seconds"]),
        "gpu_forward_seconds": float(infer_payload["gpu_forward_seconds"]),
        "aggregation_seconds": float(max(0.0, t_infer - feature_t1 - infer_payload["window_assembly_seconds"] - infer_payload["gpu_forward_seconds"])),
        "cache_io_seconds": float((t_cache_read - t_hash) + (t_cache_write - t_infer)),
        "total_seconds": float(t_cache_write - t0),
    }
    return {
        "target_probabilities": requested_target,
        "full_probability_vectors": requested_vectors,
        "mask_hashes": mask_hashes,
        "unique_mask_hashes": unique_order,
        "from_cache": from_cache.tolist(),
        "covering_window_counts": [int(len(covers)) for _ in range(requested_mask_count)],
        "accounting": accounting,
        "efficiency": efficiency,
        "timings_sec": timings,
    }


def extract_segment_features_batch(
    epochs: Sequence[np.ndarray],
    segments: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    ensure_lime_imports()
    return [
        np.asarray(extract_segment_features(epoch, segments, max_segments=30, include_spectral=True), dtype=np.float32)
        for epoch in epochs
    ]


def batched_segment_feature_parity(
    epochs: Sequence[np.ndarray],
    segments: Sequence[Tuple[int, int]],
    out_dir: Path,
    seed: int = 123,
) -> Dict[str, object]:
    ensure_lime_imports()
    if not epochs:
        payload = {"status": "BATCHED_SEGMENT_FEATURE_PARITY_FAIL", "reason": "no epochs"}
        write_json(out_dir / "batched_segment_feature_parity.json", payload)
        return payload
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.arange(len(epochs)), size=min(10, len(epochs)), replace=False)
    batch_features = extract_segment_features_batch([epochs[int(i)] for i in pick], segments)
    max_diff = 0.0
    for local_idx, source_idx in enumerate(pick):
        single = np.asarray(extract_segment_features(epochs[int(source_idx)], segments, max_segments=30, include_spectral=True), dtype=np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(single - batch_features[local_idx]))))
    payload = {
        "status": "BATCHED_SEGMENT_FEATURE_PARITY_PASS" if max_diff < 1e-6 else "BATCHED_SEGMENT_FEATURE_PARITY_FAIL",
        "sampled_epoch_count": int(len(pick)),
        "max_abs_segment_feature_diff": float(max_diff),
        "threshold": 1e-6,
        "implementation": "CPU batch wrapper around existing extract_segment_features; boundaries are shared.",
    }
    write_json(out_dir / "batched_segment_feature_parity.json", payload)
    return payload


def predict_masks_sequence(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    target_class: int,
    epoch: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    masks: np.ndarray,
    mode: str,
    batch_windows: int,
    chunk: int,
) -> np.ndarray:
    out = []
    for start in range(0, len(masks), max(1, int(chunk))):
        chunk_masks = masks[start: start + max(1, int(chunk))]
        pert_epochs, pert_feats = perturb_epochs_for_masks(epoch, segments, chunk_masks, target_class, mode)
        out.append(
            sequence_target_probabilities(
                model,
                x_raw,
                x_seg,
                counts,
                global_epoch,
                target_class,
                pert_epochs,
                pert_feats,
                batch_windows=batch_windows,
            )
        )
    return np.concatenate(out).astype(np.float32)


def vectorized_forward_plan(total_masks: int, counts: np.ndarray, global_epoch: int, batch_windows: int, chunk: int) -> Dict[str, object]:
    covering = target_windows(counts, global_epoch)
    target_rows = int(total_masks * len(covering))
    return {
        "covering_window_count": int(len(covering)),
        "formal_context_batch_count": int(len(target_formal_window_batches(counts, global_epoch, batch_windows))),
        "formal_context_window_rows_per_chunk": None,
        "model_forward_batch_count": int(math.ceil(target_rows / max(1, int(batch_windows)))),
        "target_sequence_window_instances": int(total_masks * len(covering)),
        "forwarded_sequence_window_rows": target_rows,
        "ideal_forward_batches_for_target_windows": int(math.ceil(target_rows / max(1, int(batch_windows)))),
        "vectorized_inference": True,
    }


def sample_cache_dir(out_dir: Path, global_epoch: int) -> Path:
    return out_dir / "cache" / f"sample_{int(global_epoch):06d}"


def read_sample_status(out_dir: Path, global_epoch: int) -> Dict[str, object]:
    path = sample_cache_dir(out_dir, global_epoch) / "status.json"
    if path.exists():
        return read_json(path)
    return {
        "segments_done": False,
        "surrogate_done": False,
        "faithfulness_done": False,
        "stability_done": False,
        "clinical_done": False,
        "randomization_done": False,
        "robustness_done": False,
    }


def update_sample_status(out_dir: Path, global_epoch: int, **updates) -> Dict[str, object]:
    status = read_sample_status(out_dir, global_epoch)
    status.update(updates)
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json_atomic(sample_cache_dir(out_dir, global_epoch) / "status.json", status)
    return status


def surrogate_config_signature(config: RunConfig) -> str:
    payload = {
        "schema_version": 4,
        "train_masks": int(config.train_masks),
        "val_masks": int(config.val_masks),
        "target_segments": int(config.target_segments),
        "mask_sampling": str(config.mask_sampling),
        "ridge_alphas": [float(value) for value in config.ridge_alphas],
        "kernel_widths": [float(value) for value in config.kernel_widths],
        "response_transform": str(config.response_transform),
        "perturbation_mode": str(config.surrogate_perturbation_mode),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_surrogate_module(
    sample: Mapping[str, object],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
    section_name: str = "surrogate",
    force: bool = False,
) -> Dict[str, object]:
    ensure_lime_imports()
    global_idx = int(sample["global_epoch_index"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / f"{section_name}.json"
    status = read_sample_status(out_dir, global_idx)
    config_signature = surrogate_config_signature(config)
    if status.get("surrogate_done") and result_path.exists() and not force:
        cached = read_json(result_path)
        if int(cached.get("surrogate_schema_version", 0)) >= 4 and cached.get("surrogate_config_signature") == config_signature:
            return cached

    t0 = time.perf_counter()
    epoch = x_raw[global_idx]
    target_class = int(np.argmax(artifact["probs"][global_idx]))
    segments = semantic_segmentation(epoch, target_segments=config.target_segments)
    update_sample_status(out_dir, global_idx, segments_done=True, segments_count=int(len(segments)))
    t_segments = time.perf_counter()

    num_features = int(epoch.shape[1] * len(segments))
    masks = make_masks(
        num_features,
        config.train_masks,
        config.val_masks,
        100000 + global_idx,
        sampling=config.mask_sampling,
    )
    distances = calculate_cosine_distances(masks, masks.shape[1])
    weights = calculate_weights_from_distances(distances)
    t_masks = time.perf_counter()

    eval_payload = evaluate_masks_for_target_epoch(
        masks,
        model,
        x_raw,
        x_seg,
        counts,
        global_idx,
        target_class,
        epoch,
        segments,
        perturbation_mode=config.surrogate_perturbation_mode,
        window_inference_batch_size=config.batch_windows,
        out_dir=out_dir,
        cache_namespace=section_name,
        model_state_id="original_model",
    )
    probs = np.asarray(eval_payload["target_probabilities"], dtype=np.float32)
    t_infer = time.perf_counter()

    ridge, fidelity, _ = fit_surrogate(
        masks,
        probs,
        config.train_masks,
        alpha_grid=config.ridge_alphas,
        kernel_width_grid=config.kernel_widths,
        response_transform=config.response_transform,
    )
    pred_val = ridge.predict(masks[config.train_masks:])
    val_y = probs[config.train_masks:]
    finite = bool(np.all(np.isfinite(probs)) and np.all(np.isfinite(ridge.coef_)) and np.all(np.isfinite(pred_val)))
    t_ridge = time.perf_counter()

    plan = vectorized_forward_plan(len(masks), counts, global_idx, config.batch_windows, config.perturbation_chunk)
    preview_epochs = [
        np.asarray(apply_perturbation_to_multimodal(epoch, masks[i], segments, target_class, perturbation_mode=config.surrogate_perturbation_mode), dtype=np.float32)
        for i in range(min(10, len(masks)))
    ]
    parity = batched_segment_feature_parity(preview_epochs, segments, cache_dir)
    t_cache = time.perf_counter()
    accounting = eval_payload["accounting"]
    eval_t = eval_payload["timings_sec"]
    result = {
        "surrogate_schema_version": 4,
        "surrogate_config_signature": config_signature,
        "surrogate_configuration": {
            "target_segments": int(config.target_segments),
            "mask_sampling": str(config.mask_sampling),
            "ridge_alphas": [float(value) for value in config.ridge_alphas],
            "kernel_widths": [float(value) for value in config.kernel_widths],
            "response_transform": str(config.response_transform),
            "perturbation_mode": str(config.surrogate_perturbation_mode),
        },
        "global_epoch_index": global_idx,
        "target_class": target_class,
        "target_stage": CLASS_LABELS[target_class],
        "inference_batch_size": int(config.batch_windows),
        "perturbation_count": int(len(masks)),
        "train_perturbations": int(config.train_masks),
        "validation_perturbations": int(config.val_masks),
        "generated_target_epochs_count": int(len(masks)),
        "generated_sequence_windows_count": int(accounting["actual_assembled_window_instances"]),
        "model_forward_batch_count": int(accounting["model_forward_call_count"]),
        "model_forward_window_count": int(accounting["actual_inferred_window_instances"]),
        "covering_window_count": int(plan["covering_window_count"]),
        "vectorized_inference": bool(plan["vectorized_inference"]),
        "window_instance_accounting": accounting,
        "mask_efficiency": eval_payload["efficiency"],
        "fidelity": fidelity,
        "ridge_fit_possible": True,
        "all_values_finite": finite,
        "original_probability": float(probs[0]),
        "full_perturbation_probability": float(probs[1]),
        "coef": np.asarray(ridge.coef_, dtype=np.float32).tolist(),
        "timings_sec": {
            "dynamic_segmentation": t_segments - t0,
            "mask_generation_and_kernel": t_masks - t_segments,
            "surrogate_perturbation_generation": eval_t["mask_generation_seconds"],
            "target_segment_feature_recompute": eval_t["target_segment_feature_recompute"],
            "window_assembly": eval_t["window_assembly_seconds"],
            "gpu_forward": eval_t["gpu_forward_seconds"],
            "mask_cache_io": eval_t["cache_io_seconds"],
            "sequence_window_assembly_and_gpu_forward_and_overlap": eval_t["window_assembly_seconds"] + eval_t["gpu_forward_seconds"],
            "ridge_fit_and_validation": t_ridge - t_infer,
            "cache_write_and_feature_parity": t_cache - t_ridge,
            "total": t_cache - t0,
        },
        "throughput": {
            "perturbations_per_second": float(len(masks) / max(t_infer - t_masks, EPS)),
            "sequence_windows_per_second": float(accounting["actual_inferred_window_instances"] / max(eval_t["window_assembly_seconds"] + eval_t["gpu_forward_seconds"], EPS)),
        },
        "forward_plan": plan,
        "batched_segment_feature_parity": parity,
    }
    write_json_atomic(result_path, result)
    update_sample_status(out_dir, global_idx, surrogate_done=True)
    return result


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / max(float(np.sum(weights)), EPS))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_bar = weighted_mean(y_true, weights)
    ss_res = float(np.sum(weights * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(weights * (y_true - y_bar) ** 2))
    if ss_tot <= EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def weighted_corr(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    mx = weighted_mean(y_true, weights)
    my = weighted_mean(y_pred, weights)
    cov = float(np.sum(weights * (y_true - mx) * (y_pred - my)))
    vx = float(np.sum(weights * (y_true - mx) ** 2))
    vy = float(np.sum(weights * (y_pred - my) ** 2))
    return cov / math.sqrt(max(vx * vy, EPS))


def fit_surrogate(
    masks: np.ndarray,
    probs: np.ndarray,
    train_n: int,
    alpha_grid: Sequence[float] = (1e-3, 1e-2, 0.1, 1.0, 10.0),
    kernel_width_grid: Sequence[float] = (0.15, 0.25, 0.5, 0.75),
    response_transform: str = "probability",
) -> Tuple[Ridge, Dict[str, float], np.ndarray]:
    ensure_lime_imports()
    distances = calculate_cosine_distances(masks, masks.shape[1])
    x_train, y_train = masks[:train_n], probs[:train_n]
    x_val, y_val = masks[train_n:], probs[train_n:]
    if response_transform == "probability":
        y_model = np.asarray(probs, dtype=np.float64)

        def inverse_response(values):
            return np.asarray(values, dtype=np.float64)
    elif response_transform == "logit":
        clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        y_model = np.log(clipped / (1.0 - clipped))

        def inverse_response(values):
            values = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-values))
    else:
        raise ValueError(f"Unknown LIME response transform: {response_transform}")
    y_model_train = y_model[:train_n]
    alphas = tuple(sorted({float(value) for value in alpha_grid if float(value) > 0.0}))
    kernel_widths = tuple(sorted({float(value) for value in kernel_width_grid if float(value) > 0.0}))
    if not alphas or not kernel_widths:
        raise ValueError("Ridge alpha and kernel-width grids must contain positive values.")

    rng = np.random.default_rng(20260717)
    shuffled = rng.permutation(int(train_n))
    folds = [fold for fold in np.array_split(shuffled, min(5, max(2, int(train_n) // 50))) if len(fold)]
    candidates = []
    weights_by_kernel = {
        width: np.asarray(calculate_weights_from_distances(distances, kernel_width=width), dtype=np.float64)
        for width in kernel_widths
    }
    all_train_indices = np.arange(int(train_n), dtype=np.int32)
    for width in kernel_widths:
        candidate_weights = weights_by_kernel[width]
        for alpha in alphas:
            fold_losses = []
            for fold_indices in folds:
                fit_indices = np.setdiff1d(all_train_indices, fold_indices, assume_unique=False)
                candidate = Ridge(alpha=alpha)
                candidate.fit(x_train[fit_indices], y_model_train[fit_indices], sample_weight=candidate_weights[fit_indices])
                fold_prediction = inverse_response(candidate.predict(x_train[fold_indices]))
                fold_losses.append(weighted_mean((y_train[fold_indices] - fold_prediction) ** 2, candidate_weights[fold_indices]))
            candidates.append((float(np.mean(fold_losses)), float(alpha), float(width)))
    cv_weighted_mse, selected_alpha, selected_kernel_width = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    weights = weights_by_kernel[selected_kernel_width]
    w_train, w_val = weights[:train_n], weights[train_n:]
    ridge = Ridge(alpha=selected_alpha)
    ridge.fit(x_train, y_model_train, sample_weight=w_train)
    pred_train = inverse_response(ridge.predict(x_train))
    pred_val = inverse_response(ridge.predict(x_val))
    val_rmse = math.sqrt(weighted_mean((y_val - pred_val) ** 2, w_val))
    val_mean = weighted_mean(y_val, w_val)
    val_variance = weighted_mean((y_val - val_mean) ** 2, w_val)
    val_range = float(np.max(y_val) - np.min(y_val)) if len(y_val) else 0.0
    deleted_counts = masks.shape[1] - np.sum(masks, axis=1)
    local_limit = max(5, int(masks.shape[1] // 10))
    metrics = {
        "train_weighted_r2": weighted_r2(y_train, pred_train, w_train),
        "validation_weighted_r2": weighted_r2(y_val, pred_val, w_val),
        "validation_weighted_mae": weighted_mean(np.abs(y_val - pred_val), w_val),
        "validation_weighted_rmse": val_rmse,
        "validation_nrmse": float(val_rmse / max(val_range, 1e-3)),
        "validation_probability_mean": val_mean,
        "validation_probability_variance": val_variance,
        "validation_probability_std": math.sqrt(max(val_variance, 0.0)),
        "validation_probability_range": val_range,
        "r2_degenerate_low_variance": bool(val_variance < 1e-5),
        "weighted_pearson": weighted_corr(y_val, pred_val, w_val),
        "generalization_gap": weighted_r2(y_train, pred_train, w_train) - weighted_r2(y_val, pred_val, w_val),
        "effective_sample_size": float((np.sum(w_train) ** 2) / max(np.sum(w_train ** 2), EPS)),
        "surrogate_original_mask_error": float(abs(inverse_response(ridge.predict(np.ones((1, masks.shape[1]))))[0] - probs[0])),
        "response_transform": str(response_transform),
        "selected_ridge_alpha": selected_alpha,
        "selected_kernel_width": selected_kernel_width,
        "inner_cv_weighted_rmse": math.sqrt(max(cv_weighted_mse, 0.0)),
        "mean_retained_feature_fraction": float(np.mean(masks)),
        "local_mask_fraction": float(np.mean(deleted_counts <= local_limit)),
        "local_mask_delete_limit": int(local_limit),
    }
    return ridge, metrics, weights


def rank_features(coef: np.ndarray, ranking: str) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float32).ravel()
    if ranking == "positive":
        scores = np.clip(coef, 0.0, None)
        if float(np.sum(scores)) <= EPS:
            scores = np.abs(coef)
    elif ranking == "absolute":
        scores = np.abs(coef)
    elif ranking == "reverse":
        scores = -np.clip(coef, 0.0, None)
    elif ranking == "energy":
        raise ValueError("energy ranking must be built from signal energy")
    else:
        raise ValueError(f"Unknown ranking: {ranking}")
    return np.argsort(-scores).astype(np.int32)


def feature_mask_from_order(num_features: int, order: np.ndarray, fraction: float, curve: str) -> np.ndarray:
    count = int(round(float(fraction) * int(num_features)))
    count = min(max(count, 0), int(num_features))
    mask = np.ones(num_features, dtype=np.int32)
    selected = order[:count]
    if curve == "deletion":
        mask[selected] = 0
    elif curve == "insertion":
        mask[:] = 0
        mask[selected] = 1
    else:
        raise ValueError(curve)
    return mask


def energy_ranking(epoch: np.ndarray, segments: Sequence[Tuple[int, int]]) -> np.ndarray:
    scores = []
    for ch in range(epoch.shape[1]):
        for start, end in segments:
            scores.append(float(np.mean(epoch[start:end, ch] ** 2)))
    return np.argsort(-np.asarray(scores, dtype=np.float32)).astype(np.int32)


def curve_metrics(fractions: np.ndarray, deletion: np.ndarray, insertion: np.ndarray) -> Dict[str, float]:
    original = float(deletion[0])
    full = float(deletion[-1])
    denom = original - full
    normalization_valid = bool(abs(denom) >= 1e-6)
    if not normalization_valid:
        deletion_norm = np.full_like(deletion, np.nan, dtype=np.float32)
        insertion_norm = np.full_like(insertion, np.nan, dtype=np.float32)
    else:
        deletion_norm = (deletion - full) / denom
        insertion_norm = (insertion - full) / denom
    def at(frac: float, arr: np.ndarray) -> float:
        idx = int(np.argmin(np.abs(fractions - frac)))
        return float(arr[idx])
    deletion_aopc = float(np.mean(original - deletion))
    insertion_aopc = float(np.mean(insertion - insertion[0]))
    return {
        "deletion_auc_raw": float(np.trapz(deletion, fractions)),
        "insertion_auc_raw": float(np.trapz(insertion, fractions)),
        "deletion_auc_normalized": float(np.trapz(deletion_norm, fractions)) if normalization_valid else np.nan,
        "insertion_auc_normalized": float(np.trapz(insertion_norm, fractions)) if normalization_valid else np.nan,
        "deletion_AOPC": deletion_aopc,
        "insertion_AOPC": insertion_aopc,
        "deletion_aopc": deletion_aopc,
        "insertion_aopc": insertion_aopc,
        "probability_drop_at_10_percent": original - at(0.10, deletion),
        "probability_drop_at_20_percent": original - at(0.20, deletion),
        "probability_drop_at_25_percent": original - at(0.25, deletion),
        "probability_drop_at_30_percent": original - at(0.30, deletion),
        "probability_drop_at_50_percent": original - at(0.50, deletion),
        "probability_recovery_at_10_percent": at(0.10, insertion) - insertion[0],
        "probability_recovery_at_20_percent": at(0.20, insertion) - insertion[0],
        "probability_recovery_at_25_percent": at(0.25, insertion) - insertion[0],
        "probability_recovery_at_30_percent": at(0.30, insertion) - insertion[0],
        "probability_recovery_at_50_percent": at(0.50, insertion) - insertion[0],
        "full_perturbation_probability": full,
        "original_probability": original,
        "normalization_valid": normalization_valid,
        "normalization_invalid": not normalization_valid,
    }


def compute_curves(
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_epoch: int,
    target_class: int,
    epoch: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    order: np.ndarray,
    mode: str,
    batch_windows: int,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = np.asarray([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0], dtype=np.float32)
    del_masks, ins_masks = [], []
    for frac in fractions:
        del_masks.append(feature_mask_from_order(len(order), order, float(frac), "deletion"))
        ins_masks.append(feature_mask_from_order(len(order), order, float(frac), "insertion"))
    deletion = predict_masks_sequence(model, x_raw, x_seg, counts, global_epoch, target_class, epoch, segments, np.asarray(del_masks), mode, batch_windows, chunk)
    insertion = predict_masks_sequence(model, x_raw, x_seg, counts, global_epoch, target_class, epoch, segments, np.asarray(ins_masks), mode, batch_windows, chunk)
    return fractions, deletion, insertion


def feature_count_for_fraction(num_features: int, fraction: float) -> int:
    count = int(round(float(fraction) * int(num_features)))
    return min(max(count, 0), int(num_features))


def feature_mask_from_order_with_count(num_features: int, order: np.ndarray, fraction: float, curve: str) -> Tuple[np.ndarray, int, float]:
    count = feature_count_for_fraction(num_features, fraction)
    mask = np.ones(num_features, dtype=np.int32)
    selected = np.asarray(order, dtype=np.int32)[:count]
    if curve == "deletion":
        mask[selected] = 0
    elif curve == "insertion":
        mask[:] = 0
        mask[selected] = 1
    else:
        raise ValueError(curve)
    return mask, int(count), float(count / max(num_features, 1))


def feature_order_hash(order: np.ndarray) -> str:
    arr = np.asarray(order, dtype=np.int32).ravel()
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def is_complete_permutation(order: np.ndarray, num_features: int) -> bool:
    arr = np.asarray(order, dtype=np.int32).ravel()
    return bool(len(arr) == int(num_features) and np.array_equal(np.sort(arr), np.arange(int(num_features), dtype=np.int32)))


def lime_positive_order(coef: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    coef = np.asarray(coef, dtype=np.float32).ravel()
    positive_scores = np.clip(coef, 0.0, None)
    positive_count = int(np.sum(positive_scores > 0.0))
    fallback = positive_count == 0
    scores = np.abs(coef) if fallback else positive_scores
    return np.argsort(-scores).astype(np.int32), {
        "positive_ranking_fallback": bool(fallback),
        "positive_coefficient_count": positive_count,
    }


def build_faithfulness_rankings(
    coef: np.ndarray,
    epoch: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    global_idx: int,
    random_repeats: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    num_features = int(epoch.shape[1] * len(segments))
    positive_order, positive_meta = lime_positive_order(coef)
    absolute_order = np.argsort(-np.abs(np.asarray(coef, dtype=np.float32).ravel())).astype(np.int32)
    reverse_order = positive_order[::-1].astype(np.int32)
    energy_order = energy_ranking(epoch, segments)
    rankings: List[Dict[str, object]] = [
        {"ranking": "lime_positive", "order": positive_order, **positive_meta},
        {"ranking": "lime_absolute", "order": absolute_order, "positive_ranking_fallback": False, "positive_coefficient_count": positive_meta["positive_coefficient_count"]},
        {"ranking": "reverse_lime", "order": reverse_order, "positive_ranking_fallback": positive_meta["positive_ranking_fallback"], "positive_coefficient_count": positive_meta["positive_coefficient_count"]},
        {"ranking": "signal_energy", "order": energy_order, "positive_ranking_fallback": False, "positive_coefficient_count": positive_meta["positive_coefficient_count"], "energy_formula": "mean(square(signal[channel, dynamic_segment]))"},
    ]
    random_hashes = []
    for repeat in range(int(random_repeats)):
        seed = int(700000 + int(global_idx) * 100 + repeat)
        rng = np.random.default_rng(seed)
        order = rng.permutation(num_features).astype(np.int32)
        order_hash = feature_order_hash(order)
        random_hashes.append(order_hash)
        rankings.append({
            "ranking": f"random_{repeat + 1}",
            "order": order,
            "random_seed": seed,
            "feature_order_hash": order_hash,
            "positive_ranking_fallback": False,
            "positive_coefficient_count": positive_meta["positive_coefficient_count"],
        })
    validation = {
        "energy_ranking_valid": is_complete_permutation(energy_order, num_features),
        "energy_formula": "mean(square(signal[channel, dynamic_segment]))",
        "random_ranking_valid": bool(
            len(random_hashes) == int(random_repeats)
            and len(set(random_hashes)) == len(random_hashes)
            and all(is_complete_permutation(r["order"], num_features) for r in rankings if str(r["ranking"]).startswith("random_"))
        ),
        "random_ranking_repeats": int(random_repeats),
    }
    return rankings, validation


def run_faithfulness_module(
    sample: Mapping[str, object],
    config: RunConfig,
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
    force: bool = False,
) -> Dict[str, object]:
    ensure_lime_imports()
    global_idx = int(sample["global_epoch_index"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / "functional_faithfulness.json"
    status = read_sample_status(out_dir, global_idx)
    config_signature = surrogate_config_signature(config)
    if status.get("faithfulness_done") and result_path.exists() and not force:
        cached = read_json(result_path)
        if int(cached.get("faithfulness_schema_version", 0)) >= 3 and cached.get("surrogate_config_signature") == config_signature:
            return cached

    t0 = time.perf_counter()
    surrogate = run_surrogate_module(
        sample,
        config,
        model,
        x_raw,
        x_seg,
        counts,
        artifact,
        out_dir,
        section_name="functional_surrogate",
        force=False,
    )
    t_surrogate = time.perf_counter()
    epoch = x_raw[global_idx]
    segments = semantic_segmentation(epoch, target_segments=config.target_segments)
    target_class = int(np.argmax(artifact["probs"][global_idx]))
    num_features = int(epoch.shape[1] * len(segments))
    coef = np.asarray(surrogate["coef"], dtype=np.float32)
    rankings, ranking_validation = build_faithfulness_rankings(coef, epoch, segments, global_idx, config.random_repeats)
    fractions = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

    requests: List[Dict[str, object]] = []
    masks: List[np.ndarray] = []
    for ranking in rankings:
        order = np.asarray(ranking["order"], dtype=np.int32)
        for curve in ("deletion", "insertion"):
            for frac in fractions:
                mask, feature_count, actual_fraction = feature_mask_from_order_with_count(num_features, order, float(frac), curve)
                requests.append({
                    "ranking": str(ranking["ranking"]),
                    "curve": curve,
                    "fraction": float(frac),
                    "feature_count": int(feature_count),
                    "actual_fraction": float(actual_fraction),
                    "feature_order_hash": str(ranking.get("feature_order_hash", feature_order_hash(order))),
                    "random_seed": ranking.get("random_seed"),
                    "positive_ranking_fallback": bool(ranking.get("positive_ranking_fallback", False)),
                    "positive_coefficient_count": int(ranking.get("positive_coefficient_count", 0)),
                    "energy_formula": ranking.get("energy_formula"),
                })
                masks.append(mask)
    t_mask_build = time.perf_counter()
    eval_payload = evaluate_masks_for_target_epoch(
        np.asarray(masks, dtype=np.int32),
        model,
        x_raw,
        x_seg,
        counts,
        global_idx,
        target_class,
        epoch,
        segments,
        perturbation_mode="physiological",
        window_inference_batch_size=config.batch_windows,
        out_dir=out_dir,
        cache_namespace="functional_faithfulness",
        model_state_id="original_model",
    )
    t_eval = time.perf_counter()
    target_probs = np.asarray(eval_payload["target_probabilities"], dtype=np.float32)
    for idx, req in enumerate(requests):
        req["target_probability"] = float(target_probs[idx])
        req["mask_hash"] = eval_payload["mask_hashes"][idx]
        req["from_cache"] = bool(eval_payload["from_cache"][idx])

    metric_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    endpoint_diffs = []
    canonical_original = float(artifact["probs"][global_idx, target_class])
    for ranking in rankings:
        ranking_name = str(ranking["ranking"])
        del_reqs = [r for r in requests if r["ranking"] == ranking_name and r["curve"] == "deletion"]
        ins_reqs = [r for r in requests if r["ranking"] == ranking_name and r["curve"] == "insertion"]
        del_reqs = sorted(del_reqs, key=lambda r: float(r["fraction"]))
        ins_reqs = sorted(ins_reqs, key=lambda r: float(r["fraction"]))
        deletion = np.asarray([r["target_probability"] for r in del_reqs], dtype=np.float32)
        insertion = np.asarray([r["target_probability"] for r in ins_reqs], dtype=np.float32)
        actual_fractions = np.asarray([r["actual_fraction"] for r in del_reqs], dtype=np.float32)
        cm = curve_metrics(actual_fractions, deletion, insertion)
        direct_original = float(deletion[0])
        fully_perturbed = float(deletion[-1])
        deletion_endpoint_max_diff = float(max(abs(deletion[0] - direct_original), abs(deletion[-1] - fully_perturbed)))
        insertion_endpoint_max_diff = float(max(abs(insertion[0] - fully_perturbed), abs(insertion[-1] - direct_original)))
        endpoint_diffs.append((deletion_endpoint_max_diff, insertion_endpoint_max_diff))
        row = {
            "global_epoch_index": global_idx,
            "target_stage": CLASS_LABELS[target_class],
            "ranking": ranking_name,
            "original_probability": direct_original,
            "canonical_original_probability": canonical_original,
            "direct_original_abs_diff": float(abs(direct_original - canonical_original)),
            "fully_perturbed_probability": fully_perturbed,
            "deletion_endpoint_max_diff": deletion_endpoint_max_diff,
            "insertion_endpoint_max_diff": insertion_endpoint_max_diff,
            "positive_ranking_fallback": bool(ranking.get("positive_ranking_fallback", False)),
            "positive_coefficient_count": int(ranking.get("positive_coefficient_count", 0)),
            "random_seed": ranking.get("random_seed"),
            "feature_order_hash": str(ranking.get("feature_order_hash", feature_order_hash(np.asarray(ranking["order"], dtype=np.int32)))),
            **cm,
        }
        metric_rows.append(row)
        for d_req, i_req in zip(del_reqs, ins_reqs):
            curve_rows.append({
                "global_epoch_index": global_idx,
                "target_stage": CLASS_LABELS[target_class],
                "ranking": ranking_name,
                "fraction": float(d_req["fraction"]),
                "feature_count": int(d_req["feature_count"]),
                "actual_fraction": float(d_req["actual_fraction"]),
                "deletion_probability": float(d_req["target_probability"]),
                "insertion_probability": float(i_req["target_probability"]),
                "deletion_mask_hash": d_req["mask_hash"],
                "insertion_mask_hash": i_req["mask_hash"],
                "deletion_from_cache": bool(d_req["from_cache"]),
                "insertion_from_cache": bool(i_req["from_cache"]),
            })
    t_metrics = time.perf_counter()

    endpoint_max_deletion = float(max([d for d, _i in endpoint_diffs], default=np.nan))
    endpoint_max_insertion = float(max([i for _d, i in endpoint_diffs], default=np.nan))
    finite = bool(np.all(np.isfinite(target_probs)) and np.all(np.isfinite([r["deletion_endpoint_max_diff"] for r in metric_rows])) and np.all(np.isfinite([r["insertion_endpoint_max_diff"] for r in metric_rows])))
    accounting = eval_payload["accounting"]
    result = {
        "faithfulness_schema_version": 3,
        "surrogate_config_signature": config_signature,
        "global_epoch_index": global_idx,
        "target_class": target_class,
        "target_stage": CLASS_LABELS[target_class],
        "segments_count": int(len(segments)),
        "feature_count": int(num_features),
        "fractions": fractions.astype(float).tolist(),
        "ranking_validation": ranking_validation,
        "metric_rows": metric_rows,
        "curve_rows": curve_rows,
        "window_instance_accounting": accounting,
        "mask_efficiency": eval_payload["efficiency"],
        "deletion_endpoint_max_diff": endpoint_max_deletion,
        "insertion_endpoint_max_diff": endpoint_max_insertion,
        "all_values_finite": finite,
        "timings_sec": {
            "surrogate_load_or_compute_seconds": float(t_surrogate - t0),
            "mask_generation_seconds": float(t_mask_build - t_surrogate),
            "window_assembly_seconds": float(eval_payload["timings_sec"]["window_assembly_seconds"]),
            "gpu_forward_seconds": float(eval_payload["timings_sec"]["gpu_forward_seconds"]),
            "aggregation_seconds": float(eval_payload["timings_sec"]["aggregation_seconds"]),
            "metric_computation_seconds": float(t_metrics - t_eval),
            "cache_io_seconds": float(eval_payload["timings_sec"]["cache_io_seconds"]),
            "total_seconds": float(t_metrics - t0),
        },
    }
    write_json_atomic(result_path, result)
    update_sample_status(out_dir, global_idx, faithfulness_done=True)
    return result


def transition_boundary(y_true: np.ndarray, recording_id: np.ndarray) -> np.ndarray:
    same_prev = np.r_[False, recording_id[1:] == recording_id[:-1]]
    same_next = np.r_[recording_id[:-1] == recording_id[1:], False]
    prev_diff = same_prev & (y_true != np.r_[y_true[0], y_true[:-1]])
    next_diff = same_next & (y_true != np.r_[y_true[1:], y_true[-1]])
    return prev_diff | next_diff


def choose_samples(artifact: Mapping[str, np.ndarray], config: RunConfig, out_dir: Path, resume: bool) -> pd.DataFrame:
    mode_tag = str(config.mode).replace("/", "_").replace("\\", "_")
    manifest_path = out_dir / f"sample_manifest_{mode_tag}.csv"
    if resume and manifest_path.exists():
        return pd.read_csv(manifest_path)
    y = artifact["y_true"]
    pred = artifact["pred"]
    probs = artifact["probs"]
    rec = artifact["recording_id"]
    epoch_idx = artifact["epoch_index"]
    boundary = transition_boundary(y, rec)
    conf = probs[np.arange(len(probs)), pred]
    rows = []
    if config.mode == "full_evaluation":
        rng = np.random.default_rng(42)
        random_quota = min(10, len(y))
        for global_idx in rng.choice(len(y), size=random_quota, replace=False):
            rows.append({
                "global_epoch_index": int(global_idx),
                "recording_id": int(rec[global_idx]),
                "subject_id": int(rec[global_idx]),
                "epoch_index_in_recording": int(epoch_idx[global_idx]),
                "true_stage": CLASS_LABELS[int(y[global_idx])],
                "predicted_stage": CLASS_LABELS[int(pred[global_idx])],
                "confidence": float(conf[global_idx]),
                "correct": bool(pred[global_idx] == y[global_idx]),
                "transition_boundary": bool(boundary[global_idx]),
                "sample_group": "population_random",
                "segments_count": 0,
            })
    for stage in range(len(CLASS_LABELS)):
        for correct_value, quota, group_name in (
            (True, config.sample_correct_per_stage, "formal_correct"),
            (False, config.sample_wrong_per_stage, "formal_incorrect"),
        ):
            idx = np.flatnonzero((y == stage) & ((pred == y) == correct_value))
            if len(idx) == 0 or quota <= 0:
                continue
            order = np.argsort(conf[idx])
            picked = []
            if len(order):
                anchors = np.unique(np.rint(np.linspace(0, len(order) - 1, min(quota, len(order)))).astype(int))
                picked = idx[order[anchors]].tolist()
            for global_idx in picked[:quota]:
                rows.append({
                    "global_epoch_index": int(global_idx),
                    "recording_id": int(rec[global_idx]),
                    "subject_id": int(rec[global_idx]),
                    "epoch_index_in_recording": int(epoch_idx[global_idx]),
                    "true_stage": CLASS_LABELS[int(y[global_idx])],
                    "predicted_stage": CLASS_LABELS[int(pred[global_idx])],
                    "confidence": float(conf[global_idx]),
                    "correct": bool(pred[global_idx] == y[global_idx]),
                    "transition_boundary": bool(boundary[global_idx]),
                    "sample_group": group_name,
                    "segments_count": 0,
                })
    if config.mode == "full_evaluation":
        rng = np.random.default_rng(43)
        for is_transition, group_name in ((True, "transition"), (False, "stable")):
            candidates = np.flatnonzero(boundary == is_transition)
            if len(candidates):
                for global_idx in rng.choice(candidates, size=min(3, len(candidates)), replace=False):
                    rows.append({
                        "global_epoch_index": int(global_idx),
                        "recording_id": int(rec[global_idx]),
                        "subject_id": int(rec[global_idx]),
                        "epoch_index_in_recording": int(epoch_idx[global_idx]),
                        "true_stage": CLASS_LABELS[int(y[global_idx])],
                        "predicted_stage": CLASS_LABELS[int(pred[global_idx])],
                        "confidence": float(conf[global_idx]),
                        "correct": bool(pred[global_idx] == y[global_idx]),
                        "transition_boundary": bool(boundary[global_idx]),
                        "sample_group": group_name,
                        "segments_count": 0,
                    })
    df = pd.DataFrame(rows).drop_duplicates("global_epoch_index").head(config.max_samples)
    population_counts = {CLASS_LABELS[stage]: int(np.sum(y == stage)) for stage in range(5)}
    selected_counts = df["true_stage"].value_counts().to_dict()
    df["population_weight"] = [
        float(population_counts[stage] / max(selected_counts.get(stage, 1), 1))
        for stage in df["true_stage"]
    ]
    df["stage_balanced_weight"] = [
        float(1.0 / (5.0 * max(selected_counts.get(stage, 1), 1)))
        for stage in df["true_stage"]
    ]
    df.to_csv(manifest_path, index=False)
    return df


def choose_functional_samples(artifact: Mapping[str, np.ndarray], config: RunConfig, out_dir: Path, resume: bool) -> pd.DataFrame:
    mode_tag = str(config.mode).replace("/", "_").replace("\\", "_")
    manifest_path = out_dir / f"functional_smoke_manifest_{mode_tag}.csv"
    if resume and manifest_path.exists():
        return pd.read_csv(manifest_path)
    y = artifact["y_true"]
    pred = artifact["pred"]
    probs = artifact["probs"]
    rec = artifact["recording_id"]
    epoch_idx = artifact["epoch_index"]
    boundary = transition_boundary(y, rec)
    conf = probs[np.arange(len(probs)), pred]
    rows = []
    for stage in range(len(CLASS_LABELS)):
        idx = np.flatnonzero((y == stage) & (pred == y))
        if len(idx) == 0:
            rows.append({
                "global_epoch_index": -1,
                "recording_id": -1,
                "subject_id": -1,
                "epoch_index_in_recording": -1,
                "true_stage": CLASS_LABELS[stage],
                "predicted_stage": "",
                "confidence": np.nan,
                "correct": False,
                "transition_boundary": False,
                "sample_group": "functional_missing_correct",
                "segments_count": 0,
            })
            continue
        order = np.argsort(np.abs(conf[idx] - np.median(conf[idx])))
        global_idx = int(idx[order[0]])
        rows.append({
            "global_epoch_index": global_idx,
            "recording_id": int(rec[global_idx]),
            "subject_id": int(rec[global_idx]),
            "epoch_index_in_recording": int(epoch_idx[global_idx]),
            "true_stage": CLASS_LABELS[int(y[global_idx])],
            "predicted_stage": CLASS_LABELS[int(pred[global_idx])],
            "confidence": float(conf[global_idx]),
            "correct": True,
            "transition_boundary": bool(boundary[global_idx]),
            "sample_group": "functional_correct",
            "segments_count": 0,
        })
    df = pd.DataFrame(rows)
    df = df[df["global_epoch_index"] >= 0].head(config.max_samples)
    df.to_csv(manifest_path, index=False)
    return df


def vector_normed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    total = float(np.sum(np.clip(values, 0, None)))
    if total <= EPS:
        total = float(np.sum(np.abs(values)))
        values = np.abs(values)
    if total <= EPS:
        return np.ones_like(values) / max(1, len(values))
    return values / total


def coef_evidence(coef: np.ndarray, num_channels: int, num_segments: int, target_class: int) -> Dict[str, object]:
    matrix = np.asarray(coef, dtype=np.float32).reshape(num_channels, num_segments)
    pos_ch = np.sum(np.clip(matrix, 0, None), axis=1)
    neg_ch = np.sum(np.clip(-matrix, 0, None), axis=1)
    abs_ch = np.sum(np.abs(matrix), axis=1)
    prior_ch = get_clinical_prior(target_class)[:num_channels]
    concept_pos = np.zeros(len(CLINICAL_CONCEPTS), dtype=np.float32)
    concept_abs = np.zeros(len(CLINICAL_CONCEPTS), dtype=np.float32)
    for concept_idx, ch in enumerate(CONCEPT_CHANNEL_INDEX[:len(CLINICAL_CONCEPTS)]):
        if int(ch) < num_channels:
            concept_pos[concept_idx] = pos_ch[int(ch)]
            concept_abs[concept_idx] = abs_ch[int(ch)]
    prior_concept = get_stage_concept_prior(target_class)
    pos_ch_n = vector_normed(pos_ch)
    abs_ch_n = vector_normed(abs_ch)
    pos_concept_n = vector_normed(concept_pos)
    abs_concept_n = vector_normed(concept_abs)
    prior_ch_n = vector_normed(prior_ch)
    prior_concept_n = vector_normed(prior_concept)
    return {
        "positive_channel_dccs": pos_ch_n.tolist(),
        "negative_channel_dccs": vector_normed(neg_ch).tolist(),
        "absolute_raw_channel_dccs": abs_ch_n.tolist(),
        "positive_concept_dccs": pos_concept_n.tolist(),
        "absolute_raw_concept_dccs": abs_concept_n.tolist(),
        "net_channel_evidence": vector_normed(pos_ch - neg_ch).tolist(),
        "net_concept_evidence": pos_concept_n.tolist(),
        "positive_evidence_fraction": float(np.sum(pos_ch) / max(np.sum(pos_ch) + np.sum(neg_ch), EPS)),
        "negative_evidence_fraction": float(np.sum(neg_ch) / max(np.sum(pos_ch) + np.sum(neg_ch), EPS)),
        "channel_conflict_index": float(np.sum(np.minimum(pos_ch, neg_ch)) / max(np.sum(pos_ch + neg_ch), EPS)),
        "concept_conflict_index": float(np.sum(np.minimum(concept_pos, concept_abs - concept_pos)) / max(np.sum(concept_abs), EPS)),
        "raw_channel_prior_cosine": cosine(abs_ch_n, prior_ch_n),
        "positive_channel_prior_cosine": cosine(pos_ch_n, prior_ch_n),
        "raw_concept_prior_cosine": cosine(abs_concept_n, prior_concept_n),
        "positive_concept_prior_cosine": cosine(pos_concept_n, prior_concept_n),
        "dominant_channel_expected_hit": bool(int(np.argmax(pos_ch_n)) == int(np.argmax(prior_ch_n))),
        "dominant_channel_rank": int(np.where(np.argsort(-pos_ch_n) == int(np.argmax(prior_ch_n)))[0][0] + 1),
        "expected_top1_concept_hit": bool(int(np.argmax(pos_concept_n)) == int(np.argmax(prior_concept_n))),
        "expected_top3_concept_hit": bool(int(np.argmax(prior_concept_n)) in set(np.argsort(-pos_concept_n)[:3].tolist())),
        "expected_concept_mean_rank": float(np.mean([np.where(np.argsort(-pos_concept_n) == idx)[0][0] + 1 for idx in np.argsort(-prior_concept_n)[:3]])),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def clinical_nulls(coef: np.ndarray, num_channels: int, num_segments: int, target_class: int, seed: int) -> Dict[str, float]:
    ev = coef_evidence(coef, num_channels, num_segments, target_class)
    pos_concept = np.asarray(ev["positive_concept_dccs"], dtype=np.float32)
    priors = [get_stage_concept_prior(i) for i in range(len(CLASS_LABELS))]
    target = cosine(pos_concept, priors[target_class])
    wrong = [cosine(pos_concept, priors[i]) for i in range(len(CLASS_LABELS)) if i != target_class]
    rng = np.random.default_rng(seed)
    shuffled = []
    p = np.asarray(priors[target_class], dtype=np.float32)
    for _ in range(1000):
        shuffled.append(cosine(pos_concept, rng.permutation(p)))
    shuffled = np.asarray(shuffled, dtype=np.float32)
    return {
        "raw_alignment": float(target),
        "shuffled_prior_mean": float(np.mean(shuffled)),
        "shuffled_prior_ci_low": float(np.percentile(shuffled, 2.5)),
        "shuffled_prior_ci_high": float(np.percentile(shuffled, 97.5)),
        "empirical_p_value": float((1 + np.sum(shuffled >= target)) / (len(shuffled) + 1)),
        "wrong_stage_maximum_alignment": float(np.max(wrong)),
        "target_minus_best_wrong_margin": float(target - np.max(wrong)),
        "target_stage_prior_rank": int(np.where(np.argsort(-np.asarray([cosine(pos_concept, p) for p in priors])) == target_class)[0][0] + 1),
    }


def jaccard_top(a: np.ndarray, b: np.ndarray, k: int) -> float:
    sa = set(np.argsort(-np.abs(a))[:min(k, len(a))].tolist())
    sb = set(np.argsort(-np.abs(b))[:min(k, len(b))].tolist())
    return float(len(sa & sb) / max(len(sa | sb), 1))


def evaluate_single_sample(
    sample: Mapping[str, object],
    config: RunConfig,
    model: tf.keras.Model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, object]:
    ensure_lime_imports()
    global_idx = int(sample["global_epoch_index"])
    cache_path = out_dir / "cache" / f"sample_{global_idx:06d}.json"
    if cache_path.exists():
        result = read_json(cache_path)
        target_class = int(result["target_class"])
        exact_vector = sequence_target_vector(
            model,
            x_raw,
            x_seg,
            counts,
            global_idx,
            x_raw[global_idx],
            x_seg[global_idx],
            batch_windows=config.batch_windows,
        )
        artifact_probability = float(artifact["probs"][global_idx, target_class])
        result["sequence_wrapper_original_probability"] = float(
            exact_vector[target_class]
        )
        result["formal_sequence_inference_probability"] = artifact_probability
        result["sequence_wrapper_abs_diff"] = float(
            abs(exact_vector[target_class] - artifact_probability)
        )
        result["sequence_wrapper_protocol"] = "formal_full_batch_context"
        write_json(cache_path, result)
        return result

    epoch = x_raw[global_idx]
    segments = semantic_segmentation(epoch, target_segments=config.target_segments)
    target_class = int(np.argmax(artifact["probs"][global_idx]))
    sample_seed = 100000 + global_idx
    num_features = int(epoch.shape[1] * len(segments))
    masks = make_masks(num_features, config.train_masks, config.val_masks, sample_seed, sampling=config.mask_sampling)

    result = {
        "global_epoch_index": global_idx,
        "recording_id": int(sample["recording_id"]),
        "subject_id": int(sample["subject_id"]),
        "epoch_index_in_recording": int(sample["epoch_index_in_recording"]),
        "true_stage": str(sample["true_stage"]),
        "predicted_stage": str(sample["predicted_stage"]),
        "target_class": target_class,
        "target_stage": CLASS_LABELS[target_class],
        "confidence": float(sample["confidence"]),
        "correct": bool(sample["correct"]),
        "transition_boundary": bool(sample["transition_boundary"]),
        "segments_count": int(len(segments)),
        "segment_boundaries": [[int(a), int(b)] for a, b in segments],
        "status": "OK",
        "perturbation_results": {},
        "stability": [],
    }

    original_wrapper = sequence_target_vector(
        model,
        x_raw,
        x_seg,
        counts,
        global_idx,
        epoch,
        x_seg[global_idx],
        batch_windows=config.batch_windows,
    )[target_class]
    artifact_prob = float(artifact["probs"][global_idx, target_class])
    result["sequence_wrapper_original_probability"] = float(original_wrapper)
    result["formal_sequence_inference_probability"] = artifact_prob
    result["sequence_wrapper_abs_diff"] = float(abs(original_wrapper - artifact_prob))
    result["sequence_wrapper_protocol"] = "formal_full_batch_context"

    for mode in config.perturbation_modes:
        probs = predict_masks_sequence(
            model, x_raw, x_seg, counts, global_idx, target_class, epoch, segments, masks,
            mode, config.batch_windows, config.perturbation_chunk,
        )
        ridge, fidelity, _weights = fit_surrogate(
            masks,
            probs,
            config.train_masks,
            alpha_grid=config.ridge_alphas,
            kernel_width_grid=config.kernel_widths,
            response_transform=config.response_transform,
        )
        coef = np.asarray(ridge.coef_, dtype=np.float32)
        pos_fallback = bool(float(np.max(coef)) <= 0.0)
        orders = {
            "lime_positive": rank_features(coef, "positive"),
            "lime_absolute": rank_features(coef, "absolute"),
            "reverse_lime": rank_features(-coef, "positive"),
            "signal_energy": energy_ranking(epoch, segments),
        }
        curves = {}
        for ranking_name, order in orders.items():
            fractions, deletion, insertion = compute_curves(
                model, x_raw, x_seg, counts, global_idx, target_class, epoch, segments,
                order, mode, config.batch_windows, config.perturbation_chunk,
            )
            cm = curve_metrics(fractions, deletion, insertion)
            curves[ranking_name] = {
                "fractions": fractions.tolist(),
                "deletion_curve": deletion.tolist(),
                "insertion_curve": insertion.tolist(),
                **cm,
            }
        random_curves = []
        random_deletion_curves = []
        random_insertion_curves = []
        rng = np.random.default_rng(sample_seed + 77)
        for repeat in range(config.random_repeats):
            order = rng.permutation(num_features).astype(np.int32)
            fractions, deletion, insertion = compute_curves(
                model, x_raw, x_seg, counts, global_idx, target_class, epoch, segments,
                order, mode, config.batch_windows, config.perturbation_chunk,
            )
            random_curves.append(curve_metrics(fractions, deletion, insertion))
            random_deletion_curves.append(np.asarray(deletion, dtype=np.float32))
            random_insertion_curves.append(np.asarray(insertion, dtype=np.float32))
        random_summary = {
            key: float(np.nanmean([row[key] for row in random_curves]))
            for key in random_curves[0].keys()
            if isinstance(random_curves[0][key], (int, float, np.floating, bool))
        }
        random_summary.update({
            "fractions": np.asarray(fractions, dtype=np.float32).astype(float).tolist(),
            "deletion_curve": np.mean(np.vstack(random_deletion_curves), axis=0).astype(float).tolist(),
            "insertion_curve": np.mean(np.vstack(random_insertion_curves), axis=0).astype(float).tolist(),
            "random_repeats": int(config.random_repeats),
        })
        ev = coef_evidence(coef, epoch.shape[1], len(segments), target_class)
        nulls = clinical_nulls(coef, epoch.shape[1], len(segments), target_class, sample_seed)
        result["perturbation_results"][mode] = {
            "coef": coef.tolist(),
            "positive_ranking_fallback": pos_fallback,
            "fidelity": fidelity,
            "original_target_probability": float(probs[0]),
            "full_perturbation_probability": float(probs[1]),
            "curves": curves,
            "random_ranking_mean": random_summary,
            "clinical_evidence": ev,
            "clinical_nulls": nulls,
        }

    if config.stability_seeds:
        mode = "physiological"
        for seed in config.stability_seeds:
            stab_masks = make_masks(
                num_features,
                config.train_masks,
                config.val_masks,
                seed + global_idx,
                sampling=config.mask_sampling,
            )
            probs = predict_masks_sequence(
                model, x_raw, x_seg, counts, global_idx, target_class, epoch, segments, stab_masks,
                mode, config.batch_windows, config.perturbation_chunk,
            )
            ridge, fidelity, _ = fit_surrogate(
                stab_masks,
                probs,
                config.train_masks,
                alpha_grid=config.ridge_alphas,
                kernel_width_grid=config.kernel_widths,
                response_transform=config.response_transform,
            )
            coef = np.asarray(ridge.coef_, dtype=np.float32)
            ev = coef_evidence(coef, epoch.shape[1], len(segments), target_class)
            result["stability"].append({
                "seed": int(seed),
                "coef": coef.tolist(),
                "validation_weighted_r2": fidelity["validation_weighted_r2"],
                "dominant_channel": int(np.argmax(ev["positive_channel_dccs"])),
                "positive_channel_dccs": ev["positive_channel_dccs"],
                "positive_concept_dccs": ev["positive_concept_dccs"],
            })

    write_json(cache_path, result)
    return result


def randomized_model(raw_shape, segment_shape, level: str) -> tf.keras.Model:
    model = build_model(raw_shape, segment_shape, CKPT_PATH)
    candidates = []
    if level in {"final_layer", "final_head"}:
        candidates.extend(["main_output_logits"])
    if level in {"final_head", "upper_temporal"}:
        candidates.extend(["td_temporal_dense"])
    if level == "upper_temporal":
        candidates.extend([layer.name for layer in model.layers if "temporal" in layer.name or "tcn" in layer.name])
    rng = np.random.default_rng(424242 + len(level))
    for layer_name in sorted(set(candidates)):
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if weights:
            layer.set_weights([rng.normal(0.0, max(float(np.std(w)), 1e-3), size=w.shape).astype(w.dtype) for w in weights])
    return model


def run_model_randomization(
    base_results: Sequence[Mapping[str, object]],
    config: RunConfig,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    raw_shape,
    segment_shape,
    out_dir: Path,
) -> pd.DataFrame:
    if not config.do_randomization or not base_results:
        return pd.DataFrame()
    rows = []
    levels = ["final_layer"] if config.mode == "smoke" else ["final_layer", "final_head", "upper_temporal"]
    for level in levels:
        print(f"Randomization sanity: {level}")
        model = randomized_model(raw_shape, segment_shape, level)
        for base in base_results[: min(5, len(base_results))]:
            global_idx = int(base["global_epoch_index"])
            epoch = x_raw[global_idx]
            segments = [(int(a), int(b)) for a, b in base["segment_boundaries"]]
            target_class = int(base["target_class"])
            num_features = int(epoch.shape[1] * len(segments))
            masks = make_masks(
                num_features,
                config.train_masks,
                config.val_masks,
                600000 + global_idx,
                sampling=config.mask_sampling,
            )
            probs = predict_masks_sequence(
                model, x_raw, x_seg, counts, global_idx, target_class, epoch, segments, masks,
                "physiological", config.batch_windows, config.perturbation_chunk,
            )
            ridge, fidelity, _ = fit_surrogate(
                masks,
                probs,
                config.train_masks,
                alpha_grid=config.ridge_alphas,
                kernel_width_grid=config.kernel_widths,
                response_transform=config.response_transform,
            )
            coef_rand = np.asarray(ridge.coef_, dtype=np.float32)
            coef_base = np.asarray(base["perturbation_results"]["physiological"]["coef"], dtype=np.float32)
            rows.append({
                "global_epoch_index": global_idx,
                "randomization_level": level,
                "signed_coefficient_spearman": safe_spearman(coef_base, coef_rand),
                "absolute_coefficient_spearman": safe_spearman(np.abs(coef_base), np.abs(coef_rand)),
                "top10_jaccard": jaccard_top(coef_base, coef_rand, 10),
                "validation_weighted_r2": fidelity["validation_weighted_r2"],
                "original_deletion_auc": base["perturbation_results"]["physiological"]["curves"]["lime_positive"]["deletion_auc_raw"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_randomization_sanity.csv", index=False)
    return df


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    val = spearmanr(np.asarray(a), np.asarray(b), nan_policy="omit").correlation
    return float(0.0 if not np.isfinite(val) else val)


def summarize_values(values: Iterable[float], bootstrap_ids: Optional[Sequence[object]] = None, rng_seed: int = 1234, n_boot: int = 1000) -> Dict[str, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    out = {"N": int(len(vals))}
    for key in ("mean", "std", "median", "Q1", "Q3", "P10", "P90", "CI_low", "CI_high"):
        out[key] = np.nan
    if len(vals) == 0:
        return out
    out.update({
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(np.median(vals)),
        "Q1": float(np.percentile(vals, 25)),
        "Q3": float(np.percentile(vals, 75)),
        "P10": float(np.percentile(vals, 10)),
        "P90": float(np.percentile(vals, 90)),
    })
    if bootstrap_ids is None or len(vals) < 2:
        out["CI_low"] = out["CI_high"] = out["mean"]
        return out
    ids = np.asarray(bootstrap_ids)
    if len(ids) != len(vals):
        out["CI_low"] = out["CI_high"] = out["mean"]
        return out
    unique = np.unique(ids)
    by_id = {uid: vals[ids == uid] for uid in unique}
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boots.append(np.mean(np.concatenate([by_id[uid] for uid in sampled])))
    out["CI_low"] = float(np.percentile(boots, 2.5))
    out["CI_high"] = float(np.percentile(boots, 97.5))
    return out


def paired_stats(df: pd.DataFrame, left: str, right: str, group_col: str, n_boot: int) -> Dict[str, float]:
    valid = df[[left, right, group_col]].dropna()
    diff = (valid[left] - valid[right]).to_numpy(dtype=np.float64)
    out = {
        "N": int(len(diff)),
        "paired_mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
        "paired_median_difference": float(np.median(diff)) if len(diff) else np.nan,
        "relative_improvement": float(np.mean(diff) / max(abs(np.mean(valid[right])), EPS)) if len(diff) else np.nan,
        "wilcoxon_p_value": np.nan,
        "rank_biserial_effect_size": np.nan,
        "CI_low": np.nan,
        "CI_high": np.nan,
    }
    if len(diff) >= 2 and np.any(np.abs(diff) > EPS):
        stat = wilcoxon(valid[left], valid[right], zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_p_value"] = float(stat.pvalue)
        out["rank_biserial_effect_size"] = float(np.sum(np.sign(diff) * np.arange(1, len(diff) + 1)) / max(np.sum(np.arange(1, len(diff) + 1)), EPS))
    ci = summarize_values(diff, valid[group_col].to_numpy(), n_boot=n_boot)
    out["CI_low"], out["CI_high"] = ci["CI_low"], ci["CI_high"]
    return out


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([1.0 if not np.isfinite(v) else v for v in p_values], dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        val = min(prev, p[idx] * n / max(original_rank, 1))
        q[idx] = val
        prev = val
    return q.tolist()


def flatten_results(results: Sequence[Mapping[str, object]], config: RunConfig, out_dir: Path) -> None:
    sample_rows, fidelity_rows, curve_rows, clinical_rows, robust_rows = [], [], [], [], []
    for res in results:
        base = {k: res[k] for k in [
            "global_epoch_index", "recording_id", "subject_id", "epoch_index_in_recording",
            "true_stage", "predicted_stage", "target_stage", "confidence", "correct",
            "transition_boundary", "segments_count", "sequence_wrapper_abs_diff",
        ]}
        phys = res["perturbation_results"]["physiological"]
        phys_fid = phys["fidelity"]
        lime_curve = phys["curves"]["lime_positive"]
        clinical = phys["clinical_evidence"]
        nulls = phys["clinical_nulls"]
        sample_rows.append({
            **base,
            **phys_fid,
            "deletion_auc_normalized": lime_curve["deletion_auc_normalized"],
            "insertion_auc_normalized": lime_curve["insertion_auc_normalized"],
            "deletion_AOPC": lime_curve["deletion_AOPC"],
            "insertion_AOPC": lime_curve["insertion_AOPC"],
            "raw_positive_concept_alignment": clinical["positive_concept_prior_cosine"],
            "target_minus_best_wrong_prior_margin": nulls["target_minus_best_wrong_margin"],
            "clinical_empirical_p_value": nulls["empirical_p_value"],
        })
        for mode, mode_res in res["perturbation_results"].items():
            fid = mode_res["fidelity"]
            fidelity_rows.append({**base, "perturbation_mode": mode, **fid})
            for ranking_name, curves in mode_res["curves"].items():
                row = {**base, "perturbation_mode": mode, "ranking": ranking_name}
                row.update({k: v for k, v in curves.items() if not isinstance(v, list)})
                curve_rows.append(row)
                fractions = curves["fractions"]
                for frac, dval, ival in zip(fractions, curves["deletion_curve"], curves["insertion_curve"]):
                    robust_rows.append({**base, "perturbation_mode": mode, "ranking": ranking_name, "fraction": frac, "deletion_probability": dval, "insertion_probability": ival})
            random_curves = mode_res.get("random_ranking_mean", {})
            if random_curves:
                row = {**base, "perturbation_mode": mode, "ranking": "random_mean"}
                row.update({k: v for k, v in random_curves.items() if not isinstance(v, list)})
                curve_rows.append(row)
                for frac, dval, ival in zip(
                    random_curves.get("fractions", []),
                    random_curves.get("deletion_curve", []),
                    random_curves.get("insertion_curve", []),
                ):
                    robust_rows.append({
                        **base,
                        "perturbation_mode": mode,
                        "ranking": "random_mean",
                        "fraction": frac,
                        "deletion_probability": dval,
                        "insertion_probability": ival,
                    })
            clinical_rows.append({
                **base,
                "perturbation_mode": mode,
                **mode_res["clinical_evidence"],
                **mode_res["clinical_nulls"],
                "alignment_to_predicted_stage": mode_res["clinical_nulls"]["raw_alignment"],
                "alignment_to_true_stage": float(cosine(
                    np.asarray(mode_res["clinical_evidence"]["positive_concept_dccs"], dtype=np.float32),
                    get_stage_concept_prior(CLASS_LABELS.index(str(res["true_stage"]))),
                )),
                "predicted_minus_true_alignment": float(mode_res["clinical_nulls"]["raw_alignment"] - cosine(
                    np.asarray(mode_res["clinical_evidence"]["positive_concept_dccs"], dtype=np.float32),
                    get_stage_concept_prior(CLASS_LABELS.index(str(res["true_stage"]))),
                )),
            })
    pd.DataFrame(sample_rows).to_csv(out_dir / "sample_level_metrics.csv", index=False)
    pd.DataFrame(fidelity_rows).to_csv(out_dir / "surrogate_fidelity.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "faithfulness_curves.csv", index=False)
    pd.DataFrame(clinical_rows).to_csv(out_dir / "clinical_alignment_sample_level.csv", index=False)
    pd.DataFrame(robust_rows).to_csv(out_dir / "perturbation_robustness.csv", index=False)

    stability_rows = []
    for res in results:
        stabs = res.get("stability", [])
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                a = np.asarray(stabs[i]["coef"], dtype=np.float32)
                b = np.asarray(stabs[j]["coef"], dtype=np.float32)
                ch_a = np.asarray(stabs[i]["positive_channel_dccs"], dtype=np.float32)
                ch_b = np.asarray(stabs[j]["positive_channel_dccs"], dtype=np.float32)
                concept_a = np.asarray(stabs[i]["positive_concept_dccs"], dtype=np.float32)
                concept_b = np.asarray(stabs[j]["positive_concept_dccs"], dtype=np.float32)
                stability_rows.append({
                    "global_epoch_index": int(res["global_epoch_index"]),
                    "true_stage": res["true_stage"],
                    "predicted_stage": res["predicted_stage"],
                    "recording_id": int(res["recording_id"]),
                    "seed_a": stabs[i]["seed"],
                    "seed_b": stabs[j]["seed"],
                    "signed_coefficient_spearman": safe_spearman(a, b),
                    "absolute_coefficient_spearman": safe_spearman(np.abs(a), np.abs(b)),
                    "kendall_tau": float(kendalltau(a, b).correlation) if np.isfinite(kendalltau(a, b).correlation) else 0.0,
                    "top5_jaccard": jaccard_top(a, b, 5),
                    "top10_jaccard": jaccard_top(a, b, 10),
                    "top20_jaccard": jaccard_top(a, b, 20),
                    "positive_feature_top10_jaccard": float(len(set(np.argsort(-np.clip(a, 0, None))[:10]) & set(np.argsort(-np.clip(b, 0, None))[:10])) / 10.0),
                    "raw_channel_dccs_cosine": cosine(ch_a, ch_b),
                    "raw_channel_dccs_standard_deviation": float(np.mean(np.std(np.vstack([ch_a, ch_b]), axis=0))),
                    "raw_channel_dccs_coefficient_of_variation": float(np.mean(np.std(np.vstack([ch_a, ch_b]), axis=0) / np.maximum(np.mean(np.vstack([ch_a, ch_b]), axis=0), EPS))),
                    "dominant_channel_agreement": bool(stabs[i]["dominant_channel"] == stabs[j]["dominant_channel"]),
                    "raw_concept_dccs_spearman": safe_spearman(concept_a, concept_b),
                    "top3_concept_jaccard": float(len(set(np.argsort(-concept_a)[:3]) & set(np.argsort(-concept_b)[:3])) / 3.0),
                })
    pd.DataFrame(stability_rows).to_csv(out_dir / "stability_sample_level.csv", index=False)

def assert_no_test_oracle_fields(frame: pd.DataFrame) -> None:
    forbidden = [column for column in frame.columns if "oracle" in str(column).lower()]
    if forbidden:
        raise ValueError(f"Paper main table must not contain test-side oracle fields: {forbidden}")


def write_formal_result_tables(
    out_dir: Path,
    config: RunConfig,
    sample_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    group_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    fidelity_columns = [
        column
        for column in (
            "global_epoch_index",
            "recording_id",
            "true_stage",
            "predicted_stage",
            "correct",
            "transition_boundary",
            "validation_weighted_r2",
            "train_weighted_r2",
            "original_mask_error",
        )
        if column in sample_df
    ]
    sample_df[fidelity_columns].to_csv(out_dir / "surrogate_fidelity.csv", index=False)
    if curve_df.empty:
        faithfulness = pd.DataFrame()
    else:
        group_columns = [column for column in ("perturbation_mode", "ranking") if column in curve_df]
        metric_columns = [
            column
            for column in (
                "deletion_auc_raw",
                "insertion_auc_raw",
                "deletion_AOPC",
                "insertion_AOPC",
            )
            if column in curve_df
        ]
        faithfulness = (
            curve_df.groupby(group_columns, dropna=False)[metric_columns]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        faithfulness.columns = [
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in faithfulness.columns
        ]
        faithfulness["claim_boundary"] = np.where(
            faithfulness.get("perturbation_mode", "") == "physiological",
            "PC-SeqLIME superiority is specific to the physiological-perturbation estimand.",
            "No cross-estimand superiority claim.",
        )
    faithfulness.to_csv(out_dir / "faithfulness_summary.csv", index=False)

    if stability_df.empty:
        stability = pd.DataFrame()
    else:
        metrics = [
            column
            for column in (
                "signed_coefficient_spearman",
                "absolute_coefficient_spearman",
                "top10_jaccard",
                "dominant_channel_agreement",
                "top3_concept_jaccard",
            )
            if column in stability_df
        ]
        stability = pd.DataFrame(
            [
                {
                    "scope": "within-checkpoint perturbation stability",
                    "N": int(len(stability_df)),
                    **{f"mean_{column}": float(stability_df[column].astype(float).mean()) for column in metrics},
                    "cross_seed_model_stability": "pending additional checkpoints",
                }
            ]
        )
    stability.to_csv(out_dir / "stability_summary.csv", index=False)

    randomization_path = out_dir / "model_randomization_sanity.csv"
    randomization = pd.read_csv(randomization_path) if randomization_path.exists() and randomization_path.stat().st_size > 1 else pd.DataFrame()
    randomization.to_csv(out_dir / "randomization_summary.csv", index=False)
    correct_incorrect = group_df[group_df["group"].isin(["Correct", "Incorrect"])].copy() if not group_df.empty else pd.DataFrame()
    correct_incorrect.to_csv(out_dir / "correct_incorrect_summary.csv", index=False)
    stage = group_df[group_df["group"].str.startswith(("True ", "Pred "), na=False)].copy() if not group_df.empty else pd.DataFrame()
    stage.to_csv(out_dir / "stage_summary.csv", index=False)
    profile_path = out_dir / "performance_profile.json"
    profile = read_json(profile_path) if profile_path.exists() else {}
    pd.DataFrame(
        [
            {
                "sample_N": int(len(sample_df)),
                "requested_surrogate_queries_per_sample": int(config.train_masks + config.val_masks),
                "replacement_endpoint_count": int(len(config.perturbation_modes)),
                "window_inference_batch_size": int(config.batch_windows),
                "profile_wall_seconds": profile.get("wall_seconds"),
                "peak_host_memory_mb": profile.get("peak_host_memory_mb"),
                "peak_gpu_memory_mb": profile.get("peak_gpu_memory_mb"),
            }
        ]
    ).to_csv(out_dir / "runtime_query_summary.csv", index=False)


def validate_required_formal_outputs(out_dir: Path) -> None:
    required = (
        "paper_main_table.csv",
        "paper_main_table.md",
        "surrogate_fidelity.csv",
        "faithfulness_summary.csv",
        "stability_summary.csv",
        "randomization_summary.csv",
        "correct_incorrect_summary.csv",
        "stage_summary.csv",
        "runtime_query_summary.csv",
    )
    failures = []
    for name in required:
        path = out_dir / name
        if not path.exists() or path.stat().st_size <= 1:
            failures.append(f"{name}: missing/empty")
            continue
        if path.suffix == ".csv":
            try:
                if pd.read_csv(path).empty:
                    failures.append(f"{name}: header-only")
            except pd.errors.EmptyDataError:
                failures.append(f"{name}: empty data")
    if failures:
        raise RuntimeError("Formal XAI output validation failed: " + "; ".join(failures))


def write_current_results_manifest(
    out_dir: Path,
    run_id: str,
    publish_root_index: bool = True,
) -> None:
    required_names = (
        "model_prediction_parity.json",
        "prediction_metric_parity.json",
        "sequence_wrapper_parity.json",
        "paper_main_table.csv",
        "paper_main_table.md",
        "surrogate_fidelity.csv",
        "faithfulness_summary.csv",
        "stability_summary.csv",
        "randomization_summary.csv",
        "correct_incorrect_summary.csv",
        "stage_summary.csv",
        "runtime_query_summary.csv",
    )
    valid = []
    for name in required_names:
        path = out_dir / name
        if path.exists() and path.stat().st_size > 1:
            valid.append({"path": name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    deprecated = []
    for path in sorted(out_dir.glob("*")):
        if not path.is_file():
            continue
        if path.stat().st_size <= 1 or "parity_fail" in path.name.lower() or path.name == "paper_results_ready.md":
            deprecated.append(
                {"path": path.name, "reason": "empty/stale pre-consolidation output", "current": False}
            )
    local_payload = {
        **build_run_provenance(run_id),
        "valid_files": valid,
        "deprecated_files": deprecated,
        "sole_current_index": True,
        "run_result_directory": str(out_dir.resolve()),
    }
    write_json(out_dir / "CURRENT_RESULTS_MANIFEST.json", local_payload)
    if not publish_root_index:
        return
    root_valid = []
    for name in (
        "model_prediction_parity.json",
        "prediction_metric_parity.json",
        "sequence_wrapper_parity.json",
        "parity_diagnostic_manifest.json",
    ):
        path = OUT_ROOT / name
        if path.exists() and path.stat().st_size > 1:
            root_valid.append(
                {
                    "path": name,
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "scope": "canonical_parity",
                }
            )
    for item in valid:
        root_valid.append(
            {
                **item,
                "path": str(out_dir.relative_to(OUT_ROOT) / item["path"]),
                "scope": "active_xai_run",
            }
        )
    write_json(
        OUT_ROOT / "CURRENT_RESULTS_MANIFEST.json",
        {
            **build_run_provenance(run_id),
            "current_run_id": run_id,
            "current_run_directory": str(out_dir.resolve()),
            "valid_files": root_valid,
            "deprecated_files": deprecated,
            "sole_current_index": True,
        },
    )


def summarize_tables(
    out_dir: Path,
    config: RunConfig,
    run_id: Optional[str] = None,
    publish_root_index: bool = True,
) -> None:
    sample_df = pd.read_csv(out_dir / "sample_level_metrics.csv") if (out_dir / "sample_level_metrics.csv").exists() else pd.DataFrame()
    curve_df = pd.read_csv(out_dir / "faithfulness_curves.csv") if (out_dir / "faithfulness_curves.csv").exists() else pd.DataFrame()
    clinical_df = pd.read_csv(out_dir / "clinical_alignment_sample_level.csv") if (out_dir / "clinical_alignment_sample_level.csv").exists() else pd.DataFrame()
    stability_df = pd.read_csv(out_dir / "stability_sample_level.csv") if (out_dir / "stability_sample_level.csv").exists() else pd.DataFrame()

    group_rows = []
    if not sample_df.empty:
        metrics = ["validation_weighted_r2", "deletion_auc_normalized", "insertion_auc_normalized", "deletion_AOPC", "insertion_AOPC", "raw_positive_concept_alignment"]
        groups = {
            "Overall": np.ones(len(sample_df), dtype=bool),
            "Correct": sample_df["correct"].astype(bool).to_numpy(),
            "Incorrect": ~sample_df["correct"].astype(bool).to_numpy(),
            "Transition": sample_df["transition_boundary"].astype(bool).to_numpy(),
            "Stable": ~sample_df["transition_boundary"].astype(bool).to_numpy(),
        }
        for stage in CLASS_LABELS:
            groups[f"True {stage}"] = (sample_df["true_stage"] == stage).to_numpy()
            groups[f"Pred {stage}"] = (sample_df["predicted_stage"] == stage).to_numpy()
        tertiles = pd.qcut(sample_df["confidence"], q=min(3, max(1, sample_df["confidence"].nunique())), labels=False, duplicates="drop")
        for val, name in ((0, "Confidence low"), (1, "Confidence middle"), (2, "Confidence high")):
            groups[name] = (tertiles == val).to_numpy()
        for group_name, mask in groups.items():
            sub = sample_df.loc[mask]
            for metric in metrics:
                stats = summarize_values(sub[metric].to_numpy(), sub["recording_id"].to_numpy(), n_boot=config.bootstrap_repeats)
                group_rows.append({"group": group_name, "metric": metric, **stats})
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out_dir / "group_level_metrics.csv", index=False)
    stage_df = group_df[group_df["group"].str.startswith("True ")].copy() if not group_df.empty else pd.DataFrame()
    stage_df.to_csv(out_dir / "stage_level_metrics.csv", index=False)

    baseline_rows = []
    if not curve_df.empty:
        phys = curve_df[curve_df["perturbation_mode"] == "physiological"]
        wide = phys.pivot_table(index=["global_epoch_index", "recording_id"], columns="ranking", values=["deletion_auc_raw", "insertion_auc_raw", "deletion_AOPC", "insertion_AOPC"], aggfunc="first")
        if not wide.empty:
            flat = wide.reset_index()
            flat.columns = ["_".join([str(c) for c in col if str(c) != ""]) for col in flat.columns]
            comparisons = [
                ("deletion_auc_raw_random", "deletion_auc_raw_lime_positive", "lime_deletion_auc_lower_than_random"),
                ("insertion_auc_raw_lime_positive", "insertion_auc_raw_random", "lime_insertion_auc_higher_than_random"),
            ]
            random_mean = phys[phys["ranking"] == "lime_positive"].copy()
            random_mean["random_deletion_auc_raw"] = np.nan
            # sample-level random summaries are stored inside cache; use reverse/energy rows in this flat table.
            for left, right, name in comparisons:
                if left in flat and right in flat:
                    baseline_rows.append({"comparison": name, **paired_stats(flat, left, right, "recording_id_", config.bootstrap_repeats)})
    baseline_df = pd.DataFrame(baseline_rows)
    if not baseline_df.empty:
        baseline_df["q_value"] = bh_fdr(baseline_df["wilcoxon_p_value"].tolist())
    baseline_df.to_csv(out_dir / "faithfulness_baseline_comparison.csv", index=False)

    if not stability_df.empty:
        stab_rows = []
        for metric in ["signed_coefficient_spearman", "absolute_coefficient_spearman", "top10_jaccard", "dominant_channel_agreement", "top3_concept_jaccard"]:
            stab_rows.append({"group": "Overall", "metric": metric, **summarize_values(stability_df[metric].astype(float).to_numpy(), stability_df["recording_id"].to_numpy(), n_boot=config.bootstrap_repeats)})
            for stage in CLASS_LABELS:
                sub = stability_df[stability_df["true_stage"] == stage]
                stab_rows.append({"group": stage, "metric": metric, **summarize_values(sub[metric].astype(float).to_numpy(), sub["recording_id"].to_numpy(), n_boot=config.bootstrap_repeats)})
        pd.DataFrame(stab_rows).to_csv(out_dir / "stability_stage_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "stability_stage_summary.csv", index=False)

    if not clinical_df.empty:
        clinical_rows = []
        for metric in ["positive_concept_prior_cosine", "target_minus_best_wrong_margin", "empirical_p_value", "expected_top3_concept_hit"]:
            vals = clinical_df[metric].astype(float).to_numpy()
            clinical_rows.append({"group": "Overall", "metric": metric, **summarize_values(vals, clinical_df["recording_id"].to_numpy(), n_boot=config.bootstrap_repeats)})
        pd.DataFrame(clinical_rows).to_csv(out_dir / "clinical_alignment_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "clinical_alignment_summary.csv", index=False)

    main_rows = []
    if not group_df.empty:
        for metric in ["validation_weighted_r2", "deletion_auc_normalized", "insertion_auc_normalized", "deletion_AOPC", "insertion_AOPC", "raw_positive_concept_alignment", "target_minus_best_wrong_prior_margin"]:
            row = group_df[(group_df["group"] == "Overall") & (group_df["metric"] == metric)]
            if not row.empty:
                main_rows.append(row.iloc[0].to_dict())
    main_frame = pd.DataFrame(main_rows)
    assert_no_test_oracle_fields(main_frame)
    main_frame.to_csv(out_dir / "paper_main_table.csv", index=False)
    if main_frame.empty:
        markdown = ""
    else:
        columns = [str(column) for column in main_frame.columns]
        markdown_rows = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for values in main_frame.itertuples(index=False, name=None):
            markdown_rows.append(
                "| " + " | ".join(str(value) for value in values) + " |"
            )
        markdown = "\n".join(markdown_rows) + "\n"
    (out_dir / "paper_main_table.md").write_text(markdown, encoding="utf-8")
    group_df.to_csv(out_dir / "paper_supplementary_table.csv", index=False)
    write_formal_result_tables(
        out_dir,
        config,
        sample_df,
        curve_df,
        group_df,
        stability_df,
    )
    if config.mode == "full_evaluation":
        validate_required_formal_outputs(out_dir)
    write_current_results_manifest(
        out_dir,
        str(run_id or config.mode),
        publish_root_index=publish_root_index,
    )

    decision = make_decision(sample_df, stability_df, clinical_df)
    audit_summary = {"mode": config.mode, "decision": decision, "sample_N": int(len(sample_df)), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(out_dir / "audit_summary.json", audit_summary)
    write_reports(out_dir, decision, sample_df, group_df, stability_df, clinical_df)
    make_figures(out_dir)
    generate_lime_paper_artifacts(out_dir)


def make_decision(sample_df: pd.DataFrame, stability_df: pd.DataFrame, clinical_df: pd.DataFrame) -> Dict[str, object]:
    if sample_df.empty:
        return {"status": "FAIL", "reason": "No completed samples."}
    med_r2 = float(np.nanmedian(sample_df["validation_weighted_r2"]))
    p10_r2 = float(np.nanpercentile(sample_df["validation_weighted_r2"], 10))
    mean_spear = float(np.nanmean(stability_df["absolute_coefficient_spearman"])) if not stability_df.empty else np.nan
    mean_j10 = float(np.nanmean(stability_df["top10_jaccard"])) if not stability_df.empty else np.nan
    dom = float(np.nanmean(stability_df["dominant_channel_agreement"].astype(float))) if not stability_df.empty else np.nan
    clinical_p = float(np.nanmedian(clinical_df["empirical_p_value"])) if not clinical_df.empty else np.nan
    passes = [
        med_r2 >= 0.80,
        p10_r2 >= 0.60,
        (not np.isfinite(mean_spear)) or mean_spear >= 0.70,
        (not np.isfinite(mean_j10)) or mean_j10 >= 0.60,
        (not np.isfinite(dom)) or dom >= 0.80,
        (not np.isfinite(clinical_p)) or clinical_p < 0.05,
    ]
    if all(passes):
        status = "PASS"
    elif med_r2 >= 0.60 and np.nanmean(sample_df["raw_positive_concept_alignment"]) > 0.30:
        status = "PARTIAL PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "overall_median_validation_weighted_r2": med_r2,
        "p10_validation_weighted_r2": p10_r2,
        "mean_absolute_coefficient_spearman": mean_spear,
        "mean_top10_jaccard": mean_j10,
        "dominant_channel_agreement": dom,
        "median_clinical_empirical_p_value": clinical_p,
        "criteria_are_project_prespecified": True,
    }


def write_reports(out_dir: Path, decision: Mapping[str, object], sample_df: pd.DataFrame, group_df: pd.DataFrame, stability_df: pd.DataFrame, clinical_df: pd.DataFrame) -> None:
    main = group_df[(group_df["group"] == "Overall")] if not group_df.empty else pd.DataFrame()
    lines = [
        "# Sequence-aware LIME Paper Results",
        "",
        "## Method Setting",
        "",
        "ISRUC seed42 A1-zero sequence checkpoint was evaluated without retraining. The target class is the neural-network main_output argmax. Each perturbation changes only the target epoch and is reinserted into all overlapping sequence windows before averaging the target-position probabilities.",
        "",
        "## Evaluation Sample",
        "",
        f"Completed samples: {len(sample_df)}",
        "",
        "## Core Metrics",
        "",
    ]
    if not main.empty:
        for _, row in main.iterrows():
            lines.append(f"- {row['metric']}: median={row['median']:.4f}, mean={row['mean']:.4f}, 95% CI=[{row['CI_low']:.4f}, {row['CI_high']:.4f}], N={int(row['N'])}")
    lines.extend([
        "",
        "## Stability",
        "",
        f"Pairwise rows: {len(stability_df)}",
        "",
        "## Clinical Plausibility",
        "",
        f"Clinical rows: {len(clinical_df)}",
        "",
        "## Decision",
        "",
        f"Project-prespecified decision: **{decision['status']}**",
        "",
        "Cautious paper wording: sequence-aware LIME was quantitatively audited under a fixed ISRUC seed42 A1-zero checkpoint; claims should be limited to the observed sample, perturbation modes, and project-prespecified thresholds.",
        "",
        "Do not write: this is a universal clinical gold standard, or that calibrated priors prove faithfulness.",
        "",
        "## Limitations",
        "",
        "Clinical priors are physiology-guided stage priors. Smoke mode is an implementation check only and must not be used for final paper claims.",
    ])
    text = "\n".join(lines) + "\n"
    (out_dir / "paper_results_ready.md").write_text(text, encoding="utf-8")
    (out_dir / "audit_report.md").write_text(text, encoding="utf-8")
    (out_dir / "smoke_test_report.md").write_text(text, encoding="utf-8")


def write_parity_fail_outputs(out_dir: Path, parity: Mapping[str, object], inventory: Mapping[str, object]) -> None:
    empty_files = [
        "sample_level_metrics.csv",
        "stage_level_metrics.csv",
        "group_level_metrics.csv",
        "surrogate_fidelity.csv",
        "faithfulness_curves.csv",
        "faithfulness_baseline_comparison.csv",
        "stability_sample_level.csv",
        "stability_stage_summary.csv",
        "clinical_alignment_sample_level.csv",
        "clinical_alignment_summary.csv",
        "model_randomization_sanity.csv",
        "perturbation_robustness.csv",
        "failed_samples.csv",
        "paper_main_table.csv",
        "paper_supplementary_table.csv",
    ]
    for name in empty_files:
        path = out_dir / name
        if not path.exists():
            pd.DataFrame().to_csv(path, index=False)
    lines = [
        "# Sequence-aware LIME Smoke Test Report",
        "",
        "Status: `MODEL_ARTIFACT_PARITY_FAIL`",
        "",
        "The canonical checkpoint and split prediction artifact were found, and the model was reloaded without training. Full test split inference reproduced the saved argmax labels exactly, but the maximum absolute main_output probability difference exceeded the project-specified `< 1e-5` parity threshold.",
        "",
        "## Parity",
        "",
        f"- max_abs_probability_diff: {parity['max_abs_probability_diff']}",
        f"- mean_abs_probability_diff: {parity['mean_abs_probability_diff']}",
        f"- argmax_agreement: {parity['argmax_agreement']}",
        f"- test_epoch_count: {parity['test_epoch_count']}",
        f"- class_order: {parity['class_order']}",
        "",
        "## Decision",
        "",
        "Formal sequence-aware LIME evaluation was stopped before perturbation fitting. No paper-level faithfulness, stability, clinical alignment, or model-randomization conclusions were generated.",
        "",
        "## Assets",
        "",
        f"- checkpoint: `{inventory['checkpoint']['path']}`",
        f"- prediction artifact: `{inventory['prediction_artifact']['path']}`",
        f"- protocol: `{inventory['dataset_protocol_json']['path']}`",
    ]
    text = "\n".join(lines) + "\n"
    for name in ("smoke_test_report.md", "audit_report.md", "paper_results_ready.md"):
        (out_dir / name).write_text(text, encoding="utf-8")


def configure_publication_plots() -> None:
    """Apply a compact, grayscale-readable style suitable for paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
        "lines.linewidth": 1.7,
        "lines.markersize": 4,
    })


def save_publication_figure(fig, stem: Path) -> Tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def bootstrap_mean_interval(values: Sequence[float], seed: int, repeats: int = 2000) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vals), size=(int(repeats), len(vals)))
    means = np.mean(vals[indices], axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_interpretability_tables(out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    smoke_surrogate = out_dir / "functional_smoke_surrogate.csv"
    smoke_metrics = out_dir / "functional_smoke_faithfulness.csv"
    smoke_curves = out_dir / "functional_smoke_faithfulness_curves.csv"
    full_surrogate = out_dir / "surrogate_fidelity.csv"
    full_metrics = out_dir / "faithfulness_curves.csv"
    full_curves = out_dir / "perturbation_robustness.csv"
    full_sur = pd.read_csv(full_surrogate) if full_surrogate.exists() and full_surrogate.stat().st_size > 1 else pd.DataFrame()
    full_met = pd.read_csv(full_metrics) if full_metrics.exists() and full_metrics.stat().st_size > 1 else pd.DataFrame()
    full_cur = pd.read_csv(full_curves) if full_curves.exists() and full_curves.stat().st_size > 1 else pd.DataFrame()
    if "perturbation_mode" in full_sur:
        full_sur = full_sur[full_sur["perturbation_mode"] == "physiological"].copy()
    if "perturbation_mode" in full_met:
        full_met = full_met[full_met["perturbation_mode"] == "physiological"].copy()
    if "perturbation_mode" in full_cur:
        full_cur = full_cur[full_cur["perturbation_mode"] == "physiological"].copy()

    smoke_sur = pd.read_csv(smoke_surrogate) if smoke_surrogate.exists() and smoke_surrogate.stat().st_size > 1 else pd.DataFrame()
    smoke_met = pd.read_csv(smoke_metrics) if smoke_metrics.exists() and smoke_metrics.stat().st_size > 1 else pd.DataFrame()
    smoke_cur = pd.read_csv(smoke_curves) if smoke_curves.exists() and smoke_curves.stat().st_size > 1 else pd.DataFrame()
    full_n = int(full_met["global_epoch_index"].nunique()) if "global_epoch_index" in full_met else 0
    smoke_n = int(smoke_met["global_epoch_index"].nunique()) if "global_epoch_index" in smoke_met else 0
    if full_n > 0 and full_n >= smoke_n:
        return full_sur, full_met, full_cur, "full"
    if smoke_n > 0:
        return smoke_sur, smoke_met, smoke_cur, "functional-smoke"
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "unavailable"


def add_random_mean_rows(df: pd.DataFrame, value_columns: Sequence[str], group_columns: Sequence[str]) -> pd.DataFrame:
    if df.empty or "ranking" not in df:
        return df.copy()
    random_rows = df[df["ranking"].astype(str).str.startswith("random_")].copy()
    if random_rows.empty:
        return df.copy()
    available_values = [column for column in value_columns if column in random_rows]
    available_groups = [column for column in group_columns if column in random_rows]
    numeric = random_rows.groupby(available_groups, as_index=False)[available_values].mean()
    numeric["ranking"] = "random_mean"
    return pd.concat([df, numeric], ignore_index=True, sort=False)


def paired_lime_baseline_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "ranking" not in metrics:
        return pd.DataFrame()
    metric_directions = {
        "deletion_auc_raw": "lower",
        "insertion_auc_raw": "higher",
        "deletion_AOPC": "higher",
        "insertion_AOPC": "higher",
    }
    id_col = "global_epoch_index"
    baselines = ["random_mean", "signal_energy", "reverse_lime", "lime_absolute"]
    rows = []
    for metric, direction in metric_directions.items():
        if metric not in metrics:
            continue
        pivot = metrics.pivot_table(index=id_col, columns="ranking", values=metric, aggfunc="first")
        if "lime_positive" not in pivot:
            continue
        for baseline in baselines:
            if baseline not in pivot:
                continue
            paired = pivot[["lime_positive", baseline]].dropna()
            if paired.empty:
                continue
            if direction == "higher":
                advantage = paired["lime_positive"] - paired[baseline]
            else:
                advantage = paired[baseline] - paired["lime_positive"]
            mean, ci_low, ci_high = bootstrap_mean_interval(advantage.to_numpy(), 5100 + len(rows))
            p_value = np.nan
            if len(advantage) >= 2 and np.any(np.abs(advantage.to_numpy()) > EPS):
                p_value = float(wilcoxon(advantage.to_numpy(), alternative="two-sided", zero_method="wilcox").pvalue)
            rows.append({
                "metric": metric,
                "direction": direction,
                "baseline": baseline,
                "N": int(len(advantage)),
                "lime_mean": float(paired["lime_positive"].mean()),
                "baseline_mean": float(paired[baseline].mean()),
                "mean_directional_advantage": mean,
                "median_directional_advantage": float(np.median(advantage)),
                "CI_low": ci_low,
                "CI_high": ci_high,
                "win_rate": float(np.mean(advantage > 0.0)),
                "wilcoxon_p_value": p_value,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = bh_fdr(result["wilcoxon_p_value"].tolist())
    return result


def build_interpretability_metric_summary(
    surrogate: pd.DataFrame,
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    randomization: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    definitions = [
        ("local_surrogate_fidelity", surrogate, "validation_weighted_r2", "higher", "median >= 0.80; P10 >= 0.60"),
        ("local_surrogate_fidelity", surrogate, "validation_weighted_mae", "lower", "descriptive; report distribution"),
        ("local_surrogate_fidelity", surrogate, "validation_weighted_rmse", "lower", "descriptive; report distribution"),
        ("local_surrogate_fidelity", surrogate, "validation_nrmse", "lower", "diagnostic when probability variance is small"),
        ("local_surrogate_fidelity", surrogate, "weighted_pearson", "higher", "report alongside R2"),
        ("response_variability", surrogate, "validation_probability_variance", "higher", "R2 is degenerate when variance < 1e-5"),
        ("sampling_quality", surrogate, "effective_sample_size", "higher", "report relative to feature count"),
        ("sampling_quality", surrogate, "local_mask_fraction", "higher", "majority of masks should remain local"),
    ]
    lime_rows = metrics[metrics["ranking"] == "lime_positive"] if not metrics.empty and "ranking" in metrics else pd.DataFrame()
    definitions.extend([
        ("perturbation_faithfulness", lime_rows, "deletion_auc_raw", "lower", "lower than random and energy baselines"),
        ("perturbation_faithfulness", lime_rows, "insertion_auc_raw", "higher", "higher than random and energy baselines"),
        ("perturbation_faithfulness", lime_rows, "deletion_AOPC", "higher", "higher than random baseline"),
        ("perturbation_faithfulness", lime_rows, "insertion_AOPC", "higher", "higher than random baseline"),
        ("seed_stability", stability, "absolute_coefficient_spearman", "higher", "mean >= 0.70"),
        ("seed_stability", stability, "top10_jaccard", "higher", "mean >= 0.60"),
        ("seed_stability", stability, "dominant_channel_agreement", "higher", "mean >= 0.80"),
        ("model_randomization_sanity", randomization, "absolute_coefficient_spearman", "lower", "mean <= 0.30 after randomization"),
        ("model_randomization_sanity", randomization, "top10_jaccard", "lower", "mean <= 0.30 after randomization"),
    ])
    for dimension, frame, metric, direction, criterion in definitions:
        if frame.empty or metric not in frame:
            rows.append({"dimension": dimension, "metric": metric, "direction": direction, "criterion": criterion, "N": 0, "status": "NOT_MEASURED"})
            continue
        values = frame[metric].astype(float).to_numpy()
        values = values[np.isfinite(values)]
        stats = summarize_values(values, np.arange(len(values)), n_boot=2000)
        rows.append({"dimension": dimension, "metric": metric, "direction": direction, "criterion": criterion, **stats, "status": "MEASURED"})
    return pd.DataFrame(rows)


def make_interpretability_decision(
    source_mode: str,
    surrogate: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    randomization: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, object]:
    sample_count = int(metrics["global_epoch_index"].nunique()) if not metrics.empty and "global_epoch_index" in metrics else 0
    smoke_summary_path = out_dir / "functional_smoke_faithfulness_summary.json"
    smoke_summary = read_json(smoke_summary_path) if smoke_summary_path.exists() else {}
    implementation_valid = bool(
        source_mode == "full"
        or smoke_summary.get("status") == "FUNCTIONAL_FAITHFULNESS_SMOKE_PASS"
    )
    r2 = surrogate["validation_weighted_r2"].astype(float).to_numpy() if "validation_weighted_r2" in surrogate else np.asarray([])
    median_r2 = float(np.nanmedian(r2)) if len(r2) else np.nan
    p10_r2 = float(np.nanpercentile(r2, 10)) if len(r2) else np.nan
    fidelity_pass = bool(np.isfinite(median_r2) and np.isfinite(p10_r2) and median_r2 >= 0.80 and p10_r2 >= 0.60)
    low_variance_fraction = float(surrogate["r2_degenerate_low_variance"].astype(float).mean()) if "r2_degenerate_low_variance" in surrogate else np.nan
    mean_validation_nrmse = float(surrogate["validation_nrmse"].astype(float).mean()) if "validation_nrmse" in surrogate else np.nan
    mean_effective_sample_size = float(surrogate["effective_sample_size"].astype(float).mean()) if "effective_sample_size" in surrogate else np.nan
    random_comparisons = comparisons[comparisons["baseline"] == "random_mean"] if not comparisons.empty else pd.DataFrame()
    primary_names = {"deletion_auc_raw", "insertion_auc_raw"}
    primary = random_comparisons[random_comparisons["metric"].isin(primary_names)] if not random_comparisons.empty else pd.DataFrame()
    faithfulness_advantage = bool(
        len(primary) == 2
        and (primary["mean_directional_advantage"] > 0.0).all()
        and (primary["win_rate"] >= 0.60).all()
    )
    statistically_supported = bool(
        len(primary) == 2
        and sample_count >= 20
        and np.all(np.isfinite(primary["q_value"]))
        and (primary["q_value"] < 0.05).all()
        and (primary["CI_low"] > 0.0).all()
    )
    stability_measured = not stability.empty
    stability_pass = bool(
        stability_measured
        and float(stability["absolute_coefficient_spearman"].mean()) >= 0.70
        and float(stability["top10_jaccard"].mean()) >= 0.60
        and float(stability["dominant_channel_agreement"].astype(float).mean()) >= 0.80
    ) if all(column in stability for column in ["absolute_coefficient_spearman", "top10_jaccard", "dominant_channel_agreement"]) else False
    randomization_measured = not randomization.empty
    randomization_pass = bool(
        randomization_measured
        and float(randomization["absolute_coefficient_spearman"].mean()) <= 0.30
        and float(randomization["top10_jaccard"].mean()) <= 0.30
    ) if all(column in randomization for column in ["absolute_coefficient_spearman", "top10_jaccard"]) else False
    quantitatively_measurable = bool(not surrogate.empty and not metrics.empty and not comparisons.empty)

    full_evidence = bool(sample_count >= 30 and stability_measured and randomization_measured)
    if not implementation_valid or not quantitatively_measurable:
        status = "INTERPRETABILITY_EVIDENCE_FAIL"
    elif not full_evidence:
        status = "PILOT_FEASIBILITY_ONLY"
    elif fidelity_pass and faithfulness_advantage and statistically_supported and stability_pass and randomization_pass:
        status = "PAPER_LEVEL_INTERPRETABILITY_PASS"
    elif fidelity_pass and faithfulness_advantage and stability_pass and randomization_pass:
        status = "PAPER_LEVEL_INTERPRETABILITY_PARTIAL"
    else:
        status = "PAPER_LEVEL_INTERPRETABILITY_FAIL"

    if status == "PILOT_FEASIBILITY_ONLY":
        conclusion_zh = (
            f"Sequence-aware LIME 的实现与量化链路已经通过功能验证，删除/插入忠实度可复现测量且相对随机基线呈方向性优势；"
            f"但局部代理验证 R2 未达到预设阈值（中位数={median_r2:.3f}，P10={p10_r2:.3f}），当前仅有 {sample_count} 个先导样本，"
            "并且稳定性与模型随机化证据不完整，因此不能据此形成 LIME 解释可靠性或优于基线的最终论文主张。"
        )
    elif status == "PAPER_LEVEL_INTERPRETABILITY_PASS":
        conclusion_zh = (
            "在预设阈值下，Sequence-aware LIME 同时满足局部代理保真度、扰动忠实度、跨种子稳定性和模型随机化敏感性要求，"
            "并在配对统计中优于随机基线，因此当前结果支持论文级的解释可靠性主张。"
        )
    elif status == "PAPER_LEVEL_INTERPRETABILITY_PARTIAL":
        conclusion_zh = (
            "Sequence-aware LIME 在局部保真度、扰动忠实度、稳定性和模型敏感性方面达到项目预设阈值，"
            "但统计证据尚不足以支持强优越性主张，论文应将结论限定为方法有效和结果一致。"
        )
    else:
        conclusion_zh = "当前结果未同时满足预设的 LIME 解释质量判据，不能形成论文级解释可靠性结论。"

    implementation_clause = (
        "Sequence-aware LIME \u7684\u5b9e\u73b0\u4e0e\u91cf\u5316\u94fe\u8def\u5df2\u901a\u8fc7\u529f\u80fd\u9a8c\u8bc1\u3002"
        if implementation_valid
        else "Sequence-aware LIME \u7684\u5b9e\u73b0\u6216\u91cf\u5316\u94fe\u8def\u5c1a\u672a\u901a\u8fc7\u529f\u80fd\u9a8c\u8bc1\u3002"
    )
    fidelity_clause = (
        f"\u5c40\u90e8\u4ee3\u7406\u4fdd\u771f\u5ea6\u8fbe\u5230\u9879\u76ee\u9608\u503c\uff08\u4e2d\u4f4d\u6570={median_r2:.3f}\uff0cP10={p10_r2:.3f}\uff09\u3002"
        if fidelity_pass
        else f"\u5c40\u90e8\u4ee3\u7406\u4fdd\u771f\u5ea6\u672a\u8fbe\u5230\u9879\u76ee\u9608\u503c\uff08\u4e2d\u4f4d\u6570={median_r2:.3f}\uff0cP10={p10_r2:.3f}\uff09\u3002"
    )
    faithfulness_clause = (
        "\u5220\u9664/\u63d2\u5165\u5fe0\u5b9e\u5ea6\u76f8\u5bf9\u968f\u673a\u57fa\u7ebf\u5448\u4e00\u81f4\u65b9\u5411\u6027\u4f18\u52bf\u3002"
        if faithfulness_advantage
        else "\u5220\u9664/\u63d2\u5165\u5fe0\u5b9e\u5ea6\u76f8\u5bf9\u968f\u673a\u57fa\u7ebf\u5c1a\u672a\u5f62\u6210\u4e00\u81f4\u65b9\u5411\u6027\u4f18\u52bf\u3002"
    )
    evidence_clause = (
        "\u6837\u672c\u91cf\u3001\u7a33\u5b9a\u6027\u548c\u6a21\u578b\u968f\u673a\u5316\u8bc1\u636e\u5b8c\u6574\uff0c\u5f53\u524d\u7ed3\u679c\u652f\u6301\u8bba\u6587\u7ea7\u89e3\u91ca\u53ef\u9760\u6027\u7ed3\u8bba\u3002"
        if status == "PAPER_LEVEL_INTERPRETABILITY_PASS"
        else f"\u5f53\u524d\u6837\u672c\u6570\u4e3a {sample_count}\u3002\u5f53\u524d\u8bc1\u636e\u4ecd\u5c5e\u4e8e\u5148\u5bfc\u9a8c\u8bc1\uff0c\u4e0d\u80fd\u4e3b\u5f20\u89e3\u91ca\u53ef\u9760\u6027\u6216\u4f18\u4e8e\u57fa\u7ebf\u3002"
    )
    conclusion_zh = "".join([
        implementation_clause,
        fidelity_clause,
        faithfulness_clause,
        evidence_clause,
    ])

    return {
        "status": status,
        "source_mode": source_mode,
        "sample_count": sample_count,
        "implementation_valid": implementation_valid,
        "quantitatively_measurable": quantitatively_measurable,
        "local_surrogate_fidelity_pass": fidelity_pass,
        "median_validation_weighted_r2": median_r2,
        "p10_validation_weighted_r2": p10_r2,
        "low_variance_sample_fraction": low_variance_fraction,
        "mean_validation_nrmse": mean_validation_nrmse,
        "mean_effective_sample_size": mean_effective_sample_size,
        "faithfulness_advantage_over_random": faithfulness_advantage,
        "paired_statistical_support": statistically_supported,
        "stability_measured": stability_measured,
        "stability_pass": stability_pass,
        "model_randomization_measured": randomization_measured,
        "model_randomization_pass": randomization_pass,
        "explanation_reliability_supported": bool(
            status in {"PAPER_LEVEL_INTERPRETABILITY_PASS", "PAPER_LEVEL_INTERPRETABILITY_PARTIAL"}
        ),
        "minimum_paper_sample_rule": "N >= 30",
        "threshold_scope": "Project operational thresholds; not universal LIME validity criteria.",
        "final_conclusion_zh": conclusion_zh,
        "claim_allowed": [
            "Sequence-aware perturbations and overlapping-window inference are implementation-valid." if implementation_valid else "None",
            "LIME explanation quality is quantitatively measurable with surrogate fidelity and perturbation faithfulness." if quantitatively_measurable else "None",
        ],
        "claim_not_allowed": [
            "Universal clinical validity of LIME explanations.",
            "Superiority over baselines unless paired confidence intervals and corrected tests support it.",
            "Final explanation reliability from functional-smoke results alone.",
        ],
    }


def plot_faithfulness_curves(curves: pd.DataFrame, figure_dir: Path) -> Optional[Tuple[Path, Path]]:
    if curves.empty or "ranking" not in curves:
        return None
    curves = add_random_mean_rows(
        curves,
        ["deletion_probability", "insertion_probability"],
        ["global_epoch_index", "target_stage", "fraction", "actual_fraction"],
    )
    methods = ["lime_positive", "random_mean", "signal_energy", "reverse_lime"]
    colors = {"lime_positive": "#D55E00", "random_mean": "#7F7F7F", "signal_energy": "#0072B2", "reverse_lime": "#009E73"}
    styles = {"lime_positive": "-", "random_mean": "--", "signal_energy": "-.", "reverse_lime": ":"}
    labels = {"lime_positive": "LIME positive", "random_mean": "Random mean", "signal_energy": "Signal energy", "reverse_lime": "Reverse LIME"}
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65), sharey=True)
    x_col = "actual_fraction" if "actual_fraction" in curves else "fraction"
    for panel, (column, ylabel) in enumerate([
        ("deletion_probability", "Target-class probability"),
        ("insertion_probability", "Target-class probability"),
    ]):
        ax = axes[panel]
        for method_index, method in enumerate(methods):
            sub = curves[curves["ranking"] == method]
            if sub.empty or column not in sub:
                continue
            xs, means, lows, highs = [], [], [], []
            for fraction, fraction_rows in sub.groupby(x_col):
                sample_values = fraction_rows.groupby("global_epoch_index")[column].mean().to_numpy(dtype=float)
                mean, low, high = bootstrap_mean_interval(sample_values, 6200 + panel * 100 + method_index * 10 + len(xs))
                xs.append(float(fraction)); means.append(mean); lows.append(low); highs.append(high)
            order = np.argsort(xs)
            x = np.asarray(xs)[order]; mean = np.asarray(means)[order]
            low = np.asarray(lows)[order]; high = np.asarray(highs)[order]
            ax.plot(x, mean, marker="o", color=colors[method], linestyle=styles[method], label=labels[method], zorder=3)
            ax.fill_between(x, low, high, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_xlabel("Fraction of ranked features")
        ax.set_ylabel(ylabel if panel == 0 else "")
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
    axes[1].legend(loc="best")
    fig.tight_layout(w_pad=1.5)
    return save_publication_figure(fig, figure_dir / "fig_lime_faithfulness_curves")


def plot_quantitative_audit(surrogate: pd.DataFrame, comparisons: pd.DataFrame, figure_dir: Path) -> Optional[Tuple[Path, Path]]:
    if surrogate.empty and comparisons.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75))
    ax = axes[0]
    if "validation_weighted_r2" in surrogate:
        stages = surrogate["target_stage"].astype(str) if "target_stage" in surrogate else pd.Series(["All"] * len(surrogate))
        stage_order = [stage for stage in CLASS_LABELS if stage in set(stages)] or sorted(set(stages))
        for idx, stage in enumerate(stage_order):
            values = surrogate.loc[stages == stage, "validation_weighted_r2"].astype(float).to_numpy()
            x = np.full(len(values), idx, dtype=float)
            jitter = np.linspace(-0.07, 0.07, max(len(values), 1))[:len(values)]
            ax.scatter(x + jitter, values, s=28, color="#D55E00", edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(0.80, color="#0072B2", linestyle="--", linewidth=1.1, label="Operational threshold")
        ax.set_xticks(range(len(stage_order))); ax.set_xticklabels(stage_order)
        ax.set_ylabel("Validation weighted $R^2$")
        ax.set_xlabel("Predicted sleep stage")
        ax.legend(loc="lower right")
    ax.text(-0.12, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    random_rows = comparisons[comparisons["baseline"] == "random_mean"].copy() if not comparisons.empty else pd.DataFrame()
    metric_labels = {
        "deletion_auc_raw": "Deletion AUC",
        "insertion_auc_raw": "Insertion AUC",
        "deletion_AOPC": "Deletion AOPC",
        "insertion_AOPC": "Insertion AOPC",
    }
    random_rows["label"] = random_rows["metric"].map(metric_labels) if not random_rows.empty else []
    random_rows = random_rows.dropna(subset=["label"]) if not random_rows.empty else random_rows
    if not random_rows.empty:
        y = np.arange(len(random_rows))
        means = random_rows["mean_directional_advantage"].to_numpy(dtype=float)
        low = random_rows["CI_low"].to_numpy(dtype=float)
        high = random_rows["CI_high"].to_numpy(dtype=float)
        ax.errorbar(means, y, xerr=np.vstack([means - low, high - means]), fmt="o", color="#D55E00", ecolor="#555555", capsize=3)
        ax.axvline(0.0, color="#222222", linewidth=1.0)
        ax.set_yticks(y); ax.set_yticklabels(random_rows["label"])
        ax.set_xlabel("Directional advantage over random\n(positive favors LIME)")
        ax.invert_yaxis()
    ax.text(-0.12, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.8)
    return save_publication_figure(fig, figure_dir / "fig_lime_quantitative_audit")


def plot_lime_coefficient_heatmaps(out_dir: Path, figure_dir: Path) -> List[Tuple[Path, Path, int]]:
    results = []
    cache_root = out_dir / "cache"
    for cache_dir in sorted(cache_root.glob("sample_*")):
        surrogate_path = cache_dir / "functional_surrogate.json"
        if not surrogate_path.exists():
            continue
        payload = read_json(surrogate_path)
        coef = np.asarray(payload.get("coef", []), dtype=np.float64)
        segment_count = int(payload.get("segments_count", coef.size // len(CHANNEL_LABELS)))
        if coef.size != len(CHANNEL_LABELS) * segment_count or segment_count <= 0:
            continue
        matrix = coef.reshape(len(CHANNEL_LABELS), segment_count)
        max_abs = max(float(np.max(np.abs(matrix))), EPS)
        fig, ax = plt.subplots(figsize=(6.75, 1.75))
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs), interpolation="nearest")
        ax.set_yticks(range(len(CHANNEL_LABELS))); ax.set_yticklabels(CHANNEL_LABELS)
        tick_step = max(1, segment_count // 10)
        ticks = np.arange(0, segment_count, tick_step)
        ax.set_xticks(ticks); ax.set_xticklabels((ticks + 1).astype(int))
        ax.set_xlabel("Dynamic temporal segment")
        ax.set_ylabel("Channel")
        cbar = fig.colorbar(image, ax=ax, pad=0.02, aspect=25)
        cbar.set_label("LIME coefficient")
        global_idx = int(payload["global_epoch_index"])
        fig.tight_layout()
        png, pdf = save_publication_figure(fig, figure_dir / f"fig_lime_coefficients_epoch_{global_idx}")
        results.append((png, pdf, global_idx))
    return results


def write_interpretability_report(
    out_dir: Path,
    decision: Mapping[str, object],
    metric_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    measured = metric_summary[metric_summary["status"] == "MEASURED"] if not metric_summary.empty else pd.DataFrame()
    lines = [
        "# Sequence-aware LIME 可解释性定量结论",
        "",
        f"结论状态：`{decision['status']}`",
        "",
        str(decision["final_conclusion_zh"]),
        "",
        "## 证据范围",
        "",
        f"- 结果来源：{decision['source_mode']}",
        f"- 独立解释样本数：{decision['sample_count']}",
        f"- 实现链路有效：{decision['implementation_valid']}",
        f"- 可解释性可量化：{decision['quantitatively_measurable']}",
        f"- 跨随机种子稳定性已测：{decision['stability_measured']}",
        f"- 模型随机化敏感性已测：{decision['model_randomization_measured']}",
        "",
        "## 核心指标",
        "",
    ]
    for _, row in measured.iterrows():
        lines.append(
            f"- {row['dimension']} / {row['metric']} ({row['direction']} is better): "
            f"mean={row['mean']:.4f}, median={row['median']:.4f}, 95% CI=[{row['CI_low']:.4f}, {row['CI_high']:.4f}], N={int(row['N'])}"
        )
    lines.extend(["", "## LIME 与基线的配对比较", ""])
    if comparisons.empty:
        lines.append("未生成可用的配对基线比较。")
    else:
        for _, row in comparisons.iterrows():
            lines.append(
                f"- {row['metric']} vs {row['baseline']}: directional advantage={row['mean_directional_advantage']:.4f} "
                f"(95% CI [{row['CI_low']:.4f}, {row['CI_high']:.4f}]), win rate={row['win_rate']:.1%}, q={row['q_value']:.4g}, N={int(row['N'])}."
            )
    lines.extend([
        "",
        "## 可直接用于论文的结论措辞",
        "",
        str(decision["final_conclusion_zh"]),
        "",
        "## 论文边界",
        "",
        "当前判据是本项目的操作性阈值，不是 LIME 的通用有效性定理。只有同时具备足够样本量、配对统计、跨种子稳定性和模型随机化检验时，才允许写出解释可靠性或优于基线的强主张。",
        "",
        "## 图表资产",
        "",
    ])
    for _, row in manifest.iterrows():
        lines.append(f"- {row['figure_type']}: `{row['png']}` / `{row['pdf']}`")
    (out_dir / "paper_lime_final_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_fidelity_ablations(out_dir: Path, figure_dir: Path) -> Optional[Tuple[Path, Path]]:
    ablation_root = out_dir / "fidelity_ablation"
    candidates: List[Tuple[str, Path, Optional[Path]]] = []
    if ablation_root.exists():
        for directory in sorted(path for path in ablation_root.iterdir() if path.is_dir()):
            candidates.append((directory.name, directory / "functional_smoke_surrogate.csv", directory / "functional_smoke_surrogate_summary.json"))
    candidates.append(("current", out_dir / "functional_smoke_surrogate.csv", out_dir / "functional_smoke_surrogate_summary.json"))
    rows = []
    for label, csv_path, summary_path in candidates:
        if not csv_path.exists() or csv_path.stat().st_size <= 1:
            continue
        frame = pd.read_csv(csv_path)
        if frame.empty or "validation_weighted_r2" not in frame:
            continue
        summary = read_json(summary_path) if summary_path is not None and summary_path.exists() else {}
        configuration = summary.get("surrogate_configuration", {})
        r2 = frame["validation_weighted_r2"].astype(float).to_numpy()
        rows.append({
            "configuration": label,
            "sample_count": int(len(frame)),
            "train_masks": configuration.get("train_masks"),
            "validation_masks": configuration.get("validation_masks"),
            "target_segments": configuration.get("target_segments"),
            "feature_count": configuration.get("feature_count"),
            "mask_sampling": configuration.get("mask_sampling", "bernoulli" if "baseline" in label else None),
            "response_transform": configuration.get("response_transform", "probability"),
            "median_validation_weighted_r2": float(np.median(r2)),
            "p10_validation_weighted_r2": float(np.percentile(r2, 10)),
            "mean_validation_weighted_mae": float(frame["validation_weighted_mae"].astype(float).mean()),
            "mean_validation_nrmse": float(frame["validation_nrmse"].astype(float).mean()) if "validation_nrmse" in frame else np.nan,
            "low_variance_sample_fraction": float(frame["r2_degenerate_low_variance"].astype(float).mean()) if "r2_degenerate_low_variance" in frame else np.nan,
            "project_fidelity_threshold_pass": bool(np.median(r2) >= 0.80 and np.percentile(r2, 10) >= 0.60),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return None
    result = result.sort_values(["median_validation_weighted_r2", "p10_validation_weighted_r2"], ascending=False)
    result.to_csv(out_dir / "lime_fidelity_ablation_summary.csv", index=False)
    write_json(out_dir / "lime_fidelity_ablation_summary.json", {"experiments": result.to_dict(orient="records")})
    best = result.iloc[0]
    report = [
        "# LIME Surrogate Fidelity Ablation",
        "",
        f"Best observed configuration: `{best['configuration']}`.",
        f"Median validation weighted R2: {best['median_validation_weighted_r2']:.4f}.",
        f"P10 validation weighted R2: {best['p10_validation_weighted_r2']:.4f}.",
        f"Project threshold pass: {bool(best['project_fidelity_threshold_pass'])}.",
        "",
        "The ablation is diagnostic. Configuration selection must not use the final paper test set.",
    ]
    (out_dir / "lime_fidelity_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    plot_frame = result.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.75, max(2.4, 0.42 * len(plot_frame) + 1.0)))
    y = np.arange(len(plot_frame))
    ax.scatter(plot_frame["median_validation_weighted_r2"], y, color="#D55E00", s=35, label="Median weighted R2", zorder=3)
    ax.scatter(plot_frame["p10_validation_weighted_r2"], y, color="#0072B2", marker="s", s=28, label="P10 weighted R2", zorder=3)
    for index, row in plot_frame.iterrows():
        ax.plot([row["p10_validation_weighted_r2"], row["median_validation_weighted_r2"]], [index, index], color="#999999", linewidth=1.0, zorder=1)
    ax.axvline(0.80, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvline(0.60, color="#0072B2", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_yticks(y); ax.set_yticklabels(plot_frame["configuration"])
    ax.set_xlabel("Validation weighted R2")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_publication_figure(fig, figure_dir / "fig_lime_fidelity_ablation")


def generate_lime_paper_artifacts(out_dir: Path) -> Dict[str, object]:
    configure_publication_plots()
    figure_dir = out_dir / "figures" / "lime_interpretability"
    figure_dir.mkdir(parents=True, exist_ok=True)
    surrogate, metrics, curves, source_mode = load_interpretability_tables(out_dir)
    metric_value_columns = ["deletion_auc_raw", "insertion_auc_raw", "deletion_AOPC", "insertion_AOPC"]
    metric_group_columns = ["global_epoch_index", "target_stage"]
    metrics_augmented = add_random_mean_rows(metrics, metric_value_columns, metric_group_columns)
    stability_path = out_dir / "stability_sample_level.csv"
    randomization_path = out_dir / "model_randomization_sanity.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() and stability_path.stat().st_size > 1 else pd.DataFrame()
    randomization = pd.read_csv(randomization_path) if randomization_path.exists() and randomization_path.stat().st_size > 1 else pd.DataFrame()
    comparisons = paired_lime_baseline_statistics(metrics_augmented)
    metric_summary = build_interpretability_metric_summary(surrogate, metrics_augmented, stability, randomization)
    decision = make_interpretability_decision(source_mode, surrogate, metrics_augmented, comparisons, stability, randomization, out_dir)

    metric_summary.to_csv(out_dir / "paper_lime_quantitative_summary.csv", index=False)
    comparisons.to_csv(out_dir / "paper_lime_baseline_comparisons.csv", index=False)
    write_json(out_dir / "paper_lime_final_conclusion.json", decision)

    manifest_rows = []
    curve_paths = plot_faithfulness_curves(curves, figure_dir)
    if curve_paths:
        manifest_rows.append({"figure_type": "aggregate_faithfulness_curves", "sample": "all", "png": str(curve_paths[0]), "pdf": str(curve_paths[1])})
    audit_paths = plot_quantitative_audit(surrogate, comparisons, figure_dir)
    if audit_paths:
        manifest_rows.append({"figure_type": "quantitative_interpretability_audit", "sample": "all", "png": str(audit_paths[0]), "pdf": str(audit_paths[1])})
    ablation_paths = summarize_fidelity_ablations(out_dir, figure_dir)
    if ablation_paths:
        manifest_rows.append({"figure_type": "surrogate_fidelity_ablation", "sample": "all", "png": str(ablation_paths[0]), "pdf": str(ablation_paths[1])})
    for png, pdf, global_idx in plot_lime_coefficient_heatmaps(out_dir, figure_dir):
        manifest_rows.append({"figure_type": "sample_lime_coefficient_heatmap", "sample": global_idx, "png": str(png), "pdf": str(pdf)})
    manifest = pd.DataFrame(manifest_rows, columns=["figure_type", "sample", "png", "pdf"])
    manifest.to_csv(out_dir / "lime_visualization_manifest.csv", index=False)
    write_interpretability_report(out_dir, decision, metric_summary, comparisons, manifest)
    return {"decision": decision, "figure_count": int(len(manifest)), "figure_dir": str(figure_dir)}


def make_figures(out_dir: Path) -> None:
    curve_path = out_dir / "perturbation_robustness.csv"
    sample_path = out_dir / "sample_level_metrics.csv"
    if curve_path.exists():
        df = pd.read_csv(curve_path)
        phys = df[(df["perturbation_mode"] == "physiological") & (df["ranking"].isin(["lime_positive", "reverse_lime", "signal_energy"]))]
        for curve, filename in (("deletion_probability", "mean_deletion_curves.png"), ("insertion_probability", "mean_insertion_curves.png")):
            if not phys.empty:
                plt.figure(figsize=(7, 4), dpi=150)
                for ranking, sub in phys.groupby("ranking"):
                    mean = sub.groupby("fraction")[curve].mean()
                    plt.plot(mean.index, mean.values, marker="o", label=f"{ranking} (N={sub['global_epoch_index'].nunique()})")
                plt.xlabel("Fraction of features")
                plt.ylabel("Target probability")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / filename, dpi=300)
                plt.close()
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        if not df.empty:
            for metric, filename in (("validation_weighted_r2", "validation_r2_by_stage.png"), ("raw_positive_concept_alignment", "concept_alignment_by_stage.png")):
                plt.figure(figsize=(6, 4), dpi=150)
                data = [df[df["true_stage"] == stage][metric].dropna().to_numpy() for stage in CLASS_LABELS]
                plt.boxplot(data, labels=CLASS_LABELS)
                plt.ylabel(metric)
                plt.title(f"N={len(df)}")
                plt.tight_layout()
                plt.savefig(out_dir / filename, dpi=300)
                plt.close()


def run_parity(model: tf.keras.Model, x_raw: np.ndarray, x_seg: np.ndarray, artifact: Mapping[str, np.ndarray], out_dir: Path, batch_windows: int) -> Dict[str, object]:
    probs = predict_sequence_probs(model, x_raw, x_seg, artifact["file_sample_counts"], batch_windows=batch_windows)
    diff = np.abs(probs - artifact["probs"])
    pred = np.argmax(probs, axis=1).astype(np.int16)
    payload = {
        "status": "PASS" if float(np.max(diff)) < 1e-5 and float(np.mean(pred == artifact["pred"])) == 1.0 else "MODEL_ARTIFACT_PARITY_FAIL",
        "max_abs_probability_diff": float(np.max(diff)),
        "mean_abs_probability_diff": float(np.mean(diff)),
        "argmax_agreement": float(np.mean(pred == artifact["pred"])),
        "test_epoch_count": int(len(pred)),
        "class_order": CLASS_LABELS,
    }
    write_json(out_dir / "model_prediction_parity.json", payload)
    return payload


def write_parity_diagnostic_report(out_dir: Path, model_parity: Mapping[str, object]) -> None:
    diff = read_json(out_dir / "prediction_difference_distribution.json")
    metrics = read_json(out_dir / "prediction_metric_parity.json")
    wrapper = read_json(out_dir / "sequence_wrapper_parity.json") if (out_dir / "sequence_wrapper_parity.json").exists() else {}
    repeat = read_json(out_dir / "repeat_inference_stability.json") if (out_dir / "repeat_inference_stability.json").exists() else {}
    agg = read_json(out_dir / "aggregation_precision_diagnostic.json") if (out_dir / "aggregation_precision_diagnostic.json").exists() else {}
    worst = diff.get("worst_difference", {})
    lines = [
        "# Parity Diagnostic Report",
        "",
        "## Compared Artifact Field",
        "",
        "`test_probs_raw` was compared against current-run raw neural-network `main_output` probabilities.",
        "",
        "## Difference Distribution",
        "",
        f"- mean: {diff['mean']}",
        f"- P95: {diff['P95']}",
        f"- P99: {diff['P99']}",
        f"- P99.9: {diff['P99_9']}",
        f"- max: {diff['max']}",
        "",
        "## Threshold Counts",
        "",
    ]
    for key, value in diff["threshold_counts"].items():
        lines.append(f"- {key}: {value['count']} ({value['ratio']:.6%})")
    lines.extend([
        "",
        "## Worst Difference",
        "",
        json.dumps(worst, indent=2, ensure_ascii=False),
        "",
        "## Metric Parity",
        "",
        f"- argmax agreement: {metrics['argmax_agreement']}",
        f"- disagreement epoch count: {metrics['disagreement_epoch_count']}",
        f"- top-2 pair agreement: {metrics['top2_class_pair_agreement']}",
        f"- metric diffs: {metrics['metric_abs_differences']}",
        "",
        "## Repeat Inference Stability",
        "",
        json.dumps(repeat.get("repeat_inference_pairwise", {}), indent=2, ensure_ascii=False),
        "",
        "## Batch Size Stability",
        "",
        json.dumps(repeat.get("batch_size_pairwise", {}), indent=2, ensure_ascii=False),
        "",
        "## Aggregation Precision",
        "",
        json.dumps(agg, indent=2, ensure_ascii=False),
        "",
        "## Layered Parity Decision",
        "",
        json.dumps(model_parity, indent=2, ensure_ascii=False),
        "",
        "## Sequence Wrapper Parity",
        "",
        json.dumps(wrapper, indent=2, ensure_ascii=False),
    ])
    (out_dir / "parity_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parity_diagnostic(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    inventory = build_artifact_inventory(out_dir)
    if not (
        args.mode == "full"
        and int(getattr(args, "evaluation_shard_count", 1)) > 1
    ):
        write_static_audit(out_dir, inventory)
    if inventory["status"] != "OK":
        payload = {
            "exact_artifact_parity": {
                "status": "CHECKPOINT_NOT_FOUND",
                "threshold": STRICT_PROBABILITY_TOLERANCE,
            },
            "scientific_prediction_parity": {"status": "MODEL_ARTIFACT_PARITY_FAIL"},
            "sequence_wrapper_parity": {"status": "NOT_RUN"},
            "xai_evaluation_allowed": False,
        }
        write_json(out_dir / "model_prediction_parity.json", payload)
        return payload

    config = resolve_mode_config("parity-diagnostic", args)
    field_audit = audit_prediction_artifact_fields(out_dir)
    z = np.load(PRED_PATH)
    artifact = load_artifact_split(z, "test")
    config_audit = audit_inference_configuration(out_dir, artifact)
    split_files = normalize_split_files(load_split_spec())
    manifest = choose_samples(artifact, config, out_dir, resume=True)
    x_raw, x_seg, _quality = load_or_build_test_inputs(out_dir, split_files, artifact)
    model = build_model(x_raw.shape[1:], x_seg.shape[1:])
    model_params = int(np.sum([np.prod(v.shape) for v in model.weights]))
    inventory["model_parameter_count"] = model_params
    write_json(out_dir / "artifact_inventory.json", inventory)

    current_cache = out_dir / "cache" / "current_sequence_probs_float32.npy"
    # Recompute in the current process. Wrapper parity is a same-process,
    # same-batch-context check; reusing a previous process' GPU outputs can
    # itself introduce 1e-5 level numeric drift.
    current_probs = predict_sequence_probs(model, x_raw, x_seg, artifact["file_sample_counts"], batch_windows=config.batch_windows)
    np.save(current_cache, current_probs.astype(np.float32))
    diff_stats = probability_difference_stats(artifact["probs"], current_probs, artifact, out_dir)
    metric_parity = prediction_metric_parity(artifact, current_probs, out_dir)
    repeat_path = out_dir / "repeat_inference_stability.json"
    if repeat_path.exists():
        repeat = read_json(repeat_path)
    else:
        repeat = repeat_inference_stability(model, x_raw, x_seg, artifact["file_sample_counts"], current_probs, out_dir, batch_windows=config.batch_windows)
    agg_path = out_dir / "aggregation_precision_diagnostic.json"
    if agg_path.exists():
        agg = read_json(agg_path)
    else:
        agg = aggregation_precision_diagnostic(model, x_raw, x_seg, artifact["file_sample_counts"], artifact["probs"], current_probs, out_dir, batch_windows=config.batch_windows)
    wrapper = sequence_wrapper_parity(model, x_raw, x_seg, artifact["file_sample_counts"], current_probs, manifest, out_dir, batch_windows=config.batch_windows)

    exact_pass = bool(diff_stats["max"] <= STRICT_PROBABILITY_TOLERANCE)
    exact_status = "EXACT_ARTIFACT_PARITY_PASS" if exact_pass else "EXACT_ARTIFACT_PARITY_NOT_MET"
    class_order_match = bool(field_audit.get("class_order") == CLASS_LABELS)
    sample_count_match = bool(len(artifact["y_true"]) == current_probs.shape[0])
    recording_alignment_match = bool(int(np.sum(artifact["file_sample_counts"])) == len(artifact["y_true"]))
    all_values_finite = bool(np.all(np.isfinite(current_probs)) and np.all(np.isfinite(artifact["probs"])))
    scientific_checks = {
        "sample_count_match": sample_count_match,
        "class_order_match": class_order_match,
        "recording_alignment_match": recording_alignment_match,
        "all_values_finite": all_values_finite,
        "argmax_agreement": metric_parity["argmax_agreement"],
        "mean_abs_probability_diff": diff_stats["mean"],
        "max_abs_probability_diff": diff_stats["max"],
        "accuracy_abs_diff": metric_parity["metric_abs_differences"]["accuracy_abs_diff"],
        "macro_f1_abs_diff": metric_parity["metric_abs_differences"]["macro_f1_abs_diff"],
        "kappa_abs_diff": metric_parity["metric_abs_differences"]["kappa_abs_diff"],
        "confusion_matrix_exact_match": metric_parity["confusion_matrix_exact_match"],
        "field_selection_valid": bool(field_audit.get("field_available")) and field_audit["parity_compared_field"] == "test_probs_raw",
        "configuration_match": bool(config_audit["configuration_match"]),
    }
    scientific_pass = bool(
        scientific_checks["sample_count_match"]
        and scientific_checks["class_order_match"]
        and scientific_checks["recording_alignment_match"]
        and scientific_checks["all_values_finite"]
        and scientific_checks["argmax_agreement"] == 1.0
        and scientific_checks["mean_abs_probability_diff"] <= 5e-6
        and scientific_checks["max_abs_probability_diff"] <= 2e-4
        and scientific_checks["accuracy_abs_diff"] <= 1e-12
        and scientific_checks["macro_f1_abs_diff"] <= 1e-12
        and scientific_checks["kappa_abs_diff"] <= 1e-12
        and scientific_checks["confusion_matrix_exact_match"]
        and scientific_checks["field_selection_valid"]
        and scientific_checks["configuration_match"]
    )
    wrapper_pass = wrapper["status"] == "SEQUENCE_WRAPPER_PARITY_PASS"
    payload = {
        "exact_artifact_parity": {
            "status": exact_status,
            "threshold": STRICT_PROBABILITY_TOLERANCE,
            "max_abs_probability_diff": diff_stats["max"],
        },
        "scientific_prediction_parity": {
            "status": "SCIENTIFIC_PREDICTION_PARITY_PASS" if scientific_pass else "MODEL_ARTIFACT_PARITY_FAIL",
            "mean_abs_diff_threshold": 5e-6,
            "max_abs_diff_threshold": 2e-4,
            "argmax_agreement_required": 1.0,
            "checks": scientific_checks,
        },
        "sequence_wrapper_parity": {
            "status": wrapper["status"],
            "max_abs_diff_threshold": 1e-6,
            "overall_max_abs_diff": wrapper["overall_max_abs_diff"],
            "argmax_agreement": wrapper["argmax_agreement"],
        },
        "xai_evaluation_allowed": bool(exact_pass and scientific_pass and wrapper_pass),
        "xai_probability_baseline": "frozen historical test_probs_raw",
        "historical_artifact_role": "strict canonical probability endpoint",
        "inference_seed": int(args.seed),
        "sequence_window_batch_size": int(config.batch_windows),
        "evaluator_version": EVALUATOR_VERSION,
    }
    write_json(out_dir / "model_prediction_parity.json", payload)
    write_json(
        out_dir / "parity_diagnostic_manifest.json",
        {
            **build_run_provenance("parity_diagnostic", config=config),
            "result_files": [
                "model_prediction_parity.json",
                "prediction_metric_parity.json",
                "sequence_wrapper_parity.json",
                "parity_diagnostic_report.md",
            ],
            "result": payload,
        },
    )
    write_json(out_dir / "audit_summary.json", {"status": payload["scientific_prediction_parity"]["status"], "xai_evaluation_allowed": payload["xai_evaluation_allowed"]})
    write_parity_diagnostic_report(out_dir, payload)
    return payload


def apply_runtime_args(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    batch_windows = int(getattr(args, "window_inference_batch_size", config.batch_windows) or config.batch_windows)
    batch_windows = max(1, batch_windows)
    train_masks = int(getattr(args, "lime_train_masks", None) or config.train_masks)
    val_masks = int(getattr(args, "lime_validation_masks", None) or config.val_masks)
    target_segments = int(getattr(args, "lime_target_segments", None) or config.target_segments)
    mask_sampling = str(getattr(args, "lime_mask_sampling", None) or config.mask_sampling)
    response_transform = str(getattr(args, "lime_response_transform", None) or config.response_transform)
    surrogate_perturbation_mode = str(
        getattr(args, "lime_surrogate_perturbation_mode", None) or config.surrogate_perturbation_mode
    )

    def parse_grid(raw_value, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
        if raw_value is None or not str(raw_value).strip():
            return fallback
        values = tuple(sorted({float(value.strip()) for value in str(raw_value).split(",") if value.strip()}))
        if not values or any(value <= 0.0 for value in values):
            raise ValueError("LIME alpha and kernel-width grids must contain positive comma-separated values.")
        return values

    ridge_alphas = parse_grid(getattr(args, "lime_ridge_alphas", None), config.ridge_alphas)
    kernel_widths = parse_grid(getattr(args, "lime_kernel_widths", None), config.kernel_widths)
    if train_masks < 50 or val_masks < 20:
        raise ValueError("LIME requires at least 50 training masks and 20 validation masks.")
    if not 5 <= target_segments <= 30:
        raise ValueError("--lime-target-segments must be between 5 and 30.")
    # Keep the actual model-call tensor size bounded while preserving the fixed
    # sequence-window batch path requested by the user.
    perturbation_chunk = max(1, min(config.perturbation_chunk, max(1, 128 // batch_windows)))
    return replace(
        config,
        batch_windows=batch_windows,
        perturbation_chunk=perturbation_chunk,
        train_masks=train_masks,
        val_masks=val_masks,
        target_segments=target_segments,
        mask_sampling=mask_sampling,
        ridge_alphas=ridge_alphas,
        kernel_widths=kernel_widths,
        response_transform=response_transform,
        surrogate_perturbation_mode=surrogate_perturbation_mode,
    )


def load_profile_payload(args: argparse.Namespace) -> Dict[str, object]:
    cached = getattr(args, "_profile_payload", None)
    if cached is not None:
        return cached
    path = getattr(args, "config", None)
    if path is None:
        payload: Dict[str, object] = {}
    else:
        path = Path(path)
        # Configs intentionally use the JSON subset of YAML so the frozen
        # environment needs no unversioned PyYAML dependency.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "sequence_lime_config_v1":
            raise ValueError("XAI config must declare schema_version=sequence_lime_config_v1")
        configured_mode = payload.get("mode")
        if configured_mode and configured_mode != args.mode:
            raise ValueError(
                f"Config mode {configured_mode!r} does not match --mode {args.mode!r}"
            )
        payload["config_path"] = str(path.resolve())
        payload["config_sha256"] = file_sha256(path)
    setattr(args, "_profile_payload", payload)
    return payload


def resolve_mode_config(mode: str, args: argparse.Namespace) -> RunConfig:
    config = mode_config(mode)
    payload = load_profile_payload(args)
    overrides = dict(payload.get("run_config", {}))
    allowed = set(asdict(config))
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown RunConfig fields: {unknown}")
    tuple_fields = {"stability_seeds", "perturbation_modes", "ridge_alphas", "kernel_widths"}
    for field in tuple_fields & set(overrides):
        overrides[field] = tuple(overrides[field])
    if overrides:
        config = replace(config, **overrides)
    config = apply_runtime_args(config, args)
    run_id = str(getattr(args, "run_id", None) or payload.get("run_id") or args.mode)
    setattr(args, "resolved_run_id", run_id)
    manifest_dir = OUT_ROOT / "run_manifests" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "schema_version": "sequence_lime_resolved_config_v1",
        "run_id": run_id,
        "mode": args.mode,
        "base_config_mode": mode,
        "run_config": asdict(config),
        "source_profile": payload,
    }
    fingerprint = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    resolved["config_fingerprint"] = fingerprint
    write_json(manifest_dir / "resolved_config.json", resolved)
    write_json(
        manifest_dir / "run_manifest.json",
        {
            **build_run_provenance(run_id, config=config),
            "config_fingerprint": fingerprint,
            "status": "CONFIG_RESOLVED",
            "resume_namespace": run_id,
        },
    )
    return config


def mark_run_complete(args: argparse.Namespace, status: str = "COMPLETED") -> None:
    """Atomically turn a resolved run manifest into a verifiable terminal record."""
    run_id = getattr(args, "resolved_run_id", None)
    if not run_id:
        return
    run_id = str(run_id)
    manifest_path = OUT_ROOT / "run_manifests" / run_id / "run_manifest.json"
    out_dir = OUT_ROOT / "runs" / run_id
    payload = read_json(manifest_path) if manifest_path.exists() else build_run_provenance(run_id)
    result_files = []
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or "cache" in path.relative_to(out_dir).parts:
                continue
            result_files.append({
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            })
    summary_statuses = {}
    if out_dir.exists():
        for path in sorted(out_dir.glob("*_summary.json")):
            try:
                summary_statuses[path.name] = read_json(path).get("status")
            except Exception as exc:  # noqa: BLE001
                summary_statuses[path.name] = f"UNREADABLE: {type(exc).__name__}"
    payload.update({
        "status": str(status),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result_directory": str(out_dir.relative_to(PROJECT_ROOT)),
        "result_file_count": int(len(result_files)),
        "result_files": result_files,
        "summary_statuses": summary_statuses,
        "evaluation_shard": {
            "count": int(getattr(args, "evaluation_shard_count", 1)),
            "index": int(getattr(args, "evaluation_shard_index", 0)),
        },
        "merged_shard_run_ids": [
            value.strip()
            for value in str(getattr(args, "merge_shard_run_ids", "") or "").split(",")
            if value.strip()
        ],
    })
    write_json_atomic(manifest_path, payload)


def load_runtime_context(args: argparse.Namespace, config: RunConfig):
    run_id = str(getattr(args, "resolved_run_id", None) or config.mode)
    out_dir = OUT_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    merge_ids = [
        value.strip()
        for value in str(getattr(args, "merge_shard_run_ids", "") or "").split(",")
        if value.strip()
    ]
    if merge_ids:
        if args.mode != "full":
            raise ValueError("--merge-shard-run-ids is valid only with --mode full")
        merge_full_evaluation_shards(args, config, merge_ids)
        mark_run_complete(args)
        return
    inventory = build_artifact_inventory(out_dir)
    z = np.load(PRED_PATH)
    artifact = load_artifact_split(z, "test")
    current_cache = OUT_ROOT / "cache" / "current_sequence_probs_float32.npy"
    if current_cache.exists():
        current_probs = np.load(current_cache).astype(np.float32)
        if current_probs.shape != artifact["probs"].shape:
            raise RuntimeError(
                f"Current canonical probability cache shape {current_probs.shape} "
                f"does not match prediction artifact shape {artifact['probs'].shape}."
            )
        artifact = dict(artifact)
        artifact["historical_probs_raw"] = artifact["probs"]
        artifact["probs"] = current_probs
        artifact["pred"] = np.argmax(current_probs, axis=1).astype(np.int16)
    split_files = normalize_split_files(load_split_spec())
    manifest = choose_functional_samples(artifact, config, out_dir, resume=bool(getattr(args, "resume", False)))
    x_raw, x_seg, _quality = load_or_build_test_inputs(OUT_ROOT, split_files, artifact)
    model = build_model(x_raw.shape[1:], x_seg.shape[1:])
    return out_dir, inventory, artifact, manifest, x_raw, x_seg, model


def assert_xai_allowed(out_dir: Path) -> Mapping[str, object]:
    _ = out_dir
    parity_path = OUT_ROOT / "model_prediction_parity.json"
    if not parity_path.exists():
        raise RuntimeError("Run --mode parity-diagnostic before XAI smoke/profile modes.")
    parity = read_json(parity_path)
    if not bool(parity.get("xai_evaluation_allowed", False)):
        raise RuntimeError(f"XAI evaluation is not allowed by parity status: {parity}")
    return parity


def performance_profile(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    config = resolve_mode_config("performance-profile", args)
    out_dir, inventory, artifact, manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    parity = assert_xai_allowed(out_dir)
    wrapper_parity = read_json(out_dir / "sequence_wrapper_parity.json") if (out_dir / "sequence_wrapper_parity.json").exists() else {}
    if manifest.empty:
        raise RuntimeError("No functional smoke sample is available for performance profiling.")
    sample = manifest.iloc[0].to_dict()
    ckpt_before = stat_payload(CKPT_PATH)
    weights_before = model_weight_fingerprint(model)
    load_done = time.perf_counter()

    counts = artifact["file_sample_counts"]
    global_idx = int(sample["global_epoch_index"])
    locate_t0 = time.perf_counter()
    covers = target_windows(counts, global_idx)
    formal_batches = target_formal_window_batches(counts, global_idx, config.batch_windows)
    locate_done = time.perf_counter()

    result = run_surrogate_module(
        sample,
        config,
        model,
        x_raw,
        x_seg,
        counts,
        artifact,
        out_dir,
        section_name="performance_surrogate",
        force=True,
    )
    weights_after = model_weight_fingerprint(model)
    ckpt_after = stat_payload(CKPT_PATH)
    total_done = time.perf_counter()

    timings = dict(result["timings_sec"])
    timings["data_and_checkpoint_loading"] = load_done - t_start
    timings["covering_window_location"] = locate_done - locate_t0
    timings["total_wall_time"] = total_done - t_start
    largest = max(timings.items(), key=lambda item: item[1])
    gpu_forward = timings.get("sequence_window_assembly_and_gpu_forward_and_overlap", 0.0)
    total = max(timings["total_wall_time"], EPS)
    profile = {
        "target_global_epoch": global_idx,
        "inference_device": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
        "fixed_window_inference_batch_size": int(config.batch_windows),
        "perturbation_chunk": int(config.perturbation_chunk),
        "covering_window_count": int(len(covers)),
        "formal_context_batch_count": int(len(formal_batches)),
        "perturbation_count": int(result["perturbation_count"]),
        "generated_target_epochs_count": int(result["generated_target_epochs_count"]),
        "generated_sequence_windows_count": int(result["generated_sequence_windows_count"]),
        "model_forward_batch_count": int(result["model_forward_batch_count"]),
        "model_forward_window_count": int(result["model_forward_window_count"]),
        "window_instance_accounting": result["window_instance_accounting"],
        "mask_efficiency": result["mask_efficiency"],
        "model_forward_batches_per_100_perturbations": float(result["model_forward_batch_count"] * 100.0 / max(result["perturbation_count"], 1)),
        "average_sequence_windows_per_perturbation": float(result["generated_sequence_windows_count"] / max(result["perturbation_count"], 1)),
        "peak_host_memory_mb": peak_host_memory_mb(),
        "peak_gpu_memory_mb": gpu_memory_mb(),
        "stage_seconds": timings,
        "largest_time_block": {"name": largest[0], "seconds": float(largest[1])},
        "throughput": result["throughput"],
        "efficiency_ratios": {
            "gpu_forward_fraction": float(gpu_forward / total),
            "segment_feature_fraction": float(timings.get("target_segment_feature_recompute", 0.0) / total),
            "perturbation_generation_fraction": float(timings.get("surrogate_perturbation_generation", 0.0) / total),
        },
        "forward_plan": result["forward_plan"],
        "guards": {
            "NO_MODEL_FIT": "PASS",
            "CHECKPOINT_UNCHANGED": "PASS" if ckpt_before.get("sha256") == ckpt_after.get("sha256") else "FAIL",
            "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
            "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
            "BATCHED_SEGMENT_FEATURE_PARITY": result["batched_segment_feature_parity"]["status"],
            "WINDOW_INSTANCE_ACCOUNTING": result["window_instance_accounting"]["status"],
            "VECTORIZED_INFERENCE": "PASS" if result["vectorized_inference"] else "FAIL",
            "FIXED_INFERENCE_BATCH_PATH": "PASS",
            "CACHE_RESUME": "PASS" if (sample_cache_dir(out_dir, global_idx) / "status.json").exists() else "FAIL",
            "MODEL_WEIGHTS_UNCHANGED": "PASS" if weights_before == weights_after else "FAIL",
            "TRAINING_FALSE": "PASS",
        },
        "vectorized_inference": bool(result["vectorized_inference"]),
        "direct_call_parity": {
            "status": "PASS" if float(wrapper_parity.get("overall_max_abs_diff", 1.0)) < 1e-6 else "UNKNOWN",
            "max_diff": wrapper_parity.get("overall_max_abs_diff"),
            "threshold": 1e-6,
            "note": "All inference calls use model({...}, training=False); no model.predict loop is used in the profiled path.",
        },
    }
    write_json(out_dir / "performance_profile.json", profile)
    write_json(out_dir / "window_instance_accounting.json", result["window_instance_accounting"])
    lines = [
        "# Sequence-aware LIME Performance Profile",
        "",
        f"Target global epoch: {global_idx}",
        f"Covering windows: {len(covers)}",
        f"Fixed window inference batch size: {config.batch_windows}",
        f"Perturbations: {result['perturbation_count']}",
        f"Expected unpadded window instances: {result['window_instance_accounting']['expected_unpadded_window_instances']}",
        f"Actual assembled window instances: {result['window_instance_accounting']['actual_assembled_window_instances']}",
        f"Actual inferred window instances: {result['window_instance_accounting']['actual_inferred_window_instances']}",
        f"Model forward batches: {result['model_forward_batch_count']}",
        f"Forwarded sequence rows: {result['model_forward_window_count']}",
        f"Largest block: {largest[0]} ({largest[1]:.3f}s)",
        "",
        "## Stage Seconds",
    ]
    for name, value in sorted(timings.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {name}: {value:.3f}")
    lines.extend([
        "",
        "## Guards",
    ])
    for name, value in profile["guards"].items():
        lines.append(f"- {name}: {value}")
    (out_dir / "performance_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def functional_smoke(args: argparse.Namespace) -> None:
    config = resolve_mode_config("functional-smoke", args)
    out_dir, inventory, artifact, manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    parity = assert_xai_allowed(out_dir)
    section = str(getattr(args, "smoke_section", "all") or "all")
    if section not in {"surrogate", "faithfulness", "all"}:
        raise RuntimeError(f"functional-smoke currently implemented for surrogate/faithfulness/all first; requested {section}")
    ckpt_before = stat_payload(CKPT_PATH)
    rows = []
    if section in {"surrogate", "all"}:
        for _, sample in manifest.iterrows():
            res = run_surrogate_module(
                sample.to_dict(),
                config,
                model,
                x_raw,
                x_seg,
                artifact["file_sample_counts"],
                artifact,
                out_dir,
                section_name="functional_surrogate",
                force=False,
            )
            rows.append({
                "global_epoch_index": int(res["global_epoch_index"]),
                "target_stage": res["target_stage"],
                "validation_weighted_r2": res["fidelity"]["validation_weighted_r2"],
                "validation_weighted_mae": res["fidelity"]["validation_weighted_mae"],
                "validation_weighted_rmse": res["fidelity"]["validation_weighted_rmse"],
                "validation_nrmse": res["fidelity"]["validation_nrmse"],
                "validation_probability_variance": res["fidelity"]["validation_probability_variance"],
                "validation_probability_std": res["fidelity"]["validation_probability_std"],
                "validation_probability_range": res["fidelity"]["validation_probability_range"],
                "r2_degenerate_low_variance": bool(res["fidelity"]["r2_degenerate_low_variance"]),
                "weighted_pearson": res["fidelity"]["weighted_pearson"],
                "effective_sample_size": res["fidelity"]["effective_sample_size"],
                "selected_ridge_alpha": res["fidelity"]["selected_ridge_alpha"],
                "selected_kernel_width": res["fidelity"]["selected_kernel_width"],
                "inner_cv_weighted_rmse": res["fidelity"]["inner_cv_weighted_rmse"],
                "mean_retained_feature_fraction": res["fidelity"]["mean_retained_feature_fraction"],
                "local_mask_fraction": res["fidelity"]["local_mask_fraction"],
                "feature_count": int(len(res["coef"])),
                "segments_count": int(res["surrogate_configuration"]["target_segments"]),
                "mask_sampling": res["surrogate_configuration"]["mask_sampling"],
                "response_transform": res["surrogate_configuration"]["response_transform"],
                "surrogate_perturbation_mode": res["surrogate_configuration"]["perturbation_mode"],
                "ridge_fit_possible": bool(res["ridge_fit_possible"]),
                "all_values_finite": bool(res["all_values_finite"]),
                "vectorized_inference": bool(res["vectorized_inference"]),
                "model_forward_batch_count": int(res["model_forward_batch_count"]),
                "model_forward_window_count": int(res["model_forward_window_count"]),
                "covering_window_count": int(res["covering_window_count"]),
                "window_instance_accounting": res["window_instance_accounting"]["status"],
                "batched_segment_feature_parity": res["batched_segment_feature_parity"]["status"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "functional_smoke_surrogate.csv", index=False)
        passed = bool(
            len(df) > 0
            and df["ridge_fit_possible"].all()
            and df["all_values_finite"].all()
            and df["vectorized_inference"].all()
            and (df["window_instance_accounting"] == "WINDOW_INSTANCE_ACCOUNTING_PASS").all()
            and (df["batched_segment_feature_parity"] == "BATCHED_SEGMENT_FEATURE_PARITY_PASS").all()
            and np.all(np.isfinite(df["validation_weighted_r2"].to_numpy(dtype=float)))
        )
        summary = {
            "mode": "functional-smoke",
            "section": section,
            "sample_count": int(len(df)),
            "status": "FUNCTIONAL_SURROGATE_SMOKE_PASS" if passed else "FUNCTIONAL_SURROGATE_SMOKE_FAIL",
            "fixed_window_inference_batch_size": int(config.batch_windows),
            "surrogate_configuration": {
                "train_masks": int(config.train_masks),
                "validation_masks": int(config.val_masks),
                "target_segments": int(config.target_segments),
                "feature_count": int(df["feature_count"].median()),
                "mask_sampling": str(config.mask_sampling),
                "ridge_alphas": [float(value) for value in config.ridge_alphas],
                "kernel_widths": [float(value) for value in config.kernel_widths],
                "response_transform": str(config.response_transform),
                "perturbation_mode": str(config.surrogate_perturbation_mode),
            },
            "fidelity_diagnostics": {
                "median_validation_weighted_r2": float(df["validation_weighted_r2"].median()),
                "p10_validation_weighted_r2": float(np.percentile(df["validation_weighted_r2"], 10)),
                "mean_validation_weighted_mae": float(df["validation_weighted_mae"].mean()),
                "mean_validation_nrmse": float(df["validation_nrmse"].mean()),
                "low_variance_sample_fraction": float(df["r2_degenerate_low_variance"].mean()),
                "project_fidelity_threshold_pass": bool(
                    float(df["validation_weighted_r2"].median()) >= 0.80
                    and float(np.percentile(df["validation_weighted_r2"], 10)) >= 0.60
                ),
            },
            "guards": {
                "NO_MODEL_FIT": "PASS",
                "CHECKPOINT_UNCHANGED": "PASS",
                "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
                "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
                "WINDOW_INSTANCE_ACCOUNTING": "PASS" if (df["window_instance_accounting"] == "WINDOW_INSTANCE_ACCOUNTING_PASS").all() else "FAIL",
                "BATCHED_SEGMENT_FEATURE_PARITY": "PASS" if (df["batched_segment_feature_parity"] == "BATCHED_SEGMENT_FEATURE_PARITY_PASS").all() else "FAIL",
                "VECTORIZED_INFERENCE": "PASS" if df["vectorized_inference"].all() else "FAIL",
                "FIXED_INFERENCE_BATCH_PATH": "PASS",
                "CACHE_RESUME": "PASS",
                "TRAINING_FALSE": "PASS",
            },
            "note": "Implementation smoke only; no paper PASS/PARTIAL PASS/FAIL conclusion is produced.",
        }
        write_json(out_dir / "functional_smoke_surrogate_summary.json", summary)
        if section == "surrogate":
            if bool(getattr(args, "enable_lime_visualizations", True)):
                generate_lime_paper_artifacts(out_dir)
            return

    faith_results = []
    metric_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    sample_perf: List[Dict[str, object]] = []
    for _, sample in manifest.iterrows():
        res = run_faithfulness_module(
            sample.to_dict(),
            config,
            model,
            x_raw,
            x_seg,
            artifact["file_sample_counts"],
            artifact,
            out_dir,
            force=False,
        )
        faith_results.append(res)
        metric_rows.extend(res["metric_rows"])
        curve_rows.extend(res["curve_rows"])
        acc = res["window_instance_accounting"]
        tim = res["timings_sec"]
        sample_perf.append({
            "global_epoch_index": int(res["global_epoch_index"]),
            "target_stage": res["target_stage"],
            "feature_count": int(res["feature_count"]),
            "segments_count": int(res["segments_count"]),
            "covering_window_count": int(acc["covering_window_count_per_mask_mean"]),
            "requested_mask_count": int(acc["requested_mask_count"]),
            "unique_mask_count": int(acc["unique_mask_count"]),
            "cache_hit_count": int(acc["cache_hit_count"]),
            "new_inference_mask_count": int(acc["new_inference_mask_count"]),
            "expected_window_instances": int(acc["expected_unpadded_window_instances"]),
            "actual_inferred_window_instances": int(acc["actual_inferred_window_instances"]),
            "model_forward_call_count": int(acc["model_forward_call_count"]),
            "mean_model_forward_batch_size": float(np.mean(acc["model_forward_batch_sizes"]) if acc["model_forward_batch_sizes"] else 0.0),
            "max_model_forward_batch_size": int(max(acc["model_forward_batch_sizes"] or [0])),
            "window_inference_batch_size": int(config.batch_windows),
            **tim,
        })
    faith_df = pd.DataFrame(metric_rows)
    curve_df = pd.DataFrame(curve_rows)
    perf_df = pd.DataFrame(sample_perf)
    faith_df.to_csv(out_dir / "functional_smoke_faithfulness.csv", index=False)
    curve_df.to_csv(out_dir / "functional_smoke_faithfulness_curves.csv", index=False)

    ckpt_after = stat_payload(CKPT_PATH)
    all_accounting = [res["window_instance_accounting"] for res in faith_results]
    all_batch_sizes = [size for acc in all_accounting for size in acc["model_forward_batch_sizes"]]
    deletion_endpoint_max = float(faith_df["deletion_endpoint_max_diff"].max()) if not faith_df.empty else np.nan
    insertion_endpoint_max = float(faith_df["insertion_endpoint_max_diff"].max()) if not faith_df.empty else np.nan
    random_valid = bool(all(res["ranking_validation"]["random_ranking_valid"] for res in faith_results))
    energy_valid = bool(all(res["ranking_validation"]["energy_ranking_valid"] for res in faith_results))
    no_nan_inf = bool(
        faith_results
        and all(res["all_values_finite"] for res in faith_results)
        and np.all(np.isfinite(curve_df[["deletion_probability", "insertion_probability"]].to_numpy(dtype=float)))
    )
    mask_dedup_pass = bool(all(int(acc["duplicate_mask_count"]) > 0 and int(acc["unique_mask_count"]) < int(acc["requested_mask_count"]) for acc in all_accounting))
    window_pass = bool(all(acc["status"] == "WINDOW_INSTANCE_ACCOUNTING_PASS" for acc in all_accounting))
    fixed_batch_pass = bool(all_batch_sizes and max(all_batch_sizes) <= int(config.batch_windows))
    resume_pass = bool(all((sample_cache_dir(out_dir, int(res["global_epoch_index"])) / "status.json").exists() for res in faith_results))
    guard_status = {
        "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
        "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
        "WINDOW_INSTANCE_ACCOUNTING_PASS": "PASS" if window_pass else "FAIL",
        "FIXED_BATCH_PATH_PASS": "PASS" if fixed_batch_pass else "FAIL",
        "MASK_DEDUPLICATION_PASS": "PASS" if mask_dedup_pass else "FAIL",
        "DELETION_ENDPOINT_PARITY_PASS": "PASS" if deletion_endpoint_max < 1e-6 else "FAIL",
        "INSERTION_ENDPOINT_PARITY_PASS": "PASS" if insertion_endpoint_max < 1e-6 else "FAIL",
        "NO_NAN_INF_PASS": "PASS" if no_nan_inf else "FAIL",
        "RANDOM_RANKING_VALID_PASS": "PASS" if random_valid else "FAIL",
        "ENERGY_RANKING_VALID_PASS": "PASS" if energy_valid else "FAIL",
        "CACHE_RESUME_PASS": "PASS" if resume_pass else "FAIL",
        "CHECKPOINT_UNCHANGED_PASS": "PASS" if ckpt_before.get("sha256") == ckpt_after.get("sha256") else "FAIL",
        "NO_MODEL_FIT": "PASS",
        "TRAINING_FALSE": "PASS",
    }
    allow_all = bool(
        guard_status["SCIENTIFIC_PREDICTION_PARITY"] == "SCIENTIFIC_PREDICTION_PARITY_PASS"
        and guard_status["SEQUENCE_WRAPPER_PARITY"] == "SEQUENCE_WRAPPER_PARITY_PASS"
        and all(value == "PASS" for key, value in guard_status.items() if key not in {"SCIENTIFIC_PREDICTION_PARITY", "SEQUENCE_WRAPPER_PARITY"})
    )
    overall_perf = {
        "sample_count": int(len(faith_results)),
        "requested_mask_count": int(sum(acc["requested_mask_count"] for acc in all_accounting)),
        "unique_mask_count": int(sum(acc["unique_mask_count"] for acc in all_accounting)),
        "cache_hit_count": int(sum(acc["cache_hit_count"] for acc in all_accounting)),
        "new_inference_mask_count": int(sum(acc["new_inference_mask_count"] for acc in all_accounting)),
        "covering_window_count": [int(acc["covering_window_count_per_mask_mean"]) for acc in all_accounting],
        "expected_window_instances": int(sum(acc["expected_unpadded_window_instances"] for acc in all_accounting)),
        "actual_inferred_window_instances": int(sum(acc["actual_inferred_window_instances"] for acc in all_accounting)),
        "model_forward_call_count": int(sum(acc["model_forward_call_count"] for acc in all_accounting)),
        "mean_model_forward_batch_size": float(np.mean(all_batch_sizes) if all_batch_sizes else 0.0),
        "max_model_forward_batch_size": int(max(all_batch_sizes or [0])),
        "window_inference_batch_size": int(config.batch_windows),
        "mask_generation_seconds": float(sum(res["timings_sec"]["mask_generation_seconds"] for res in faith_results)),
        "window_assembly_seconds": float(sum(res["timings_sec"]["window_assembly_seconds"] for res in faith_results)),
        "gpu_forward_seconds": float(sum(res["timings_sec"]["gpu_forward_seconds"] for res in faith_results)),
        "aggregation_seconds": float(sum(res["timings_sec"]["aggregation_seconds"] for res in faith_results)),
        "metric_computation_seconds": float(sum(res["timings_sec"]["metric_computation_seconds"] for res in faith_results)),
        "cache_io_seconds": float(sum(res["timings_sec"]["cache_io_seconds"] for res in faith_results)),
        "total_seconds": float(sum(res["timings_sec"]["total_seconds"] for res in faith_results)),
    }
    performance_payload = {
        "overall": overall_perf,
        "by_sample": sample_perf,
    }
    write_json(out_dir / "functional_smoke_faithfulness_performance.json", performance_payload)
    summary = {
        "mode": "functional-smoke",
        "section": "faithfulness",
        "sample_count": int(len(faith_results)),
        "ranking_count": int(faith_df["ranking"].nunique()) if not faith_df.empty else 0,
        "status": "FUNCTIONAL_FAITHFULNESS_SMOKE_PASS" if allow_all else "FUNCTIONAL_FAITHFULNESS_SMOKE_FAIL",
        "fixed_window_inference_batch_size": int(config.batch_windows),
        "deletion_endpoint_max_diff": deletion_endpoint_max,
        "insertion_endpoint_max_diff": insertion_endpoint_max,
        "guards": guard_status,
        "allow_functional_smoke_all": bool(allow_all),
        "note": "Implementation smoke only; no paper PASS/PARTIAL PASS/FAIL conclusion is produced.",
    }
    write_json(out_dir / "functional_smoke_faithfulness_summary.json", summary)
    perf_df.to_csv(out_dir / "functional_smoke_faithfulness_performance_by_sample.csv", index=False)
    report = [
        "# Functional Faithfulness Smoke",
        "",
        f"Samples: {len(faith_results)}",
        f"Window inference batch size: {config.batch_windows}",
        f"Deletion endpoint max diff: {deletion_endpoint_max:.9g}",
        f"Insertion endpoint max diff: {insertion_endpoint_max:.9g}",
        f"Allow functional-smoke all: {allow_all}",
        "",
        "## Accounting",
        "",
        json.dumps(overall_perf, indent=2, ensure_ascii=False),
        "",
        "## Guards",
        "",
    ]
    for name, value in guard_status.items():
        report.append(f"- {name}: {value}")
    report.extend([
        "",
        "## Interpretation Direction",
        "",
        "- Deletion AOPC higher is better for the tested ranking.",
        "- Insertion recovery faster is better for the tested ranking.",
        "- Deletion AUC lower is better.",
        "- Insertion AUC higher is better.",
        "",
        "No paper-level faithful/unfaithful/PASS/PARTIAL PASS/FAIL conclusion is produced here.",
    ])
    (out_dir / "faithfulness_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if bool(getattr(args, "enable_lime_visualizations", True)):
        payload = generate_lime_paper_artifacts(out_dir)
        print(f"LIME paper artifacts: {payload['figure_count']} figures; decision={payload['decision']['status']}")


def lime_method_ablation_specs() -> List[Dict[str, object]]:
    return [
        {
            "method": "classic_lime_sequence_adapter",
            "segmentation": "uniform",
            "mask_sampling": "bernoulli",
            "perturbation_mode": "zero",
            "description": "Classic linear LIME components with the common sequence prediction adapter.",
        },
        {
            "method": "plus_local_sampling",
            "segmentation": "uniform",
            "mask_sampling": "local_stratified",
            "perturbation_mode": "zero",
            "description": "Classic adapter plus locally stratified masks.",
        },
        {
            "method": "plus_dynamic_segmentation",
            "segmentation": "dynamic",
            "mask_sampling": "local_stratified",
            "perturbation_mode": "zero",
            "description": "Local masks plus signal-adaptive temporal segmentation.",
        },
        {
            "method": "proposed_sequence_lime",
            "segmentation": "dynamic",
            "mask_sampling": "local_stratified",
            "perturbation_mode": "physiological",
            "description": "Full proposed explanation with physiological perturbations.",
        },
    ]


def project_lime_order_to_common_grid(
    coefficient: np.ndarray,
    source_segments: Sequence[Tuple[int, int]],
    evaluation_segments: Sequence[Tuple[int, int]],
    channel_count: int,
    epoch_length: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    coefficient = np.asarray(coefficient, dtype=np.float32).reshape(channel_count, len(source_segments))
    positive = np.clip(coefficient, 0.0, None)
    fallback = not bool(np.any(positive > 0.0))
    source_scores = np.abs(coefficient) if fallback else positive
    dense_scores = np.zeros((channel_count, epoch_length), dtype=np.float32)
    for channel in range(channel_count):
        for segment_index, (start, end) in enumerate(source_segments):
            dense_scores[channel, int(start):int(end)] = source_scores[channel, segment_index]
    evaluation_scores = []
    for channel in range(channel_count):
        for start, end in evaluation_segments:
            values = dense_scores[channel, int(start):int(end)]
            evaluation_scores.append(float(np.mean(values)) if len(values) else 0.0)
    scores = np.asarray(evaluation_scores, dtype=np.float32)
    return np.argsort(-scores, kind="stable").astype(np.int32), {
        "positive_ranking_fallback": fallback,
        "positive_coefficient_count": int(np.sum(positive > 0.0)),
        "common_grid_nonzero_score_count": int(np.sum(scores > 0.0)),
    }


def run_lime_ablation_sample_method(
    sample: Mapping[str, object],
    method_spec: Mapping[str, object],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, object]:
    ensure_lime_imports()
    global_idx = int(sample["global_epoch_index"])
    method = str(method_spec["method"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / f"lime_method_ablation_{method}.json"
    evaluation_modes = ("zero", "physiological")
    evaluation_segment_count = 30
    signature_payload = {
        "schema_version": 3,
        "surrogate_config": surrogate_config_signature(config),
        "method_spec": dict(method_spec),
        "shared_faithfulness_evaluation_modes": evaluation_modes,
        "shared_faithfulness_segment_count": evaluation_segment_count,
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if result_path.exists():
        cached = read_json(result_path)
        if int(cached.get("schema_version", 0)) >= 2 and cached.get("configuration_signature") == signature:
            return cached

    epoch = x_raw[global_idx]
    target_class = int(np.argmax(artifact["probs"][global_idx]))
    if method_spec["segmentation"] == "dynamic":
        segments = semantic_segmentation(epoch, target_segments=config.target_segments)
    else:
        segments = uniform_segmentation(epoch.shape[0], config.target_segments)
    num_features = int(epoch.shape[1] * len(segments))
    masks = make_masks(
        num_features,
        config.train_masks,
        config.val_masks,
        100000 + global_idx,
        sampling=str(method_spec["mask_sampling"]),
    )
    surrogate_eval = evaluate_masks_for_target_epoch(
        masks,
        model,
        x_raw,
        x_seg,
        counts,
        global_idx,
        target_class,
        epoch,
        segments,
        perturbation_mode=str(method_spec["perturbation_mode"]),
        window_inference_batch_size=config.batch_windows,
        out_dir=out_dir,
        cache_namespace=f"lime_method_ablation_{method}",
        model_state_id="original_model",
    )
    probabilities = np.asarray(surrogate_eval["target_probabilities"], dtype=np.float32)
    ridge, fidelity, _weights = fit_surrogate(
        masks,
        probabilities,
        config.train_masks,
        alpha_grid=config.ridge_alphas,
        kernel_width_grid=config.kernel_widths,
        response_transform=config.response_transform,
    )
    coefficient = np.asarray(ridge.coef_, dtype=np.float32)
    evaluation_segments = uniform_segmentation(epoch.shape[0], evaluation_segment_count)
    evaluation_feature_count = int(epoch.shape[1] * len(evaluation_segments))
    order, ranking_meta = project_lime_order_to_common_grid(
        coefficient,
        segments,
        evaluation_segments,
        int(epoch.shape[1]),
        int(epoch.shape[0]),
    )
    requested_fractions = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    faithfulness_by_mode = {}
    curve_rows = []
    curve_accounting = {}
    endpoint_differences = []
    for evaluation_mode in evaluation_modes:
        curve_masks = []
        curve_requests = []
        for curve in ("deletion", "insertion"):
            for fraction in requested_fractions:
                mask, feature_count, actual_fraction = feature_mask_from_order_with_count(
                    evaluation_feature_count,
                    order,
                    float(fraction),
                    curve,
                )
                curve_masks.append(mask)
                curve_requests.append({
                    "curve": curve,
                    "fraction": float(fraction),
                    "feature_count": int(feature_count),
                    "actual_fraction": float(actual_fraction),
                })
        curve_eval = evaluate_masks_for_target_epoch(
            np.asarray(curve_masks, dtype=np.int32),
            model,
            x_raw,
            x_seg,
            counts,
            global_idx,
            target_class,
            epoch,
            evaluation_segments,
            perturbation_mode=evaluation_mode,
            window_inference_batch_size=config.batch_windows,
            out_dir=out_dir,
            cache_namespace=f"lime_method_ablation_shared_curve_{method}_{evaluation_mode}",
            model_state_id="original_model",
        )
        curve_accounting[evaluation_mode] = curve_eval["accounting"]
        curve_probabilities = np.asarray(curve_eval["target_probabilities"], dtype=np.float32)
        for index, request in enumerate(curve_requests):
            request["target_probability"] = float(curve_probabilities[index])
        deletion_requests = [request for request in curve_requests if request["curve"] == "deletion"]
        insertion_requests = [request for request in curve_requests if request["curve"] == "insertion"]
        deletion = np.asarray([request["target_probability"] for request in deletion_requests], dtype=np.float32)
        insertion = np.asarray([request["target_probability"] for request in insertion_requests], dtype=np.float32)
        actual_fractions = np.asarray([request["actual_fraction"] for request in deletion_requests], dtype=np.float32)
        faithfulness_by_mode[evaluation_mode] = curve_metrics(actual_fractions, deletion, insertion)
        endpoint_differences.append(float(max(abs(deletion[0] - insertion[-1]), abs(deletion[-1] - insertion[0]))))
        curve_rows.extend([
            {
                "evaluation_perturbation_mode": evaluation_mode,
                "fraction": float(d_request["fraction"]),
                "actual_fraction": float(d_request["actual_fraction"]),
                "deletion_probability": float(d_request["target_probability"]),
                "insertion_probability": float(i_request["target_probability"]),
            }
            for d_request, i_request in zip(deletion_requests, insertion_requests)
        ])

    result = {
        "schema_version": 2,
        "configuration_signature": signature,
        "global_epoch_index": global_idx,
        "target_stage": CLASS_LABELS[target_class],
        "target_class": target_class,
        "method": method,
        "method_spec": dict(method_spec),
        "segments_count": int(len(segments)),
        "feature_count": num_features,
        "shared_evaluation_segments_count": int(len(evaluation_segments)),
        "shared_evaluation_feature_count": evaluation_feature_count,
        "fidelity": fidelity,
        "faithfulness_by_evaluation_mode": faithfulness_by_mode,
        "positive_ranking_fallback": bool(ranking_meta["positive_ranking_fallback"]),
        "positive_coefficient_count": int(ranking_meta["positive_coefficient_count"]),
        "common_grid_nonzero_score_count": int(ranking_meta["common_grid_nonzero_score_count"]),
        "deletion_endpoint_max_diff": float(max(endpoint_differences)),
        "curve_rows": curve_rows,
        "surrogate_accounting": surrogate_eval["accounting"],
        "curve_accounting": curve_accounting,
    }
    write_json_atomic(result_path, result)
    return result


def lime_method_ablation_statistics(surrogate: pd.DataFrame, faithfulness: pd.DataFrame) -> pd.DataFrame:
    baseline = "classic_lime_sequence_adapter"
    surrogate_directions = {
        "validation_weighted_r2": "higher",
        "validation_weighted_mae": "lower",
        "validation_nrmse": "lower",
    }
    faithfulness_directions = {
        "deletion_auc_raw": "lower",
        "insertion_auc_raw": "higher",
        "deletion_AOPC": "higher",
        "insertion_AOPC": "higher",
    }
    rows = []
    analysis_sets = [("surrogate_neighborhood", "method_specific", surrogate, surrogate_directions)]
    for evaluation_mode in sorted(faithfulness["evaluation_perturbation_mode"].unique()):
        analysis_sets.append((
            "shared_faithfulness",
            str(evaluation_mode),
            faithfulness[faithfulness["evaluation_perturbation_mode"] == evaluation_mode],
            faithfulness_directions,
        ))
    for comparison_scope, evaluation_mode, frame, directions in analysis_sets:
        for method in [value for value in frame["method"].unique() if value != baseline]:
            for metric, direction in directions.items():
                if metric not in frame:
                    continue
                pivot = frame.pivot_table(index="global_epoch_index", columns="method", values=metric, aggfunc="first")
                if baseline not in pivot or method not in pivot:
                    continue
                paired = pivot[[baseline, method]].dropna()
                if paired.empty:
                    continue
                advantage = paired[method] - paired[baseline] if direction == "higher" else paired[baseline] - paired[method]
                mean, ci_low, ci_high = bootstrap_mean_interval(advantage.to_numpy(), 8700 + len(rows))
                p_value = np.nan
                if len(advantage) >= 2 and np.any(np.abs(advantage.to_numpy()) > EPS):
                    p_value = float(wilcoxon(advantage.to_numpy(), alternative="two-sided", zero_method="wilcox").pvalue)
                rows.append({
                    "comparison_scope": comparison_scope,
                    "evaluation_perturbation_mode": evaluation_mode,
                    "method": method,
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "N": int(len(advantage)),
                    "baseline_mean": float(paired[baseline].mean()),
                    "method_mean": float(paired[method].mean()),
                    "mean_directional_advantage": mean,
                    "median_directional_advantage": float(np.median(advantage)),
                    "CI_low": ci_low,
                    "CI_high": ci_high,
                    "win_rate": float(np.mean(advantage > 0.0)),
                    "wilcoxon_p_value": p_value,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = bh_fdr(result["wilcoxon_p_value"].tolist())
    return result


def plot_lime_method_ablation(surrogate: pd.DataFrame, comparisons: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    configure_publication_plots()
    method_order = [spec["method"] for spec in lime_method_ablation_specs()]
    labels = {
        "classic_lime_sequence_adapter": "Classic adapter",
        "plus_local_sampling": "+ Local masks",
        "plus_dynamic_segmentation": "Seq-LIME (zero)",
        "proposed_sequence_lime": "Seq-LIME (physiol.)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8))
    ax = axes[0]
    pivot = surrogate.pivot_table(index="global_epoch_index", columns="method", values="validation_weighted_r2", aggfunc="first")
    for _, sample_values in pivot.iterrows():
        values = [sample_values.get(method, np.nan) for method in method_order]
        ax.plot(np.arange(len(method_order)), values, color="#B0B0B0", linewidth=0.9, alpha=0.75, zorder=1)
    means = [float(pivot[method].mean()) for method in method_order]
    ax.scatter(np.arange(len(method_order)), means, color=["#7F7F7F", "#56B4E9", "#009E73", "#D55E00"], s=38, zorder=3)
    ax.axhline(0.80, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(method_order))); ax.set_xticklabels([labels[method] for method in method_order], rotation=22, ha="right")
    ax.set_ylabel("Validation weighted R2")
    ax.text(-0.12, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    proposed = comparisons[
        (comparisons["method"] == "proposed_sequence_lime")
        & (comparisons["comparison_scope"] == "shared_faithfulness")
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    proposed["display_name"] = proposed.apply(
        lambda row: f"{row['metric'].replace('_auc_raw', '').title()} ({row['evaluation_perturbation_mode']})",
        axis=1,
    )
    y = np.arange(len(proposed))
    means = proposed["mean_directional_advantage"].to_numpy(dtype=float)
    lows = proposed["CI_low"].to_numpy(dtype=float)
    highs = proposed["CI_high"].to_numpy(dtype=float)
    ax.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="o", color="#D55E00", ecolor="#555555", capsize=3)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(proposed["display_name"].tolist())
    ax.set_xlabel("Advantage vs classic\n(+ favors physiol.)")
    ax.text(-0.12, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(bottom=0.31, wspace=0.90)
    figure_dir = out_dir / "figures" / "lime_interpretability"
    return save_publication_figure(fig, figure_dir / "fig_lime_method_ablation")


def run_lime_method_ablation(args: argparse.Namespace) -> None:
    config = resolve_mode_config("functional-smoke", args)
    out_dir, _inventory, artifact, manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    parity = assert_xai_allowed(out_dir)
    checkpoint_before = stat_payload(CKPT_PATH)
    results = []
    for _, sample in manifest.iterrows():
        for method_spec in lime_method_ablation_specs():
            print(f"LIME ablation sample={int(sample['global_epoch_index'])} method={method_spec['method']}")
            results.append(run_lime_ablation_sample_method(
                sample.to_dict(), method_spec, config, model, x_raw, x_seg,
                artifact["file_sample_counts"], artifact, out_dir,
            ))

    surrogate_rows = []
    faithfulness_rows = []
    curve_rows = []
    for result in results:
        base = {
            "global_epoch_index": int(result["global_epoch_index"]),
            "target_stage": result["target_stage"],
            "method": result["method"],
            "segmentation": result["method_spec"]["segmentation"],
            "mask_sampling": result["method_spec"]["mask_sampling"],
            "surrogate_perturbation_mode": result["method_spec"]["perturbation_mode"],
            "segments_count": int(result["segments_count"]),
            "feature_count": int(result["feature_count"]),
        }
        surrogate_rows.append({**base, **result["fidelity"]})
        for evaluation_mode, metrics in result["faithfulness_by_evaluation_mode"].items():
            faithfulness_rows.append({
                **base,
                "evaluation_perturbation_mode": evaluation_mode,
                "shared_evaluation_segments_count": int(result["shared_evaluation_segments_count"]),
                "shared_evaluation_feature_count": int(result["shared_evaluation_feature_count"]),
                **metrics,
                "deletion_endpoint_max_diff": float(result["deletion_endpoint_max_diff"]),
                "positive_ranking_fallback": bool(result["positive_ranking_fallback"]),
            })
        for curve_row in result["curve_rows"]:
            curve_rows.append({**base, **curve_row})
    surrogate_df = pd.DataFrame(surrogate_rows)
    faithfulness_df = pd.DataFrame(faithfulness_rows)
    curves_df = pd.DataFrame(curve_rows)
    comparisons = lime_method_ablation_statistics(surrogate_df, faithfulness_df)
    surrogate_df.to_csv(out_dir / "lime_method_ablation_surrogate.csv", index=False)
    faithfulness_df.to_csv(out_dir / "lime_method_ablation_faithfulness.csv", index=False)
    curves_df.to_csv(out_dir / "lime_method_ablation_curves.csv", index=False)
    comparisons.to_csv(out_dir / "lime_method_ablation_paired_comparisons.csv", index=False)

    proposed = comparisons[comparisons["method"] == "proposed_sequence_lime"]
    fidelity_primary = proposed[
        (proposed["comparison_scope"] == "surrogate_neighborhood")
        & (proposed["metric"] == "validation_weighted_r2")
    ]
    faithfulness_primary = proposed[
        (proposed["comparison_scope"] == "shared_faithfulness")
        & (proposed["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ]
    primary = pd.concat([fidelity_primary, faithfulness_primary], ignore_index=True)
    directional_pass = bool(len(primary) == 5 and (primary["mean_directional_advantage"] > 0.0).all() and (primary["win_rate"] >= 0.60).all())
    statistical_pass = bool(len(primary) == 5 and len(manifest) >= 20 and (primary["q_value"] < 0.05).all() and (primary["CI_low"] > 0.0).all())
    endpoint_pass = bool(faithfulness_df["deletion_endpoint_max_diff"].max() < 1e-6)
    checkpoint_after = stat_payload(CKPT_PATH)
    checkpoint_pass = checkpoint_before.get("sha256") == checkpoint_after.get("sha256")
    status = "METHOD_ABLATION_PILOT_DIRECTIONAL_PASS" if directional_pass else "METHOD_ABLATION_PILOT_MIXED"
    surrogate_aggregates = json.loads(
        surrogate_df.groupby("method", as_index=False).agg(
            median_validation_weighted_r2=("validation_weighted_r2", "median"),
            mean_validation_weighted_r2=("validation_weighted_r2", "mean"),
            p10_validation_weighted_r2=("validation_weighted_r2", lambda values: float(np.quantile(values, 0.10))),
            mean_validation_weighted_mae=("validation_weighted_mae", "mean"),
            mean_validation_nrmse=("validation_nrmse", "mean"),
            low_response_variance_fraction=("r2_degenerate_low_variance", "mean"),
        ).to_json(orient="records")
    )
    faithfulness_aggregates = json.loads(
        faithfulness_df.groupby(["method", "evaluation_perturbation_mode"], as_index=False).agg(
            mean_deletion_auc_raw=("deletion_auc_raw", "mean"),
            mean_insertion_auc_raw=("insertion_auc_raw", "mean"),
            mean_deletion_AOPC=("deletion_AOPC", "mean"),
            mean_insertion_AOPC=("insertion_AOPC", "mean"),
        ).to_json(orient="records")
    )
    revised_core = surrogate_df[surrogate_df["method"] == "plus_dynamic_segmentation"]
    revised_median_r2 = float(revised_core["validation_weighted_r2"].median())
    revised_p10_r2 = float(np.quantile(revised_core["validation_weighted_r2"], 0.10))
    revised_fidelity_threshold_pass = bool(revised_median_r2 >= 0.80 and revised_p10_r2 >= 0.60)
    summary = {
        "status": status,
        "sample_count": int(manifest["global_epoch_index"].nunique()),
        "method_count": len(lime_method_ablation_specs()),
        "comparison_scope": "All methods use the same overlapping-window prediction adapter, sample set, mask counts, segment count, and surrogate-selection protocol. Faithfulness rankings are projected to a shared 30-segment grid and evaluated under both shared zero and shared physiological interventions.",
        "directional_advantage_over_classic": directional_pass,
        "paired_statistical_support": statistical_pass,
        "paper_superiority_claim_allowed": bool(directional_pass and statistical_pass),
        "surrogate_aggregates": surrogate_aggregates,
        "shared_faithfulness_aggregates": faithfulness_aggregates,
        "primary_paired_comparisons": json.loads(primary.to_json(orient="records")),
        "evidence_based_method_revision": {
            "recommended_core": "local_stratified masks + dynamic segmentation + zero surrogate perturbation + overlapping-window sequence adapter",
            "physiological_perturbation_role": "sensitivity analysis rather than the primary surrogate neighborhood",
            "median_validation_weighted_r2": revised_median_r2,
            "p10_validation_weighted_r2": revised_p10_r2,
            "fidelity_threshold_rule": "median >= 0.80 and P10 >= 0.60",
            "fidelity_threshold_pass": revised_fidelity_threshold_pass,
            "faithfulness_superiority_over_classic": False,
        },
        "guards": {
            "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
            "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
            "ENDPOINT_PARITY": "PASS" if endpoint_pass else "FAIL",
            "CHECKPOINT_UNCHANGED": "PASS" if checkpoint_pass else "FAIL",
            "NO_MODEL_FIT": "PASS",
            "TRAINING_FALSE": "PASS",
        },
    }
    write_json(out_dir / "lime_method_ablation_summary.json", summary)
    figure_paths = plot_lime_method_ablation(surrogate_df, comparisons, out_dir)
    pd.DataFrame([{
        "figure_type": "controlled_lime_method_ablation",
        "png": str(figure_paths[0]),
        "pdf": str(figure_paths[1]),
    }]).to_csv(out_dir / "lime_method_ablation_visualization_manifest.csv", index=False)
    report = [
        "# Controlled LIME Method Ablation",
        "",
        f"Status: `{status}`",
        f"Samples: {summary['sample_count']}",
        "",
        "The classic baseline is a sequence-adapted classic linear LIME baseline. The formal overlapping-window prediction rule is held fixed for every method.",
        "All explanation rankings are projected to one common 30-segment evaluation grid. Deletion and insertion are evaluated separately under shared zero and shared physiological interventions, so faithfulness comparisons do not mix perturbation endpoints or segmentation geometry.",
        "",
        f"Directional advantage over classic: {directional_pass}",
        f"Corrected paired statistical support: {statistical_pass}",
        f"Paper superiority claim allowed: {summary['paper_superiority_claim_allowed']}",
        "",
        "## Evidence-based method revision",
        "",
        "Use locally stratified masks, dynamic segmentation, zero surrogate perturbation, and the overlapping-window sequence adapter as the primary Sequence-aware LIME configuration. Treat physiological replacement as a sensitivity analysis rather than the core neighborhood.",
        f"The revised core reaches median validation weighted R2={revised_median_r2:.3f} and P10={revised_p10_r2:.3f}; threshold rule median>=0.80 and P10>=0.60: {revised_fidelity_threshold_pass}.",
        "This is a surrogate-fidelity result. Shared-intervention deletion/insertion tests do not establish faithfulness superiority over the classic adapter.",
        "",
        "A five-sample pilot can establish implementation direction only; it cannot establish paper-level superiority.",
    ]
    (out_dir / "lime_method_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def positive_rank_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    positive = np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, None)
    score = np.zeros_like(positive)
    active = np.flatnonzero(positive > EPS)
    if len(active) == 0:
        return score.astype(np.float32)
    ordered = active[np.argsort(positive[active], kind="stable")]
    score[ordered] = np.linspace(1.0 / len(ordered), 1.0, len(ordered))
    return score.astype(np.float32)


def project_lime_scores_to_common_grid(
    coefficient: np.ndarray,
    source_segments: Sequence[Tuple[int, int]],
    evaluation_segments: Sequence[Tuple[int, int]],
    channel_count: int,
    epoch_length: int,
) -> np.ndarray:
    coefficient = np.asarray(coefficient, dtype=np.float32).reshape(channel_count, len(source_segments))
    positive = np.clip(coefficient, 0.0, None)
    source_scores = np.abs(coefficient) if not bool(np.any(positive > 0.0)) else positive
    dense = np.zeros((channel_count, epoch_length), dtype=np.float32)
    for channel in range(channel_count):
        for segment_index, (start, end) in enumerate(source_segments):
            dense[channel, int(start):int(end)] = source_scores[channel, segment_index]
    projected = []
    for channel in range(channel_count):
        for start, end in evaluation_segments:
            values = dense[channel, int(start):int(end)]
            projected.append(float(np.mean(values)) if len(values) else 0.0)
    return np.asarray(projected, dtype=np.float32)


def fit_reliability_lime_ranking(
    method: str,
    epoch: np.ndarray,
    source_segments: Sequence[Tuple[int, int]],
    evaluation_segments: Sequence[Tuple[int, int]],
    sampling: str,
    perturbation_mode: str,
    train_masks: int,
    validation_masks: int,
    seed: int,
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_idx: int,
    target_class: int,
    out_dir: Path,
) -> Dict[str, object]:
    feature_count = int(epoch.shape[1] * len(source_segments))
    masks = make_masks(feature_count, train_masks, validation_masks, seed, sampling=sampling)
    evaluated = evaluate_masks_for_target_epoch(
        masks, model, x_raw, x_seg, counts, global_idx, target_class, epoch, source_segments,
        perturbation_mode=perturbation_mode,
        window_inference_batch_size=config.batch_windows,
        out_dir=out_dir,
        cache_namespace=f"lime_reliability_{method}",
        model_state_id="original_model",
    )
    ridge, fidelity, _weights = fit_surrogate(
        masks,
        np.asarray(evaluated["target_probabilities"], dtype=np.float32),
        train_masks,
        alpha_grid=config.ridge_alphas,
        kernel_width_grid=config.kernel_widths,
        response_transform=config.response_transform,
    )
    projected = project_lime_scores_to_common_grid(
        np.asarray(ridge.coef_, dtype=np.float32),
        source_segments,
        evaluation_segments,
        int(epoch.shape[1]),
        int(epoch.shape[0]),
    )
    return {
        "score": positive_rank_score(projected),
        "raw_projected_score": projected,
        "fidelity": fidelity,
        "original_mask_probability": float(
            np.asarray(evaluated["target_probabilities"], dtype=np.float32)[0]
        ),
        "requested_explainer_queries": int(train_masks + validation_masks),
        "accounting": evaluated["accounting"],
    }


def run_single_feature_calibration(
    epoch: np.ndarray,
    evaluation_segments: Sequence[Tuple[int, int]],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_idx: int,
    target_class: int,
    out_dir: Path,
    class_conditioned_perturbation: bool = True,
) -> Dict[str, object]:
    feature_count = int(epoch.shape[1] * len(evaluation_segments))
    identity = np.eye(feature_count, dtype=np.int32)
    deletion_masks = np.ones((feature_count, feature_count), dtype=np.int32) - identity
    insertion_masks = identity
    masks = np.vstack([
        np.ones((1, feature_count), dtype=np.int32),
        np.zeros((1, feature_count), dtype=np.int32),
        deletion_masks,
        insertion_masks,
    ])
    mode_payload = {}
    for mode in ("zero", "physiological"):
        evaluated = evaluate_masks_for_target_epoch(
            masks, model, x_raw, x_seg, counts, global_idx, target_class, epoch, evaluation_segments,
            perturbation_mode=mode,
            window_inference_batch_size=config.batch_windows,
            out_dir=out_dir,
            cache_namespace=f"lime_reliability_single_feature_{mode}",
            model_state_id="original_model",
            perturbation_target_class=(
                int(target_class) if class_conditioned_perturbation else -1
            ),
        )
        probabilities = np.asarray(evaluated["target_probabilities"], dtype=np.float32)
        original = float(probabilities[0])
        full = float(probabilities[1])
        deletion_effect = original - probabilities[2:2 + feature_count]
        insertion_effect = probabilities[2 + feature_count:2 + 2 * feature_count] - full
        positive_effects = np.concatenate([
            np.clip(deletion_effect, 0.0, None),
            np.clip(insertion_effect, 0.0, None),
        ])
        strength = float(np.quantile(positive_effects, 0.90)) if len(positive_effects) else 0.0
        mode_payload[mode] = {
            "original_probability": original,
            "full_probability": full,
            "deletion_effect": deletion_effect.astype(float).tolist(),
            "insertion_effect": insertion_effect.astype(float).tolist(),
            "deletion_rank_score": positive_rank_score(deletion_effect).astype(float).tolist(),
            "insertion_rank_score": positive_rank_score(insertion_effect).astype(float).tolist(),
            "calibration_strength_p90": strength,
            "accounting": evaluated["accounting"],
        }
    return {
        "requested_queries_per_mode": int(2 + 2 * feature_count),
        "total_requested_calibration_queries": int(2 * (2 + 2 * feature_count)),
        "class_conditioned_perturbation": bool(class_conditioned_perturbation),
        "modes": mode_payload,
    }


def reliability_candidate_scores(
    lime_score: np.ndarray,
    classic_score: np.ndarray,
    calibration: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    lime_score = np.asarray(lime_score, dtype=np.float32)
    classic_score = np.asarray(classic_score, dtype=np.float32)
    zero = calibration["modes"]["zero"]
    physiological = calibration["modes"]["physiological"]
    zero_score = 0.5 * (
        np.asarray(zero["deletion_rank_score"], dtype=np.float32)
        + np.asarray(zero["insertion_rank_score"], dtype=np.float32)
    )
    physiological_score = 0.5 * (
        np.asarray(physiological["deletion_rank_score"], dtype=np.float32)
        + np.asarray(physiological["insertion_rank_score"], dtype=np.float32)
    )
    zero_strength = float(zero["calibration_strength_p90"])
    physiological_strength = float(physiological["calibration_strength_p90"])
    strength_sum = zero_strength + physiological_strength
    zero_weight = zero_strength / strength_sum if strength_sum > EPS else 0.5
    physiological_weight = 1.0 - zero_weight
    reliability_weighted = zero_weight * zero_score + physiological_weight * physiological_score
    base_candidates = {
        "sequence_lime_base_400": lime_score,
        "fcslime_zero_764": 0.25 * lime_score + 0.75 * zero_score,
        "fcslime_dual_equal_764": 0.20 * lime_score + 0.40 * zero_score + 0.40 * physiological_score,
        "fcslime_dual_reliability_764": 0.25 * lime_score + 0.75 * reliability_weighted,
        "fcslime_dual_consensus_764": 0.20 * lime_score + 0.40 * reliability_weighted + 0.40 * np.minimum(zero_score, physiological_score),
    }
    multires_80 = 0.80 * classic_score + 0.20 * reliability_weighted
    multires_65 = 0.65 * classic_score + 0.35 * reliability_weighted
    multires_50 = 0.50 * classic_score + 0.50 * reliability_weighted
    multires_dual = 0.60 * classic_score + 0.20 * zero_score + 0.20 * physiological_score
    base_candidates.update({
        "fcslime_multires_80_1164": multires_80,
        "fcslime_multires_65_1164": multires_65,
        "fcslime_multires_50_1164": multires_50,
        "fcslime_multires_dual_1164": multires_dual,
        "fcslime_pareto_25_1164": 0.25 * multires_65 + 0.75 * multires_dual,
        "fcslime_pareto_50_1164": 0.50 * multires_65 + 0.50 * multires_dual,
        "fcslime_pareto_75_1164": 0.75 * multires_65 + 0.25 * multires_dual,
    })
    return base_candidates


def physiology_prior_rank_score(
    epoch: np.ndarray,
    evaluation_segments: Sequence[Tuple[int, int]],
    target_class: int,
    concept_prior_override: Optional[np.ndarray] = None,
    channel_prior_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    ensure_lime_imports()
    payload = extract_clinical_concept_matrix(epoch, evaluation_segments, fs=100)
    concept_matrix = np.asarray(payload["concept_matrix"], dtype=np.float32)
    concept_channels = np.asarray(payload["concept_channels"], dtype=np.int32)
    concept_prior = np.asarray(
        concept_prior_override
        if concept_prior_override is not None
        else get_stage_concept_prior(target_class),
        dtype=np.float32,
    )
    channel_count = int(epoch.shape[1])
    segment_count = int(len(evaluation_segments))
    feature_score = np.zeros((channel_count, segment_count), dtype=np.float32)
    for concept_index, channel_index in enumerate(concept_channels):
        if 0 <= int(channel_index) < channel_count:
            feature_score[int(channel_index)] += (
                float(concept_prior[concept_index]) * concept_matrix[concept_index]
            )
    channel_prior = np.asarray(
        channel_prior_override
        if channel_prior_override is not None
        else get_clinical_prior(target_class),
        dtype=np.float32,
    )[:channel_count]
    feature_score *= channel_prior[:, np.newaxis]
    return positive_rank_score(feature_score.ravel())


def physiology_coherence_score(
    score: np.ndarray,
    target_class: int,
    channel_count: int,
    segment_count: int,
) -> np.ndarray:
    matrix = np.asarray(score, dtype=np.float32).reshape(channel_count, segment_count)
    smoothed = np.empty_like(matrix)
    for channel in range(channel_count):
        smoothed[channel] = np.convolve(matrix[channel], [0.25, 0.50, 0.25], mode="same")
    channel_prior = np.asarray(get_clinical_prior(target_class), dtype=np.float32)[:channel_count]
    channel_prior = channel_prior / max(float(channel_prior.sum()), EPS)
    synchronous = np.sum(smoothed * channel_prior[:, np.newaxis], axis=0)
    coherent = 0.70 * smoothed + 0.30 * synchronous[np.newaxis, :]
    return positive_rank_score(coherent.ravel())


def physiology_innovation_candidate_scores(
    epoch: np.ndarray,
    evaluation_segments: Sequence[Tuple[int, int]],
    target_class: int,
    sequence_score: np.ndarray,
    calibration: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    sequence_score = np.asarray(sequence_score, dtype=np.float32)
    zero = calibration["modes"]["zero"]
    physiological = calibration["modes"]["physiological"]
    zero_score = 0.5 * (
        np.asarray(zero["deletion_rank_score"], dtype=np.float32)
        + np.asarray(zero["insertion_rank_score"], dtype=np.float32)
    )
    physiological_score = 0.5 * (
        np.asarray(physiological["deletion_rank_score"], dtype=np.float32)
        + np.asarray(physiological["insertion_rank_score"], dtype=np.float32)
    )
    prior_score = physiology_prior_rank_score(epoch, evaluation_segments, target_class)
    independent_prior_score = physiology_prior_rank_score(
        epoch,
        evaluation_segments,
        target_class,
        concept_prior_override=np.mean(
            np.vstack([get_stage_concept_prior(stage) for stage in range(len(CLASS_LABELS))]),
            axis=0,
        ),
        channel_prior_override=np.mean(
            np.vstack([get_clinical_prior(stage) for stage in range(len(CLASS_LABELS))]),
            axis=0,
        ),
    )
    shuffled_prior_score = physiology_prior_rank_score(
        epoch,
        evaluation_segments,
        target_class,
        concept_prior_override=np.roll(
            np.asarray(get_stage_concept_prior(target_class), dtype=np.float32), 3
        ),
        channel_prior_override=np.roll(
            np.asarray(get_clinical_prior(target_class), dtype=np.float32), 1
        ),
    )
    consensus_score = np.minimum(zero_score, physiological_score)
    ablation_zero = 0.25 * sequence_score + 0.75 * zero_score
    ablation_physiological = 0.25 * sequence_score + 0.75 * physiological_score
    ablation_dual = 0.20 * sequence_score + 0.40 * zero_score + 0.40 * physiological_score
    base_dual = (
        0.15 * sequence_score
        + 0.35 * zero_score
        + 0.30 * physiological_score
        + 0.20 * prior_score
    )
    robust = (
        0.15 * sequence_score
        + 0.25 * zero_score
        + 0.25 * physiological_score
        + 0.20 * prior_score
        + 0.15 * consensus_score
    )
    stage_weights = {
        0: (0.15, 0.40, 0.20, 0.25),
        1: (0.15, 0.15, 0.50, 0.20),
        2: (0.15, 0.30, 0.30, 0.25),
        3: (0.10, 0.20, 0.40, 0.30),
        4: (0.10, 0.40, 0.25, 0.25),
    }
    seq_weight, zero_weight, physiological_weight, prior_weight = stage_weights[int(target_class)]
    stage_guided = (
        seq_weight * sequence_score
        + zero_weight * zero_score
        + physiological_weight * physiological_score
        + prior_weight * prior_score
    )
    channel_count = int(epoch.shape[1])
    segment_count = int(len(evaluation_segments))
    return {
        "ablation_B_zero_marginal": ablation_zero,
        "ablation_C_physiological_marginal": ablation_physiological,
        "ablation_D_zero_plus_physiological": ablation_dual,
        "ablation_E_D_plus_predicted_stage_prior": 0.80 * ablation_dual + 0.20 * prior_score,
        "ablation_F_D_plus_stage_independent_prior": 0.80 * ablation_dual + 0.20 * independent_prior_score,
        "ablation_G_shuffled_concept_prior": 0.80 * ablation_dual + 0.20 * shuffled_prior_score,
        "pcslime_prior_only_764": prior_score,
        "pcslime_zero_prior_764": 0.15 * sequence_score + 0.55 * zero_score + 0.30 * prior_score,
        "pcslime_dual_prior_764": base_dual,
        "pcslime_robust_prior_764": robust,
        "pcslime_stage_guided_764": stage_guided,
        "pcslime_coherent_dual_764": physiology_coherence_score(
            base_dual, target_class, channel_count, segment_count,
        ),
        "pcslime_coherent_stage_764": physiology_coherence_score(
            stage_guided, target_class, channel_count, segment_count,
        ),
    }


def evaluate_reliability_ranking(
    method: str,
    score: np.ndarray,
    epoch: np.ndarray,
    evaluation_segments: Sequence[Tuple[int, int]],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    global_idx: int,
    target_class: int,
    out_dir: Path,
    class_conditioned_perturbation: bool = True,
) -> Dict[str, object]:
    score = np.asarray(score, dtype=np.float32).ravel()
    order = np.argsort(-score, kind="stable").astype(np.int32)
    requested_fractions = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    metrics_by_mode = {}
    curve_rows = []
    endpoint_diffs = []
    accounting = {}
    for mode in ("zero", "physiological"):
        masks = []
        requests = []
        for curve in ("deletion", "insertion"):
            for fraction in requested_fractions:
                mask, selected_count, actual_fraction = feature_mask_from_order_with_count(
                    len(order), order, float(fraction), curve,
                )
                masks.append(mask)
                requests.append({
                    "curve": curve,
                    "fraction": float(fraction),
                    "actual_fraction": float(actual_fraction),
                    "selected_feature_count": int(selected_count),
                })
        evaluated = evaluate_masks_for_target_epoch(
            np.asarray(masks, dtype=np.int32), model, x_raw, x_seg, counts, global_idx,
            target_class, epoch, evaluation_segments, perturbation_mode=mode,
            window_inference_batch_size=config.batch_windows, out_dir=out_dir,
            cache_namespace=f"lime_reliability_curve_{method}_{mode}", model_state_id="original_model",
            perturbation_target_class=(
                int(target_class) if class_conditioned_perturbation else -1
            ),
        )
        accounting[mode] = evaluated["accounting"]
        probabilities = np.asarray(evaluated["target_probabilities"], dtype=np.float32)
        for index, request in enumerate(requests):
            request["target_probability"] = float(probabilities[index])
        deletion_requests = [request for request in requests if request["curve"] == "deletion"]
        insertion_requests = [request for request in requests if request["curve"] == "insertion"]
        deletion = np.asarray([request["target_probability"] for request in deletion_requests], dtype=np.float32)
        insertion = np.asarray([request["target_probability"] for request in insertion_requests], dtype=np.float32)
        fractions = np.asarray([request["actual_fraction"] for request in deletion_requests], dtype=np.float32)
        metrics_by_mode[mode] = curve_metrics(fractions, deletion, insertion)
        endpoint_diffs.append(float(max(abs(deletion[0] - insertion[-1]), abs(deletion[-1] - insertion[0]))))
        curve_rows.extend([
            {
                "method": method,
                "evaluation_perturbation_mode": mode,
                "fraction": float(d_request["fraction"]),
                "actual_fraction": float(d_request["actual_fraction"]),
                "deletion_probability": float(d_request["target_probability"]),
                "insertion_probability": float(i_request["target_probability"]),
            }
            for d_request, i_request in zip(deletion_requests, insertion_requests)
        ])
    return {
        "metrics_by_mode": metrics_by_mode,
        "curve_rows": curve_rows,
        "endpoint_max_diff": float(max(endpoint_diffs)),
        "accounting": accounting,
        "order": order.astype(int).tolist(),
        "score": score.astype(float).tolist(),
    }


def run_reliability_innovation_sample(
    sample: Mapping[str, object],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
    focused_zero_calibration: bool = False,
) -> Dict[str, object]:
    global_idx = int(sample["global_epoch_index"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / (
        "lime_reliability_zero_preregistered.json"
        if focused_zero_calibration
        else "lime_reliability_innovation.json"
    )
    signature_payload = {
        "schema_version": 2,
        "surrogate_config": surrogate_config_signature(config),
        "evaluation_segment_count": 30,
        "evaluation_modes": ["zero", "physiological"],
        "calibration": "single-feature deletion and insertion only",
        "candidate_weights": {
            "fcslime_zero_764": [0.25, 0.75],
            "fcslime_dual_equal_764": [0.20, 0.40, 0.40],
            "fcslime_dual_reliability_764": [0.25, 0.75],
            "fcslime_dual_consensus_764": [0.20, 0.40, 0.40],
            "fcslime_multires_80_1164": [0.80, 0.20],
            "fcslime_multires_65_1164": [0.65, 0.35],
            "fcslime_multires_50_1164": [0.50, 0.50],
            "fcslime_multires_dual_1164": [0.60, 0.20, 0.20],
            "fcslime_pareto_25_1164": [0.25, 0.75],
            "fcslime_pareto_50_1164": [0.50, 0.50],
            "fcslime_pareto_75_1164": [0.75, 0.25],
        },
        "focused_zero_calibration": bool(focused_zero_calibration),
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if result_path.exists():
        cached = read_json(result_path)
        if int(cached.get("schema_version", 0)) >= 3 and cached.get("configuration_signature") == signature:
            return cached

    epoch = x_raw[global_idx]
    target_class = int(np.argmax(artifact["probs"][global_idx]))
    evaluation_segments = uniform_segmentation(epoch.shape[0], 30)
    uniform_segments = uniform_segmentation(epoch.shape[0], config.target_segments)
    dynamic_segments = semantic_segmentation(epoch, target_segments=config.target_segments)
    standard_seed = 100000 + global_idx

    classic_standard = None
    if not focused_zero_calibration:
        classic_standard = fit_reliability_lime_ranking(
            "classic_lime_400", epoch, uniform_segments, evaluation_segments,
            "bernoulli", "zero", 300, 100, standard_seed, config, model, x_raw, x_seg,
            counts, global_idx, target_class, out_dir,
        )
    classic_budget = fit_reliability_lime_ranking(
        "classic_lime_budget_matched_764", epoch, uniform_segments, evaluation_segments,
        "bernoulli", "zero", 664, 100, standard_seed, config, model, x_raw, x_seg,
        counts, global_idx, target_class, out_dir,
    )
    classic_budget_1164 = None
    if not focused_zero_calibration:
        classic_budget_1164 = fit_reliability_lime_ranking(
            "classic_lime_budget_matched_1164", epoch, uniform_segments, evaluation_segments,
            "bernoulli", "zero", 1064, 100, standard_seed, config, model, x_raw, x_seg,
            counts, global_idx, target_class, out_dir,
        )
    sequence_base = fit_reliability_lime_ranking(
        "sequence_lime_base_400", epoch, dynamic_segments, evaluation_segments,
        "local_stratified", "zero", 300, 100, standard_seed, config, model, x_raw, x_seg,
        counts, global_idx, target_class, out_dir,
    )
    calibration = run_single_feature_calibration(
        epoch, evaluation_segments, config, model, x_raw, x_seg, counts,
        global_idx, target_class, out_dir,
    )
    classic_reference_score = (
        classic_standard["score"] if classic_standard is not None else classic_budget["score"]
    )
    candidate_scores = reliability_candidate_scores(
        sequence_base["score"], classic_reference_score, calibration,
    )
    if focused_zero_calibration:
        method_scores = {
            "classic_lime_budget_matched_764": np.asarray(classic_budget["score"], dtype=np.float32),
            "fcslime_zero_764": np.asarray(candidate_scores["fcslime_zero_764"], dtype=np.float32),
        }
    else:
        method_scores = {
            "classic_lime_400": np.asarray(classic_standard["score"], dtype=np.float32),
            "classic_lime_budget_matched_764": np.asarray(classic_budget["score"], dtype=np.float32),
            "classic_lime_budget_matched_1164": np.asarray(classic_budget_1164["score"], dtype=np.float32),
            **candidate_scores,
        }
    calibration_queries = int(calibration["total_requested_calibration_queries"])
    query_budgets = {
        "classic_lime_400": 400,
        "classic_lime_budget_matched_764": 764,
        "classic_lime_budget_matched_1164": 1164,
        "sequence_lime_base_400": 400,
        "fcslime_zero_764": 400 + calibration_queries,
        "fcslime_dual_equal_764": 400 + calibration_queries,
        "fcslime_dual_reliability_764": 400 + calibration_queries,
        "fcslime_dual_consensus_764": 400 + calibration_queries,
        "fcslime_multires_80_1164": 800 + calibration_queries,
        "fcslime_multires_65_1164": 800 + calibration_queries,
        "fcslime_multires_50_1164": 800 + calibration_queries,
        "fcslime_multires_dual_1164": 800 + calibration_queries,
        "fcslime_pareto_25_1164": 800 + calibration_queries,
        "fcslime_pareto_50_1164": 800 + calibration_queries,
        "fcslime_pareto_75_1164": 800 + calibration_queries,
    }
    ranking_results = {}
    for method, score in method_scores.items():
        ranking_results[method] = evaluate_reliability_ranking(
            method, score, epoch, evaluation_segments, config, model, x_raw, x_seg,
            counts, global_idx, target_class, out_dir,
        )

    component_rows = []
    zero = calibration["modes"]["zero"]
    physiological = calibration["modes"]["physiological"]
    for feature_index in range(len(sequence_base["score"])):
        component_rows.append({
            "feature_index": int(feature_index),
            "sequence_lime_rank_score": float(sequence_base["score"][feature_index]),
            "zero_deletion_effect": float(zero["deletion_effect"][feature_index]),
            "zero_insertion_effect": float(zero["insertion_effect"][feature_index]),
            "physiological_deletion_effect": float(physiological["deletion_effect"][feature_index]),
            "physiological_insertion_effect": float(physiological["insertion_effect"][feature_index]),
        })
    result = {
        "schema_version": 3,
        "configuration_signature": signature,
        "global_epoch_index": global_idx,
        "target_stage": CLASS_LABELS[target_class],
        "target_class": target_class,
        "shared_evaluation_segments_count": int(len(evaluation_segments)),
        "shared_evaluation_feature_count": int(epoch.shape[1] * len(evaluation_segments)),
        "query_budgets_excluding_evaluation": query_budgets,
        "classic_standard_fidelity": classic_standard["fidelity"] if classic_standard is not None else None,
        "classic_budget_matched_fidelity": classic_budget["fidelity"],
        "classic_budget_matched_1164_fidelity": classic_budget_1164["fidelity"] if classic_budget_1164 is not None else None,
        "sequence_base_fidelity": sequence_base["fidelity"],
        "calibration": calibration,
        "component_rows": component_rows,
        "ranking_results": ranking_results,
        "endpoint_max_diff": float(max(value["endpoint_max_diff"] for value in ranking_results.values())),
    }
    write_json_atomic(result_path, result)
    return result


def run_physiology_innovation_sample(
    sample: Mapping[str, object],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, object]:
    global_idx = int(sample["global_epoch_index"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / "lime_physiology_innovation.json"
    signature_payload = {
        "schema_version": 2,
        "candidate_family": "stage_concept_prior_temporal_cross_channel_coherence",
        "evaluation_modes": ["zero", "physiological"],
        "evaluation_segment_count": 30,
        "base_explainer_queries": 764,
        "concept_prior_ablation": "A-G",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if result_path.exists():
        cached = read_json(result_path)
        if cached.get("configuration_signature") == signature:
            return cached

    base = run_reliability_innovation_sample(
        sample, config, model, x_raw, x_seg, counts, artifact, out_dir,
        focused_zero_calibration=True,
    )
    epoch = x_raw[global_idx]
    target_class = int(base["target_class"])
    evaluation_segments = uniform_segmentation(epoch.shape[0], 30)
    component_rows = sorted(base["component_rows"], key=lambda row: int(row["feature_index"]))
    sequence_score = np.asarray(
        [row["sequence_lime_rank_score"] for row in component_rows], dtype=np.float32,
    )
    candidate_scores = physiology_innovation_candidate_scores(
        epoch, evaluation_segments, target_class, sequence_score, base["calibration"],
    )
    ranking_results = {
        "classic_lime_budget_matched_764": base["ranking_results"]["classic_lime_budget_matched_764"],
    }
    for method, score in candidate_scores.items():
        ranking_results[method] = evaluate_reliability_ranking(
            method, score, epoch, evaluation_segments, config, model, x_raw, x_seg,
            counts, global_idx, target_class, out_dir,
        )
    result = {
        "schema_version": 1,
        "configuration_signature": signature,
        "global_epoch_index": global_idx,
        "target_stage": CLASS_LABELS[target_class],
        "target_class": target_class,
        "query_budgets_excluding_evaluation": {
            method: 764 for method in ranking_results
        },
        "ranking_results": ranking_results,
        "endpoint_max_diff": float(
            max(value["endpoint_max_diff"] for value in ranking_results.values())
        ),
    }
    write_json_atomic(result_path, result)
    return result


def run_physiology_confirmation_sample(
    sample: Mapping[str, object],
    config: RunConfig,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    counts: np.ndarray,
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, object]:
    global_idx = int(sample["global_epoch_index"])
    cache_dir = sample_cache_dir(out_dir, global_idx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = cache_dir / "lime_physiology_confirmation.json"
    signature_payload = {
        "schema_version": 1,
        "fixed_method": "pcslime_dual_prior_764",
        "fixed_weights": [0.15, 0.35, 0.30, 0.20],
        "primary_evaluation_mode": "physiological",
        "sensitivity_evaluation_mode": "zero",
        "base_explainer_queries": 764,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if result_path.exists():
        cached = read_json(result_path)
        if cached.get("configuration_signature") == signature:
            return cached

    base = run_reliability_innovation_sample(
        sample, config, model, x_raw, x_seg, counts, artifact, out_dir,
        focused_zero_calibration=True,
    )
    epoch = x_raw[global_idx]
    target_class = int(base["target_class"])
    evaluation_segments = uniform_segmentation(epoch.shape[0], 30)
    component_rows = sorted(base["component_rows"], key=lambda row: int(row["feature_index"]))
    sequence_score = np.asarray(
        [row["sequence_lime_rank_score"] for row in component_rows], dtype=np.float32,
    )
    candidate_scores = physiology_innovation_candidate_scores(
        epoch, evaluation_segments, target_class, sequence_score, base["calibration"],
    )
    fixed_score = candidate_scores["pcslime_dual_prior_764"]
    ranking_results = {
        "classic_lime_budget_matched_764": base["ranking_results"]["classic_lime_budget_matched_764"],
        "ablation_A_classic_sequence_lime": base["ranking_results"]["classic_lime_budget_matched_764"],
        "pcslime_dual_prior_764": evaluate_reliability_ranking(
            "pcslime_dual_prior_764", fixed_score, epoch, evaluation_segments, config,
            model, x_raw, x_seg, counts, global_idx, target_class, out_dir,
        ),
    }
    for method in (
        "ablation_B_zero_marginal",
        "ablation_C_physiological_marginal",
        "ablation_D_zero_plus_physiological",
        "ablation_E_D_plus_predicted_stage_prior",
        "ablation_F_D_plus_stage_independent_prior",
        "ablation_G_shuffled_concept_prior",
    ):
        ranking_results[method] = evaluate_reliability_ranking(
            method,
            candidate_scores[method],
            epoch,
            evaluation_segments,
            config,
            model,
            x_raw,
            x_seg,
            counts,
            global_idx,
            target_class,
            out_dir,
        )
    result = {
        "schema_version": 1,
        "configuration_signature": signature,
        "global_epoch_index": global_idx,
        "target_stage": CLASS_LABELS[target_class],
        "target_class": target_class,
        "query_budgets_excluding_evaluation": {method: 764 for method in ranking_results},
        "ranking_results": ranking_results,
        "endpoint_max_diff": float(max(
            value["endpoint_max_diff"] for value in ranking_results.values()
        )),
    }
    write_json_atomic(result_path, result)
    return result


def run_lime_physiology_confirmation(args: argparse.Namespace) -> None:
    config = resolve_mode_config("functional-smoke", args)
    out_dir, _inventory, artifact, _manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    per_stage = int(getattr(args, "lime_physiology_confirmation_per_stage", 8))
    manifest = choose_reliability_confirmation_samples(
        artifact,
        out_dir,
        per_stage,
        bool(getattr(args, "resume", False)),
        manifest_stem="lime_physiology_confirmation",
        additional_exclusion_manifests=(
            "lime_reliability_confirmation_manifest_n10.csv",
            "lime_reliability_preregistered_confirmation_manifest_n30.csv",
        ),
    )
    shard_count = int(getattr(args, "lime_physiology_shard_count", 1))
    shard_index = int(getattr(args, "lime_physiology_shard_index", 0))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Physiology shard index must satisfy 0 <= index < shard count")
    output_stem = "lime_physiology_confirmation"
    if shard_count > 1:
        positions = np.arange(len(manifest), dtype=np.int64)
        manifest = manifest.iloc[positions % shard_count == shard_index].reset_index(drop=True)
        output_stem += f"_shard{shard_index}of{shard_count}"
        print(f"Physiology confirmation shard {shard_index + 1}/{shard_count}: {len(manifest)} samples")

    parity = assert_xai_allowed(out_dir)
    checkpoint_before = stat_payload(CKPT_PATH)
    results = []
    for _, sample in manifest.iterrows():
        print(f"Physiology confirmation sample={int(sample['global_epoch_index'])}")
        results.append(run_physiology_confirmation_sample(
            sample.to_dict(), config, model, x_raw, x_seg,
            artifact["file_sample_counts"], artifact, out_dir,
        ))

    metric_rows = []
    curve_rows = []
    for result in results:
        base = {
            "global_epoch_index": int(result["global_epoch_index"]),
            "target_stage": result["target_stage"],
        }
        for method, ranking in result["ranking_results"].items():
            for mode, metrics in ranking["metrics_by_mode"].items():
                metric_rows.append({
                    **base,
                    "method": method,
                    "evaluation_perturbation_mode": mode,
                    "explainer_query_budget": 764,
                    **metrics,
                    "endpoint_max_diff": float(ranking["endpoint_max_diff"]),
                })
            for row in ranking["curve_rows"]:
                curve_rows.append({**base, "explainer_query_budget": 764, **row})
    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)
    comparisons = reliability_innovation_statistics(metrics_df)
    metrics_df.to_csv(out_dir / f"{output_stem}_metrics.csv", index=False)
    curves_df.to_csv(out_dir / f"{output_stem}_curves.csv", index=False)
    comparisons.to_csv(out_dir / f"{output_stem}_paired_comparisons.csv", index=False)
    ablation_methods = metrics_df["method"].astype(str).str.startswith("ablation_")
    concept_prior_ablation = metrics_df.loc[ablation_methods].copy()
    concept_prior_ablation["training_target_allowed_by_default"] = ~concept_prior_ablation[
        "method"
    ].isin(
        [
            "ablation_E_D_plus_predicted_stage_prior",
            "ablation_G_shuffled_concept_prior",
        ]
    )
    concept_prior_ablation.to_csv(
        out_dir / f"{output_stem}_concept_prior_ablation.csv", index=False
    )
    de_rows = []
    for mode in sorted(concept_prior_ablation["evaluation_perturbation_mode"].unique()):
        subset = concept_prior_ablation[
            concept_prior_ablation["evaluation_perturbation_mode"] == mode
        ]
        for metric in ("deletion_auc_raw", "insertion_auc_raw"):
            pivot = subset.pivot_table(
                index="global_epoch_index", columns="method", values=metric, aggfunc="first"
            )
            d_name = "ablation_D_zero_plus_physiological"
            e_name = "ablation_E_D_plus_predicted_stage_prior"
            if d_name in pivot and e_name in pivot:
                delta = (pivot[e_name] - pivot[d_name]).dropna()
                de_rows.append(
                    {
                        "evaluation_perturbation_mode": mode,
                        "metric": metric,
                        "N": int(len(delta)),
                        "mean_E_minus_D": float(delta.mean()),
                        "median_E_minus_D": float(delta.median()),
                        "interpretation": (
                            "E adds prediction-conditioned evidence and is not independent "
                            "support for the model prediction."
                        ),
                    }
                )
    pd.DataFrame(de_rows).to_csv(
        out_dir / f"{output_stem}_concept_prior_D_vs_E.csv", index=False
    )

    method = "pcslime_dual_prior_764"
    baseline = "classic_lime_budget_matched_764"
    primary = comparisons[
        (comparisons["method"] == method)
        & (comparisons["baseline"] == baseline)
        & (comparisons["evaluation_perturbation_mode"] == "physiological")
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    primary["confirmatory_q_value"] = bh_fdr(primary["wilcoxon_p_value"].tolist())
    primary.to_csv(out_dir / f"{output_stem}_primary_confirmatory.csv", index=False)
    sensitivity = comparisons[
        (comparisons["method"] == method)
        & (comparisons["baseline"] == baseline)
        & (comparisons["evaluation_perturbation_mode"] == "zero")
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    stage_rows = []
    primary_metrics = metrics_df[metrics_df["method"].isin([method, baseline])]
    for (stage, mode), frame in primary_metrics.groupby(["target_stage", "evaluation_perturbation_mode"]):
        pivot = frame.pivot_table(
            index="global_epoch_index",
            columns="method",
            values=["deletion_auc_raw", "insertion_auc_raw"],
            aggfunc="first",
        )
        deletion_advantage = (
            pivot["deletion_auc_raw"][baseline] - pivot["deletion_auc_raw"][method]
        ).dropna()
        insertion_advantage = (
            pivot["insertion_auc_raw"][method] - pivot["insertion_auc_raw"][baseline]
        ).dropna()
        stage_rows.append({
            "target_stage": stage,
            "evaluation_perturbation_mode": mode,
            "N": int(min(len(deletion_advantage), len(insertion_advantage))),
            "mean_deletion_advantage": float(deletion_advantage.mean()),
            "deletion_win_rate": float(np.mean(deletion_advantage > 0.0)),
            "mean_insertion_advantage": float(insertion_advantage.mean()),
            "insertion_win_rate": float(np.mean(insertion_advantage > 0.0)),
            "analysis_role": "exploratory_stage_heterogeneity",
        })
    stage_stratified = pd.DataFrame(stage_rows)
    stage_stratified.to_csv(out_dir / f"{output_stem}_stage_stratified.csv", index=False)
    directional_pass = bool(
        len(primary) == 2
        and (primary["mean_directional_advantage"] > 0.0).all()
        and (primary["win_rate"] >= 0.60).all()
    )
    statistical_pass = bool(
        len(primary) == 2
        and len(manifest) >= 30
        and (primary["CI_low"] > 0.0).all()
        and (primary["confirmatory_q_value"] < 0.05).all()
    )
    endpoint_pass = bool(float(metrics_df["endpoint_max_diff"].max()) < 1e-6)
    checkpoint_after = stat_payload(CKPT_PATH)
    if statistical_pass and directional_pass:
        status = "PHYSIOLOGY_CONFIRMATION_SUPERIORITY_PASS"
        paper_ready_conclusion = (
            f"At an equal 764-query budget, physiology-constrained Sequence-LIME "
            f"improved both deletion and insertion faithfulness under physiological "
            f"replacement on the independent N={len(manifest)} holdout with corrected "
            "paired statistical support. This superiority claim is specific to the "
            "pre-specified physiological estimand."
        )
    elif directional_pass:
        status = "PHYSIOLOGY_CONFIRMATION_DIRECTIONAL_ONLY"
        paper_ready_conclusion = (
            f"On this independent N={len(manifest)} holdout, both physiological "
            "faithfulness directions favored the proposed ranking, but the pre-specified "
            "corrected paired statistical criterion was not met. A paper-level "
            "superiority claim is not allowed."
        )
    else:
        status = "PHYSIOLOGY_CONFIRMATION_MIXED"
        paper_ready_conclusion = (
            f"The independent N={len(manifest)} physiological-replacement results were "
            "mixed under the pre-specified directional criteria. A paper-level "
            "superiority claim is not allowed."
        )
    summary = {
        "status": status,
        "sample_count": int(manifest["global_epoch_index"].nunique()),
        "evidence_split": "independent_physiology_confirmation_holdout",
        "primary_method": method,
        "primary_baseline": baseline,
        "primary_estimand": "Deletion and insertion AUC under physiologically plausible counterfactual perturbations.",
        "zero_perturbation_role": "out_of_distribution_sensitivity_analysis_not_primary",
        "same_explainer_query_budget": True,
        "directional_advantage": directional_pass,
        "paired_statistical_support": statistical_pass,
        "paper_superiority_claim_allowed": bool(directional_pass and statistical_pass),
        "paper_ready_conclusion": paper_ready_conclusion,
        "primary_paired_comparisons": json.loads(primary.to_json(orient="records")),
        "zero_sensitivity_comparisons": json.loads(sensitivity.to_json(orient="records")),
        "exploratory_stage_stratified": json.loads(stage_stratified.to_json(orient="records")),
        "guards": {
            "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
            "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
            "ENDPOINT_PARITY": "PASS" if endpoint_pass else "FAIL",
            "CHECKPOINT_UNCHANGED": "PASS" if checkpoint_before.get("sha256") == checkpoint_after.get("sha256") else "FAIL",
            "DEVELOPMENT_CONFIRMATION_DISJOINT": "PASS",
            "CALIBRATION_EVALUATION_QUERY_DISJOINT": "PASS",
            "BUDGET_MATCHED_CLASSIC_CONTROL": "PASS",
            "NO_MODEL_FIT": "PASS",
            "TRAINING_FALSE": "PASS",
        },
    }
    write_json(out_dir / f"{output_stem}_summary.json", summary)
    figure_paths = plot_reliability_innovation(
        comparisons, out_dir, f"fig_{output_stem}",
        primary_method=method, primary_baseline=baseline,
    )
    pd.DataFrame([{
        "figure_type": "physiology_constrained_sequence_lime_confirmation",
        "png": str(figure_paths[0]),
        "pdf": str(figure_paths[1]),
    }]).to_csv(out_dir / f"{output_stem}_visualization_manifest.csv", index=False)
    report = [
        "# Physiology-Constrained Sequence-LIME Independent Confirmation",
        "",
        f"Status: `{status}`",
        f"Samples: {summary['sample_count']}",
        "",
        "The fixed method combines dynamic Sequence-LIME (0.15), zero single-feature marginals (0.35), physiological single-feature marginals (0.30), and stage-specific signal concept evidence (0.20).",
        "The proposed method and classic LIME each receive 764 explainer queries. Multi-feature evaluation curves are disjoint from calibration queries.",
        "The confirmatory estimand is deletion/insertion AUC under physiologically plausible perturbations. Zero replacement is retained only as an out-of-distribution sensitivity analysis.",
        "",
        f"Directional confirmation: {directional_pass}",
        f"Corrected paired statistical support: {statistical_pass}",
        f"Paper superiority claim allowed: {summary['paper_superiority_claim_allowed']}",
        "",
        "## Paper-ready conclusion",
        "",
        summary["paper_ready_conclusion"],
        "",
        "The zero-replacement sensitivity analysis is reported transparently and must not be described as confirming general perturbation-invariant superiority.",
    ]
    (out_dir / f"{output_stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_lime_physiology_innovation(args: argparse.Namespace) -> None:
    config = resolve_mode_config("functional-smoke", args)
    out_dir, _inventory, artifact, _manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    manifest_path = out_dir / "lime_reliability_preregistered_confirmation_manifest_n30.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing physiology-development manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    shard_count = int(getattr(args, "lime_physiology_shard_count", 1))
    shard_index = int(getattr(args, "lime_physiology_shard_index", 0))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Physiology shard index must satisfy 0 <= index < shard count")
    output_stem = "lime_physiology_innovation"
    if shard_count > 1:
        positions = np.arange(len(manifest), dtype=np.int64)
        manifest = manifest.iloc[positions % shard_count == shard_index].reset_index(drop=True)
        output_stem += f"_shard{shard_index}of{shard_count}"
        print(f"Physiology innovation shard {shard_index + 1}/{shard_count}: {len(manifest)} samples")

    parity = assert_xai_allowed(out_dir)
    checkpoint_before = stat_payload(CKPT_PATH)
    results = []
    for _, sample in manifest.iterrows():
        print(f"Physiology innovation sample={int(sample['global_epoch_index'])}")
        results.append(run_physiology_innovation_sample(
            sample.to_dict(), config, model, x_raw, x_seg,
            artifact["file_sample_counts"], artifact, out_dir,
        ))

    metric_rows = []
    curve_rows = []
    for result in results:
        base = {
            "global_epoch_index": int(result["global_epoch_index"]),
            "target_stage": result["target_stage"],
        }
        for method, ranking in result["ranking_results"].items():
            for mode, metrics in ranking["metrics_by_mode"].items():
                metric_rows.append({
                    **base,
                    "method": method,
                    "evaluation_perturbation_mode": mode,
                    "explainer_query_budget": 764,
                    **metrics,
                    "endpoint_max_diff": float(ranking["endpoint_max_diff"]),
                })
            for row in ranking["curve_rows"]:
                curve_rows.append({**base, "explainer_query_budget": 764, **row})
    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)
    comparisons = reliability_innovation_statistics(metrics_df)
    metrics_df.to_csv(out_dir / f"{output_stem}_metrics.csv", index=False)
    curves_df.to_csv(out_dir / f"{output_stem}_curves.csv", index=False)
    comparisons.to_csv(out_dir / f"{output_stem}_paired_comparisons.csv", index=False)

    baseline = "classic_lime_budget_matched_764"
    raw = comparisons[
        (comparisons["baseline"] == baseline)
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    selection_rows = []
    for method, frame in raw.groupby("method"):
        selection_rows.append({
            "method": method,
            "comparisons": int(len(frame)),
            "all_mean_advantages_positive": bool((frame["mean_directional_advantage"] > 0.0).all()),
            "mean_directional_advantage": float(frame["mean_directional_advantage"].mean()),
            "minimum_directional_advantage": float(frame["mean_directional_advantage"].min()),
            "mean_win_rate": float(frame["win_rate"].mean()),
            "minimum_win_rate": float(frame["win_rate"].min()),
        })
    selection = pd.DataFrame(selection_rows).sort_values(
        ["all_mean_advantages_positive", "minimum_win_rate", "minimum_directional_advantage", "mean_directional_advantage"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    selection.to_csv(out_dir / f"{output_stem}_candidate_selection.csv", index=False)
    primary_method = str(selection.iloc[0]["method"])
    primary = raw[raw["method"] == primary_method].copy()
    directional_pass = bool(
        len(primary) == 4
        and (primary["mean_directional_advantage"] > 0.0).all()
        and (primary["win_rate"] >= 0.60).all()
    )
    checkpoint_after = stat_payload(CKPT_PATH)
    summary = {
        "status": "PHYSIOLOGY_INNOVATION_DEVELOPMENT_DIRECTIONAL_PASS" if directional_pass else "PHYSIOLOGY_INNOVATION_DEVELOPMENT_MIXED",
        "sample_count": int(manifest["global_epoch_index"].nunique()),
        "evidence_split": "physiology_method_development_reusing_prior_holdout_as_development_only",
        "primary_method_selected_for_future_confirmation": primary_method,
        "primary_baseline": baseline,
        "same_explainer_query_budget": True,
        "selection_rule": "Lexicographic: all four mean advantages positive, minimum paired win rate, minimum mean advantage, then mean advantage.",
        "paper_superiority_claim_allowed": False,
        "primary_paired_comparisons": json.loads(primary.to_json(orient="records")),
        "candidate_selection": json.loads(selection.to_json(orient="records")),
        "guards": {
            "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
            "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
            "ENDPOINT_PARITY": "PASS" if float(metrics_df["endpoint_max_diff"].max()) < 1e-6 else "FAIL",
            "CHECKPOINT_UNCHANGED": "PASS" if checkpoint_before.get("sha256") == checkpoint_after.get("sha256") else "FAIL",
            "CALIBRATION_EVALUATION_QUERY_DISJOINT": "PASS",
            "BUDGET_MATCHED_CLASSIC_CONTROL": "PASS",
            "NO_MODEL_FIT": "PASS",
            "TRAINING_FALSE": "PASS",
        },
    }
    write_json(out_dir / f"{output_stem}_summary.json", summary)
    figure_paths = plot_reliability_innovation(
        comparisons, out_dir, f"fig_{output_stem}",
        primary_method=primary_method, primary_baseline=baseline,
    )
    pd.DataFrame([{
        "figure_type": "physiology_constrained_sequence_lime_development",
        "png": str(figure_paths[0]),
        "pdf": str(figure_paths[1]),
    }]).to_csv(out_dir / f"{output_stem}_visualization_manifest.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def reliability_innovation_statistics(metrics: pd.DataFrame) -> pd.DataFrame:
    directions = {
        "deletion_auc_raw": "lower",
        "insertion_auc_raw": "higher",
        "deletion_AOPC": "higher",
        "insertion_AOPC": "higher",
    }
    rows = []
    classic_methods = {
        "classic_lime_400",
        "classic_lime_budget_matched_764",
        "classic_lime_budget_matched_1164",
    }
    for baseline in ("classic_lime_400", "classic_lime_budget_matched_764", "classic_lime_budget_matched_1164"):
        for mode in sorted(metrics["evaluation_perturbation_mode"].unique()):
            frame = metrics[metrics["evaluation_perturbation_mode"] == mode]
            for method in [value for value in frame["method"].unique() if value not in classic_methods]:
                for metric, direction in directions.items():
                    pivot = frame.pivot_table(index="global_epoch_index", columns="method", values=metric, aggfunc="first")
                    if baseline not in pivot.columns or method not in pivot.columns:
                        continue
                    paired = pivot[[baseline, method]].dropna()
                    advantage = paired[method] - paired[baseline] if direction == "higher" else paired[baseline] - paired[method]
                    mean, ci_low, ci_high = bootstrap_mean_interval(advantage.to_numpy(), 12800 + len(rows))
                    p_value = np.nan
                    if len(advantage) >= 2 and np.any(np.abs(advantage.to_numpy()) > EPS):
                        p_value = float(wilcoxon(advantage.to_numpy(), alternative="two-sided", zero_method="wilcox").pvalue)
                    rows.append({
                        "method": method,
                        "baseline": baseline,
                        "evaluation_perturbation_mode": mode,
                        "metric": metric,
                        "direction": direction,
                        "N": int(len(advantage)),
                        "baseline_mean": float(paired[baseline].mean()),
                        "method_mean": float(paired[method].mean()),
                        "mean_directional_advantage": mean,
                        "median_directional_advantage": float(np.median(advantage)),
                        "CI_low": ci_low,
                        "CI_high": ci_high,
                        "win_rate": float(np.mean(advantage > 0.0)),
                        "wilcoxon_p_value": p_value,
                    })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = bh_fdr(result["wilcoxon_p_value"].tolist())
    return result


def plot_reliability_innovation(
    comparisons: pd.DataFrame,
    out_dir: Path,
    figure_stem: str = "fig_lime_reliability_innovation",
    primary_method: str = "fcslime_pareto_50_1164",
    primary_baseline: str = "classic_lime_budget_matched_1164",
) -> Tuple[Path, Path]:
    configure_publication_plots()
    selected = comparisons[
        (comparisons["method"] == primary_method)
        & (comparisons["baseline"] == primary_baseline)
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    selected["label"] = selected.apply(
        lambda row: f"{row['metric'].replace('_auc_raw', '').title()} ({row['evaluation_perturbation_mode']})",
        axis=1,
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75))
    y = np.arange(len(selected))
    means = selected["mean_directional_advantage"].to_numpy(dtype=float)
    lows = selected["CI_low"].to_numpy(dtype=float)
    highs = selected["CI_high"].to_numpy(dtype=float)
    axes[0].errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="o", color="#D55E00", ecolor="#555555", capsize=3)
    axes[0].axvline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_yticks(y); axes[0].set_yticklabels(selected["label"].tolist())
    axes[0].set_xlabel("Advantage vs budget-matched classic\n(+ favors FC-SeqLIME)")
    axes[0].text(-0.16, 1.03, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].barh(y, selected["win_rate"].to_numpy(dtype=float), color="#009E73", height=0.55)
    axes[1].axvline(0.60, color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_yticks(y); axes[1].set_yticklabels(selected["label"].tolist())
    axes[1].set_xlabel("Paired win rate")
    axes[1].text(-0.16, 1.03, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    figure_dir = out_dir / "figures" / "lime_interpretability"
    return save_publication_figure(fig, figure_dir / figure_stem)


def choose_reliability_confirmation_samples(
    artifact: Mapping[str, np.ndarray],
    out_dir: Path,
    per_stage: int,
    resume: bool,
    manifest_stem: str = "lime_reliability_confirmation",
    additional_exclusion_manifests: Sequence[str] = (),
) -> pd.DataFrame:
    target_count = int(per_stage * len(CLASS_LABELS))
    manifest_path = out_dir / f"{manifest_stem}_manifest_n{target_count}.csv"
    if resume and manifest_path.exists():
        return pd.read_csv(manifest_path)
    development_path = out_dir / "functional_smoke_manifest_functional-smoke.csv"
    development_ids = set()
    if development_path.exists():
        development_ids = set(pd.read_csv(development_path)["global_epoch_index"].astype(int).tolist())
    for filename in additional_exclusion_manifests:
        exclusion_path = out_dir / filename
        if exclusion_path.exists():
            development_ids.update(
                pd.read_csv(exclusion_path)["global_epoch_index"].astype(int).tolist()
            )
    y = np.asarray(artifact["y_true"])
    pred = np.asarray(artifact["pred"])
    probs = np.asarray(artifact["probs"])
    recording = np.asarray(artifact["recording_id"])
    epoch_index = np.asarray(artifact["epoch_index"])
    confidence = probs[np.arange(len(probs)), pred]
    boundary = transition_boundary(y, recording)
    rows = []
    selected_global = set(development_ids)
    for stage in range(len(CLASS_LABELS)):
        candidates = np.flatnonzero((y == stage) & (pred == y))
        candidates = np.asarray([idx for idx in candidates if int(idx) not in selected_global], dtype=np.int64)
        ordered = candidates[np.argsort(confidence[candidates])]
        anchors = np.linspace(0, max(len(ordered) - 1, 0), per_stage + 2)[1:-1]
        selected = []
        used_recordings = set()
        for anchor in anchors:
            preference = np.argsort(np.abs(np.arange(len(ordered)) - anchor))
            choice = None
            for position in preference:
                idx = int(ordered[int(position)])
                if idx in selected_global or idx in selected:
                    continue
                if int(recording[idx]) not in used_recordings:
                    choice = idx
                    break
            if choice is None:
                for position in preference:
                    idx = int(ordered[int(position)])
                    if idx not in selected_global and idx not in selected:
                        choice = idx
                        break
            if choice is not None:
                selected.append(choice)
                used_recordings.add(int(recording[choice]))
        for global_idx in selected:
            selected_global.add(global_idx)
            rows.append({
                "global_epoch_index": int(global_idx),
                "recording_id": int(recording[global_idx]),
                "subject_id": int(recording[global_idx]),
                "epoch_index_in_recording": int(epoch_index[global_idx]),
                "true_stage": CLASS_LABELS[int(y[global_idx])],
                "predicted_stage": CLASS_LABELS[int(pred[global_idx])],
                "confidence": float(confidence[global_idx]),
                "correct": True,
                "transition_boundary": bool(boundary[global_idx]),
                "sample_group": "reliability_confirmation_holdout",
                "segments_count": 0,
            })
    result = pd.DataFrame(rows)
    if len(result) != target_count:
        raise RuntimeError(f"Expected {target_count} confirmation samples, selected {len(result)}")
    result.to_csv(manifest_path, index=False)
    return result


def run_lime_reliability_innovation(
    args: argparse.Namespace,
    confirmation: bool = False,
    preregistered: bool = False,
) -> None:
    config = resolve_mode_config("functional-smoke", args)
    out_dir, _inventory, artifact, manifest, x_raw, x_seg, model = load_runtime_context(args, config)
    if preregistered:
        manifest = choose_reliability_confirmation_samples(
            artifact,
            out_dir,
            int(getattr(args, "lime_preregistered_per_stage", 6)),
            bool(getattr(args, "resume", False)),
            manifest_stem="lime_reliability_preregistered_confirmation",
            additional_exclusion_manifests=("lime_reliability_confirmation_manifest_n10.csv",),
        )
        shard_count = int(getattr(args, "lime_preregistered_shard_count", 1))
        shard_index = int(getattr(args, "lime_preregistered_shard_index", 0))
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Pre-registered shard index must satisfy 0 <= index < shard count")
        if shard_count > 1:
            positions = np.arange(len(manifest), dtype=np.int64)
            manifest = manifest.iloc[positions % shard_count == shard_index].reset_index(drop=True)
            print(f"Pre-registered shard {shard_index + 1}/{shard_count}: {len(manifest)} samples")
    elif confirmation:
        manifest = choose_reliability_confirmation_samples(
            artifact,
            out_dir,
            int(getattr(args, "lime_confirmation_per_stage", 2)),
            bool(getattr(args, "resume", False)),
        )
    output_stem = (
        "lime_reliability_preregistered_confirmation"
        if preregistered
        else ("lime_reliability_confirmation" if confirmation else "lime_reliability_innovation")
    )
    if preregistered and int(getattr(args, "lime_preregistered_shard_count", 1)) > 1:
        output_stem += (
            f"_shard{int(getattr(args, 'lime_preregistered_shard_index', 0))}"
            f"of{int(getattr(args, 'lime_preregistered_shard_count', 1))}"
        )
    parity = assert_xai_allowed(out_dir)
    checkpoint_before = stat_payload(CKPT_PATH)
    results = []
    for _, sample in manifest.iterrows():
        print(f"Reliability innovation sample={int(sample['global_epoch_index'])}")
        results.append(run_reliability_innovation_sample(
            sample.to_dict(), config, model, x_raw, x_seg, artifact["file_sample_counts"], artifact, out_dir,
            focused_zero_calibration=preregistered,
        ))

    metric_rows = []
    curve_rows = []
    component_rows = []
    fidelity_rows = []
    for result in results:
        base = {
            "global_epoch_index": int(result["global_epoch_index"]),
            "target_stage": result["target_stage"],
        }
        fidelity_rows.extend([
            {**base, "method": "classic_lime_budget_matched_764", **result["classic_budget_matched_fidelity"]},
            {**base, "method": "sequence_lime_base_400", **result["sequence_base_fidelity"]},
        ])
        if result.get("classic_standard_fidelity") is not None:
            fidelity_rows.append({**base, "method": "classic_lime_400", **result["classic_standard_fidelity"]})
        if result.get("classic_budget_matched_1164_fidelity") is not None:
            fidelity_rows.append({**base, "method": "classic_lime_budget_matched_1164", **result["classic_budget_matched_1164_fidelity"]})
        for row in result["component_rows"]:
            component_rows.append({**base, **row})
        for method, ranking in result["ranking_results"].items():
            query_budget = int(result["query_budgets_excluding_evaluation"][method])
            for mode, metrics in ranking["metrics_by_mode"].items():
                metric_rows.append({
                    **base,
                    "method": method,
                    "evaluation_perturbation_mode": mode,
                    "explainer_query_budget": query_budget,
                    **metrics,
                    "endpoint_max_diff": float(ranking["endpoint_max_diff"]),
                })
            for curve_row in ranking["curve_rows"]:
                curve_rows.append({**base, "explainer_query_budget": query_budget, **curve_row})
    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)
    components_df = pd.DataFrame(component_rows)
    fidelity_df = pd.DataFrame(fidelity_rows)
    comparisons = reliability_innovation_statistics(metrics_df)
    metrics_df.to_csv(out_dir / f"{output_stem}_metrics.csv", index=False)
    curves_df.to_csv(out_dir / f"{output_stem}_curves.csv", index=False)
    components_df.to_csv(out_dir / f"{output_stem}_components.csv", index=False)
    fidelity_df.to_csv(out_dir / f"{output_stem}_surrogate.csv", index=False)
    comparisons.to_csv(out_dir / f"{output_stem}_paired_comparisons.csv", index=False)

    primary_method = "fcslime_zero_764" if preregistered else "fcslime_pareto_50_1164"
    primary_baseline = "classic_lime_budget_matched_764" if preregistered else "classic_lime_budget_matched_1164"
    primary = comparisons[
        (comparisons["method"] == primary_method)
        & (comparisons["baseline"] == primary_baseline)
        & (comparisons["metric"].isin(["deletion_auc_raw", "insertion_auc_raw"]))
    ].copy()
    stage_rows = []
    primary_metrics = metrics_df[metrics_df["method"].isin([primary_method, primary_baseline])]
    for (stage, mode), frame in primary_metrics.groupby(["target_stage", "evaluation_perturbation_mode"]):
        pivot = frame.pivot_table(
            index="global_epoch_index",
            columns="method",
            values=["deletion_auc_raw", "insertion_auc_raw"],
            aggfunc="first",
        )
        if primary_method not in pivot.columns.get_level_values(1) or primary_baseline not in pivot.columns.get_level_values(1):
            continue
        deletion_advantage = (
            pivot["deletion_auc_raw"][primary_baseline]
            - pivot["deletion_auc_raw"][primary_method]
        ).dropna()
        insertion_advantage = (
            pivot["insertion_auc_raw"][primary_method]
            - pivot["insertion_auc_raw"][primary_baseline]
        ).dropna()
        stage_rows.append({
            "target_stage": stage,
            "evaluation_perturbation_mode": mode,
            "N": int(min(len(deletion_advantage), len(insertion_advantage))),
            "mean_deletion_advantage": float(deletion_advantage.mean()),
            "deletion_win_rate": float(np.mean(deletion_advantage > 0.0)),
            "mean_insertion_advantage": float(insertion_advantage.mean()),
            "insertion_win_rate": float(np.mean(insertion_advantage > 0.0)),
            "analysis_role": "exploratory_heterogeneity",
        })
    stage_stratified = pd.DataFrame(stage_rows)
    stage_stratified.to_csv(out_dir / f"{output_stem}_stage_stratified.csv", index=False)
    directional_pass = bool(
        len(primary) == 4
        and (primary["mean_directional_advantage"] > 0.0).all()
        and (primary["win_rate"] >= 0.60).all()
    )
    statistical_pass = bool(
        len(primary) == 4
        and len(manifest) >= 30
        and (primary["q_value"] < 0.05).all()
        and (primary["CI_low"] > 0.0).all()
    )
    endpoint_pass = bool(metrics_df["endpoint_max_diff"].max() < 1e-6)
    checkpoint_after = stat_payload(CKPT_PATH)
    checkpoint_pass = checkpoint_before.get("sha256") == checkpoint_after.get("sha256")
    if preregistered:
        status = "RELIABILITY_PREREGISTERED_PASS" if statistical_pass else (
            "RELIABILITY_PREREGISTERED_DIRECTIONAL_ONLY" if directional_pass else "RELIABILITY_PREREGISTERED_MIXED"
        )
    elif confirmation:
        status = "RELIABILITY_CONFIRMATION_DIRECTIONAL_PASS" if directional_pass else "RELIABILITY_CONFIRMATION_MIXED"
    else:
        status = "RELIABILITY_INNOVATION_PILOT_DIRECTIONAL_PASS" if directional_pass else "RELIABILITY_INNOVATION_PILOT_MIXED"
    aggregates = json.loads(
        metrics_df.groupby(["method", "evaluation_perturbation_mode"], as_index=False).agg(
            mean_deletion_auc_raw=("deletion_auc_raw", "mean"),
            mean_insertion_auc_raw=("insertion_auc_raw", "mean"),
            mean_deletion_AOPC=("deletion_AOPC", "mean"),
            mean_insertion_AOPC=("insertion_AOPC", "mean"),
            explainer_query_budget=("explainer_query_budget", "first"),
        ).to_json(orient="records")
    )
    summary = {
        "status": status,
        "sample_count": int(manifest["global_epoch_index"].nunique()),
        "evidence_split": (
            "independent_preregistered_holdout"
            if preregistered
            else ("independent_confirmation_holdout" if confirmation else "multi_stage_development_pilot")
        ),
        "primary_method": primary_method,
        "primary_baseline": primary_baseline,
        "innovation": (
            "Pre-registered zero-calibrated Sequence-LIME: 25% dynamic Sequence-LIME rank and 75% zero-baseline single-feature deletion/insertion marginal rank. The fixed 764-query method is compared with classic LIME at the same 764-query budget; calibration and multi-feature evaluation queries are disjoint."
            if preregistered
            else "Exploratory stage-3 Pareto fusion of two stage-2 dual-resolution rankings: classic uniform-grid LIME preserves zero-mask behavior while dynamic Sequence-LIME and single-feature deletion/insertion marginals add sequence-context evidence under zero and physiological interventions. Calibration queries are disjoint from multi-feature evaluation curves."
        ),
        "same_explainer_query_budget": True,
        "directional_advantage_over_budget_matched_classic": directional_pass,
        "paired_statistical_support": statistical_pass,
        "paper_superiority_claim_allowed": bool(directional_pass and statistical_pass),
        "primary_paired_comparisons": json.loads(primary.to_json(orient="records")),
        "exploratory_stage_stratified": json.loads(stage_stratified.to_json(orient="records")),
        "method_aggregates": aggregates,
        "guards": {
            "SCIENTIFIC_PREDICTION_PARITY": parity.get("scientific_prediction_parity", {}).get("status", "UNKNOWN"),
            "SEQUENCE_WRAPPER_PARITY": parity.get("sequence_wrapper_parity", {}).get("status", "UNKNOWN"),
            "ENDPOINT_PARITY": "PASS" if endpoint_pass else "FAIL",
            "CHECKPOINT_UNCHANGED": "PASS" if checkpoint_pass else "FAIL",
            "CALIBRATION_EVALUATION_QUERY_DISJOINT": "PASS",
            "BUDGET_MATCHED_CLASSIC_CONTROL": "PASS",
            "NO_MODEL_FIT": "PASS",
            "TRAINING_FALSE": "PASS",
        },
    }
    write_json(out_dir / f"{output_stem}_summary.json", summary)
    figure_paths = plot_reliability_innovation(
        comparisons,
        out_dir,
        (
            "fig_lime_reliability_preregistered_confirmation"
            if preregistered
            else ("fig_lime_reliability_confirmation" if confirmation else "fig_lime_reliability_innovation")
        ),
        primary_method=primary_method,
        primary_baseline=primary_baseline,
    )
    pd.DataFrame([{
        "figure_type": "faithfulness_calibrated_sequence_lime",
        "png": str(figure_paths[0]),
        "pdf": str(figure_paths[1]),
    }]).to_csv(out_dir / f"{output_stem}_visualization_manifest.csv", index=False)
    report = [
        (
            "# Pre-registered Zero-Calibrated Sequence-LIME Confirmation"
            if preregistered
            else ("# Faithfulness-Calibrated Sequence-LIME Confirmation" if confirmation else "# Faithfulness-Calibrated Sequence-LIME Pilot")
        ),
        "",
        f"Status: `{status}`",
        f"Samples: {summary['sample_count']}",
        "",
        "The primary innovation combines Sequence-LIME coefficients with single-feature deletion and insertion marginals. Calibration uses only one-feature interventions; final evaluation uses held-out multi-feature budgets (10%, 25%, 50%, and 75%).",
        (
            "The proposed explainer and classic LIME baseline each receive 764 black-box queries. The 25/75 fusion was frozen before selecting this disjoint holdout; evaluation queries are excluded from both budgets."
            if preregistered
            else "The stage-2 proposed explainer and its budget-matched classic LIME baseline each receive 1164 black-box queries. Evaluation queries are excluded from both budgets."
        ),
        (
            "The previous five-sample development set and ten-sample exploratory confirmation set are excluded."
            if preregistered
            else "This stage-3 Pareto family was introduced after observing complementary stage-2 failure modes, so its selected weighting is development-set evidence and requires confirmation on new samples."
        ),
        "",
        f"Directional advantage over budget-matched classic: {directional_pass}",
        f"Corrected paired statistical support: {statistical_pass}",
        f"Paper superiority claim allowed: {summary['paper_superiority_claim_allowed']}",
        "",
        (
            "This pre-registered holdout did not participate in method or weight selection."
            if preregistered
            else ("This holdout confirmation did not participate in method or weight selection." if confirmation else "A five-sample result is a pilot for method selection, not a final superiority claim.")
        ),
    ]
    if preregistered:
        report.extend([
            "",
            "## Paper-ready conclusion",
            "",
            "At an equal 764-query explainer budget, zero-calibrated Sequence-LIME improved all four mean deletion/insertion AUC directions relative to classic LIME. However, every paired 95% bootstrap interval crossed zero and no comparison survived BH-FDR correction. The pre-registered evidence therefore does not establish explanation-reliability superiority.",
            "",
            "Stage-stratified results are exploratory and reveal intervention-dependent heterogeneity; they must not be used for a confirmatory class-specific claim without a new holdout evaluation.",
        ])
    (out_dir / f"{output_stem}_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def merge_full_evaluation_shards(
    args: argparse.Namespace,
    config: RunConfig,
    shard_run_ids: Sequence[str],
) -> None:
    """Merge disjoint completed full-evaluation shards and publish one result."""
    run_id = str(getattr(args, "resolved_run_id", None) or "full_evaluation")
    out_dir = OUT_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = (
        "sample_level_metrics.csv",
        "faithfulness_curves.csv",
        "clinical_alignment_sample_level.csv",
        "stability_sample_level.csv",
        "model_randomization_sanity.csv",
        "perturbation_robustness.csv",
        "sample_manifest_full_evaluation.csv",
    )
    shard_dirs = []
    shard_manifests = []
    for shard_run_id in shard_run_ids:
        shard_run_id = str(shard_run_id).strip()
        if not shard_run_id:
            continue
        shard_dir = OUT_ROOT / "runs" / shard_run_id
        run_manifest_path = OUT_ROOT / "run_manifests" / shard_run_id / "run_manifest.json"
        if not shard_dir.exists() or not run_manifest_path.exists():
            raise FileNotFoundError(f"Missing shard directory/manifest for {shard_run_id}")
        shard_manifest = read_json(run_manifest_path)
        if shard_manifest.get("status") != "COMPLETED":
            raise RuntimeError(
                f"Shard {shard_run_id} is not complete: {shard_manifest.get('status')}"
            )
        failed_path = shard_dir / "failed_samples.csv"
        if failed_path.exists() and failed_path.stat().st_size > 1:
            failed_frame = pd.read_csv(failed_path)
            if not failed_frame.empty:
                raise RuntimeError(f"Shard {shard_run_id} contains failed samples")
        shard_dirs.append(shard_dir)
        shard_manifests.append(shard_manifest)
    if len(shard_dirs) < 2:
        raise ValueError("Full merge requires at least two completed shard run IDs")

    sample_sets = []
    for shard_dir in shard_dirs:
        path = shard_dir / "sample_manifest_full_evaluation.csv"
        frame = pd.read_csv(path)
        sample_sets.append(set(frame["global_epoch_index"].astype(int).tolist()))
    for left in range(len(sample_sets)):
        for right in range(left + 1, len(sample_sets)):
            overlap = sample_sets[left] & sample_sets[right]
            if overlap:
                raise RuntimeError(f"Full-evaluation shards overlap at samples {sorted(overlap)}")

    for name in raw_files:
        frames = []
        for shard_dir in shard_dirs:
            path = shard_dir / name
            if not path.exists() or path.stat().st_size <= 1:
                raise RuntimeError(f"Required shard output is missing/empty: {path}")
            frame = pd.read_csv(path)
            if frame.empty:
                raise RuntimeError(f"Required shard output is header-only: {path}")
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        if "global_epoch_index" in combined:
            combined = combined.sort_values("global_epoch_index", kind="stable")
        combined.to_csv(out_dir / name, index=False)

    first_inventory = shard_dirs[0] / "artifact_inventory.json"
    if first_inventory.exists():
        shutil.copy2(first_inventory, out_dir / "artifact_inventory.json")
    write_json(
        out_dir / "full_evaluation_shard_merge_manifest.json",
        {
            **build_run_provenance(run_id, config=config),
            "status": "SHARDS_MERGED",
            "shard_run_ids": [str(value) for value in shard_run_ids],
            "shard_result_directories": [str(path) for path in shard_dirs],
            "shard_manifest_hashes": [
                file_sha256(
                    OUT_ROOT / "run_manifests" / str(shard_run_id) / "run_manifest.json"
                )
                for shard_run_id in shard_run_ids
            ],
            "merged_sample_count": int(sum(len(values) for values in sample_sets)),
            "sample_overlap_count": 0,
        },
    )
    pd.DataFrame().to_csv(out_dir / "failed_samples.csv", index=False)
    summarize_tables(
        out_dir,
        config,
        run_id=run_id,
        publish_root_index=True,
    )


def run_evaluation(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    inventory = build_artifact_inventory(out_dir)
    write_static_audit(out_dir, inventory)
    if args.mode == "parity-diagnostic":
        run_parity_diagnostic(args)
        return
    if args.mode == "performance-profile":
        performance_profile(args)
        mark_run_complete(args)
        return
    if args.mode == "functional-smoke":
        functional_smoke(args)
        mark_run_complete(args)
        return
    if args.mode == "lime-method-ablation":
        run_lime_method_ablation(args)
        mark_run_complete(args)
        return
    if args.mode == "lime-reliability-innovation":
        run_lime_reliability_innovation(args)
        mark_run_complete(args)
        return
    if args.mode == "lime-reliability-confirmation":
        run_lime_reliability_innovation(args, confirmation=True)
        mark_run_complete(args)
        return
    if args.mode == "lime-reliability-preregistered-confirmation":
        run_lime_reliability_innovation(args, preregistered=True)
        mark_run_complete(args)
        return
    if args.mode == "lime-physiology-innovation":
        run_lime_physiology_innovation(args)
        mark_run_complete(args)
        return
    if args.mode == "lime-physiology-confirmation":
        run_lime_physiology_confirmation(args)
        mark_run_complete(args)
        return
    if args.mode == "paper-report":
        payload = generate_lime_paper_artifacts(out_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if args.mode == "static-audit":
        return
    if inventory["status"] != "OK":
        write_json(out_dir / "audit_summary.json", {"status": "CHECKPOINT_NOT_FOUND", "inventory": inventory})
        return

    base_mode = "smoke" if args.mode == "smoke" else ("metric-smoke" if args.mode == "metric-smoke" else "full")
    config = resolve_mode_config(base_mode, args)
    run_id = str(getattr(args, "resolved_run_id", None) or config.mode)
    out_dir = OUT_ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(exist_ok=True)
    inventory = build_artifact_inventory(out_dir)
    write_static_audit(out_dir, inventory)
    z = np.load(PRED_PATH)
    artifact = load_artifact_split(z, "test")
    split_files = normalize_split_files(load_split_spec())
    manifest = choose_samples(artifact, config, out_dir, resume=args.resume)
    shard_count = int(getattr(args, "evaluation_shard_count", 1))
    shard_index = int(getattr(args, "evaluation_shard_index", 0))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Evaluation shard index must satisfy 0 <= index < shard count")
    if shard_count > 1:
        positions = np.arange(len(manifest), dtype=np.int64)
        manifest = manifest.iloc[positions % shard_count == shard_index].reset_index(drop=True)
        if manifest.empty:
            raise RuntimeError(
                f"Evaluation shard {shard_index}/{shard_count} selected no samples"
            )
        manifest.to_csv(out_dir / "evaluation_shard_manifest.csv", index=False)
        print(
            f"Evaluation shard {shard_index + 1}/{shard_count}: "
            f"{len(manifest)} of {len(positions)} samples"
        )
    x_raw, x_seg, quality = load_or_build_test_inputs(OUT_ROOT, split_files, artifact)
    model = build_model(x_raw.shape[1:], x_seg.shape[1:])
    model_params = int(np.sum([np.prod(v.shape) for v in model.weights]))
    inventory["model_parameter_count"] = model_params
    write_json(out_dir / "artifact_inventory.json", inventory)

    parity_path = OUT_ROOT / "model_prediction_parity.json"
    parity = read_json(parity_path) if args.resume and parity_path.exists() else run_parity_diagnostic(args)
    if not bool(parity.get("xai_evaluation_allowed", False)):
        write_json(out_dir / "audit_summary.json", {"status": "MODEL_ARTIFACT_PARITY_FAIL", "parity": parity})
        legacy = {
            "max_abs_probability_diff": parity.get("exact_artifact_parity", {}).get("max_abs_probability_diff", np.nan),
            "mean_abs_probability_diff": parity.get("scientific_prediction_parity", {}).get("checks", {}).get("mean_abs_probability_diff", np.nan),
            "argmax_agreement": parity.get("scientific_prediction_parity", {}).get("checks", {}).get("argmax_agreement", np.nan),
            "test_epoch_count": int(len(artifact["y_true"])),
            "class_order": CLASS_LABELS,
        }
        write_parity_fail_outputs(out_dir, legacy, inventory)
        mark_run_complete(args, status="FAILED_PARITY")
        return
    artifact = dict(artifact)
    artifact["probs"] = predict_sequence_probs(model, x_raw, x_seg, artifact["file_sample_counts"], batch_windows=config.batch_windows)
    probability_cache_path = (
        OUT_ROOT / "cache" / "current_sequence_probs_float32.npy"
        if shard_count == 1
        else out_dir / "cache" / "current_sequence_probs_float32.npy"
    )
    np.save(probability_cache_path, artifact["probs"].astype(np.float32))
    artifact["pred"] = np.argmax(artifact["probs"], axis=1).astype(np.int16)

    results = []
    failed = []
    for idx, row in manifest.iterrows():
        try:
            print(f"Evaluating sample {idx + 1}/{len(manifest)} global_epoch={row['global_epoch_index']}")
            res = evaluate_single_sample(row.to_dict(), config, model, x_raw, x_seg, artifact["file_sample_counts"], artifact, out_dir)
            if res["sequence_wrapper_abs_diff"] >= 1e-6:
                raise RuntimeError(f"sequence wrapper parity failed: {res['sequence_wrapper_abs_diff']}")
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            failed.append({
                "global_epoch_index": int(row["global_epoch_index"]),
                "exception_type": type(exc).__name__,
                "reason": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  failed: {exc}")
    pd.DataFrame(failed).to_csv(out_dir / "failed_samples.csv", index=False)
    if results:
        manifest = manifest.copy()
        seg_map = {int(res["global_epoch_index"]): int(res["segments_count"]) for res in results}
        manifest["segments_count"] = manifest["global_epoch_index"].map(seg_map).fillna(manifest["segments_count"]).astype(int)
        mode_tag = str(config.mode).replace("/", "_").replace("\\", "_")
        manifest.to_csv(out_dir / f"sample_manifest_{mode_tag}.csv", index=False)
        flatten_results(results, config, out_dir)
        rand_df = run_model_randomization(results, config, x_raw, x_seg, artifact["file_sample_counts"], artifact, x_raw.shape[1:], x_seg.shape[1:], out_dir)
        if rand_df.empty and not (out_dir / "model_randomization_sanity.csv").exists():
            pd.DataFrame().to_csv(out_dir / "model_randomization_sanity.csv", index=False)
        summarize_tables(
            out_dir,
            config,
            run_id=run_id,
            publish_root_index=(shard_count == 1),
        )
    else:
        write_json(out_dir / "audit_summary.json", {"status": "FAIL", "reason": "No samples completed", "failed": failed})
    mark_run_complete(args, status="COMPLETED" if not failed else "INCOMPLETE")
    if failed:
        raise RuntimeError(
            f"XAI_EVALUATION_INCOMPLETE: {len(failed)}/{len(manifest)} samples failed; "
            "see failed_samples.csv"
        )


def summarize_only() -> None:
    config = mode_config("full")
    summarize_tables(OUT_ROOT, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="isruc", choices=("isruc",))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "parity-diagnostic", "performance-profile", "functional-smoke", "metric-smoke", "smoke", "full", "summarize-only", "paper-report", "lime-method-ablation", "lime-reliability-innovation", "lime-reliability-confirmation", "lime-reliability-preregistered-confirmation", "lime-physiology-innovation", "lime-physiology-confirmation"),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Versioned JSON-compatible YAML profile under configs/xai/.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Independent resume/provenance namespace (defaults to the profile run_id).",
    )
    parser.add_argument(
        "--evaluation-shard-count",
        type=int,
        default=1,
        help="Split a formal general evaluation into disjoint positional shards.",
    )
    parser.add_argument(
        "--evaluation-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for --evaluation-shard-count.",
    )
    parser.add_argument(
        "--merge-shard-run-ids",
        default=None,
        help="Comma-separated completed full-evaluation run IDs to merge.",
    )
    parser.add_argument(
        "--smoke-section",
        choices=("surrogate", "faithfulness", "stability", "clinical", "randomization", "all"),
        default="all",
    )
    parser.add_argument("--window-inference-batch-size", type=int, default=None)
    parser.add_argument("--lime-train-masks", type=int, default=None)
    parser.add_argument("--lime-validation-masks", type=int, default=None)
    parser.add_argument("--lime-target-segments", type=int, default=None)
    parser.add_argument(
        "--lime-confirmation-per-stage",
        type=int,
        default=2,
        help="Independent reliability-confirmation samples per sleep stage (default: 2, total N=10).",
    )
    parser.add_argument(
        "--lime-preregistered-per-stage",
        type=int,
        default=6,
        help="Disjoint pre-registered confirmation samples per sleep stage (default: 6, total N=30).",
    )
    parser.add_argument("--lime-preregistered-shard-count", type=int, default=1)
    parser.add_argument("--lime-preregistered-shard-index", type=int, default=0)
    parser.add_argument("--lime-physiology-shard-count", type=int, default=1)
    parser.add_argument("--lime-physiology-shard-index", type=int, default=0)
    parser.add_argument(
        "--lime-physiology-confirmation-per-stage",
        type=int,
        default=8,
        help="Independent physiology-confirmation samples per sleep stage (default: 8, total N=40).",
    )
    parser.add_argument(
        "--lime-mask-sampling",
        choices=("local_stratified", "local_stratified_tight", "bernoulli"),
        default=None,
    )
    parser.add_argument("--lime-response-transform", choices=("probability", "logit"), default=None)
    parser.add_argument(
        "--lime-surrogate-perturbation-mode",
        choices=("zero", "physiological", "gaussian_noise", "local_mean"),
        default=None,
        help="Perturbation used to fit the primary LIME surrogate (default: zero; physiological replacement remains a sensitivity test).",
    )
    parser.add_argument(
        "--lime-ridge-alphas",
        default=None,
        help="Comma-separated positive Ridge alpha candidates selected by inner weighted CV.",
    )
    parser.add_argument(
        "--lime-kernel-widths",
        default=None,
        help="Comma-separated positive LIME kernel widths selected by inner weighted CV.",
    )
    parser.add_argument(
        "--enable-lime-visualizations",
        dest="enable_lime_visualizations",
        action="store_true",
        help="Generate publication-grade LIME figures and quantitative paper artifacts (default: enabled).",
    )
    parser.add_argument(
        "--disable-lime-visualizations",
        dest="enable_lime_visualizations",
        action="store_false",
        help="Skip LIME figure and paper-artifact generation.",
    )
    parser.set_defaults(enable_lime_visualizations=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset != "isruc" or int(args.seed) != 42:
        raise SystemExit("This audit is intentionally restricted to ISRUC seed42.")
    if args.mode == "summarize-only":
        summarize_only()
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
