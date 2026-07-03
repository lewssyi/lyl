#!/usr/bin/env python
"""Export train/val/test C10 predictions for fixed-split diagnostics.

The exporter does not train or modify the model. It loads a saved Keras model,
replays fixed train/val/test NPZ splits, and writes split-prefixed prediction
arrays consumed by tools/c10_error_diagnostics.py and C12 comparison audits.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CLASS_LABELS = ("W", "N1", "N2", "N3", "REM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved C10 Keras model and export train/val/test logits, "
            "probabilities, labels, recording ids, and epoch indices."
        )
    )
    parser.add_argument("--model", default=None, help="Saved full Keras model path (.keras/.h5/SavedModel).")
    parser.add_argument("--weights", default=None, help="Weights-only .weights.h5 path; rebuild architecture first.")
    parser.add_argument(
        "--dataset-preset",
        default=None,
        help=(
            "Preset from tools/run_main_with_protocol_split.py used for --weights mode, "
            "for example isruc_c10_late_concept_residual or edf_c10_late_concept_residual."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("sequence", "hybrid"),
        default="sequence",
        help="Architecture to rebuild in --weights mode.",
    )
    parser.add_argument("--split-json", required=True, help="Fixed protocol split.json.")
    parser.add_argument("--npz-dir", default=None, help="Override NPZ directory from split.json.")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-mode", action="store_true", default=True)
    parser.add_argument("--single-epoch", action="store_true", help="Disable sequence-window inference.")
    parser.add_argument("--seq-len", type=int, default=75)
    parser.add_argument("--sequence-stride", type=int, default=5)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--filter-profile", default="clinical_canonical")
    parser.add_argument("--normalization-mode", default="recording_robust_zscore")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--include-concepts", action="store_true")
    parser.add_argument("--concept-version", choices=("v1", "v2"), default="v1")
    return parser.parse_args()


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def parse_float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def apply_dataset_preset(preset: str | None) -> None:
    if not preset:
        return
    from tools.run_main_with_protocol_split import DATASET_PRESETS  # noqa: WPS433

    if preset not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown --dataset-preset {preset!r}. Available presets include: "
            + ", ".join(sorted(DATASET_PRESETS))
        )
    for name, value in DATASET_PRESETS[preset].items():
        os.environ[name] = str(value)


def generalization_defaults() -> tuple[bool, bool, str, float]:
    preset = os.environ.get("LFTS_GENERALIZATION_PRESET", "mixstyle_stage_coral").strip().lower()
    use_mixstyle = parse_bool_env(
        "LFTS_USE_DOMAIN_MIXSTYLE",
        preset in {
            "mixstyle",
            "mixstyle_coral",
            "mixstyle_mmd",
            "mixstyle_stage_coral",
            "mixstyle_stage_mmd",
            "mixstyle_coral_tent",
        },
    )
    use_alignment = parse_bool_env(
        "LFTS_USE_DOMAIN_ALIGNMENT",
        preset in {
            "mixstyle_coral",
            "mixstyle_mmd",
            "mixstyle_stage_coral",
            "mixstyle_stage_mmd",
            "mixstyle_coral_tent",
            "coral",
            "mmd",
            "stage_coral",
            "stage_mmd",
        },
    )
    domain_mode = os.environ.get(
        "LFTS_DOMAIN_ALIGNMENT_MODE",
        (
            "stage_mmd"
            if preset in {"stage_mmd", "mixstyle_stage_mmd"}
            else "stage_coral"
            if preset in {"stage_coral", "mixstyle_stage_coral"}
            else "mmd"
            if preset in {"mmd", "mixstyle_mmd"}
            else "coral"
        ),
    ).strip().lower()
    alignment_weight = parse_float_env(
        "LFTS_DOMAIN_ALIGNMENT_LOSS_WEIGHT",
        0.005 if use_alignment and domain_mode.startswith("stage") else 0.01 if use_alignment else 0.0,
    )
    return use_mixstyle, use_alignment, domain_mode, alignment_weight


def load_split_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def normalize_split_files(split_data: Mapping[str, object]) -> dict[str, list[str]]:
    raw = split_data.get("split_files")
    if not isinstance(raw, dict):
        raise ValueError("split.json must contain split_files.")
    normalized: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        values = raw.get(split) or raw.get(split.capitalize())
        if not isinstance(values, list):
            raise ValueError(f"split_files.{split} must be a list.")
        normalized[split] = [str(item) for item in values]
    return normalized


def resolve_npz_dir(split_data: Mapping[str, object], override: str | None) -> Path:
    raw = override or split_data.get("source_npz_dir") or split_data.get("npz_dir")
    if not raw:
        raise ValueError("NPZ directory not provided. Pass --npz-dir or include it in split.json.")
    path = Path(str(raw)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"NPZ directory does not exist: {path}")
    return path


def _coerce_file_sample_counts(total_len: int, counts: Sequence[int] | None) -> list[int]:
    if counts is None:
        return [int(total_len)]
    values = [int(count) for count in counts if int(count) > 0]
    if sum(values) != int(total_len):
        raise ValueError(f"file_sample_counts sum={sum(values)} does not match epochs={total_len}.")
    return values


def get_file_sample_counts(npz_dir: Path, files: Sequence[str]) -> np.ndarray:
    counts = []
    for filename in files:
        with np.load(npz_dir / filename, mmap_mode="r") as data:
            counts.append(int(data["x"].shape[0]))
    return np.asarray(counts, dtype=np.int64)


def recording_axes(split: str, files: Sequence[str], counts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    recording_id: list[str] = []
    epoch_index: list[int] = []
    for filename, count in zip(files, counts):
        stem = Path(filename).stem
        recording_id.extend([stem] * int(count))
        epoch_index.extend(range(int(count)))
    if not recording_id:
        return (
            np.asarray([f"{split}_recording_000"], dtype=object)[:0],
            np.asarray([], dtype=np.int64),
        )
    return np.asarray(recording_id, dtype=object), np.asarray(epoch_index, dtype=np.int64)


def build_sequence_windows(
    total_len: int,
    file_sample_counts: Sequence[int] | None,
    seq_len: int,
    stride: int,
) -> list[tuple[int, int]]:
    seq_len = max(1, int(seq_len))
    stride = max(1, int(stride))
    windows: list[tuple[int, int]] = []
    offset = 0
    for count in _coerce_file_sample_counts(total_len, file_sample_counts):
        record_start = offset
        record_end = offset + int(count)
        if count < seq_len:
            windows.append((record_start, int(count)))
            offset = record_end
            continue
        last_start = record_end - seq_len
        starts = list(range(record_start, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows.extend((start, seq_len) for start in starts)
        offset = record_end
    return windows


def window_epoch_indices(start: int, valid_len: int, seq_len: int) -> np.ndarray:
    valid_len = max(1, int(valid_len))
    seq_len = max(1, int(seq_len))
    indices = np.arange(int(start), int(start) + valid_len, dtype=np.int64)
    if valid_len < seq_len:
        pad = np.full(seq_len - valid_len, indices[-1], dtype=np.int64)
        indices = np.concatenate([indices, pad])
    return indices


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(np.sum(exp_values, axis=-1, keepdims=True), 1e-12)


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    row_sums = np.sum(probs, axis=-1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return probs / row_sums


def output_to_dict(outputs: object, output_names: Sequence[str]) -> dict[str, np.ndarray]:
    if isinstance(outputs, dict):
        return {str(key): np.asarray(value) for key, value in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    return {output_names[0]: np.asarray(outputs)}


def load_full_model(path: Path) -> tf.keras.Model:
    if path.name.endswith(".weights.h5"):
        raise ValueError(
            "This is a weights-only .weights.h5 file. Use --weights instead of --model."
        )
    from src import model_training as mt  # noqa: WPS433

    custom_objects = {
        name: value
        for name, value in vars(mt).items()
        if callable(value) and not name.startswith("__")
    }
    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)


def transition_boundary_num_classes() -> int:
    use_ot_tacb = parse_bool_env("LFTS_USE_OT_TACB", True)
    use_fine_transition = parse_bool_env("LFTS_USE_FINE_TRANSITION_AUX", True)
    return 8 if use_ot_tacb or use_fine_transition else 1


def build_model_for_weights(
    raw_shape: Sequence[int],
    segment_shape: Sequence[int],
    args: argparse.Namespace,
) -> tf.keras.Model:
    from src.model_training import create_hybrid_model_v2, create_sequence_sleep_model  # noqa: WPS433

    use_mixstyle, use_alignment, domain_mode, alignment_weight = generalization_defaults()
    learning_rate = parse_float_env("LFTS_EXPORT_LEARNING_RATE", 0.001)
    clinical_version = os.environ.get("LFTS_CLINICAL_CONCEPT_VERSION", args.concept_version).strip().lower()

    if args.architecture == "hybrid":
        return create_hybrid_model_v2(
            raw_shape=tuple(raw_shape),
            segment_shape=tuple(segment_shape),
            num_classes=len(DEFAULT_CLASS_LABELS),
            learning_rate=learning_rate,
            attention_loss_weight=parse_float_env("LFTS_ATTENTION_FEEDBACK_LOSS_WEIGHT", 0.0),
            concept_loss_weight=parse_float_env("LFTS_CONCEPT_OUTPUT_LOSS_WEIGHT", 0.0),
            concept_bottleneck_loss_weight=parse_float_env("LFTS_CONCEPT_BOTTLENECK_LOSS_WEIGHT", 0.0),
            use_channel_quality_routing=parse_bool_env("LFTS_USE_CHANNEL_QUALITY_ROUTING", False),
            channel_quality_bias_strength=parse_float_env("LFTS_CHANNEL_QUALITY_BIAS_STRENGTH", 0.35),
            loss_name=os.environ.get("LFTS_LOSS_NAME", "categorical_crossentropy"),
            use_domain_mixstyle=use_mixstyle,
            mixstyle_p=parse_float_env("LFTS_DOMAIN_MIXSTYLE_P", 0.50),
            mixstyle_alpha=parse_float_env("LFTS_DOMAIN_MIXSTYLE_ALPHA", 0.30),
            use_domain_alignment=use_alignment,
            domain_alignment_loss_weight=alignment_weight,
            domain_alignment_mode=domain_mode,
            clinical_concept_version=clinical_version,
            clinical_prior_trainable=parse_bool_env("LFTS_CLINICAL_PRIOR_TRAINABLE", False),
        )

    use_ot_tacb = parse_bool_env("LFTS_USE_OT_TACB", False)
    use_mpdg = parse_bool_env("LFTS_USE_MPDG", False)
    concept_language = bool(
        parse_bool_env("LFTS_USE_SEQUENCE_CLINICAL_CONCEPTS", False)
        and parse_bool_env("LFTS_USE_CONCEPT_LANGUAGE", False)
        and (
            parse_float_env("LFTS_CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT", 0.0) > 0.0
            or parse_float_env("LFTS_CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT", 0.0) > 0.0
        )
    )
    transition_loss = parse_float_env(
        "LFTS_TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT",
        0.020,
    )
    return create_sequence_sleep_model(
        raw_shape=tuple(raw_shape),
        segment_shape=tuple(segment_shape),
        seq_len=int(args.seq_len),
        num_classes=len(DEFAULT_CLASS_LABELS),
        learning_rate=learning_rate,
        sequence_backbone=os.environ.get("LFTS_SEQUENCE_BACKBONE", "v3_tcn").strip().lower(),
        epoch_encoder_variant=os.environ.get("LFTS_EPOCH_ENCODER_VARIANT", "v3_multiscale").strip().lower(),
        use_transformer=parse_bool_env("LFTS_SEQUENCE_USE_TRANSFORMER", True),
        loss_name=os.environ.get("LFTS_LOSS_NAME", "categorical_crossentropy"),
        use_domain_mixstyle=use_mixstyle,
        mixstyle_p=parse_float_env("LFTS_DOMAIN_MIXSTYLE_P", 0.50),
        mixstyle_alpha=parse_float_env("LFTS_DOMAIN_MIXSTYLE_ALPHA", 0.30),
        use_domain_alignment=use_alignment,
        domain_alignment_loss_weight=alignment_weight,
        domain_alignment_mode=domain_mode,
        use_clinical_concept_features=parse_bool_env("LFTS_USE_SEQUENCE_CLINICAL_CONCEPTS", False),
        clinical_concept_version=clinical_version,
        clinical_prior_trainable=parse_bool_env("LFTS_CLINICAL_PRIOR_TRAINABLE", False),
        use_late_clinical_concept_residual=parse_bool_env("LFTS_USE_LATE_CLINICAL_CONCEPT_RESIDUAL", True),
        late_concept_residual_scale=parse_float_env("LFTS_LATE_CONCEPT_RESIDUAL_SCALE", 0.05),
        late_concept_residual_trainable=parse_bool_env("LFTS_LATE_CONCEPT_RESIDUAL_TRAINABLE", True),
        late_concept_stop_gradient=parse_bool_env("LFTS_LATE_CONCEPT_STOP_GRADIENT", True),
        use_late_concept_confidence_gate=parse_bool_env("LFTS_USE_LATE_CONCEPT_CONFIDENCE_GATE", False),
        late_concept_gate_low_confidence=parse_float_env("LFTS_LATE_CONCEPT_GATE_LOW_CONFIDENCE", 0.65),
        late_concept_gate_high_confidence=parse_float_env("LFTS_LATE_CONCEPT_GATE_HIGH_CONFIDENCE", 0.85),
        late_concept_gate_mid_value=parse_float_env("LFTS_LATE_CONCEPT_GATE_MID_VALUE", 0.50),
        use_concept_language=concept_language,
        concept_loss_weight=parse_float_env("LFTS_CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT", 0.0),
        concept_bottleneck_loss_weight=parse_float_env("LFTS_CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT", 0.0),
        use_ot_tacb=use_ot_tacb,
        ot_sinkhorn_epsilon=parse_float_env("LFTS_OT_TACB_SINKHORN_EPSILON", 0.08),
        ot_sinkhorn_iterations=int(os.environ.get("LFTS_OT_TACB_SINKHORN_ITERATIONS", "8")),
        ot_cosine_weight=parse_float_env("LFTS_OT_TACB_COSINE_WEIGHT", 1.0),
        use_mpdg=use_mpdg,
        mpdg_concept_dim=int(os.environ.get("LFTS_MPDG_CONCEPT_DIM", "16")),
        dpcm_mask_amplitude=parse_float_env("LFTS_DPCM_MASK_AMPLITUDE", 1.0),
        dpcm_prior_init_scale=parse_float_env("LFTS_DPCM_PRIOR_INIT_SCALE", 0.05),
        dpcm_boundary_contrast_loss_weight=parse_float_env("LFTS_DPCM_BOUNDARY_CONTRAST_LOSS_WEIGHT", 0.015),
        dpcm_boundary_contrast_margin=parse_float_env("LFTS_DPCM_BOUNDARY_CONTRAST_MARGIN", 0.35),
        use_transition_boundary_auxiliary=parse_bool_env("LFTS_USE_TRANSITION_BOUNDARY_AUX", True)
        and transition_loss > 0.0,
        transition_boundary_loss_weight=transition_loss,
        transition_boundary_num_classes=transition_boundary_num_classes(),
        use_concept_derivative_auxiliary=False,
        concept_derivative_loss_weight=0.0,
        use_n1_supervised_contrastive=parse_bool_env("LFTS_USE_N1_SUPCON", False),
        n1_contrastive_loss_weight=parse_float_env("LFTS_N1_SUPCON_LOSS_WEIGHT", 0.040),
        n1_contrastive_projection_dim=int(os.environ.get("LFTS_N1_SUPCON_PROJECTION_DIM", "64")),
    )


def load_weights_model(
    weights_path: Path,
    raw_shape: Sequence[int],
    segment_shape: Sequence[int],
    args: argparse.Namespace,
) -> tf.keras.Model:
    model = build_model_for_weights(raw_shape, segment_shape, args)
    model.load_weights(str(weights_path))
    return model


def make_probe_model(model: tf.keras.Model) -> tuple[tf.keras.Model, list[str]]:
    layer_candidates = (
        "main_output",
        "main_output_late_concept_residual_logits",
        "main_output_logits",
        "late_td_clinical_concept_residual_logits",
        "late_td_clinical_concept_residual_logits_unscaled",
    )
    outputs = []
    names = []
    for layer_name in layer_candidates:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue
        outputs.append(layer.output)
        names.append(layer_name)
    if not outputs:
        outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
        names = ["main_output"] if len(outputs) == 1 else [f"output_{idx}" for idx in range(len(outputs))]
    return tf.keras.Model(inputs=model.inputs, outputs=outputs), names


def predict_single_epoch(
    model: tf.keras.Model,
    output_names: Sequence[str],
    x_raw: np.ndarray,
    x_segment: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in output_names}
    for start in range(0, len(x_raw), max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), len(x_raw))
        outputs = model(
            {"raw_input": x_raw[start:end], "segment_input": x_segment[start:end]},
            training=False,
        )
        output_dict = output_to_dict(outputs, output_names)
        for name in output_names:
            chunks[name].append(np.asarray(output_dict[name]))
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def predict_sequence(
    model: tf.keras.Model,
    output_names: Sequence[str],
    x_raw: np.ndarray,
    x_segment: np.ndarray,
    counts: Sequence[int],
    batch_size: int,
    seq_len: int,
    stride: int,
) -> dict[str, np.ndarray]:
    windows = build_sequence_windows(len(x_raw), counts, seq_len=seq_len, stride=stride)
    sums: dict[str, np.ndarray] = {}
    output_counts = np.zeros((len(x_raw), 1), dtype=np.float64)
    for batch_start in range(0, len(windows), max(1, int(batch_size))):
        batch_windows = windows[batch_start: batch_start + max(1, int(batch_size))]
        raw_batch = []
        segment_batch = []
        for start, valid_len in batch_windows:
            indices = window_epoch_indices(start, valid_len, seq_len)
            raw_batch.append(x_raw[indices])
            segment_batch.append(x_segment[indices])
        outputs = model(
            {
                "raw_input": np.asarray(raw_batch, dtype=np.float32),
                "segment_input": np.asarray(segment_batch, dtype=np.float32),
            },
            training=False,
        )
        output_dict = output_to_dict(outputs, output_names)
        for name, values in output_dict.items():
            values = np.asarray(values)
            if values.ndim == 2:
                values = np.repeat(values[:, np.newaxis, :], int(seq_len), axis=1)
            if values.ndim != 3:
                continue
            if name not in sums:
                sums[name] = np.zeros((len(x_raw), values.shape[-1]), dtype=np.float64)
            for row_idx, (start, valid_len) in enumerate(batch_windows):
                epoch_indices = window_epoch_indices(start, valid_len, seq_len)[:valid_len]
                sums[name][epoch_indices] += values[row_idx, :valid_len]
        for start, valid_len in batch_windows:
            epoch_indices = window_epoch_indices(start, valid_len, seq_len)[:valid_len]
            output_counts[epoch_indices] += 1.0
    missing = output_counts[:, 0] <= 0
    if np.any(missing):
        raise ValueError(f"Sequence prediction missed {int(np.sum(missing))} epoch(s).")
    return {name: values / np.maximum(output_counts, 1.0) for name, values in sums.items()}


def pick_main_prob(outputs: Mapping[str, np.ndarray]) -> tuple[str, np.ndarray]:
    for key in ("main_output", "output_0"):
        if key in outputs:
            return key, normalize_probabilities(outputs[key])
    first_key = next(iter(outputs))
    values = np.asarray(outputs[first_key])
    if values.ndim == 2 and np.all(values >= -1e-6) and np.all(values <= 1.0 + 1e-6):
        return first_key, normalize_probabilities(values)
    return first_key, softmax(values)


def split_payload(
    split: str,
    files: Sequence[str],
    counts: np.ndarray,
    y: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    quality: np.ndarray | None,
    concepts: np.ndarray | None,
) -> dict[str, np.ndarray]:
    main_key, probs = pick_main_prob(outputs)
    logits_key = "main_output_late_concept_residual_logits"
    if logits_key in outputs:
        final_logits = np.asarray(outputs[logits_key], dtype=np.float32)
    elif "main_output_logits" in outputs:
        final_logits = np.asarray(outputs["main_output_logits"], dtype=np.float32)
    else:
        final_logits = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)
    recording_id, epoch_index = recording_axes(split, files, counts)
    payload = {
        f"{split}_y_true": np.asarray(y, dtype=np.int16),
        f"{split}_probs_raw": np.asarray(probs, dtype=np.float32),
        f"{split}_final_logits": np.asarray(final_logits, dtype=np.float32),
        f"{split}_recording_id": np.asarray(recording_id),
        f"{split}_epoch_index": np.asarray(epoch_index, dtype=np.int32),
        f"{split}_file_sample_counts": np.asarray(counts, dtype=np.int32),
        f"{split}_files": np.asarray(list(files)),
        f"{split}_main_probability_key": np.asarray(main_key),
    }
    if "main_output_logits" in outputs:
        payload[f"{split}_base_logits"] = np.asarray(outputs["main_output_logits"], dtype=np.float32)
    if "late_td_clinical_concept_residual_logits" in outputs:
        payload[f"{split}_late_concept_residual_logits"] = np.asarray(
            outputs["late_td_clinical_concept_residual_logits"],
            dtype=np.float32,
        )
    if "late_td_clinical_concept_residual_logits_unscaled" in outputs:
        payload[f"{split}_late_concept_residual_logits_unscaled"] = np.asarray(
            outputs["late_td_clinical_concept_residual_logits_unscaled"],
            dtype=np.float32,
        )
    if quality is not None:
        payload[f"{split}_quality"] = np.asarray(quality, dtype=np.float32)
    if concepts is not None:
        payload[f"{split}_clinical_concept_proxy"] = np.asarray(concepts, dtype=np.float32)
    return payload


def default_output_path(dataset: str) -> Path:
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(dataset).lower()) or "unknown"
    return PROJECT_ROOT / "outputs" / "predictions" / f"c10a_split_predictions_{safe}_{timestamp}.npz"


def main() -> None:
    args = parse_args()
    if bool(args.model) == bool(args.weights):
        raise SystemExit("Pass exactly one of --model or --weights.")
    apply_dataset_preset(args.dataset_preset)

    split_path = Path(args.split_json).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve() if args.model else None
    weights_path = Path(args.weights).expanduser().resolve() if args.weights else None
    split_data = load_split_json(split_path)
    split_files = normalize_split_files(split_data)
    npz_dir = resolve_npz_dir(split_data, args.npz_dir)

    from src.data_processing import load_npz_data  # noqa: WPS433
    from src.lime_explanation import extract_segment_features, semantic_segmentation  # noqa: WPS433
    from src.model_training import extract_clinical_concept_targets_from_segment_features  # noqa: WPS433

    probe_model: tf.keras.Model | None = None
    output_names: list[str] | None = None
    model_source = str(model_path) if model_path is not None else ""
    weights_source = str(weights_path) if weights_path is not None else ""

    payload: dict[str, np.ndarray] = {
        "dataset": np.asarray(args.dataset),
        "model_path": np.asarray(model_source),
        "weights_path": np.asarray(weights_source),
        "dataset_preset": np.asarray(str(args.dataset_preset or "")),
        "architecture": np.asarray(str(args.architecture)),
        "split_json": np.asarray(str(split_path)),
        "npz_dir": np.asarray(str(npz_dir)),
        "class_labels": np.asarray(DEFAULT_CLASS_LABELS),
        "sequence_mode": np.asarray(not bool(args.single_epoch)),
        "seq_len": np.asarray(int(args.seq_len), dtype=np.int32),
        "sequence_stride": np.asarray(int(args.sequence_stride), dtype=np.int32),
    }

    for split in ("train", "val", "test"):
        files = split_files[split]
        counts = get_file_sample_counts(npz_dir, files)
        print(f"\nLoading {split}: files={len(files)} epochs={int(np.sum(counts))}")
        loaded = load_npz_data(
            str(npz_dir),
            file_subset=files,
            preprocess=not bool(args.no_preprocess),
            filter_profile=args.filter_profile,
            normalization_mode=args.normalization_mode,
            return_quality=not bool(args.no_quality),
        )
        if args.no_quality:
            x_raw, y = loaded
            quality = None
        else:
            x_raw, y, quality = loaded

        x_segment = []
        for instance in x_raw:
            segments = semantic_segmentation(instance, target_segments=30)
            x_segment.append(
                extract_segment_features(
                    instance,
                    segments,
                    max_segments=30,
                    include_spectral=True,
                )
            )
        x_segment = np.asarray(x_segment, dtype=np.float32)

        concepts = None
        if args.include_concepts:
            concepts = extract_clinical_concept_targets_from_segment_features(
                x_segment,
                labels=None,
                concept_prior_weight=0.0,
                concept_version=args.concept_version,
            )

        if probe_model is None or output_names is None:
            if model_path is not None:
                model = load_full_model(model_path)
                print(f"Loaded full model: {model_path}")
            else:
                assert weights_path is not None
                if not weights_path.exists():
                    raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
                model = load_weights_model(
                    weights_path,
                    raw_shape=x_raw.shape[1:],
                    segment_shape=x_segment.shape[1:],
                    args=args,
                )
                print(
                    "Rebuilt model and loaded weights: "
                    f"{weights_path} | preset={args.dataset_preset or 'env/manual'} "
                    f"architecture={args.architecture}"
                )
            probe_model, output_names = make_probe_model(model)
            print(f"Probe outputs: {', '.join(output_names)}")

        if args.single_epoch:
            outputs = predict_single_epoch(
                probe_model,
                output_names,
                x_raw,
                x_segment,
                batch_size=args.batch_size,
            )
        else:
            outputs = predict_sequence(
                probe_model,
                output_names,
                x_raw,
                x_segment,
                counts=counts,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                stride=args.sequence_stride,
            )

        payload.update(split_payload(split, files, counts, y, outputs, quality, concepts))
        print(f"Exported {split}: probs={payload[f'{split}_probs_raw'].shape}")

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args.dataset).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    print(f"\nSaved split predictions: {output_path}")


if __name__ == "__main__":
    main()
