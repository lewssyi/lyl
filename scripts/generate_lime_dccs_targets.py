#!/usr/bin/env python
"""Precompute reliable LIME-DCCS feedback targets for run_main.py.

This script is intentionally separate from training. It uses an already trained
baseline/clinical model to predict train/validation samples, then runs the
expensive LIME-DCCS explanation only on reliable samples by default:

    predicted_label == true_label and confidence >= --min-confidence

Rows that are not explained are filled with a uniform channel target and receive
mask=0 in the saved .npz file. run_main.py will then keep main_output training on
all samples while applying channel_attention_weights feedback only to rows whose
mask/correct/confidence pass the filter.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf  # noqa: E402

try:  # noqa: E402
    from src.data_processing import (  # noqa: E402
        preprocess_multichannel_epochs as project_preprocess_multichannel_epochs,
        sanitize_signal_epochs as project_sanitize_signal_epochs,
    )
except Exception as exc:  # noqa: BLE001 - preprocessing is optional for this standalone generator.
    project_preprocess_multichannel_epochs = None
    project_sanitize_signal_epochs = None
    DATA_PROCESSING_IMPORT_ERROR = exc
else:
    DATA_PROCESSING_IMPORT_ERROR = None

from src.lime_explanation import (  # noqa: E402
    extract_segment_features,
    run_multimodal_lime,
    semantic_segmentation,
)
from src.model_training import (  # noqa: E402
    concept_bottleneck_loss,
    create_hybrid_model_v2,
    explanation_feedback_attention_loss,
)


NUM_CLASSES = 5
CLASS_LABELS = ("W", "N1", "N2", "N3", "REM")
CURRENT_RUN_MAIN_OUTPUTS = {
    "main_output",
    "channel_attention_weights",
    "concept_output",
    "concept_bottleneck_output",
}
DEFAULT_DATA_DIR = PROJECT_DIR.parent / "sleepEDF" / "multichannel_extracted"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "outputs" / "lime_dccs_targets.npz"
DEFAULT_WEIGHTS_PATH = PROJECT_DIR / "outputs" / "current_create_hybrid_model_v2.weights.h5"


class MainHeadWrapper:
    """LIME expects model.predict([raw, segment]) to return main probabilities."""

    def __init__(self, model, batch_size=64, expected_segment_features=None):
        self.model = model
        self.batch_size = int(batch_size)
        self.expected_segment_features = expected_segment_features

    def predict(self, inputs, **kwargs):
        verbose = kwargs.pop("verbose", 0)
        batch_size = int(kwargs.pop("batch_size", self.batch_size))
        inputs = adapt_model_inputs(inputs, expected_segment_features=self.expected_segment_features)
        outputs = self.model.predict(inputs, batch_size=batch_size, verbose=verbose, **kwargs)
        return extract_main_probabilities(outputs)


@contextlib.contextmanager
def maybe_suppress_stdout(verbose):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _as_3d_float(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., np.newaxis]
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (epochs, samples, channels), got {X.shape}")
    return X


def sanitize_signal_epochs_local(X):
    X = _as_3d_float(X)
    if project_sanitize_signal_epochs is not None:
        return project_sanitize_signal_epochs(X)
    return np.nan_to_num(X, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess_multichannel_epochs_local(X, fs=100.0):
    X = _as_3d_float(X)
    if project_preprocess_multichannel_epochs is not None:
        return project_preprocess_multichannel_epochs(
            X,
            fs=fs,
            apply_filters=True,
            normalize=True,
            clip_sigma=None,
            enable_amplitude_clipping=False,
        )

    if DATA_PROCESSING_IMPORT_ERROR is not None:
        print(
            "WARNING: src.data_processing preprocessing helpers could not be imported; "
            f"using numerical sanitization only. Reason: {DATA_PROCESSING_IMPORT_ERROR}"
        )
    return sanitize_signal_epochs_local(X)


def get_npz_file_list(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    return sorted(path.name for path in folder_path.iterdir() if path.suffix == ".npz")


def split_npz_files(files, test_size=0.15, val_size=0.15, seed=42):
    files = np.asarray(sorted(files))
    if len(files) == 0:
        return [], [], []

    rng = np.random.default_rng(seed)
    shuffled = files[rng.permutation(len(files))]

    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_files = sorted(shuffled[:n_test].tolist())
    train_val_files = shuffled[n_test:]

    n_val = max(1, int(round(len(train_val_files) * val_size))) if len(train_val_files) else 0
    val_files = sorted(train_val_files[:n_val].tolist())
    train_files = sorted(train_val_files[n_val:].tolist())
    return train_files, val_files, test_files


def load_npz_data(folder_path, file_subset=None, max_files=None, preprocess=False):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if file_subset is None:
        files = get_npz_file_list(folder_path)
    else:
        files = sorted(str(filename) for filename in file_subset if str(filename).endswith(".npz"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise ValueError(f"No .npz files found in {folder_path}")

    print(f"Loading {len(files)} files from {folder_path}...")
    total_samples = 0
    sample_shape = None
    for filename in files:
        with np.load(folder_path / filename, mmap_mode="r") as data:
            x_shape = _as_3d_float(data["x"]).shape
            total_samples += int(x_shape[0])
            if sample_shape is None:
                sample_shape = x_shape[1:]

    print(f"Scan complete, total samples: {total_samples}. Preallocating memory...")
    X = np.empty((total_samples, *sample_shape), dtype=np.float32)
    y = np.empty((total_samples,), dtype=np.int32)

    current_idx = 0
    for file_idx, filename in enumerate(files):
        with np.load(folder_path / filename) as data:
            x_data = _as_3d_float(data["x"])
            y_data = np.asarray(data["y"], dtype=np.int32).reshape(-1)
            fs = float(data["fs"]) if "fs" in data.files else 100.0

        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(f"{filename}: x/y sample count mismatch: {x_data.shape[0]} vs {y_data.shape[0]}")
        if x_data.shape[1:] != sample_shape:
            raise ValueError(f"{filename}: expected sample shape {sample_shape}, got {x_data.shape[1:]}")

        if preprocess:
            x_data = preprocess_multichannel_epochs_local(x_data, fs=fs)
        else:
            x_data = sanitize_signal_epochs_local(x_data)

        next_idx = current_idx + x_data.shape[0]
        X[current_idx:next_idx] = x_data
        y[current_idx:next_idx] = y_data
        current_idx = next_idx

        if (file_idx + 1) % 5 == 0 or (file_idx + 1) == len(files):
            print(f"Progress: loaded {file_idx + 1}/{len(files)} files.")

    return X, y


def infer_raw_shape_from_npz(folder_path, files):
    if not files:
        raise ValueError("Cannot infer raw shape because the file list is empty.")
    with np.load(Path(folder_path) / files[0], mmap_mode="r") as data:
        return tuple(int(dim) for dim in _as_3d_float(data["x"]).shape[1:])


def segment_feature_dim(n_channels, include_spectral=True):
    # Base features: length + mean/std/max/energy per channel.
    per_channel_features = 4 + (5 if include_spectral else 0)
    return int(1 + int(n_channels) * per_channel_features)


def infer_include_spectral_from_segment_shape(segment_shape, n_channels):
    if segment_shape is None or len(segment_shape) < 2 or segment_shape[-1] is None:
        return None
    feature_dim = int(segment_shape[-1])
    if feature_dim == segment_feature_dim(n_channels, include_spectral=True):
        return True
    if feature_dim == segment_feature_dim(n_channels, include_spectral=False):
        return False
    return None


def get_model_segment_shape(model):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and len(input_shape) >= 2:
        shape = input_shape[1][1:]
    elif isinstance(input_shape, tuple) and len(input_shape) >= 2 and isinstance(input_shape[0], (list, tuple)):
        shape = input_shape[1][1:]
    else:
        return None
    return tuple(None if dim is None else int(dim) for dim in shape)


def adapt_segment_features(X_segment, expected_segment_features=None):
    if expected_segment_features is None:
        return X_segment
    X_segment = np.asarray(X_segment, dtype=np.float32)
    expected_segment_features = int(expected_segment_features)
    current_features = int(X_segment.shape[-1])
    if current_features == expected_segment_features:
        return X_segment
    if current_features > expected_segment_features:
        return X_segment[..., :expected_segment_features]
    pad_width = [(0, 0)] * X_segment.ndim
    pad_width[-1] = (0, expected_segment_features - current_features)
    return np.pad(X_segment, pad_width, mode="constant").astype(np.float32)


def adapt_model_inputs(inputs, expected_segment_features=None):
    if expected_segment_features is None or not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        return inputs
    adapted = list(inputs)
    adapted[1] = adapt_segment_features(adapted[1], expected_segment_features=expected_segment_features)
    return adapted


def validate_current_run_main_model(model, model_path):
    output_names = set(getattr(model, "output_names", []) or [])
    if CURRENT_RUN_MAIN_OUTPUTS.issubset(output_names):
        return

    tensor_names = {str(getattr(output, "name", "")) for output in model.outputs}
    if all(any(name in tensor_name for tensor_name in tensor_names) for name in CURRENT_RUN_MAIN_OUTPUTS):
        return

    raise ValueError(
        f"Model {model_path} does not match the current run_main.py architecture. "
        f"Expected output heads {sorted(CURRENT_RUN_MAIN_OUTPUTS)}, got "
        f"output_names={sorted(output_names) or 'unknown'} and tensors={sorted(tensor_names)}. "
        "Use a model saved from the current create_hybrid_model_v2 only."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate outputs/lime_dccs_targets.npz for filtered LIME-DCCS "
            "attention feedback training."
        )
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="SleepEDF multichannel .npz directory.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_WEIGHTS_PATH) if DEFAULT_WEIGHTS_PATH.exists() else "",
        help=(
            "Trained model file or weights file. Use a full .keras/.h5 model, or "
            "outputs/current_create_hybrid_model_v2.weights.h5 saved by run_main.py."
        ),
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Output .npz path.")
    parser.add_argument("--seed", type=int, default=42, help="Split and sampling seed. Must match run_main.py.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Held-out test file ratio. Must match run_main.py.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation ratio after test split.")
    parser.add_argument("--batch-size", type=int, default=64, help="Prediction batch size.")
    parser.add_argument("--target-segments", type=int, default=30, help="Number of semantic segments.")
    parser.add_argument("--num-perturbations", type=int, default=80, help="LIME perturbations per explained sample.")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Reliable-sample confidence threshold.")
    parser.add_argument(
        "--max-train-lime-samples",
        type=int,
        default=1000,
        help="Class-balanced cap for train LIME samples. Use 0 to explain every eligible train sample.",
    )
    parser.add_argument(
        "--max-val-lime-samples",
        type=int,
        default=0,
        help="Class-balanced cap for validation LIME samples. Default 0 because run_main disables val feedback.",
    )
    parser.add_argument(
        "--raw-dccs",
        action="store_true",
        help="Save raw_dccs instead of calibrated_dccs as the LIME target.",
    )
    parser.add_argument(
        "--preprocess-signals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same signal preprocessing used by run_main.py.",
    )
    parser.add_argument(
        "--include-spectral-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include spectral segment features. Should match run_main.py.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output .npz.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from output-path.partial.npz if a previous LIME generation run was interrupted.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save a partial checkpoint after this many newly processed LIME samples. Use 0 to disable.",
    )
    parser.add_argument("--verbose-lime", action="store_true", help="Do not suppress per-sample LIME prints.")
    parser.add_argument(
        "--save-uncompressed",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for faster saving and larger files.",
    )
    return parser.parse_args()


def normalize_channel_target(values, n_channels=3):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values[:n_channels], 0.0, None)
    if values.shape[0] < n_channels:
        values = np.pad(values, (0, n_channels - values.shape[0]))
    total = float(np.sum(values))
    if total <= 1e-8:
        return np.ones(n_channels, dtype=np.float32) / float(n_channels)
    return (values / total).astype(np.float32)


def build_segment_feature_matrix(X_raw, target_segments=30, include_spectral=True):
    X_segments = []
    for instance in tqdm(X_raw, desc="segment feature extraction"):
        segments = semantic_segmentation(instance, target_segments=target_segments)
        X_segments.append(
            extract_segment_features(
                instance,
                segments,
                max_segments=target_segments,
                include_spectral=include_spectral,
            )
        )
    return np.asarray(X_segments, dtype=np.float32)


def load_prediction_model(model_path, raw_shape, target_segments, preferred_include_spectral, batch_size):
    model_path = str(model_path or "").strip()
    if not model_path:
        raise ValueError(
            "No model path was provided. Pass --model-path, for example "
            "outputs/current_create_hybrid_model_v2.weights.h5 or a trained .keras model."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    custom_objects = {
        "explanation_feedback_attention_loss": explanation_feedback_attention_loss,
        "concept_bottleneck_loss": concept_bottleneck_loss,
    }

    if model_path.endswith(".weights.h5"):
        n_channels = int(raw_shape[-1])
        segment_shapes = []
        for include_spectral in [preferred_include_spectral, not preferred_include_spectral]:
            candidate = (int(target_segments), segment_feature_dim(n_channels, include_spectral=include_spectral))
            if candidate not in segment_shapes:
                segment_shapes.append(candidate)

        errors = []
        for segment_shape in segment_shapes:
            try:
                tf.keras.backend.clear_session()
                model = create_hybrid_model_v2(
                    raw_shape=raw_shape,
                    segment_shape=segment_shape,
                    num_classes=NUM_CLASSES,
                    attention_loss_weight=0.0,
                    concept_loss_weight=0.0,
                    concept_bottleneck_loss_weight=0.0,
                )
                model.load_weights(model_path)
                validate_current_run_main_model(model, model_path)
                print(
                    f"Loaded weights into current create_hybrid_model_v2: {model_path} "
                    f"(raw_shape={raw_shape}, segment_shape={segment_shape})"
                )
                return model
            except Exception as exc:  # noqa: BLE001 - report all architecture attempts below.
                errors.append(f"segment_shape={segment_shape}: {type(exc).__name__}: {exc}")

        detail = "\n  - ".join(errors)
        raise ValueError(
            "The .weights.h5 file is not compatible with the current run_main.py model. "
            "This generator intentionally only uses the current create_hybrid_model_v2 "
            "architecture: outputs=[main_output, channel_attention_weights, "
            "concept_output, concept_bottleneck_output]. Do not use older or different "
            "architecture weights here; they would select reliable samples from a different model. "
            "Pass a full current .keras/.h5 model or regenerate a current-compatible "
            "*.weights.h5 file first.\n"
            f"Current-architecture load attempts:\n  - {detail}"
        )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )
    validate_current_run_main_model(model, model_path)
    print(f"Loaded full Keras model: {model_path}")
    _ = batch_size
    return model


def extract_main_probabilities(outputs):
    if isinstance(outputs, dict):
        if "main_output" in outputs:
            return np.asarray(outputs["main_output"], dtype=np.float32)
        first_key = next(iter(outputs))
        return np.asarray(outputs[first_key], dtype=np.float32)
    if isinstance(outputs, (list, tuple)):
        return np.asarray(outputs[0], dtype=np.float32)
    return np.asarray(outputs, dtype=np.float32)


def predict_main_probabilities(model, X_raw, X_segment, batch_size):
    outputs = model.predict([X_raw, X_segment], batch_size=batch_size, verbose=1)
    probabilities = extract_main_probabilities(outputs)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected main probabilities with shape (samples, {NUM_CLASSES}), got {probabilities.shape}")
    return probabilities.astype(np.float32)


def make_uniform_targets(n_samples, n_channels=3):
    return np.ones((int(n_samples), int(n_channels)), dtype=np.float32) / float(n_channels)


def partial_checkpoint_path(output_path):
    output_path = Path(output_path)
    if output_path.suffix == ".npz":
        return output_path.with_name(f"{output_path.stem}.partial.npz")
    return output_path.with_name(f"{output_path.name}.partial.npz")


def _validate_partial_targets(values, expected_len, split_name):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (expected_len, 3):
        raise ValueError(
            f"Partial checkpoint {split_name} targets shape mismatch: "
            f"expected {(expected_len, 3)}, got {values.shape}"
        )
    return values


def _validate_partial_mask(values, expected_len, split_name):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.shape[0] != expected_len:
        raise ValueError(
            f"Partial checkpoint {split_name} mask length mismatch: "
            f"expected {expected_len}, got {values.shape[0]}"
        )
    return values


def load_partial_checkpoint(path, train_len, val_len):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as data:
        checkpoint = {
            "train_targets": _validate_partial_targets(
                data["train_lime_dccs_targets"],
                train_len,
                "train",
            ),
            "val_targets": _validate_partial_targets(
                data["val_lime_dccs_targets"],
                val_len,
                "validation",
            ),
            "train_mask": _validate_partial_mask(data["train_mask"], train_len, "train"),
            "val_mask": _validate_partial_mask(data["val_mask"], val_len, "validation"),
        }
    print(
        f"Resuming partial checkpoint: {path} | "
        f"train_done={int(np.sum(checkpoint['train_mask'] > 0.0))}/{train_len}, "
        f"val_done={int(np.sum(checkpoint['val_mask'] > 0.0))}/{val_len}"
    )
    return checkpoint


def save_lime_checkpoint(
    path,
    train_targets,
    train_mask,
    val_targets,
    val_mask,
    train_failures=None,
    val_failures=None,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_lime_dccs_targets=np.asarray(train_targets, dtype=np.float32),
        val_lime_dccs_targets=np.asarray(val_targets, dtype=np.float32),
        train_mask=np.asarray(train_mask, dtype=np.float32),
        val_mask=np.asarray(val_mask, dtype=np.float32),
        train_failures_json=np.asarray(json.dumps(train_failures or [], ensure_ascii=False)),
        val_failures_json=np.asarray(json.dumps(val_failures or [], ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False, indent=2)),
    )
    print(
        f"Saved partial checkpoint: {path} | "
        f"train_done={int(np.sum(np.asarray(train_mask) > 0.0))}, "
        f"val_done={int(np.sum(np.asarray(val_mask) > 0.0))}"
    )


def select_lime_indices(labels, predicted_labels, confidence, min_confidence, max_samples, seed):
    labels = np.asarray(labels, dtype=np.int32)
    eligible = np.where((predicted_labels == labels) & (confidence >= float(min_confidence)))[0]
    if max_samples is None or int(max_samples) <= 0 or len(eligible) <= int(max_samples):
        return eligible.astype(np.int64), eligible.astype(np.int64)

    rng = np.random.default_rng(seed)
    max_samples = int(max_samples)
    selected_parts = []
    per_class = max(1, max_samples // NUM_CLASSES)

    for class_idx in range(NUM_CLASSES):
        class_indices = eligible[labels[eligible] == class_idx]
        if len(class_indices) == 0:
            continue
        take = min(per_class, len(class_indices))
        selected_parts.append(rng.choice(class_indices, size=take, replace=False))

    selected = np.concatenate(selected_parts) if selected_parts else np.empty((0,), dtype=np.int64)
    remaining_budget = max_samples - len(selected)
    if remaining_budget > 0:
        selected_set = set(selected.tolist())
        remaining = np.array([idx for idx in eligible if int(idx) not in selected_set], dtype=np.int64)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(remaining_budget, len(remaining)), replace=False)
            selected = np.concatenate([selected, extra])

    selected = np.sort(selected.astype(np.int64))
    return eligible.astype(np.int64), selected


def explain_selected_samples(
    split_name,
    model_wrapper,
    X_raw,
    selected_indices,
    target_segments,
    num_perturbations,
    use_raw_dccs,
    verbose_lime,
    targets=None,
    mask=None,
    failed_indices=None,
    checkpoint_callback=None,
    checkpoint_every=0,
):
    targets = make_uniform_targets(len(X_raw)) if targets is None else np.asarray(targets, dtype=np.float32)
    mask = np.zeros((len(X_raw),), dtype=np.float32) if mask is None else np.asarray(mask, dtype=np.float32)
    failed_indices = [] if failed_indices is None else failed_indices
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    already_done = int(np.sum(mask[selected_indices] > 0.0)) if len(selected_indices) else 0
    if already_done:
        print(f"{split_name}: resuming, {already_done}/{len(selected_indices)} selected LIME samples already done.")

    processed_since_checkpoint = 0
    for sample_idx in tqdm(selected_indices, desc=f"{split_name} LIME-DCCS"):
        sample_idx = int(sample_idx)
        if mask[sample_idx] > 0.0:
            continue
        try:
            instance = X_raw[sample_idx]
            segments = semantic_segmentation(instance, target_segments=target_segments)
            with maybe_suppress_stdout(verbose_lime):
                explanation = run_multimodal_lime(
                    model=model_wrapper,
                    instance_multimodal=instance,
                    segments=segments,
                    num_perturbations=num_perturbations,
                )
            dccs_key = "raw_dccs" if use_raw_dccs else "calibrated_dccs"
            targets[sample_idx] = normalize_channel_target(explanation[dccs_key], n_channels=3)
            mask[sample_idx] = 1.0
        except Exception as exc:  # noqa: BLE001 - keep long generation jobs alive.
            failed_indices.append((sample_idx, str(exc)))

        processed_since_checkpoint += 1
        if (
            checkpoint_callback is not None
            and int(checkpoint_every) > 0
            and processed_since_checkpoint >= int(checkpoint_every)
        ):
            checkpoint_callback()
            processed_since_checkpoint = 0

    if checkpoint_callback is not None and processed_since_checkpoint > 0:
        checkpoint_callback()

    if failed_indices:
        print(f"{split_name}: {len(failed_indices)} LIME explanations failed; those rows keep mask=0.")
        for sample_idx, message in failed_indices[:10]:
            print(f"  failed index {sample_idx}: {message}")
    return targets, mask, failed_indices


def summarize_split(name, labels, predicted_labels, confidence, eligible_indices, selected_indices):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    correct = int(np.sum(predicted_labels == labels))
    eligible = int(len(eligible_indices))
    selected = int(len(selected_indices))
    summary = ", ".join(f"{label}:{count}" for label, count in zip(CLASS_LABELS, counts))
    print(
        f"{name}: samples={len(labels)}, distribution=({summary}), "
        f"correct={correct} ({correct / max(len(labels), 1):.1%}), "
        f"confidence_mean={float(np.mean(confidence)):.3f}, "
        f"eligible_correct_high_conf={eligible} ({eligible / max(len(labels), 1):.1%}), "
        f"lime_explained={selected}"
    )


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    checkpoint_path = partial_checkpoint_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    print("LIME-DCCS target generation")
    print(f"  project: {PROJECT_DIR}")
    print(f"  data_dir: {args.data_dir}")
    print(f"  model_path: {args.model_path or 'missing'}")
    print(f"  output_path: {output_path}")
    print(f"  min_confidence: {args.min_confidence}")
    print(f"  train_lime_cap: {args.max_train_lime_samples}")
    print(f"  val_lime_cap: {args.max_val_lime_samples}")
    print(f"  perturbations: {args.num_perturbations}")
    print(f"  resume: {args.resume}")
    print(f"  checkpoint_path: {checkpoint_path}")
    print(f"  checkpoint_every: {args.checkpoint_every}")

    all_files = get_npz_file_list(args.data_dir)
    train_files, val_files, test_files = split_npz_files(
        all_files,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(f"Found {len(all_files)} files. train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    print(f"Train files: {train_files}")
    print(f"Val files:   {val_files}")

    raw_shape = infer_raw_shape_from_npz(args.data_dir, train_files or all_files)
    model = load_prediction_model(
        args.model_path,
        raw_shape=raw_shape,
        target_segments=args.target_segments,
        preferred_include_spectral=args.include_spectral_features,
        batch_size=args.batch_size,
    )
    model_segment_shape = get_model_segment_shape(model)
    if model_segment_shape is None:
        effective_target_segments = args.target_segments
        effective_include_spectral = args.include_spectral_features
        expected_segment_features = None
    else:
        effective_target_segments = int(model_segment_shape[0]) if model_segment_shape[0] is not None else args.target_segments
        expected_segment_features = int(model_segment_shape[-1]) if model_segment_shape[-1] is not None else None
        inferred_include_spectral = infer_include_spectral_from_segment_shape(model_segment_shape, raw_shape[-1])
        effective_include_spectral = (
            args.include_spectral_features
            if inferred_include_spectral is None
            else bool(inferred_include_spectral)
        )
    print(
        "Model input alignment: "
        f"raw_shape={raw_shape}, "
        f"segment_shape={model_segment_shape or 'unknown'}, "
        f"target_segments={effective_target_segments}, "
        f"include_spectral_features={effective_include_spectral}"
    )

    X_train_raw, y_train_raw = load_npz_data(
        args.data_dir,
        file_subset=train_files,
        preprocess=args.preprocess_signals,
    )
    X_val_raw, y_val_raw = load_npz_data(
        args.data_dir,
        file_subset=val_files,
        preprocess=args.preprocess_signals,
    )

    X_train_seg = build_segment_feature_matrix(
        X_train_raw,
        target_segments=effective_target_segments,
        include_spectral=effective_include_spectral,
    )
    X_val_seg = build_segment_feature_matrix(
        X_val_raw,
        target_segments=effective_target_segments,
        include_spectral=effective_include_spectral,
    )
    X_train_seg = adapt_segment_features(X_train_seg, expected_segment_features=expected_segment_features)
    X_val_seg = adapt_segment_features(X_val_seg, expected_segment_features=expected_segment_features)
    model_wrapper = MainHeadWrapper(
        model,
        batch_size=args.batch_size,
        expected_segment_features=expected_segment_features,
    )

    print("Predicting train split...")
    train_probs = predict_main_probabilities(model, X_train_raw, X_train_seg, batch_size=args.batch_size)
    print("Predicting validation split...")
    val_probs = predict_main_probabilities(model, X_val_raw, X_val_seg, batch_size=args.batch_size)

    train_confidence = np.max(train_probs, axis=1).astype(np.float32)
    val_confidence = np.max(val_probs, axis=1).astype(np.float32)
    train_predicted_labels = np.argmax(train_probs, axis=1).astype(np.int32)
    val_predicted_labels = np.argmax(val_probs, axis=1).astype(np.int32)
    train_correct = np.equal(train_predicted_labels, y_train_raw).astype(np.float32)
    val_correct = np.equal(val_predicted_labels, y_val_raw).astype(np.float32)

    train_eligible, train_selected = select_lime_indices(
        y_train_raw,
        train_predicted_labels,
        train_confidence,
        min_confidence=args.min_confidence,
        max_samples=args.max_train_lime_samples,
        seed=args.seed,
    )
    val_eligible, val_selected = select_lime_indices(
        y_val_raw,
        val_predicted_labels,
        val_confidence,
        min_confidence=args.min_confidence,
        max_samples=args.max_val_lime_samples,
        seed=args.seed + 1,
    )
    if args.max_val_lime_samples <= 0:
        val_selected = np.empty((0,), dtype=np.int64)
    summarize_split("Train", y_train_raw, train_predicted_labels, train_confidence, train_eligible, train_selected)
    summarize_split("Validation", y_val_raw, val_predicted_labels, val_confidence, val_eligible, val_selected)

    partial = load_partial_checkpoint(checkpoint_path, len(y_train_raw), len(y_val_raw)) if args.resume else None
    if partial is None:
        train_targets = make_uniform_targets(len(y_train_raw))
        train_mask = np.zeros((len(y_train_raw),), dtype=np.float32)
        val_targets = make_uniform_targets(len(y_val_raw))
        val_mask = np.zeros((len(y_val_raw),), dtype=np.float32)
    else:
        train_targets = partial["train_targets"]
        train_mask = partial["train_mask"]
        val_targets = partial["val_targets"]
        val_mask = partial["val_mask"]

    train_failures = []
    val_failures = []
    checkpoint_metadata = {
        "output_path": str(output_path),
        "model_path": str(args.model_path),
        "seed": int(args.seed),
        "target_segments": int(effective_target_segments),
        "include_spectral_features": bool(effective_include_spectral),
        "num_perturbations": int(args.num_perturbations),
        "min_confidence": float(args.min_confidence),
        "max_train_lime_samples": int(args.max_train_lime_samples),
        "max_val_lime_samples": int(args.max_val_lime_samples),
        "train_selected_count": int(len(train_selected)),
        "val_selected_count": int(len(val_selected)),
    }

    def save_checkpoint():
        save_lime_checkpoint(
            checkpoint_path,
            train_targets,
            train_mask,
            val_targets,
            val_mask,
            train_failures=train_failures,
            val_failures=val_failures,
            metadata=checkpoint_metadata,
        )

    start = time.time()
    train_targets, train_mask, train_failures = explain_selected_samples(
        "Train",
        model_wrapper,
        X_train_raw,
        train_selected,
        target_segments=effective_target_segments,
        num_perturbations=args.num_perturbations,
        use_raw_dccs=args.raw_dccs,
        verbose_lime=args.verbose_lime,
        targets=train_targets,
        mask=train_mask,
        failed_indices=train_failures,
        checkpoint_callback=save_checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    val_targets, val_mask, val_failures = explain_selected_samples(
        "Validation",
        model_wrapper,
        X_val_raw,
        val_selected,
        target_segments=effective_target_segments,
        num_perturbations=args.num_perturbations,
        use_raw_dccs=args.raw_dccs,
        verbose_lime=args.verbose_lime,
        targets=val_targets,
        mask=val_mask,
        failed_indices=val_failures,
        checkpoint_callback=save_checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - start

    metadata = {
        "created_by": "scripts/generate_lime_dccs_targets.py",
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "target_segments": int(effective_target_segments),
        "include_spectral_features": bool(effective_include_spectral),
        "model_segment_shape": list(model_segment_shape) if model_segment_shape is not None else None,
        "num_perturbations": int(args.num_perturbations),
        "min_confidence": float(args.min_confidence),
        "max_train_lime_samples": int(args.max_train_lime_samples),
        "max_val_lime_samples": int(args.max_val_lime_samples),
        "dccs_kind": "raw_dccs" if args.raw_dccs else "calibrated_dccs",
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_eligible_count": int(len(train_eligible)),
        "train_explained_count": int(np.sum(train_mask > 0.0)),
        "val_eligible_count": int(len(val_eligible)),
        "val_explained_count": int(np.sum(val_mask > 0.0)),
        "train_failure_count": int(len(train_failures)),
        "val_failure_count": int(len(val_failures)),
        "elapsed_seconds": float(elapsed),
    }

    save_fn = np.savez if args.save_uncompressed else np.savez_compressed
    save_fn(
        output_path,
        train_lime_dccs_targets=train_targets.astype(np.float32),
        val_lime_dccs_targets=val_targets.astype(np.float32),
        train_confidence=train_confidence.astype(np.float32),
        val_confidence=val_confidence.astype(np.float32),
        train_predicted_labels=train_predicted_labels.astype(np.int32),
        val_predicted_labels=val_predicted_labels.astype(np.int32),
        train_correct=train_correct.astype(np.float32),
        val_correct=val_correct.astype(np.float32),
        train_mask=train_mask.astype(np.float32),
        val_mask=val_mask.astype(np.float32),
        train_labels=y_train_raw.astype(np.int32),
        val_labels=y_val_raw.astype(np.int32),
        train_files=np.asarray(train_files),
        val_files=np.asarray(val_files),
        test_files=np.asarray(test_files),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2)),
    )
    print(f"Saved LIME-DCCS targets to: {output_path}")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed partial checkpoint after successful save: {checkpoint_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
