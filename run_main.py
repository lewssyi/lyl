import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import datetime
import inspect
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

NUM_CLASSES = 5
CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
MULTI_CHANNEL_PATH = os.path.join(PROJECT_DIR, "EEG_data", "eeg_eog_emg")

PREPROCESS_SIGNALS = True
INCLUDE_SPECTRAL_FEATURES = True
N1_OVERSAMPLE_FACTOR = 1
N1_WEIGHT_BOOST = 1.20
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42
USE_METRIC_SELECTOR = False


class Logger:
    def __init__(self, filename="Run_Results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SleepDataGenerator(Sequence):
    def __init__(
        self,
        X_raw,
        X_segment,
        y_main,
        batch_size=32,
        shuffle=True,
        sample_weights=None,
        indices=None,
    ):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.y_main = y_main
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.indices = np.array(indices if indices is not None else np.arange(len(self.X_raw)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        batch_X_raw = self.X_raw[batch_indices]
        batch_X_segment = self.X_segment[batch_indices]
        batch_y_main = self.y_main[batch_indices]
        batch_weights = (
            self.sample_weights[batch_indices]
            if self.sample_weights is not None
            else np.ones(len(batch_indices), dtype=np.float32)
        )

        return (
            {"raw_input": batch_X_raw, "segment_input": batch_X_segment},
            {
                "main_output": batch_y_main,
                "channel_attention_weights": batch_y_main,
            },
            {"main_output": batch_weights},
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class ModelWrapperForLIME:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, inputs, **kwargs):
        preds = self.base_model.predict(inputs, **kwargs)
        return preds[0] if isinstance(preds, list) else preds


def normalize_lime_explanation(explanation, num_channels, num_segments):
    if isinstance(explanation, dict):
        return explanation

    if isinstance(explanation, (list, tuple)) and len(explanation) >= 5:
        target_class, original_pred, segment_importance, raw_dccs, calibrated_dccs = explanation[:5]
        zeros_matrix = np.zeros((num_channels, num_segments), dtype=np.float32)
        zeros_vector = np.zeros((num_segments,), dtype=np.float32)
        return {
            "target_class": int(target_class),
            "original_pred": np.asarray(original_pred, dtype=np.float32),
            "segment_importance": np.asarray(segment_importance, dtype=np.float32),
            "raw_dccs": np.asarray(raw_dccs, dtype=np.float32),
            "calibrated_dccs": np.asarray(calibrated_dccs, dtype=np.float32),
            "segment_matrix": zeros_matrix,
            "prior_weighted_matrix": zeros_matrix,
            "temporal_net": zeros_vector,
            "cooperation_score": zeros_vector,
            "conflict_score": zeros_vector,
            "prior_alignment": 0.0,
            "channel_disagreement": np.zeros((num_channels,), dtype=np.float32),
            "clinical_concept_names": [],
            "clinical_concept_matrix": np.zeros((0, num_segments), dtype=np.float32),
            "clinical_concept_evidence_matrix": np.zeros((0, num_segments), dtype=np.float32),
            "raw_concept_dccs": np.zeros((0,), dtype=np.float32),
            "calibrated_concept_dccs": np.zeros((0,), dtype=np.float32),
            "concept_prior": np.zeros((0,), dtype=np.float32),
            "concept_alignment": 0.0,
            "perturbation_mode": "legacy",
        }

    raise TypeError(f"Unsupported LIME explanation type: {type(explanation)!r}")


def call_with_supported_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(*args, **kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **supported_kwargs)


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def print_class_distribution(name, y_raw, class_names):
    counts = np.bincount(y_raw, minlength=len(class_names))
    total = counts.sum()
    ratio = counts / max(total, 1)
    summary = ", ".join(
        f"{label}:{count}({pct:.1%})"
        for label, count, pct in zip(class_names, counts, ratio)
    )
    print(f"{name} distribution: {summary}")


def build_segment_feature_matrix(X_raw, semantic_segmentation, extract_segment_features):
    X_segments = []
    for instance in tqdm(X_raw, desc="segment feature extraction"):
        segments = semantic_segmentation(instance, target_segments=30)
        X_segments.append(
            extract_segment_features(
                instance,
                segments,
                max_segments=30,
                include_spectral=INCLUDE_SPECTRAL_FEATURES,
            )
        )
    return np.asarray(X_segments, dtype=np.float32)


def compute_main_sample_weights(
    y_raw,
    num_classes,
    n1_boost=1.20,
    min_weight=0.85,
    max_weight=1.60,
):
    counts = np.bincount(y_raw, minlength=num_classes).astype(np.float32)
    nonzero_counts = counts[counts > 0]
    reference_count = float(np.median(nonzero_counts)) if len(nonzero_counts) else 1.0
    class_weights = np.sqrt(reference_count / np.maximum(counts, 1.0))
    class_weights = np.clip(class_weights, min_weight, max_weight)
    if num_classes > 1:
        class_weights[1] = min(max_weight, class_weights[1] * n1_boost)
    sample_weights = class_weights[y_raw].astype(np.float32)
    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)
    return sample_weights, class_weights.astype(np.float32)


def predict_with_n1_threshold(probabilities, n1_threshold, n1_margin=0.03):
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    if probabilities.shape[1] <= 1:
        return predictions

    non_n1_probs = np.delete(probabilities, 1, axis=1)
    strongest_non_n1 = np.max(non_n1_probs, axis=1)
    n1_mask = (
        (probabilities[:, 1] >= n1_threshold) &
        (probabilities[:, 1] >= strongest_non_n1 - n1_margin)
    )
    predictions[n1_mask] = 1
    return predictions


def tune_n1_threshold(y_true, probabilities):
    base_predictions = np.argmax(probabilities, axis=1)
    base_accuracy = float(np.mean(base_predictions == y_true))
    thresholds = np.linspace(0.12, 0.42, 16)

    best_threshold = 0.50
    best_predictions = base_predictions
    best_accuracy = base_accuracy
    best_macro_f1 = float(f1_score(y_true, base_predictions, average="macro", zero_division=0))
    best_objective = best_accuracy + 0.20 * best_macro_f1

    for threshold in thresholds:
        predictions = predict_with_n1_threshold(probabilities, threshold)
        accuracy = float(np.mean(predictions == y_true))
        macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
        if accuracy < base_accuracy - 0.015:
            continue
        objective = accuracy + 0.20 * macro_f1
        if objective > best_objective:
            best_threshold = float(threshold)
            best_predictions = predictions
            best_accuracy = accuracy
            best_macro_f1 = macro_f1
            best_objective = objective

    return best_threshold, {
        "base_accuracy": base_accuracy,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "n1_recall": float(np.mean(best_predictions[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0,
    }


def build_oversampled_indices(y_raw, n1_factor=2):
    base_indices = np.arange(len(y_raw))
    if n1_factor <= 1:
        return base_indices
    n1_indices = np.where(y_raw == 1)[0]
    if len(n1_indices) == 0:
        return base_indices
    extra_indices = np.tile(n1_indices, n1_factor - 1)
    return np.concatenate([base_indices, extra_indices])


def get_file_sample_counts(folder_path, files):
    counts = []
    for filename in files:
        with np.load(os.path.join(folder_path, filename), mmap_mode="r") as data:
            counts.append(int(data["x"].shape[0]))
    return counts


def smooth_sequence_predictions(
    probabilities,
    window_size=5,
    min_confidence=0.35,
    base_predictions=None,
    protected_classes=None,
    protection_thresholds=None,
):
    if len(probabilities) == 0:
        return np.array([], dtype=np.int32)

    half_window = window_size // 2
    smoothed = (
        np.asarray(base_predictions, dtype=np.int32).copy()
        if base_predictions is not None
        else np.argmax(probabilities, axis=1).astype(np.int32)
    )
    protected_classes = set(protected_classes or [])
    protection_thresholds = protection_thresholds or {}

    for idx in range(len(smoothed)):
        left = max(0, idx - half_window)
        right = min(len(smoothed), idx + half_window + 1)
        local_probs = probabilities[left:right].sum(axis=0)
        candidate = int(np.argmax(local_probs))
        current = smoothed[idx]
        protection_threshold = protection_thresholds.get(current, min_confidence)
        if current in protected_classes and probabilities[idx, current] >= protection_threshold:
            continue
        if probabilities[idx, current] < min_confidence:
            smoothed[idx] = candidate
        elif candidate != current and local_probs[candidate] > local_probs[current] * 1.05:
            smoothed[idx] = candidate

    cleaned = smoothed.copy()
    for idx in range(1, len(cleaned) - 1):
        current = cleaned[idx]
        protection_threshold = protection_thresholds.get(current, min_confidence)
        if current in protected_classes and probabilities[idx, current] >= protection_threshold:
            continue
        if cleaned[idx - 1] == cleaned[idx + 1] and cleaned[idx] != cleaned[idx - 1]:
            cleaned[idx] = cleaned[idx - 1]

    return cleaned


def smooth_predictions_by_recording(
    probabilities,
    file_sample_counts,
    window_size=5,
    min_confidence=0.35,
    base_predictions=None,
    protected_classes=None,
    protection_thresholds=None,
):
    smoothed_parts = []
    start = 0
    for count in file_sample_counts:
        end = start + count
        base_part = None if base_predictions is None else base_predictions[start:end]
        smoothed_parts.append(
            smooth_sequence_predictions(
                probabilities[start:end],
                window_size=window_size,
                min_confidence=min_confidence,
                base_predictions=base_part,
                protected_classes=protected_classes,
                protection_thresholds=protection_thresholds,
            )
        )
        start = end
    return np.concatenate(smoothed_parts) if smoothed_parts else np.array([], dtype=np.int32)


class BestValidationSelector(Callback):
    def __init__(
        self,
        X_val_raw,
        X_val_seg,
        y_val_raw,
        val_file_sample_counts,
        batch_size=64,
        patience=8,
        min_delta=1e-4,
        macro_weight=0.20,
    ):
        super().__init__()
        self.X_val_raw = X_val_raw
        self.X_val_seg = X_val_seg
        self.y_val_raw = y_val_raw
        self.val_file_sample_counts = val_file_sample_counts
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.macro_weight = macro_weight
        self.wait = 0
        self.best_epoch = 0
        self.best_score = -np.inf
        self.best_weights = None
        self.best_n1_threshold = 0.50
        self.best_metrics = None

    def _predict_probabilities(self):
        outputs = self.model.predict(
            [self.X_val_raw, self.X_val_seg],
            batch_size=self.batch_size,
            verbose=0,
        )
        return outputs[0] if isinstance(outputs, list) else outputs

    def _evaluate_current_model(self):
        probabilities = self._predict_probabilities()
        n1_threshold, threshold_metrics = tune_n1_threshold(self.y_val_raw, probabilities)
        calibrated_predictions = predict_with_n1_threshold(probabilities, n1_threshold)
        smoothed_predictions = smooth_predictions_by_recording(
            probabilities,
            self.val_file_sample_counts,
            window_size=5,
            min_confidence=0.35,
            base_predictions=calibrated_predictions,
            protected_classes={1},
            protection_thresholds={1: n1_threshold},
        )

        argmax_predictions = np.argmax(probabilities, axis=1)
        smoothed_accuracy = float(np.mean(smoothed_predictions == self.y_val_raw))
        smoothed_macro_f1 = float(
            f1_score(self.y_val_raw, smoothed_predictions, average="macro", zero_division=0)
        )
        n1_mask = self.y_val_raw == 1
        n1_predictions = smoothed_predictions == 1
        n1_true_positive = float(np.sum(n1_mask & n1_predictions))
        n1_precision = n1_true_positive / max(float(np.sum(n1_predictions)), 1.0)
        n1_recall = n1_true_positive / max(float(np.sum(n1_mask)), 1.0)
        n1_f1 = 0.0 if n1_precision + n1_recall == 0 else (
            2.0 * n1_precision * n1_recall / (n1_precision + n1_recall)
        )
        score = smoothed_accuracy + self.macro_weight * smoothed_macro_f1

        return {
            "score": score,
            "n1_threshold": n1_threshold,
            "argmax_accuracy": float(np.mean(argmax_predictions == self.y_val_raw)),
            "calibrated_accuracy": threshold_metrics["best_accuracy"],
            "smoothed_accuracy": smoothed_accuracy,
            "smoothed_macro_f1": smoothed_macro_f1,
            "n1_precision": n1_precision,
            "n1_recall": n1_recall,
            "n1_f1": n1_f1,
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = self._evaluate_current_model()
        logs["val_selection_score"] = metrics["score"]
        logs["val_smoothed_accuracy"] = metrics["smoothed_accuracy"]
        logs["val_smoothed_macro_f1"] = metrics["smoothed_macro_f1"]

        improved = metrics["score"] > self.best_score + self.min_delta
        if improved:
            self.best_score = metrics["score"]
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.best_n1_threshold = metrics["n1_threshold"]
            self.best_metrics = metrics
            self.wait = 0
            status = "new best"
        else:
            self.wait += 1
            status = f"wait={self.wait}/{self.patience}"

        print(
            "\nValidation selector "
            f"({status}) | score={metrics['score']:.4f} "
            f"acc={metrics['smoothed_accuracy']:.4f} "
            f"macro_f1={metrics['smoothed_macro_f1']:.4f} "
            f"N1_P={metrics['n1_precision']:.4f} "
            f"N1_R={metrics['n1_recall']:.4f} "
            f"thr={metrics['n1_threshold']:.3f}"
        )

        if self.wait >= self.patience:
            self.model.stop_training = True
            print(
                "Validation selector stopping: "
                f"no score improvement for {self.patience} epochs."
            )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                "Validation selector restored best weights from "
                f"epoch {self.best_epoch} | score={self.best_score:.4f} "
                f"thr={self.best_n1_threshold:.3f}"
            )


def find_transition_windows(labels, window_radius=2):
    transitions = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            left = max(0, idx - window_radius)
            right = min(len(labels), idx + window_radius + 1)
            transitions.append((idx - 1, idx, left, right))
    return transitions


def save_academic_results(y_true, y_pred, class_names, output_dir):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    report_df = df.drop(index=["accuracy"], errors="ignore").drop(columns=["support"], errors="ignore")
    report_df.columns = ["Precision", "Recall", "F1-Score"]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        report_df.astype(float),
        annot=True,
        cmap="RdBu_r",
        fmt=".4f",
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=1.5,
        annot_kws={"size": 13, "weight": "bold"},
    )
    accuracy = report_dict.get("accuracy", 0)
    plt.title(f"Classification Report | Accuracy: {accuracy:.4f}", fontweight="bold", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_report_visual.png"), dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"result_{timestamp}.txt")
    sys.stdout = Logger(log_path)

    set_random_seed(SEED)

    from src.data_processing import get_npz_file_list, load_npz_data, split_npz_files
    from src.grad_cam import run_hybrid_grad_cam  # noqa: F401
    from src.lime_explanation import (
        extract_segment_features,
        run_multimodal_lime,
        semantic_segmentation,
        summarize_transition_explanation,
    )
    from src.model_training import create_hybrid_model_v2
    from src.visualization import plot_xai_multimodal_dashboard
    try:
        from src.visualization import plot_transition_explanation_dashboard
    except ImportError:
        plot_transition_explanation_dashboard = None

    print("Core modules loaded.")

    all_files = get_npz_file_list(MULTI_CHANNEL_PATH)
    train_files, val_files, test_files = split_npz_files(
        all_files,
        test_size=0.15,
        val_size=0.15,
        seed=SEED,
    )

    print(f"Found {len(all_files)} recording files.")
    print(f"Train files ({len(train_files)}): {train_files}")
    print(f"Val files   ({len(val_files)}): {val_files}")
    print(f"Test files  ({len(test_files)}): {test_files}")

    X_train_raw, y_train_raw = load_npz_data(
        MULTI_CHANNEL_PATH,
        file_subset=train_files,
        preprocess=PREPROCESS_SIGNALS,
    )
    X_val_raw, y_val_raw = load_npz_data(
        MULTI_CHANNEL_PATH,
        file_subset=val_files,
        preprocess=PREPROCESS_SIGNALS,
    )
    X_test_raw, y_test_raw = load_npz_data(
        MULTI_CHANNEL_PATH,
        file_subset=test_files,
        preprocess=PREPROCESS_SIGNALS,
    )
    val_file_sample_counts = get_file_sample_counts(MULTI_CHANNEL_PATH, val_files)
    test_file_sample_counts = get_file_sample_counts(MULTI_CHANNEL_PATH, test_files)

    print_class_distribution("Train", y_train_raw, CLASS_LABELS)
    print_class_distribution("Val", y_val_raw, CLASS_LABELS)
    print_class_distribution("Test", y_test_raw, CLASS_LABELS)

    X_train_seg = build_segment_feature_matrix(X_train_raw, semantic_segmentation, extract_segment_features)
    X_val_seg = build_segment_feature_matrix(X_val_raw, semantic_segmentation, extract_segment_features)
    X_test_seg = build_segment_feature_matrix(X_test_raw, semantic_segmentation, extract_segment_features)

    y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val_raw, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

    train_sample_weights, class_weights = compute_main_sample_weights(
        y_train_raw,
        NUM_CLASSES,
        n1_boost=N1_WEIGHT_BOOST,
    )
    train_indices = build_oversampled_indices(y_train_raw, n1_factor=N1_OVERSAMPLE_FACTOR)
    class_weight_summary = ", ".join(
        f"{label}:{weight:.3f}" for label, weight in zip(CLASS_LABELS, class_weights)
    )
    print(f"Moderate class weights: {class_weight_summary}")
    print(
        f"N1 balancing: factor={N1_OVERSAMPLE_FACTOR}, "
        f"training samples per epoch={len(train_indices)}, raw={len(y_train_raw)}"
    )

    model = create_hybrid_model_v2(
        raw_shape=X_train_raw.shape[1:],
        segment_shape=X_train_seg.shape[1:],
        num_classes=NUM_CLASSES,
        learning_rate=0.001,
        attention_loss_weight=0.01,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_main_output_loss",
            patience=8,
            restore_best_weights=True,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_main_output_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    validation_selector = None
    if USE_METRIC_SELECTOR:
        validation_selector = BestValidationSelector(
            X_val_raw,
            X_val_seg,
            y_val_raw,
            val_file_sample_counts,
            batch_size=64,
            patience=8,
            macro_weight=0.20,
        )
        callbacks.insert(0, validation_selector)

    train_gen = SleepDataGenerator(
        X_train_raw,
        X_train_seg,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        sample_weights=train_sample_weights,
        indices=train_indices,
    )
    val_gen = SleepDataGenerator(
        X_val_raw,
        X_val_seg,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        "\nTraining with file-level split, signal preprocessing, "
        "spectral segment features, and moderate class balancing."
    )
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2,
    )

    print("\n==========================================")
    print("4. Global test inference")
    print("==========================================")

    if validation_selector is not None and validation_selector.best_metrics is not None:
        n1_threshold = validation_selector.best_n1_threshold
        print(
            "Selected validation model: "
            f"epoch={validation_selector.best_epoch} "
            f"score={validation_selector.best_score:.4f} "
            f"val_smoothed_acc={validation_selector.best_metrics['smoothed_accuracy']:.4f} "
            f"val_macro_f1={validation_selector.best_metrics['smoothed_macro_f1']:.4f} "
            f"val_N1_P={validation_selector.best_metrics['n1_precision']:.4f} "
            f"val_N1_R={validation_selector.best_metrics['n1_recall']:.4f} "
            f"thr={n1_threshold:.3f}"
        )
    else:
        val_outputs = model.predict([X_val_raw, X_val_seg], batch_size=64, verbose=0)
        y_val_probs_all = val_outputs[0] if isinstance(val_outputs, list) else val_outputs
        n1_threshold, threshold_metrics = tune_n1_threshold(y_val_raw, y_val_probs_all)
        print(
            "Validation-calibrated N1 threshold: "
            f"{n1_threshold:.3f} | "
            f"val_acc={threshold_metrics['best_accuracy']:.4f} "
            f"val_macro_f1={threshold_metrics['best_macro_f1']:.4f} "
            f"val_n1_recall={threshold_metrics['n1_recall']:.4f}"
        )

    pred_outputs = model.predict([X_test_raw, X_test_seg], batch_size=64, verbose=1)
    y_pred_probs_all = pred_outputs[0] if isinstance(pred_outputs, list) else pred_outputs
    y_true = np.argmax(y_test, axis=1)
    y_pred_argmax = np.argmax(y_pred_probs_all, axis=1)
    y_pred_final = predict_with_n1_threshold(y_pred_probs_all, n1_threshold)
    y_pred_smoothed = smooth_predictions_by_recording(
        y_pred_probs_all,
        test_file_sample_counts,
        window_size=5,
        min_confidence=0.35,
        base_predictions=y_pred_final,
        protected_classes={1},
        protection_thresholds={1: n1_threshold},
    )

    print("\nFinal classification report (argmax raw)")
    print(classification_report(y_true, y_pred_argmax, target_names=CLASS_LABELS, digits=4))
    print("\nFinal classification report (N1-calibrated raw)")
    print(classification_report(y_true, y_pred_final, target_names=CLASS_LABELS, digits=4))
    print("\nFinal classification report (N1-protected smoothed)")
    print(classification_report(y_true, y_pred_smoothed, target_names=CLASS_LABELS, digits=4))

    argmax_accuracy = np.mean(y_pred_argmax == y_true)
    raw_accuracy = np.mean(y_pred_final == y_true)
    smoothed_accuracy = np.mean(y_pred_smoothed == y_true)
    print(
        f"\nAccuracy summary | argmax={argmax_accuracy:.4f} "
        f"calibrated={raw_accuracy:.4f} "
        f"smoothed={smoothed_accuracy:.4f} delta={smoothed_accuracy - raw_accuracy:+.4f}"
    )

    save_academic_results(y_true, y_pred_smoothed, CLASS_LABELS, OUTPUT_DIR)
    save_confusion_matrix(y_true, y_pred_smoothed, CLASS_LABELS, OUTPUT_DIR)

    print("\n5. Running representative LIME dashboards")
    lime_wrapper = ModelWrapperForLIME(model)

    for stage_idx, stage_name in enumerate(CLASS_LABELS):
        correct_indices = np.where((y_true == stage_idx) & (y_pred_final == stage_idx))[0]
        if len(correct_indices) == 0:
            print(f"Stage {stage_name}: no correctly predicted sample; skipping.")
            continue

        target_idx = correct_indices[0]
        instance_raw = X_test_raw[target_idx]
        print(f"Explaining stage {stage_name}, test index {target_idx}")

        segments = semantic_segmentation(instance_raw, target_segments=30)
        explanation = run_multimodal_lime(
            model=lime_wrapper,
            instance_multimodal=instance_raw,
            segments=segments,
            num_perturbations=300,
        )
        explanation = normalize_lime_explanation(
            explanation,
            num_channels=instance_raw.shape[1],
            num_segments=len(segments),
        )
        save_path = os.path.join(OUTPUT_DIR, f"XAI_Diagnosis_{stage_name}.png")
        call_with_supported_kwargs(
            plot_xai_multimodal_dashboard,
            signal=instance_raw,
            segments=segments,
            segment_importance=explanation["segment_importance"],
            raw_dccs=explanation["raw_dccs"],
            calibrated_dccs=explanation["calibrated_dccs"],
            segment_matrix=explanation["segment_matrix"],
            prior_weighted_matrix=explanation["prior_weighted_matrix"],
            temporal_net=explanation["temporal_net"],
            cooperation_score=explanation["cooperation_score"],
            conflict_score=explanation["conflict_score"],
            prior_alignment=explanation["prior_alignment"],
            channel_disagreement=explanation["channel_disagreement"],
            clinical_concept_names=explanation.get("clinical_concept_names"),
            clinical_concept_matrix=explanation.get("clinical_concept_matrix"),
            clinical_concept_evidence_matrix=explanation.get("clinical_concept_evidence_matrix"),
            raw_concept_dccs=explanation.get("raw_concept_dccs"),
            calibrated_concept_dccs=explanation.get("calibrated_concept_dccs"),
            concept_prior=explanation.get("concept_prior"),
            concept_alignment=explanation.get("concept_alignment"),
            perturbation_mode=explanation.get("perturbation_mode"),
            channel_names=["EEG", "EOG", "EMG"],
            target_class_name=CLASS_LABELS[explanation["target_class"]],
            save_path=save_path,
        )

    print("\n6. Running transition-aware XAI dashboards")
    if plot_transition_explanation_dashboard is None:
        print("Transition dashboard function is not available; skipping transition XAI figures.")
        print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")
        return

    transition_generated = 0
    offset = 0
    for file_idx, sample_count in enumerate(test_file_sample_counts):
        file_true = y_true[offset: offset + sample_count]
        file_pred = y_pred_smoothed[offset: offset + sample_count]
        transitions = find_transition_windows(file_true, window_radius=2)
        for _, transition_idx, left, right in transitions:
            if file_pred[transition_idx - 1] != file_true[transition_idx - 1]:
                continue
            if file_pred[transition_idx] != file_true[transition_idx]:
                continue

            epoch_explanations = []
            for local_idx in range(left, right):
                global_idx = offset + local_idx
                instance_raw = X_test_raw[global_idx]
                segments = semantic_segmentation(instance_raw, target_segments=30)
                explanation = run_multimodal_lime(
                    model=lime_wrapper,
                    instance_multimodal=instance_raw,
                    segments=segments,
                    num_perturbations=200,
                )
                explanation = normalize_lime_explanation(
                    explanation,
                    num_channels=instance_raw.shape[1],
                    num_segments=len(segments),
                )
                epoch_explanations.append(explanation)

            from_stage = CLASS_LABELS[file_true[transition_idx - 1]]
            to_stage = CLASS_LABELS[file_true[transition_idx]]
            transition_summary = summarize_transition_explanation(
                epoch_explanations,
                ["EEG", "EOG", "EMG"],
                from_stage=from_stage,
                to_stage=to_stage,
            )
            if transition_summary is None:
                continue

            transition_path = os.path.join(
                OUTPUT_DIR,
                f"XAI_Transition_{test_files[file_idx].replace('.npz', '')}_{from_stage}_to_{to_stage}.png",
            )
            call_with_supported_kwargs(
                plot_transition_explanation_dashboard,
                transition_summary,
                from_stage=from_stage,
                to_stage=to_stage,
                save_path=transition_path,
            )
            transition_generated += 1
            break
        offset += sample_count

    print(f"Transition dashboards generated: {transition_generated}")

    print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
