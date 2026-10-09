import copy
import hashlib

import numpy as np
import scipy.signal as signal
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

try:
    from .clinical_concepts import (
        RAW_PHYSIOLOGY_VERSION,
        concept_metadata,
        concepts_from_raw_waveform,
        stage_channel_prior,
        stage_concept_prior,
    )
except ImportError:  # run_main.py also supports importing src modules by path
    from clinical_concepts import (
        RAW_PHYSIOLOGY_VERSION,
        concept_metadata,
        concepts_from_raw_waveform,
        stage_channel_prior,
        stage_concept_prior,
    )


STAGE_LABELS = ["W", "N1", "N2", "N3", "REM"]
CLINICAL_CONCEPTS = concept_metadata(RAW_PHYSIOLOGY_VERSION)
_STAGE_CONCEPT_PRIOR_MATRIX = stage_concept_prior()
STAGE_CONCEPT_PRIORS = {
    stage: {
        item["name"]: float(_STAGE_CONCEPT_PRIOR_MATRIX[index, stage])
        for index, item in enumerate(CLINICAL_CONCEPTS)
    }
    for stage in range(len(STAGE_LABELS))
}

TRANSITION_CONCEPT_PRIORS = {
    ("W", "N1"): {"EEG Theta": 0.34, "EOG Slow Eye": 0.30, "EEG Alpha": 0.18, "EMG Low Tone": 0.10, "EEG Delta": 0.08},
    ("N1", "N2"): {"EEG Sigma": 0.44, "EEG Theta": 0.22, "EEG Delta": 0.14, "EOG Slow Eye": 0.12, "EMG Muscle Activation": 0.08},
    ("N2", "N3"): {"EEG Delta": 0.62, "EEG Theta": 0.18, "EEG Sigma": 0.12, "EMG Low Tone": 0.08},
    ("N3", "N2"): {"EEG Sigma": 0.40, "EEG Delta": 0.28, "EEG Theta": 0.18, "EMG Muscle Activation": 0.14},
    ("N2", "REM"): {"EOG Rapid Eye": 0.42, "EMG Low Tone": 0.34, "EEG Theta": 0.16, "EEG Beta": 0.08},
    ("REM", "W"): {"EEG Alpha": 0.36, "EMG Muscle Activation": 0.30, "EEG Beta": 0.18, "EOG Rapid Eye": 0.10, "EEG Theta": 0.06},
    ("REM", "N1"): {"EEG Theta": 0.32, "EOG Slow Eye": 0.24, "EMG Low Tone": 0.18, "EEG Alpha": 0.16, "EOG Rapid Eye": 0.10},
}


def _normalize_positive(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 1e-8:
        return np.ones_like(values, dtype=np.float32) / max(len(values), 1)
    return (values / total).astype(np.float32)


def _cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-8
    return float(np.dot(left, right) / denom)


def _stage_to_index(stage):
    if isinstance(stage, str):
        return STAGE_LABELS.index(stage) if stage in STAGE_LABELS else None
    if stage is None:
        return None
    return int(stage)


def _concept_prior_vector(prior_map):
    values = np.array([prior_map.get(item["name"], 0.02) for item in CLINICAL_CONCEPTS], dtype=np.float32)
    return _normalize_positive(values)


def _coerce_rng(seed=42, rng=None):
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of np.random.Generator")
        return rng
    return np.random.default_rng(42 if seed is None else int(seed))


def perturb_total_mean(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = np.mean(signal_1d)


def perturb_mean(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = np.mean(signal_1d[start_idx:end_idx])


def perturb_noise(signal_1d, start_idx, end_idx, rng=None):
    if start_idx == end_idx:
        return
    rng = _coerce_rng(seed=None, rng=rng)
    signal_1d[start_idx:end_idx] = rng.uniform(
        np.min(signal_1d), np.max(signal_1d), end_idx - start_idx
    )


def perturb_zero(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = 0.0


def perturb_gaussian_noise(signal_1d, start_idx, end_idx, rng=None):
    if start_idx == end_idx:
        return
    reference = signal_1d[start_idx:end_idx]
    sigma = float(np.std(reference))
    if sigma < 1e-8:
        sigma = float(np.std(signal_1d))
    rng = _coerce_rng(seed=None, rng=rng)
    signal_1d[start_idx:end_idx] = rng.normal(
        float(np.mean(reference)),
        max(sigma, 1e-8),
        end_idx - start_idx,
    )


def _blend_segment_edges(original, replacement, edge_width=8):
    if len(replacement) <= 2:
        return replacement
    edge_width = min(edge_width, max(1, len(replacement) // 4))
    blended = replacement.copy()
    ramp = np.linspace(0.0, 1.0, edge_width, dtype=np.float32)
    blended[:edge_width] = original[:edge_width] * (1.0 - ramp) + replacement[:edge_width] * ramp
    blended[-edge_width:] = replacement[-edge_width:] * (1.0 - ramp) + original[-edge_width:] * ramp
    return blended


def _match_segment_statistics(replacement, reference):
    ref_mean = float(np.mean(reference))
    ref_std = float(np.std(reference) + 1e-8)
    rep_std = float(np.std(replacement) + 1e-8)
    matched = (replacement - float(np.mean(replacement))) / rep_std
    return matched * ref_std + ref_mean


def _phase_randomized_surrogate(segment, rng=None):
    if len(segment) < 8 or np.std(segment) < 1e-8:
        return np.full_like(segment, np.mean(segment), dtype=np.float32)
    centered = segment - np.mean(segment)
    spectrum = np.fft.rfft(centered)
    rng = _coerce_rng(seed=None, rng=rng)
    random_phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    random_phase[0] = 0.0
    if len(random_phase) > 1 and len(segment) % 2 == 0:
        random_phase[-1] = 0.0
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * random_phase), n=len(segment))
    return _match_segment_statistics(surrogate, segment).astype(np.float32)


def _low_frequency_surrogate(segment, rng=None):
    if len(segment) < 6:
        return np.full_like(segment, np.mean(segment), dtype=np.float32)
    window_len = min(len(segment) // 2 * 2 - 1, 51)
    if window_len < 5:
        trend = np.full_like(segment, np.mean(segment), dtype=np.float32)
    else:
        trend = signal.savgol_filter(segment, window_length=window_len, polyorder=2, mode="interp")
    residual_std = float(np.std(segment - trend))
    rng = _coerce_rng(seed=None, rng=rng)
    slow_noise = rng.normal(0.0, residual_std + 1e-8, size=len(segment))
    if len(segment) >= 9:
        slow_noise = signal.savgol_filter(slow_noise, window_length=min(len(segment) // 2 * 2 - 1, 31), polyorder=2)
    surrogate = trend + 0.35 * slow_noise
    return _match_segment_statistics(surrogate, segment).astype(np.float32)


def _envelope_preserving_surrogate(segment, rng=None):
    if len(segment) < 4 or np.std(segment) < 1e-8:
        return np.full_like(segment, np.mean(segment), dtype=np.float32)
    segment_centered = segment - np.mean(segment)
    envelope_window = max(5, min(25, len(segment) // 4))
    envelope = np.sqrt(
        np.convolve(segment_centered ** 2, np.ones(envelope_window) / envelope_window, mode="same") + 1e-8
    )
    rng = _coerce_rng(seed=None, rng=rng)
    high_freq_noise = rng.normal(0.0, 1.0, size=len(segment))
    high_freq_noise = high_freq_noise - np.convolve(
        high_freq_noise,
        np.ones(envelope_window) / envelope_window,
        mode="same",
    )
    surrogate = high_freq_noise * envelope
    return _match_segment_statistics(surrogate, segment).astype(np.float32)


def _physiological_segment_surrogate(
    signal_1d, start_idx, end_idx, channel_idx, target_class, rng=None
):
    segment_values = signal_1d[start_idx:end_idx].astype(np.float32)
    if len(segment_values) == 0:
        return segment_values

    if channel_idx == 0:
        replacement = _phase_randomized_surrogate(segment_values, rng=rng)
    elif channel_idx == 1:
        replacement = _low_frequency_surrogate(segment_values, rng=rng)
    elif channel_idx == 2:
        replacement = _envelope_preserving_surrogate(segment_values, rng=rng)
        if target_class == 4:
            replacement = np.mean(segment_values) + 0.65 * (replacement - np.mean(replacement))
    else:
        replacement = _phase_randomized_surrogate(segment_values, rng=rng)

    return _blend_segment_edges(segment_values, replacement)


def get_clinical_prior(target_class):
    stage_idx = _stage_to_index(target_class)
    priors = stage_channel_prior()
    if stage_idx is None or stage_idx < 0 or stage_idx >= len(priors):
        return np.asarray([0.33, 0.33, 0.34], dtype=np.float32)
    return priors[stage_idx].copy()


def get_stage_concept_prior(target_class):
    stage_idx = _stage_to_index(target_class)
    return _concept_prior_vector(STAGE_CONCEPT_PRIORS.get(stage_idx, {}))


def get_transition_concept_prior(from_stage=None, to_stage=None):
    from_idx = _stage_to_index(from_stage)
    to_idx = _stage_to_index(to_stage)
    from_label = STAGE_LABELS[from_idx] if from_idx is not None and 0 <= from_idx < len(STAGE_LABELS) else from_stage
    to_label = STAGE_LABELS[to_idx] if to_idx is not None and 0 <= to_idx < len(STAGE_LABELS) else to_stage

    prior_map = TRANSITION_CONCEPT_PRIORS.get((from_label, to_label))
    if prior_map is None:
        prior_map = STAGE_CONCEPT_PRIORS.get(to_idx, {}) if to_idx is not None else {}
    return _concept_prior_vector(prior_map)


def apply_stage_aware_perturbation(
    signal_1d,
    start_idx,
    end_idx,
    channel_idx,
    target_class,
    perturbation_mode="physiological",
    rng=None,
):
    if perturbation_mode == "physiological":
        signal_1d[start_idx:end_idx] = _physiological_segment_surrogate(
            signal_1d,
            start_idx,
            end_idx,
            channel_idx,
            target_class,
            rng=rng,
        )
        return

    if perturbation_mode in {"zero", "zero_out", "zero-out"}:
        perturb_zero(signal_1d, start_idx, end_idx)
        return

    if perturbation_mode in {"gaussian", "gaussian_noise", "noise"}:
        perturb_gaussian_noise(signal_1d, start_idx, end_idx, rng=rng)
        return

    if perturbation_mode in {"mean", "local_mean"}:
        perturb_mean(signal_1d, start_idx, end_idx)
        return

    if channel_idx == 2:
        if target_class == 4:
            perturb_noise(signal_1d, start_idx, end_idx, rng=rng)
        else:
            perturb_mean(signal_1d, start_idx, end_idx)
    elif channel_idx == 1:
        if target_class in [0, 4]:
            perturb_mean(signal_1d, start_idx, end_idx)
        else:
            perturb_noise(signal_1d, start_idx, end_idx, rng=rng)
    else:
        perturb_mean(signal_1d, start_idx, end_idx)


def generate_random_perturbations(num_perturbations, num_features, seed=42, rng=None):
    rng = _coerce_rng(seed=seed, rng=rng)
    return rng.binomial(1, 0.5, size=(num_perturbations, num_features))


def calculate_cosine_distances(random_perturbations, num_features):
    original_rep = np.ones((1, num_features))
    return pairwise_distances(random_perturbations, original_rep, metric="cosine").ravel()


def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    return np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))


def apply_perturbation_to_multimodal(
    signal_multimodal,
    perturbation,
    segments,
    target_class,
    perturbation_mode="physiological",
    rng=None,
):
    rng = _coerce_rng(seed=None, rng=rng)
    perturbed_signal = copy.deepcopy(signal_multimodal)
    num_segments = len(segments)
    num_channels = signal_multimodal.shape[1] if len(signal_multimodal.shape) == 2 else 1

    for channel_idx in range(num_channels):
        channel_mask = perturbation[channel_idx * num_segments: (channel_idx + 1) * num_segments]
        for segment_idx, active in enumerate(channel_mask):
            if not active:
                start_idx, end_idx = segments[segment_idx]
                if num_channels > 1:
                    apply_stage_aware_perturbation(
                        perturbed_signal[:, channel_idx],
                        start_idx,
                        end_idx,
                        channel_idx,
                        target_class,
                        perturbation_mode=perturbation_mode,
                        rng=rng,
                    )
                else:
                    channel_signal = (
                        perturbed_signal[:, 0]
                        if getattr(perturbed_signal, "ndim", 1) == 2
                        else perturbed_signal
                    )
                    apply_stage_aware_perturbation(
                        channel_signal,
                        start_idx,
                        end_idx,
                        0,
                        target_class,
                        perturbation_mode=perturbation_mode,
                        rng=rng,
                    )
    return perturbed_signal


def predict_perturbations(
    model,
    instance_multimodal,
    random_perturbations,
    segments,
    target_class,
    perturbation_mode="physiological",
    prediction_batch_size=16,
    rng=None,
):
    rng = _coerce_rng(seed=None, rng=rng)
    perturbation_predictions = []
    prediction_batch_size = max(1, int(prediction_batch_size))
    for batch_start in range(0, len(random_perturbations), prediction_batch_size):
        signal_batch = []
        segment_batch = []
        for perturbation in random_perturbations[batch_start: batch_start + prediction_batch_size]:
            perturbed_signal = apply_perturbation_to_multimodal(
                instance_multimodal,
                perturbation,
                segments,
                target_class,
                perturbation_mode=perturbation_mode,
                rng=rng,
            )
            signal_batch.append(np.asarray(perturbed_signal, dtype=np.float32))
            segment_batch.append(
                extract_segment_features(perturbed_signal, segments, max_segments=30)
            )
        model_prediction = _predict_model_main_head(
            model,
            np.asarray(signal_batch, dtype=np.float32),
            np.asarray(segment_batch, dtype=np.float32),
        )
        perturbation_predictions.extend(np.asarray(model_prediction, dtype=np.float32))
    return np.asarray(perturbation_predictions, dtype=np.float32)


def _predict_model_main_head(model, raw_batch, segment_batch):
    prediction = model.predict([raw_batch, segment_batch], verbose=0)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    return np.asarray(prediction, dtype=np.float32)


def _predict_target_probability(model, signal_multimodal, segments, target_class, max_segments=30):
    segment_features = extract_segment_features(signal_multimodal, segments, max_segments=max_segments)
    prediction = _predict_model_main_head(
        model,
        signal_multimodal[np.newaxis, :, :],
        segment_features[np.newaxis, :, :],
    )
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 1:
        return float(prediction[int(target_class)])
    return float(prediction.reshape(prediction.shape[0], -1)[0, int(target_class)])


def rank_lime_features(segment_importance, ranking="positive"):
    """Return feature indices ordered by class-supporting LIME evidence."""
    scores = np.asarray(segment_importance, dtype=np.float32).ravel()
    if ranking == "absolute":
        rank_scores = np.abs(scores)
    elif ranking == "signed":
        rank_scores = scores
    elif ranking == "positive":
        rank_scores = np.clip(scores, 0.0, None)
        if float(np.sum(rank_scores)) <= 1e-8:
            rank_scores = np.abs(scores)
    else:
        raise ValueError("ranking must be one of: positive, absolute, signed")
    return np.argsort(-rank_scores).astype(np.int32)


def _feature_count_schedule(num_features, num_steps):
    num_features = int(num_features)
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if num_steps is None or int(num_steps) >= num_features:
        return np.arange(num_features + 1, dtype=np.int32)
    counts = np.rint(np.linspace(0, num_features, int(num_steps) + 1)).astype(np.int32)
    return np.unique(np.clip(counts, 0, num_features)).astype(np.int32)


def _mask_for_ranked_features(num_features, feature_order, feature_count, curve_type):
    mask = np.ones(num_features, dtype=np.int32)
    selected = feature_order[:feature_count]
    if curve_type == "deletion":
        mask[selected] = 0
    elif curve_type == "insertion":
        mask[:] = 0
        mask[selected] = 1
    else:
        raise ValueError("curve_type must be deletion or insertion")
    return mask


def compute_deletion_insertion_auc(
    model,
    instance_multimodal,
    segment_importance,
    segments,
    target_class=None,
    perturbation_mode="physiological",
    ranking="positive",
    num_steps=20,
    max_segments=30,
    random_state=None,
    rng=None,
):
    """
    Compute LIME fidelity curves for a perturbation strategy.

    Deletion starts from the original epoch and progressively replaces the most
    class-supporting features. Lower deletion AUC means the explanation removes
    the model evidence quickly. Insertion starts from a fully perturbed epoch
    and restores features in the same order. Higher insertion AUC means the
    explanation recovers the model evidence quickly.
    """
    instance_multimodal = np.asarray(instance_multimodal, dtype=np.float32)
    if instance_multimodal.ndim == 1:
        instance_multimodal = instance_multimodal[:, np.newaxis]

    num_segments = len(segments)
    num_channels = instance_multimodal.shape[1]
    num_features = num_channels * num_segments
    flat_importance = np.asarray(segment_importance, dtype=np.float32).ravel()
    if flat_importance.size != num_features:
        raise ValueError(
            "segment_importance length must match channels * segments: "
            f"{flat_importance.size} vs {num_features}"
        )

    if target_class is None:
        original_features = extract_segment_features(
            instance_multimodal,
            segments,
            max_segments=max_segments,
        )
        original_prediction = _predict_model_main_head(
            model,
            instance_multimodal[np.newaxis, :, :],
            original_features[np.newaxis, :, :],
        )
        target_class = int(np.argmax(original_prediction[0]))
    else:
        target_class = int(target_class)

    feature_order = rank_lime_features(flat_importance, ranking=ranking)
    feature_counts = _feature_count_schedule(num_features, num_steps)
    fractions = feature_counts.astype(np.float32) / float(num_features)
    deletion_curve = []
    insertion_curve = []

    rng = _coerce_rng(seed=random_state, rng=rng)
    for feature_count in feature_counts:
        deletion_mask = _mask_for_ranked_features(
            num_features,
            feature_order,
            int(feature_count),
            "deletion",
        )
        deletion_signal = apply_perturbation_to_multimodal(
            instance_multimodal,
            deletion_mask,
            segments,
            target_class,
            perturbation_mode=perturbation_mode,
            rng=rng,
        )
        deletion_curve.append(
            _predict_target_probability(
                model,
                deletion_signal,
                segments,
                target_class,
                max_segments=max_segments,
            )
        )

        insertion_mask = _mask_for_ranked_features(
            num_features,
            feature_order,
            int(feature_count),
            "insertion",
        )
        insertion_signal = apply_perturbation_to_multimodal(
            instance_multimodal,
            insertion_mask,
            segments,
            target_class,
            perturbation_mode=perturbation_mode,
            rng=rng,
        )
        insertion_curve.append(
            _predict_target_probability(
                model,
                insertion_signal,
                segments,
                target_class,
                max_segments=max_segments,
            )
        )

    deletion_curve = np.asarray(deletion_curve, dtype=np.float32)
    insertion_curve = np.asarray(insertion_curve, dtype=np.float32)
    deletion_auc = float(np.trapz(deletion_curve, fractions))
    insertion_auc = float(np.trapz(insertion_curve, fractions))
    original_score = float(deletion_curve[0])
    baseline_score = float(insertion_curve[0])

    return {
        "target_class": target_class,
        "perturbation_mode": perturbation_mode,
        "ranking": ranking,
        "feature_order": feature_order,
        "feature_counts": feature_counts,
        "fractions": fractions.astype(np.float32),
        "original_score": original_score,
        "baseline_score": baseline_score,
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "fidelity_minus": float(original_score - deletion_auc),
        "fidelity_plus": float(insertion_auc - baseline_score),
    }


def compare_perturbation_fidelity(
    model,
    instance_multimodal,
    segment_importance,
    segments,
    target_class=None,
    perturbation_modes=("physiological", "zero", "gaussian_noise"),
    ranking="positive",
    num_steps=20,
    max_segments=30,
    random_state=None,
):
    """Evaluate deletion/insertion AUC for several perturbation strategies."""
    return {
        mode: compute_deletion_insertion_auc(
            model,
            instance_multimodal,
            segment_importance,
            segments,
            target_class=target_class,
            perturbation_mode=mode,
            ranking=ranking,
            num_steps=num_steps,
            max_segments=max_segments,
            random_state=random_state,
        )
        for mode in perturbation_modes
    }


def fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class):
    if len(perturbation_predictions.shape) == 1:
        y_target = perturbation_predictions
    elif perturbation_predictions.shape[1] == 1:
        y_target = perturbation_predictions.ravel()
    else:
        y_target = perturbation_predictions[:, target_class]

    clf = Ridge(alpha=1.0)
    clf.fit(random_perturbations, y_target, sample_weight=weights)
    return clf.coef_


def calculate_dccs(segment_importance, num_channels, num_segments, target_class):
    channel_scores = []
    for channel_idx in range(num_channels):
        channel_weights = segment_importance[channel_idx * num_segments: (channel_idx + 1) * num_segments]
        channel_scores.append(np.sum(np.abs(channel_weights)))

    total_score = np.sum(channel_scores)
    if total_score == 0:
        raw_dccs = [1.0 / num_channels] * num_channels
    else:
        raw_dccs = [score / total_score for score in channel_scores]

    prior = get_clinical_prior(target_class)[:num_channels]
    calibrated_dccs = [raw_dccs[idx] * prior[idx] for idx in range(num_channels)]
    calibrated_total = np.sum(calibrated_dccs)

    if calibrated_total > 0:
        final_dccs = [score / calibrated_total for score in calibrated_dccs]
    else:
        final_dccs = raw_dccs

    return raw_dccs, final_dccs


def extract_clinical_concept_matrix(signal_multimodal, segments, fs=100):
    return concepts_from_raw_waveform(signal_multimodal, segments, fs=fs)


def calculate_concept_dccs(
    segment_importance,
    concept_matrix,
    concept_channels,
    target_class,
    num_channels,
    num_segments,
):
    segment_matrix = np.asarray(segment_importance, dtype=np.float32).reshape(num_channels, num_segments)
    concept_prior = get_stage_concept_prior(target_class)
    raw_scores = []
    concept_evidence_matrix = np.zeros_like(concept_matrix, dtype=np.float32)

    for concept_idx, channel_idx in enumerate(concept_channels):
        if channel_idx >= num_channels:
            raw_scores.append(0.0)
            continue
        channel_importance = np.abs(segment_matrix[channel_idx])
        concept_activation = np.asarray(concept_matrix[concept_idx], dtype=np.float32)
        concept_evidence_matrix[concept_idx] = channel_importance * concept_activation
        raw_scores.append(float(np.sum(concept_evidence_matrix[concept_idx])))

    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    raw_dccs = _normalize_positive(raw_scores)
    calibrated_scores = raw_scores * concept_prior
    calibrated_dccs = _normalize_positive(calibrated_scores)
    concept_alignment = _cosine_similarity(raw_dccs, concept_prior)

    return {
        "raw_concept_scores": raw_scores.astype(np.float32),
        "raw_concept_dccs": raw_dccs.astype(np.float32),
        "calibrated_concept_dccs": calibrated_dccs.astype(np.float32),
        "clinical_concept_evidence_matrix": concept_evidence_matrix.astype(np.float32),
        "concept_prior": concept_prior.astype(np.float32),
        "concept_alignment": concept_alignment,
    }


def decompose_multimodal_importance(
    segment_importance,
    num_channels,
    num_segments,
    target_class,
    signal_multimodal=None,
    segments=None,
):
    segment_matrix = np.asarray(segment_importance, dtype=np.float32).reshape(num_channels, num_segments)
    prior = get_clinical_prior(target_class)[:num_channels]
    prior_weighted_matrix = segment_matrix * prior[:, np.newaxis]

    positive_matrix = np.clip(segment_matrix, 0.0, None)
    negative_matrix = np.clip(-segment_matrix, 0.0, None)
    temporal_positive = np.sum(positive_matrix, axis=0)
    temporal_negative = np.sum(negative_matrix, axis=0)
    temporal_net = np.sum(segment_matrix, axis=0)

    cooperation_score = []
    conflict_score = []
    dominant_channel = []
    for seg_idx in range(num_segments):
        seg_values = segment_matrix[:, seg_idx]
        pos_sum = float(np.sum(np.clip(seg_values, 0.0, None)))
        neg_sum = float(np.sum(np.clip(-seg_values, 0.0, None)))
        cooperation_score.append(pos_sum)
        conflict_score.append(min(pos_sum, neg_sum))
        dominant_channel.append(int(np.argmax(np.abs(seg_values))))

    abs_channel_scores = np.sum(np.abs(segment_matrix), axis=1)
    prior_alignment = float(
        np.dot(abs_channel_scores, prior) /
        ((np.linalg.norm(abs_channel_scores) * np.linalg.norm(prior)) + 1e-8)
    )
    calibrated_channel_scores = np.sum(np.abs(prior_weighted_matrix), axis=1)
    channel_disagreement = np.abs(abs_channel_scores - calibrated_channel_scores)

    result = {
        "segment_matrix": segment_matrix,
        "prior_weighted_matrix": prior_weighted_matrix,
        "positive_matrix": positive_matrix,
        "negative_matrix": negative_matrix,
        "temporal_positive": np.asarray(temporal_positive, dtype=np.float32),
        "temporal_negative": np.asarray(temporal_negative, dtype=np.float32),
        "temporal_net": np.asarray(temporal_net, dtype=np.float32),
        "cooperation_score": np.asarray(cooperation_score, dtype=np.float32),
        "conflict_score": np.asarray(conflict_score, dtype=np.float32),
        "dominant_channel": np.asarray(dominant_channel, dtype=np.int32),
        "prior_alignment": prior_alignment,
        "channel_disagreement": np.asarray(channel_disagreement, dtype=np.float32),
        "prior_vector": prior.astype(np.float32),
    }

    if signal_multimodal is not None and segments is not None:
        concept_info = extract_clinical_concept_matrix(signal_multimodal, segments)
        concept_dccs = calculate_concept_dccs(
            segment_importance,
            concept_info["concept_matrix"],
            concept_info["concept_channels"],
            target_class,
            num_channels,
            num_segments,
        )
        result.update(
            {
                "clinical_concept_names": concept_info["concept_names"],
                "clinical_concept_channels": concept_info["concept_channels"],
                "clinical_concept_matrix": concept_info["concept_matrix"],
                "clinical_concept_evidence_matrix": concept_dccs["clinical_concept_evidence_matrix"],
                "raw_concept_scores": concept_dccs["raw_concept_scores"],
                "raw_concept_dccs": concept_dccs["raw_concept_dccs"],
                "calibrated_concept_dccs": concept_dccs["calibrated_concept_dccs"],
                "concept_prior": concept_dccs["concept_prior"],
                "concept_alignment": concept_dccs["concept_alignment"],
            }
        )

    return result


def summarize_transition_explanation(epoch_explanations, channel_names, from_stage=None, to_stage=None):
    if not epoch_explanations:
        return None

    target_segments = max(int(len(item["temporal_net"])) for item in epoch_explanations)
    concept_names = epoch_explanations[0].get(
        "clinical_concept_names",
        [item["name"] for item in CLINICAL_CONCEPTS],
    )
    concept_count = len(concept_names)

    def align_vector(values):
        values = np.asarray(values, dtype=np.float32)
        if len(values) >= target_segments:
            return values[:target_segments]
        return np.pad(values, (0, target_segments - len(values)), mode="constant")

    def align_matrix(values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape[1] >= target_segments:
            return values[:, :target_segments]
        pad_width = target_segments - values.shape[1]
        return np.pad(values, ((0, 0), (0, pad_width)), mode="constant")

    def align_concept_vector(values):
        values = np.asarray(values, dtype=np.float32)
        if len(values) >= concept_count:
            return values[:concept_count]
        return np.pad(values, (0, concept_count - len(values)), mode="constant")

    temporal_nets = np.stack([align_vector(item["temporal_net"]) for item in epoch_explanations], axis=0)
    cooperation_scores = np.stack([align_vector(item["cooperation_score"]) for item in epoch_explanations], axis=0)
    conflict_scores = np.stack([align_vector(item["conflict_score"]) for item in epoch_explanations], axis=0)
    segment_matrices = np.stack([align_matrix(item["segment_matrix"]) for item in epoch_explanations], axis=0)
    prior_alignments = np.array([item["prior_alignment"] for item in epoch_explanations], dtype=np.float32)
    concept_scores = np.stack(
        [
            align_concept_vector(item.get("calibrated_concept_dccs", np.zeros(concept_count, dtype=np.float32)))
            for item in epoch_explanations
        ],
        axis=0,
    )

    mean_segment_matrix = np.mean(segment_matrices, axis=0)
    mean_temporal_net = np.mean(temporal_nets, axis=0)
    transition_shift = temporal_nets[-1] - temporal_nets[0]
    dominant_transition_channels = np.argmax(np.abs(mean_segment_matrix), axis=0)

    channel_shift = np.mean(np.abs(segment_matrices[-1]), axis=1) - np.mean(np.abs(segment_matrices[0]), axis=1)
    mean_concept_scores = np.mean(concept_scores, axis=0)
    transition_concept_shift = concept_scores[-1] - concept_scores[0]

    if from_stage is None and "target_class" in epoch_explanations[0]:
        from_stage = int(epoch_explanations[0]["target_class"])
    if to_stage is None and "target_class" in epoch_explanations[-1]:
        to_stage = int(epoch_explanations[-1]["target_class"])

    transition_concept_prior = get_transition_concept_prior(from_stage, to_stage)
    shift_positive = np.clip(transition_concept_shift, 0.0, None)
    if float(np.sum(shift_positive)) <= 1e-8:
        shift_positive = np.abs(transition_concept_shift)
    transition_prior_alignment = _cosine_similarity(
        _normalize_positive(mean_concept_scores + shift_positive),
        transition_concept_prior,
    )
    transition_plausibility = float(0.5 * np.mean(prior_alignments) + 0.5 * transition_prior_alignment)

    return {
        "mean_segment_matrix": mean_segment_matrix,
        "mean_temporal_net": mean_temporal_net,
        "transition_shift": transition_shift.astype(np.float32),
        "mean_cooperation": np.mean(cooperation_scores, axis=0).astype(np.float32),
        "mean_conflict": np.mean(conflict_scores, axis=0).astype(np.float32),
        "prior_alignment_mean": float(np.mean(prior_alignments)),
        "prior_alignment_trend": prior_alignments.astype(np.float32),
        "dominant_transition_channels": dominant_transition_channels.astype(np.int32),
        "channel_shift": channel_shift.astype(np.float32),
        "channel_names": list(channel_names),
        "clinical_concept_names": list(concept_names),
        "mean_concept_scores": mean_concept_scores.astype(np.float32),
        "transition_concept_shift": transition_concept_shift.astype(np.float32),
        "transition_concept_prior": transition_concept_prior.astype(np.float32),
        "transition_prior_alignment": transition_prior_alignment,
        "transition_plausibility": transition_plausibility,
    }


def fit_segments_to_count(segments, total_length, target_segments):
    segments = [
        (max(0, int(start_idx)), min(total_length, int(end_idx)))
        for start_idx, end_idx in segments
        if int(end_idx) > int(start_idx)
    ]
    segments = sorted(segments, key=lambda item: item[0])

    if not segments:
        slice_width = max(1, total_length // target_segments)
        return [
            (idx * slice_width, total_length if idx == target_segments - 1 else min((idx + 1) * slice_width, total_length))
            for idx in range(target_segments)
        ]

    if segments[0][0] > 0:
        segments[0] = (0, segments[0][1])
    if segments[-1][1] < total_length:
        segments[-1] = (segments[-1][0], total_length)

    while len(segments) > target_segments:
        lengths = np.array([end_idx - start_idx for start_idx, end_idx in segments])
        merge_idx = int(np.argmin(lengths))
        if merge_idx == 0:
            segments[1] = (segments[0][0], segments[1][1])
            del segments[0]
        elif merge_idx == len(segments) - 1:
            segments[-2] = (segments[-2][0], segments[-1][1])
            del segments[-1]
        else:
            left_len = segments[merge_idx - 1][1] - segments[merge_idx - 1][0]
            right_len = segments[merge_idx + 1][1] - segments[merge_idx + 1][0]
            if left_len <= right_len:
                segments[merge_idx - 1] = (segments[merge_idx - 1][0], segments[merge_idx][1])
            else:
                segments[merge_idx + 1] = (segments[merge_idx][0], segments[merge_idx + 1][1])
            del segments[merge_idx]

    while len(segments) < target_segments:
        lengths = np.array([end_idx - start_idx for start_idx, end_idx in segments])
        split_idx = int(np.argmax(lengths))
        start_idx, end_idx = segments[split_idx]
        if end_idx - start_idx <= 1:
            break
        mid_idx = (start_idx + end_idx) // 2
        segments[split_idx: split_idx + 1] = [(start_idx, mid_idx), (mid_idx, end_idx)]

    if len(segments) != target_segments:
        boundaries = np.linspace(0, total_length, target_segments + 1, dtype=int)
        segments = [(int(boundaries[idx]), int(boundaries[idx + 1])) for idx in range(target_segments)]

    return segments


def run_multimodal_lime(
    model,
    instance_multimodal,
    segments,
    num_perturbations=500,
    perturbation_mode="physiological",
    prediction_batch_size=16,
    seed=None,
    rng=None,
    **kwargs,
):
    """Run deterministic offline LIME with an explicit local generator."""
    effective_seed = 42 if seed is None and rng is None else seed
    rng = _coerce_rng(seed=effective_seed, rng=rng)
    orig_seg_features = extract_segment_features(instance_multimodal, segments, max_segments=30)
    original_pred = model.predict(
        [instance_multimodal[np.newaxis, :, :], orig_seg_features[np.newaxis, :, :]],
        verbose=0,
    )[0]
    target_class = int(np.argmax(original_pred))

    num_segments = len(segments)
    num_channels = instance_multimodal.shape[1]
    num_features = num_segments * num_channels

    print(
        f"XAI Engine: target stage={target_class}, "
        f"segments={num_segments}, channels={num_channels}, "
        f"perturbations={num_perturbations}, mode={perturbation_mode}, "
        f"prediction_batch_size={max(1, int(prediction_batch_size))}"
    )

    random_perturbations = generate_random_perturbations(
        num_perturbations,
        num_features,
        seed=None,
        rng=rng,
    )
    predictions = predict_perturbations(
        model,
        instance_multimodal,
        random_perturbations,
        segments,
        target_class,
        perturbation_mode=perturbation_mode,
        prediction_batch_size=prediction_batch_size,
        rng=rng,
    )

    distances = calculate_cosine_distances(random_perturbations, num_features)
    weights = calculate_weights_from_distances(distances)
    segment_importance = fit_explainable_model(predictions, random_perturbations, weights, target_class)
    raw_dccs, calibrated_dccs = calculate_dccs(segment_importance, num_channels, num_segments, target_class)
    explanation = decompose_multimodal_importance(
        segment_importance,
        num_channels,
        num_segments,
        target_class,
        signal_multimodal=instance_multimodal,
        segments=segments,
    )

    return {
        "target_class": target_class,
        "seed": None if effective_seed is None else int(effective_seed),
        "perturbation_mask_sha256": hashlib.sha256(
            np.asarray(random_perturbations, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "original_pred": original_pred,
        "perturbation_mode": perturbation_mode,
        "segment_importance": np.asarray(segment_importance, dtype=np.float32),
        "raw_dccs": np.asarray(raw_dccs, dtype=np.float32),
        "calibrated_dccs": np.asarray(calibrated_dccs, dtype=np.float32),
        **explanation,
    }


def semantic_segmentation(instance_multimodal, target_segments=30, min_len=50):
    total_length = instance_multimodal.shape[0]
    num_channels = instance_multimodal.shape[1]
    smoothed_energies = []
    window_size = 100

    for channel_idx in range(num_channels):
        channel_signal = instance_multimodal[:, channel_idx]
        energy = channel_signal ** 2
        smoothed = np.convolve(energy, np.ones(window_size) / window_size, mode="same")
        min_val, max_val = np.min(smoothed), np.max(smoothed)
        if max_val - min_val > 1e-6:
            smoothed_norm = (smoothed - min_val) / (max_val - min_val)
        else:
            smoothed_norm = smoothed
        smoothed_energies.append(smoothed_norm)

    weights = [0.5, 0.25, 0.25]
    total_energy = np.zeros(total_length)
    for channel_idx in range(num_channels):
        total_energy += smoothed_energies[channel_idx] * weights[channel_idx]

    inverted_energy = -total_energy
    valleys, _ = signal.find_peaks(inverted_energy, distance=min_len, prominence=0.005)

    segments = []
    start_idx = 0
    for valley_idx in valleys:
        if valley_idx - start_idx >= min_len:
            segments.append((start_idx, valley_idx))
            start_idx = valley_idx

    if start_idx < total_length:
        segments.append((start_idx, total_length))

    if len(segments) < target_segments // 2:
        slice_width = total_length // target_segments
        segments = [(idx * slice_width, min((idx + 1) * slice_width, total_length)) for idx in range(target_segments)]

    return fit_segments_to_count(segments, total_length, target_segments)


def extract_segment_features(signal_multimodal, segments, max_segments=30, fs=100, include_spectral=True):
    features = []
    for start_idx, end_idx in segments:
        seg_len = end_idx - start_idx
        if seg_len == 0:
            continue

        seg_data = signal_multimodal[start_idx:end_idx, :]
        mean_val = np.mean(seg_data, axis=0)
        std_val = np.std(seg_data, axis=0)
        max_val = np.max(seg_data, axis=0)
        energy = np.sum(seg_data ** 2, axis=0) / seg_len
        feat_parts = [[seg_len], mean_val, std_val, max_val, energy]

        if include_spectral and seg_len >= 8:
            freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
            spectrum = np.abs(np.fft.rfft(seg_data, axis=0)) ** 2
            total_power = np.sum(spectrum, axis=0) + 1e-8
            band_features = []
            for low, high in [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.sum(spectrum[mask], axis=0) / total_power
                else:
                    band_power = np.zeros(seg_data.shape[1])
                band_features.append(band_power)
            feat_parts.append(np.concatenate(band_features))
        elif include_spectral:
            feat_parts.append(np.zeros(seg_data.shape[1] * 5))

        features.append(np.concatenate(feat_parts))

    features = np.asarray(features, dtype=np.float32)
    channel_count = signal_multimodal.shape[1] if len(signal_multimodal.shape) == 2 else 1
    num_features = features.shape[1] if len(features) > 0 else 1 + channel_count * (4 + (5 if include_spectral else 0))

    if len(features) < max_segments:
        pad_width = max_segments - len(features)
        features = np.vstack((features, np.zeros((pad_width, num_features), dtype=np.float32)))
    else:
        features = features[:max_segments]

    return features
