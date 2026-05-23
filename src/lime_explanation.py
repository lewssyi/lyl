import copy

import numpy as np
import scipy.signal as signal
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


def perturb_total_mean(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = np.mean(signal_1d)


def perturb_mean(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = np.mean(signal_1d[start_idx:end_idx])


def perturb_noise(signal_1d, start_idx, end_idx):
    if start_idx == end_idx:
        return
    signal_1d[start_idx:end_idx] = np.random.uniform(
        np.min(signal_1d), np.max(signal_1d), end_idx - start_idx
    )


def get_clinical_prior(target_class):
    clinical_priors = {
        0: [0.30, 0.20, 0.50],
        1: [0.50, 0.40, 0.10],
        2: [0.85, 0.05, 0.10],
        3: [0.90, 0.05, 0.05],
        4: [0.20, 0.30, 0.50],
    }
    return np.array(clinical_priors.get(target_class, [0.33, 0.33, 0.34]), dtype=np.float32)


def apply_stage_aware_perturbation(signal_1d, start_idx, end_idx, channel_idx, target_class):
    if channel_idx == 2:
        if target_class == 4:
            perturb_noise(signal_1d, start_idx, end_idx)
        else:
            perturb_mean(signal_1d, start_idx, end_idx)
    elif channel_idx == 1:
        if target_class in [0, 4]:
            perturb_mean(signal_1d, start_idx, end_idx)
        else:
            perturb_noise(signal_1d, start_idx, end_idx)
    else:
        perturb_mean(signal_1d, start_idx, end_idx)


def generate_random_perturbations(num_perturbations, num_features):
    return np.random.binomial(1, 0.5, size=(num_perturbations, num_features))


def calculate_cosine_distances(random_perturbations, num_features):
    original_rep = np.ones((1, num_features))
    return pairwise_distances(random_perturbations, original_rep, metric="cosine").ravel()


def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    return np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))


def apply_perturbation_to_multimodal(signal_multimodal, perturbation, segments, target_class):
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
                        perturbed_signal[:, channel_idx], start_idx, end_idx, channel_idx, target_class
                    )
                else:
                    apply_stage_aware_perturbation(perturbed_signal, start_idx, end_idx, 0, target_class)
    return perturbed_signal


def predict_perturbations(model, instance_multimodal, random_perturbations, segments, target_class):
    perturbation_predictions = []
    for perturbation in random_perturbations:
        perturbed_signal = apply_perturbation_to_multimodal(
            instance_multimodal, perturbation, segments, target_class
        )
        perturbed_seg_features = extract_segment_features(perturbed_signal, segments, max_segments=30)
        sig_reshaped = perturbed_signal[np.newaxis, :, :]
        seg_reshaped = perturbed_seg_features[np.newaxis, :, :]
        model_prediction = model.predict([sig_reshaped, seg_reshaped], verbose=0)
        perturbation_predictions.append(model_prediction[0])
    return np.asarray(perturbation_predictions, dtype=np.float32)


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


def decompose_multimodal_importance(segment_importance, num_channels, num_segments, target_class):
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

    return {
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


def summarize_transition_explanation(epoch_explanations, channel_names):
    if not epoch_explanations:
        return None

    target_segments = max(int(len(item["temporal_net"])) for item in epoch_explanations)

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

    temporal_nets = np.stack([align_vector(item["temporal_net"]) for item in epoch_explanations], axis=0)
    cooperation_scores = np.stack([align_vector(item["cooperation_score"]) for item in epoch_explanations], axis=0)
    conflict_scores = np.stack([align_vector(item["conflict_score"]) for item in epoch_explanations], axis=0)
    segment_matrices = np.stack([align_matrix(item["segment_matrix"]) for item in epoch_explanations], axis=0)
    prior_alignments = np.array([item["prior_alignment"] for item in epoch_explanations], dtype=np.float32)

    mean_segment_matrix = np.mean(segment_matrices, axis=0)
    mean_temporal_net = np.mean(temporal_nets, axis=0)
    transition_shift = temporal_nets[-1] - temporal_nets[0]
    dominant_transition_channels = np.argmax(np.abs(mean_segment_matrix), axis=0)

    channel_shift = np.mean(np.abs(segment_matrices[-1]), axis=1) - np.mean(np.abs(segment_matrices[0]), axis=1)

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


def run_multimodal_lime(model, instance_multimodal, segments, num_perturbations=500, **kwargs):
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
        f"segments={num_segments}, channels={num_channels}, perturbations={num_perturbations}"
    )

    random_perturbations = generate_random_perturbations(num_perturbations, num_features)
    predictions = predict_perturbations(model, instance_multimodal, random_perturbations, segments, target_class)

    distances = calculate_cosine_distances(random_perturbations, num_features)
    weights = calculate_weights_from_distances(distances)
    segment_importance = fit_explainable_model(predictions, random_perturbations, weights, target_class)
    raw_dccs, calibrated_dccs = calculate_dccs(segment_importance, num_channels, num_segments, target_class)
    explanation = decompose_multimodal_importance(segment_importance, num_channels, num_segments, target_class)

    return {
        "target_class": target_class,
        "original_pred": original_pred,
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
