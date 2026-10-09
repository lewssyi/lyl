"""Canonical, versioned clinical-concept definitions for A1-zero and XAI.

Two representations are deliberately exposed. ``segment_proxy_v1`` consumes
the model's 30x28 engineered segment tensor. ``raw_physiology_v1`` derives
evidence directly from waveforms. They share names and channel ownership but
are not claimed to be mathematically equivalent.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np


SEGMENT_PROXY_VERSION = "segment_proxy_v1"
RAW_PHYSIOLOGY_VERSION = "raw_physiology_v1"
SEGMENT_PROXY_V2 = "segment_proxy_v2"

CONCEPT_NAMES_V1 = (
    "EEG Delta",
    "EEG Theta",
    "EEG Alpha",
    "EEG Sigma",
    "EEG Beta",
    "EOG Slow Eye",
    "EOG Rapid Eye",
    "EMG Low Tone",
    "EMG Muscle Activation",
)
CONCEPT_NAMES_V2 = (
    "Slow-wave dominance",
    "Theta light sleep",
    "Alpha wake",
    "Sigma spindle",
    "Beta arousal",
    "EOG slow eye",
    "EOG rapid eye",
    "EMG atonia",
    "EMG active tone",
)
CONCEPT_CHANNEL_INDEX = np.asarray([0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int32)

# Rows are concepts and columns are W/N1/N2/N3/REM.  This is an optional
# analysis prior, never a ground-truth concept annotation.
BASE_STAGE_CONCEPT_PRIOR = np.asarray(
    [
        [0.04, 0.12, 0.18, 0.62, 0.04],
        [0.08, 0.32, 0.20, 0.16, 0.18],
        [0.34, 0.18, 0.04, 0.02, 0.08],
        [0.03, 0.04, 0.36, 0.10, 0.02],
        [0.18, 0.04, 0.02, 0.02, 0.10],
        [0.10, 0.26, 0.12, 0.05, 0.04],
        [0.02, 0.02, 0.02, 0.01, 0.34],
        [0.02, 0.02, 0.02, 0.07, 0.30],
        [0.30, 0.12, 0.14, 0.05, 0.02],
    ],
    dtype=np.float32,
)
BASE_STAGE_CONCEPT_PRIOR /= np.maximum(
    np.sum(BASE_STAGE_CONCEPT_PRIOR, axis=0, keepdims=True), 1e-8
)

BASE_STAGE_CHANNEL_PRIOR = np.asarray(
    [
        [0.30, 0.20, 0.50],
        [0.50, 0.40, 0.10],
        [0.85, 0.05, 0.10],
        [0.90, 0.05, 0.05],
        [0.20, 0.30, 0.50],
    ],
    dtype=np.float32,
)
BASE_STAGE_CHANNEL_PRIOR /= np.maximum(
    np.sum(BASE_STAGE_CHANNEL_PRIOR, axis=1, keepdims=True), 1e-8
)


def normalize_concept_version(version: str | None) -> str:
    key = str(version or SEGMENT_PROXY_VERSION).strip().lower().replace("-", "_")
    aliases = {
        "v1": SEGMENT_PROXY_VERSION,
        "clinical_v1": SEGMENT_PROXY_VERSION,
        SEGMENT_PROXY_VERSION: SEGMENT_PROXY_VERSION,
        RAW_PHYSIOLOGY_VERSION: RAW_PHYSIOLOGY_VERSION,
        "raw": RAW_PHYSIOLOGY_VERSION,
        "v2": SEGMENT_PROXY_V2,
        "clinical_v2": SEGMENT_PROXY_V2,
        SEGMENT_PROXY_V2: SEGMENT_PROXY_V2,
    }
    if key not in aliases:
        raise ValueError(f"Unknown clinical concept version: {version!r}")
    return aliases[key]


def concept_names(version: str | None = SEGMENT_PROXY_VERSION) -> list[str]:
    version = normalize_concept_version(version)
    return list(CONCEPT_NAMES_V2 if version == SEGMENT_PROXY_V2 else CONCEPT_NAMES_V1)


def concept_metadata(version: str | None = SEGMENT_PROXY_VERSION) -> list[dict[str, object]]:
    return [
        {"name": name, "channel": int(channel), "definition_version": normalize_concept_version(version)}
        for name, channel in zip(concept_names(version), CONCEPT_CHANNEL_INDEX)
    ]


def stage_concept_prior(num_concepts: int = 9, num_classes: int = 5) -> np.ndarray:
    return BASE_STAGE_CONCEPT_PRIOR[: int(num_concepts), : int(num_classes)].copy()


def stage_channel_prior(num_classes: int = 5, num_channels: int = 3) -> np.ndarray:
    return BASE_STAGE_CHANNEL_PRIOR[: int(num_classes), : int(num_channels)].copy()


def _numpy_segment_feature(values: np.ndarray, index: int) -> np.ndarray:
    if index >= values.shape[-1]:
        return np.zeros(values.shape[:-1], dtype=np.float32)
    return values[..., index].astype(np.float32)


def _segment_proxy_v1_numpy(values: np.ndarray) -> np.ndarray:
    eeg_delta = _numpy_segment_feature(values, 13)
    eog_delta = _numpy_segment_feature(values, 14)
    eeg_theta = _numpy_segment_feature(values, 16)
    eog_theta = _numpy_segment_feature(values, 17)
    eeg_alpha = _numpy_segment_feature(values, 19)
    eeg_sigma = _numpy_segment_feature(values, 22)
    eeg_beta = _numpy_segment_feature(values, 25)
    eog_beta = _numpy_segment_feature(values, 26)
    emg_beta = _numpy_segment_feature(values, 27)
    eog_energy = _numpy_segment_feature(values, 11)
    emg_energy = _numpy_segment_feature(values, 12)
    eog_slow = 0.70 * eog_delta + 0.30 * eog_theta
    eog_rapid = 0.55 * eog_energy + 0.30 * eog_theta + 0.15 * eog_beta
    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
    emg_low_tone = 1.0 / (1.0 + np.maximum(emg_activation, 0.0))
    return np.stack(
        [
            eeg_delta,
            eeg_theta,
            eeg_alpha,
            eeg_sigma,
            eeg_beta,
            eog_slow,
            eog_rapid,
            emg_low_tone,
            emg_activation,
        ],
        axis=-1,
    ).astype(np.float32)


def _segment_proxy_v2_numpy(values: np.ndarray) -> np.ndarray:
    eps = 1e-6
    eeg_delta = _numpy_segment_feature(values, 13)
    eog_delta = _numpy_segment_feature(values, 14)
    eeg_theta = _numpy_segment_feature(values, 16)
    eog_theta = _numpy_segment_feature(values, 17)
    eeg_alpha = _numpy_segment_feature(values, 19)
    eeg_sigma = _numpy_segment_feature(values, 22)
    eeg_beta = _numpy_segment_feature(values, 25)
    eog_beta = _numpy_segment_feature(values, 26)
    emg_beta = _numpy_segment_feature(values, 27)
    eog_energy = _numpy_segment_feature(values, 11)
    emg_energy = _numpy_segment_feature(values, 12)
    eeg_total = eeg_delta + eeg_theta + eeg_alpha + eeg_sigma + eeg_beta + eps
    sigma_median = np.median(eeg_sigma, axis=-1, keepdims=True)
    sigma_p90 = np.percentile(eeg_sigma, 90, axis=-1, keepdims=True)
    sigma_burstiness = (sigma_p90 - sigma_median) / (sigma_median + eps)
    if eog_energy.shape[-1] > 1:
        variation = np.abs(np.diff(eog_energy, axis=-1))
        variation = np.concatenate([variation, variation[..., -1:]], axis=-1)
        eog_rapid = 0.60 * variation + 0.25 * eog_energy + 0.15 * eog_beta
    else:
        eog_rapid = 0.70 * eog_energy + 0.30 * eog_beta
    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
    emg_center = np.median(emg_activation, axis=-1, keepdims=True)
    emg_iqr = np.percentile(emg_activation, 75, axis=-1, keepdims=True) - np.percentile(
        emg_activation, 25, axis=-1, keepdims=True
    )
    emg_z = np.clip((emg_activation - emg_center) / (emg_iqr + eps), -60.0, 60.0)
    return np.stack(
        [
            eeg_delta / eeg_total,
            eeg_theta / eeg_total,
            eeg_alpha / eeg_total,
            (eeg_sigma / eeg_total) * (1.0 + 0.5 * sigma_burstiness),
            eeg_beta / eeg_total,
            0.70 * eog_delta + 0.30 * eog_theta,
            eog_rapid,
            1.0 / (1.0 + np.exp(emg_z)),
            1.0 / (1.0 + np.exp(-emg_z)),
        ],
        axis=-1,
    ).astype(np.float32)


def concepts_from_segment_features(
    segment_features: np.ndarray,
    version: str | None = SEGMENT_PROXY_VERSION,
    reduce_segments: bool = False,
) -> np.ndarray:
    values = np.asarray(segment_features, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("segment_features must end with (segments, features)")
    version = normalize_concept_version(version)
    concepts = (
        _segment_proxy_v2_numpy(values)
        if version == SEGMENT_PROXY_V2
        else _segment_proxy_v1_numpy(values)
    )
    if reduce_segments:
        concepts = np.mean(np.clip(concepts, 0.0, None), axis=-2)
        row_max = np.max(concepts, axis=-1, keepdims=True)
        concepts = concepts / np.maximum(row_max, 1e-8)
    return np.nan_to_num(concepts, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def concepts_from_segment_tensor(segment_tensor, version: str | None = SEGMENT_PROXY_VERSION):
    """TensorFlow counterpart used by the classifier without changing its graph."""
    import tensorflow as tf

    version = normalize_concept_version(version)
    x = segment_tensor
    eeg_delta, eog_delta = x[..., 13:14], x[..., 14:15]
    eeg_theta, eog_theta = x[..., 16:17], x[..., 17:18]
    eeg_alpha, eeg_sigma = x[..., 19:20], x[..., 22:23]
    eeg_beta, eog_beta, emg_beta = x[..., 25:26], x[..., 26:27], x[..., 27:28]
    eog_energy, emg_energy = x[..., 11:12], x[..., 12:13]
    if version != SEGMENT_PROXY_V2:
        eog_slow = 0.70 * eog_delta + 0.30 * eog_theta
        eog_rapid = 0.55 * eog_energy + 0.30 * eog_theta + 0.15 * eog_beta
        emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
        return tf.concat(
            [
                eeg_delta,
                eeg_theta,
                eeg_alpha,
                eeg_sigma,
                eeg_beta,
                eog_slow,
                eog_rapid,
                1.0 / (1.0 + tf.nn.relu(emg_activation)),
                emg_activation,
            ],
            axis=-1,
        )
    eps = tf.cast(1e-6, x.dtype)
    eeg_total = eeg_delta + eeg_theta + eeg_alpha + eeg_sigma + eeg_beta + eps
    sorted_sigma = tf.sort(eeg_sigma, axis=-2)
    n_steps = tf.shape(sorted_sigma)[-2]

    def percentile(p):
        position = (p / 100.0) * tf.cast(tf.maximum(n_steps - 1, 0), tf.float32)
        return tf.gather(sorted_sigma, tf.cast(tf.round(position), tf.int32), axis=-2)

    sigma_median = tf.expand_dims(percentile(50.0), axis=-2)
    sigma_p90 = tf.expand_dims(percentile(90.0), axis=-2)
    sigma_burstiness = (sigma_p90 - sigma_median) / (sigma_median + eps)

    def rapid_multi():
        variation = tf.abs(eog_energy[..., 1:, :] - eog_energy[..., :-1, :])
        variation = tf.concat([variation, variation[..., -1:, :]], axis=-2)
        return 0.60 * variation + 0.25 * eog_energy + 0.15 * eog_beta

    eog_rapid = tf.cond(
        tf.shape(eog_energy)[-2] > 1,
        rapid_multi,
        lambda: 0.70 * eog_energy + 0.30 * eog_beta,
    )
    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
    sorted_emg = tf.sort(emg_activation, axis=-2)

    def emg_percentile(p):
        position = (p / 100.0) * tf.cast(tf.maximum(tf.shape(sorted_emg)[-2] - 1, 0), tf.float32)
        return tf.expand_dims(
            tf.gather(sorted_emg, tf.cast(tf.round(position), tf.int32), axis=-2), axis=-2
        )

    emg_z = tf.clip_by_value(
        (emg_activation - emg_percentile(50.0))
        / (emg_percentile(75.0) - emg_percentile(25.0) + eps),
        -60.0,
        60.0,
    )
    return tf.concat(
        [
            eeg_delta / eeg_total,
            eeg_theta / eeg_total,
            eeg_alpha / eeg_total,
            (eeg_sigma / eeg_total) * (1.0 + 0.5 * sigma_burstiness),
            eeg_beta / eeg_total,
            0.70 * eog_delta + 0.30 * eog_theta,
            eog_rapid,
            tf.sigmoid(-emg_z),
            tf.sigmoid(emg_z),
        ],
        axis=-1,
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _relative_band_power(values: np.ndarray, fs: float, low: float, high: float) -> float:
    if len(values) < 8 or float(np.std(values)) < 1e-8:
        return 0.0
    freqs = np.fft.rfftfreq(len(values), d=1.0 / float(fs))
    power = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask]) / (np.sum(power) + 1e-8)) if np.any(mask) else 0.0


def concepts_from_raw_waveform(
    waveform: np.ndarray,
    segments: Sequence[tuple[int, int]],
    fs: float = 100.0,
) -> Mapping[str, object]:
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    matrix = np.zeros((len(CONCEPT_NAMES_V1), len(segments)), dtype=np.float32)
    for segment_index, (start, end) in enumerate(segments):
        piece = values[int(start) : int(end)]
        if not len(piece):
            continue
        if piece.shape[1] > 0:
            eeg = piece[:, 0]
            for concept_index, band in enumerate(
                ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
            ):
                matrix[concept_index, segment_index] = _relative_band_power(eeg, fs, *band)
        if piece.shape[1] > 1:
            eog = piece[:, 1]
            matrix[5, segment_index] = _relative_band_power(eog, fs, 0.1, 1.0) + 0.10 * np.std(eog)
            matrix[6, segment_index] = _relative_band_power(eog, fs, 1.0, 5.0) + 0.10 * np.max(np.abs(eog))
        if piece.shape[1] > 2:
            emg = piece[:, 2]
            matrix[8, segment_index] = np.sqrt(np.mean(emg**2) + 1e-8) + 0.25 * np.percentile(np.abs(emg), 95)
    matrix = np.stack([_minmax(row) for row in matrix], axis=0)
    if values.shape[1] > 2:
        matrix[7] = 1.0 - matrix[8]
    for index, channel in enumerate(CONCEPT_CHANNEL_INDEX):
        if channel >= values.shape[1]:
            matrix[index] = 0.0
    return {
        "definition_version": RAW_PHYSIOLOGY_VERSION,
        "concept_names": concept_names(RAW_PHYSIOLOGY_VERSION),
        "concept_channels": CONCEPT_CHANNEL_INDEX.copy(),
        "concept_matrix": matrix.astype(np.float32),
    }


def validate_raw_segment_concept_parity(
    waveform: np.ndarray,
    segment_features: np.ndarray,
    segments: Sequence[tuple[int, int]],
    fs: float = 100.0,
) -> dict[str, object]:
    raw = concepts_from_raw_waveform(waveform, segments, fs=fs)["concept_matrix"]
    proxy = concepts_from_segment_features(segment_features, SEGMENT_PROXY_VERSION)
    proxy = np.asarray(proxy, dtype=np.float32).T
    count = min(raw.shape[1], proxy.shape[1])
    raw, proxy = raw[:, :count], proxy[:, :count]
    per_concept_correlation = []
    for raw_row, proxy_row in zip(raw, proxy):
        if np.std(raw_row) <= 1e-8 or np.std(proxy_row) <= 1e-8:
            per_concept_correlation.append(None)
        else:
            per_concept_correlation.append(float(np.corrcoef(raw_row, proxy_row)[0, 1]))
    return {
        "segment_definition_version": SEGMENT_PROXY_VERSION,
        "raw_definition_version": RAW_PHYSIOLOGY_VERSION,
        "mathematically_equivalent": False,
        "reason": "The segment proxy uses engineered 30x28 features; raw physiology recomputes waveform evidence and per-epoch scaling.",
        "segments_compared": int(count),
        "mean_absolute_difference": float(np.mean(np.abs(raw - proxy))) if count else None,
        "per_concept_pearson": per_concept_correlation,
    }


def assert_label_conditioning_allowed(
    split: str,
    labels,
    concept_prior_weight: float,
    label_conditioned: bool = False,
) -> None:
    uses_labels = labels is not None and float(concept_prior_weight) > 0.0
    if uses_labels and (str(split).lower() != "train" or not label_conditioned):
        raise ValueError(
            "Label-conditioned concept targets require split='train' and explicit label_conditioned=True."
        )
    if uses_labels:
        warnings.warn(
            "LABEL-CONDITIONED CONCEPT TARGET ENABLED: this is an explicit ablation, not an independent clinical annotation.",
            RuntimeWarning,
            stacklevel=2,
        )
