import os

import numpy as np
from scipy import signal
from tensorflow.keras.utils import to_categorical


def _as_3d_float(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[..., np.newaxis]
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (epochs, samples, channels), got {X.shape}")
    return X


def _channel_names(n_channels, channel_names=None):
    if channel_names is None:
        defaults = ["EEG", "EOG", "EMG"]
        return [
            defaults[idx] if idx < len(defaults) else f"CH{idx + 1}"
            for idx in range(n_channels)
        ]

    names = list(channel_names)
    if len(names) < n_channels:
        names.extend(f"CH{idx + 1}" for idx in range(len(names), n_channels))
    return names[:n_channels]


def _emit(logger, message):
    if logger is None:
        print(message)
    elif hasattr(logger, "info"):
        logger.info(message)
    elif hasattr(logger, "write"):
        logger.write(f"{message}\n")
    elif callable(logger):
        logger(message)
    else:
        print(message)


def sanitize_signal_epochs(X, clip_value=None):
    """
    Keep this as numerical safety only.

    The project no longer performs global amplitude clipping because large
    physiological waves are important for N2/N3 staging. The clip_value
    argument is accepted for backward compatibility and intentionally ignored.
    """
    X = _as_3d_float(X)
    if np.all(np.isfinite(X)):
        return X.astype(np.float32)
    return np.nan_to_num(X, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def infer_subject_id_from_npz(filename):
    """Infer the subject identifier used for leakage-safe splitting."""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    upper_stem = stem.upper()
    if (
        len(upper_stem) >= 5
        and upper_stem[:2] in {"SC", "ST"}
        and upper_stem[2:5].isdigit()
    ):
        return upper_stem[:5]
    return stem


def _robust_positive_z(metric):
    metric = np.asarray(metric, dtype=np.float32)
    center = np.median(metric, axis=0, keepdims=True)
    mad = np.median(np.abs(metric - center), axis=0, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.std(metric, axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, fallback + 1e-6, scale)
    return (metric - center) / scale


def recording_robust_zscore(X, eps=1e-6):
    """
    Robust z-score within a single recording and channel.

    This intentionally uses only the current recording's signal statistics, so
    deployment can apply the same transform without train/test population fits.
    """
    X = sanitize_signal_epochs(X)
    center = np.median(X, axis=(0, 1), keepdims=True)
    mad = np.median(np.abs(X - center), axis=(0, 1), keepdims=True)
    robust_scale = 1.4826 * mad
    fallback_scale = np.std(X, axis=(0, 1), keepdims=True)
    scale = np.where(robust_scale > eps, robust_scale, fallback_scale)
    scale = np.where(scale > eps, scale, 1.0)
    return sanitize_signal_epochs((X - center) / (scale + eps))


def _secondwise_step_metric(X_safe, fs):
    n_epochs, n_samples, n_channels = X_safe.shape
    samples_per_bin = max(1, int(round(float(fs) if fs is not None else 100.0)))
    n_bins = n_samples // samples_per_bin
    if n_bins < 2:
        return np.zeros((n_epochs, n_channels), dtype=np.float32)

    trimmed = X_safe[:, : n_bins * samples_per_bin, :]
    second_means = trimmed.reshape(n_epochs, n_bins, samples_per_bin, n_channels).mean(axis=2)
    return np.max(np.abs(np.diff(second_means, axis=1)), axis=1).astype(np.float32)


def compute_epoch_quality_scores(
    X,
    fs=100,
    flat_std_threshold=1e-4,
    flat_std_ratio=0.02,
    step_z_threshold=8.0,
    noise_z_threshold=10.0,
    min_quality=0.10,
    max_quality=1.00,
    min_step_channel_ratio=0.67,
    return_flags=False,
    channel_names=None,
    return_report=False,
    report_name=None,
    logger=None,
):
    """
    Compute a continuous quality confidence Q_t in [0.1, 1.0] for each epoch.

    This function is report/postprocessing oriented: it does not clip signal
    amplitudes, does not repair samples, and does not create sample weights.
    Low Q_t means the model emission should be trusted less by Viterbi.
    """
    X = _as_3d_float(X)
    n_epochs, _, n_channels = X.shape
    names = _channel_names(n_channels, channel_names)

    finite_channel = np.all(np.isfinite(X), axis=1)
    finite_epoch = np.all(finite_channel, axis=1)
    X_safe = np.nan_to_num(X, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    channel_std = np.std(X_safe, axis=1)
    reference_std = np.median(channel_std, axis=0, keepdims=True)
    flat_limit = np.maximum(flat_std_threshold, reference_std * flat_std_ratio)
    flat_severity_channel = np.clip((flat_limit - channel_std) / (flat_limit + 1e-6), 0.0, 1.0)
    flat_channel = flat_severity_channel > 0.50
    flat_severity = np.max(flat_severity_channel, axis=1)

    step_metric = _secondwise_step_metric(X_safe, fs)
    step_z = _robust_positive_z(step_metric)
    step_hit_channel = step_z >= step_z_threshold
    step_support = np.mean(step_hit_channel, axis=1)
    step_severity_channel = np.clip((step_z - step_z_threshold) / max(step_z_threshold, 1e-6), 0.0, 1.0)
    step_raw_severity = np.max(step_severity_channel, axis=1)
    step_context_factor = np.where(step_support >= min_step_channel_ratio, 1.0, 0.45)
    step_severity = step_raw_severity * step_context_factor

    if X_safe.shape[1] > 1:
        diff_abs = np.abs(np.diff(X_safe, axis=1))
        noise_metric = np.percentile(diff_abs, 99.5, axis=1).astype(np.float32)
        noise_z = _robust_positive_z(noise_metric)
        noise_severity_channel = np.clip((noise_z - noise_z_threshold) / max(noise_z_threshold, 1e-6), 0.0, 1.0)
        noise_hit_channel = noise_z >= noise_z_threshold
        noise_severity = np.max(noise_severity_channel, axis=1) * 0.70
    else:
        noise_metric = np.zeros((n_epochs, n_channels), dtype=np.float32)
        noise_z = np.zeros((n_epochs, n_channels), dtype=np.float32)
        noise_hit_channel = np.zeros((n_epochs, n_channels), dtype=bool)
        noise_severity = np.zeros(n_epochs, dtype=np.float32)

    nonfinite_severity = np.mean(~finite_channel, axis=1).astype(np.float32)
    component_severity = np.stack(
        [
            nonfinite_severity,
            0.95 * flat_severity,
            0.95 * step_severity,
            noise_severity,
        ],
        axis=1,
    )
    severity = np.clip(np.max(component_severity, axis=1), 0.0, 1.0)
    quality_scores = 1.0 - 0.90 * severity
    quality_scores = np.clip(quality_scores, min_quality, max_quality).astype(np.float32)

    flat_epoch = np.any(flat_channel, axis=1)
    step_epoch = step_support >= min_step_channel_ratio
    noise_epoch = np.any(noise_hit_channel, axis=1)
    flagged_epoch = (~finite_epoch) | flat_epoch | step_epoch | noise_epoch

    flags = {
        "nonfinite": ~finite_epoch,
        "flat": flat_epoch,
        "step": step_epoch,
        "noise": noise_epoch,
        "flagged": flagged_epoch,
        "nonfinite_channel": ~finite_channel,
        "flat_channel": flat_channel,
        "step_channel": step_hit_channel,
        "noise_channel": noise_hit_channel,
        "step_z": step_z.astype(np.float32),
        "noise_z": noise_z.astype(np.float32),
    }

    report = {
        "name": report_name,
        "epochs": int(n_epochs),
        "channels": int(n_channels),
        "mean_quality": float(np.mean(quality_scores)) if n_epochs else 1.0,
        "min_quality": float(np.min(quality_scores)) if n_epochs else 1.0,
        "low_quality_epochs": int(np.sum(quality_scores < 0.70)),
        "severe_quality_epochs": int(np.sum(quality_scores < 0.40)),
        "flagged_epochs": int(np.sum(flagged_epoch)),
        "flagged_ratio": float(np.mean(flagged_epoch)) if n_epochs else 0.0,
        "nonfinite_epochs": int(np.sum(~finite_epoch)),
        "flat_epochs": int(np.sum(flat_epoch)),
        "step_epochs": int(np.sum(step_epoch)),
        "noise_epochs": int(np.sum(noise_epoch)),
        "channel_reports": [],
    }
    for channel_idx, name in enumerate(names):
        report["channel_reports"].append(
            {
                "channel": name,
                "nonfinite_epochs": int(np.sum(~finite_channel[:, channel_idx])),
                "flat_epochs": int(np.sum(flat_channel[:, channel_idx])),
                "step_epochs": int(np.sum(step_hit_channel[:, channel_idx])),
                "noise_epochs": int(np.sum(noise_hit_channel[:, channel_idx])),
                "median_std": float(np.median(channel_std[:, channel_idx])) if n_epochs else 0.0,
                "max_step_z": float(np.max(step_z[:, channel_idx])) if n_epochs else 0.0,
                "max_noise_z": float(np.max(noise_z[:, channel_idx])) if n_epochs else 0.0,
            }
        )

    if logger is not None or report_name is not None:
        log_quality_report(report, logger=logger)

    if return_flags and return_report:
        return quality_scores, flags, report
    if return_report:
        return quality_scores, report
    if return_flags:
        return quality_scores, flags
    return quality_scores


def compute_epoch_channel_quality_scores(
    X,
    fs=100,
    flat_std_threshold=1e-4,
    flat_std_ratio=0.02,
    step_z_threshold=8.0,
    noise_z_threshold=10.0,
    min_quality=0.10,
    max_quality=1.00,
):
    """Compute per-epoch, per-channel quality scores in [min_quality, max_quality]."""
    X = _as_3d_float(X)
    finite_channel = np.all(np.isfinite(X), axis=1)
    X_safe = np.nan_to_num(X, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    channel_std = np.std(X_safe, axis=1)
    reference_std = np.median(channel_std, axis=0, keepdims=True)
    flat_limit = np.maximum(flat_std_threshold, reference_std * flat_std_ratio)
    flat_severity = np.clip((flat_limit - channel_std) / (flat_limit + 1e-6), 0.0, 1.0)

    step_metric = _secondwise_step_metric(X_safe, fs)
    step_z = _robust_positive_z(step_metric)
    step_severity = np.clip((step_z - step_z_threshold) / max(step_z_threshold, 1e-6), 0.0, 1.0)

    if X_safe.shape[1] > 1:
        diff_abs = np.abs(np.diff(X_safe, axis=1))
        noise_metric = np.percentile(diff_abs, 99.5, axis=1).astype(np.float32)
        noise_z = _robust_positive_z(noise_metric)
        noise_severity = np.clip((noise_z - noise_z_threshold) / max(noise_z_threshold, 1e-6), 0.0, 1.0)
    else:
        noise_severity = np.zeros_like(channel_std, dtype=np.float32)

    nonfinite_severity = (~finite_channel).astype(np.float32)
    severity = np.maximum.reduce(
        [
            nonfinite_severity,
            0.95 * flat_severity,
            0.95 * step_severity,
            0.70 * noise_severity,
        ]
    )
    channel_quality = 1.0 - 0.90 * np.clip(severity, 0.0, 1.0)
    return np.clip(channel_quality, min_quality, max_quality).astype(np.float32)


def compute_epoch_quality_weights(
    X,
    flat_std_threshold=1e-4,
    extreme_z_threshold=10.0,
    extreme_ratio_threshold=0.02,
    jump_z_threshold=8.0,
    min_weight=0.10,
    return_flags=False,
    enable_quality_downweighting=False,
    channel_names=None,
    return_report=False,
    report_name=None,
    logger=None,
):
    """
    Backward-compatible wrapper.

    Older code called these values "quality weights". They are now quality
    confidence scores Q_t for adaptive Viterbi and must not be used as a
    default training sample-weighting mechanism.
    """
    _ = extreme_ratio_threshold, enable_quality_downweighting
    return compute_epoch_quality_scores(
        X,
        flat_std_threshold=flat_std_threshold,
        step_z_threshold=jump_z_threshold,
        noise_z_threshold=extreme_z_threshold,
        min_quality=min_weight,
        return_flags=return_flags,
        channel_names=channel_names,
        return_report=return_report,
        report_name=report_name,
        logger=logger,
    )


def log_quality_report(report, logger=None):
    prefix = f"{report['name']} " if report.get("name") else ""
    _emit(
        logger,
        (
            f"{prefix}Clinical data quality report: "
            f"epochs={report['epochs']}, "
            f"mean_Q={report['mean_quality']:.3f}, "
            f"min_Q={report['min_quality']:.3f}, "
            f"low_Q<0.70={report['low_quality_epochs']}, "
            f"severe_Q<0.40={report['severe_quality_epochs']}, "
            f"flat={report['flat_epochs']}, "
            f"step={report['step_epochs']}, "
            f"noise={report['noise_epochs']}, "
            f"nonfinite={report['nonfinite_epochs']}"
        ),
    )
    for item in report["channel_reports"]:
        _emit(
            logger,
            (
                f"  {item['channel']}: "
                f"flat={item['flat_epochs']}, "
                f"step={item['step_epochs']}, "
                f"noise={item['noise_epochs']}, "
                f"nonfinite={item['nonfinite_epochs']}, "
                f"median_std={item['median_std']:.4g}, "
                f"max_step_z={item['max_step_z']:.2f}, "
                f"max_noise_z={item['max_noise_z']:.2f}"
            ),
        )


def summarize_quality_reports(reports, quality_scores, report_name=None):
    quality_scores = np.asarray(quality_scores, dtype=np.float32)
    total_epochs = int(sum(item["epochs"] for item in reports))
    n_channels = int(max((item["channels"] for item in reports), default=0))
    channel_names = _channel_names(n_channels)
    channel_reports = []

    for channel_idx, channel_name in enumerate(channel_names):
        channel_reports.append(
            {
                "channel": channel_name,
                "nonfinite_epochs": int(sum(
                    item["channel_reports"][channel_idx]["nonfinite_epochs"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                )),
                "flat_epochs": int(sum(
                    item["channel_reports"][channel_idx]["flat_epochs"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                )),
                "step_epochs": int(sum(
                    item["channel_reports"][channel_idx]["step_epochs"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                )),
                "noise_epochs": int(sum(
                    item["channel_reports"][channel_idx]["noise_epochs"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                )),
                "median_std": float(np.median([
                    item["channel_reports"][channel_idx]["median_std"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                ])) if reports else 0.0,
                "max_step_z": float(max([
                    item["channel_reports"][channel_idx]["max_step_z"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                ], default=0.0)),
                "max_noise_z": float(max([
                    item["channel_reports"][channel_idx]["max_noise_z"]
                    for item in reports
                    if channel_idx < len(item["channel_reports"])
                ], default=0.0)),
            }
        )

    return {
        "name": report_name,
        "epochs": total_epochs,
        "channels": n_channels,
        "mean_quality": float(np.mean(quality_scores)) if len(quality_scores) else 1.0,
        "min_quality": float(np.min(quality_scores)) if len(quality_scores) else 1.0,
        "low_quality_epochs": int(np.sum(quality_scores < 0.70)),
        "severe_quality_epochs": int(np.sum(quality_scores < 0.40)),
        "flagged_epochs": int(sum(item["flagged_epochs"] for item in reports)),
        "flagged_ratio": (
            float(sum(item["flagged_epochs"] for item in reports) / max(total_epochs, 1))
            if reports else 0.0
        ),
        "nonfinite_epochs": int(sum(item["nonfinite_epochs"] for item in reports)),
        "flat_epochs": int(sum(item["flat_epochs"] for item in reports)),
        "step_epochs": int(sum(item["step_epochs"] for item in reports)),
        "noise_epochs": int(sum(item["noise_epochs"] for item in reports)),
        "channel_reports": channel_reports,
    }


def _profile_key(value):
    return str(value or "clinical_canonical").strip().lower().replace("-", "_")


def _clamped_band(low, high, fs):
    fs = float(fs) if fs is not None else 100.0
    nyquist = fs / 2.0
    low = max(float(low), 1e-4)
    high = min(float(high), nyquist * 0.95)
    if high <= low:
        return None
    return low, high


def _apply_bandpass_to_channel(values, low, high, fs, backend="scipy"):
    band = _clamped_band(low, high, fs)
    if band is None:
        return values.astype(np.float32)

    low, high = band
    backend_key = str(backend or "scipy").strip().lower()
    if backend_key == "mne":
        try:
            import mne
        except ImportError as exc:
            raise ImportError(
                "LFTS_PREPROCESS_FILTER_BACKEND=mne requires mne. Install it under "
                "the project-local .deps directory or switch the backend to scipy."
            ) from exc
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[1])
        filtered = mne.filter.filter_data(
            flat,
            sfreq=float(fs),
            l_freq=float(low),
            h_freq=float(high),
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
        return filtered.reshape(values.shape).astype(np.float32)

    if backend_key not in {"scipy", "sos", "butter"}:
        raise ValueError(f"Unknown filter backend: {backend!r}")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=float(fs), output="sos")
    try:
        return signal.sosfiltfilt(sos, values, axis=1).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, values, axis=1).astype(np.float32)


def clinical_filter_bands(filter_profile="clinical_canonical", fs=100):
    """
    Return channel-specific bands for the shared clinical concept space.

    The canonical profile intentionally keeps EOG below 0.5 Hz for slow eye
    movement concepts and gives EMG its own muscle-activity band. The legacy
    profile preserves the historical project behavior for reproducibility.
    """
    key = _profile_key(filter_profile)
    if key in {"none", "off", "false", "no_filter", "nofilter"}:
        return {}
    if key in {"legacy", "baseline", "sleepedf_legacy"}:
        eeg_high = min(30.0, float(fs) * 0.5 * 0.95)
        return {0: (0.5, eeg_high), 1: (0.5, eeg_high)}
    if key in {"clinical_canonical", "clinical", "sleepedf_like", "harmonized"}:
        return {
            0: (0.5, 30.0),  # EEG: delta/theta/alpha/sigma/beta concepts.
            1: (0.1, 15.0),  # EOG: preserve slow eye movement, suppress muscle noise.
            2: (10.0, 45.0),  # EMG: muscle tone/activation band.
        }
    if key in {"spindle_safe", "spindle_safe_0p3_35", "wide_0p3_35", "k_complex_safe"}:
        return {
            0: (0.3, 35.0),  # EEG: preserve K-complex slow energy and sigma spindles.
            1: (0.05, 18.0),
            2: (8.0, 45.0),
        }
    if key in {"sigma_preserve", "sigma_preserve_0p3_40", "wide_0p3_40"}:
        return {
            0: (0.3, 40.0),
            1: (0.05, 18.0),
            2: (8.0, 45.0),
        }
    if key in {"canonical_0p5_35", "eeg_0p5_35"}:
        return {
            0: (0.5, 35.0),
            1: (0.1, 18.0),
            2: (10.0, 45.0),
        }
    raise ValueError(
        "filter_profile must be one of 'clinical_canonical', 'legacy', or 'none', "
        f"got {filter_profile!r}"
    )


def preprocess_multichannel_epochs(
    X,
    fs=100,
    apply_filters=True,
    normalize=True,
    normalization_mode="recording_robust_zscore",
    filter_profile="clinical_canonical",
    filter_backend="scipy",
    clip_sigma=None,
    enable_amplitude_clipping=False,
):
    """
    Apply dataset-agnostic clinical-concept harmonized preprocessing.

    The default profile maps all datasets to the same EEG/EOG/EMG clinical
    signal space before shared priors are applied:
    EEG 0.5-30 Hz, EOG 0.1-15 Hz, EMG 10-45 Hz, followed by per-recording
    normalization.

    No global amplitude clipping is applied. clip_sigma and
    enable_amplitude_clipping are kept only so run_main.py remains compatible.
    """
    _ = clip_sigma, enable_amplitude_clipping
    X = sanitize_signal_epochs(X)

    if apply_filters and X.shape[1] > 15:
        fs = float(fs) if fs is not None else 100.0
        processed = X.copy()
        for channel_idx, band in clinical_filter_bands(filter_profile, fs=fs).items():
            if channel_idx < X.shape[-1]:
                processed[:, :, channel_idx] = _apply_bandpass_to_channel(
                    processed[:, :, channel_idx],
                    band[0],
                    band[1],
                    fs,
                    backend=filter_backend,
                )

        X = processed

    if normalize:
        mode = str(normalization_mode or "recording_robust_zscore").strip().lower()
        if mode in {"recording_robust_zscore", "robust", "robust_zscore"}:
            X = recording_robust_zscore(X)
        elif mode in {"recording_zscore", "zscore", "standard", "channel_zscore"}:
            mean = np.mean(X, axis=(0, 1), keepdims=True)
            std = np.std(X, axis=(0, 1), keepdims=True)
            X = sanitize_signal_epochs((X - mean) / (std + 1e-6))
        elif mode in {"recording_minmax", "minmax", "channel_minmax"}:
            data_min = np.min(X, axis=(0, 1), keepdims=True)
            data_max = np.max(X, axis=(0, 1), keepdims=True)
            X = sanitize_signal_epochs(2.0 * (X - data_min) / (data_max - data_min + 1e-6) - 1.0)
        elif mode in {"recording_robust_minmax", "robust_minmax"}:
            lo = np.percentile(X, 1.0, axis=(0, 1), keepdims=True)
            hi = np.percentile(X, 99.0, axis=(0, 1), keepdims=True)
            X = np.clip((X - lo) / (hi - lo + 1e-6), 0.0, 1.0)
            X = sanitize_signal_epochs(2.0 * X - 1.0)
        else:
            raise ValueError(
                "normalization_mode must be recording_robust_zscore, recording_zscore, "
                f"recording_minmax, or recording_robust_minmax; got {normalization_mode!r}"
            )

    return X.astype(np.float32)


def _subject_groups_from_files(files):
    groups = {}
    for filename in sorted(files):
        groups.setdefault(infer_subject_id_from_npz(filename), []).append(filename)
    return {subject: sorted(subject_files) for subject, subject_files in groups.items()}


def _split_subject_ids(subject_ids, test_size=0.15, val_size=0.15, seed=42):
    subject_ids = np.array(sorted(subject_ids), dtype=object)
    if len(subject_ids) == 0:
        return [], [], []

    rng = np.random.default_rng(seed)
    shuffled = subject_ids[rng.permutation(len(subject_ids))]
    n_subjects = len(shuffled)

    if n_subjects == 1:
        return sorted(shuffled.tolist()), [], []

    n_test = int(round(n_subjects * float(test_size)))
    if test_size > 0:
        n_test = max(1, n_test)
    n_test = min(n_test, n_subjects - 1)

    remaining = n_subjects - n_test
    n_val = int(round(remaining * float(val_size)))
    if val_size > 0 and remaining >= 2:
        n_val = max(1, n_val)
    n_val = min(n_val, max(remaining - 1, 0))

    test_subjects = sorted(shuffled[:n_test].tolist())
    val_subjects = sorted(shuffled[n_test:n_test + n_val].tolist())
    train_subjects = sorted(shuffled[n_test + n_val:].tolist())
    return train_subjects, val_subjects, test_subjects


def _subject_label_counts(subject_groups, folder_path, class_names=None):
    n_classes = len(class_names) if class_names is not None else 0
    subject_counts = {}
    for subject, subject_files in subject_groups.items():
        labels = []
        for filename in subject_files:
            with np.load(os.path.join(folder_path, filename), mmap_mode="r") as data:
                file_labels = np.asarray(data["y"], dtype=np.int32).reshape(-1)
            labels.append(file_labels)
            if file_labels.size:
                n_classes = max(n_classes, int(file_labels.max()) + 1)
        if labels:
            labels = np.concatenate(labels)
            subject_counts[subject] = np.bincount(labels, minlength=n_classes).astype(np.float64)
        else:
            subject_counts[subject] = np.zeros(n_classes, dtype=np.float64)

    for subject, counts in list(subject_counts.items()):
        if counts.shape[0] < n_classes:
            padded = np.zeros(n_classes, dtype=np.float64)
            padded[:counts.shape[0]] = counts
            subject_counts[subject] = padded
    return subject_counts


def _split_subject_ids_stratified(
    subject_groups,
    folder_path,
    test_size=0.15,
    val_size=0.15,
    seed=42,
    class_names=None,
):
    base_train, base_val, base_test = _split_subject_ids(
        subject_groups.keys(),
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    target_subject_counts = {
        "train": len(base_train),
        "val": len(base_val),
        "test": len(base_test),
    }
    if target_subject_counts["val"] == 0 and target_subject_counts["test"] == 0:
        return base_train, base_val, base_test

    subject_counts = _subject_label_counts(subject_groups, folder_path, class_names=class_names)
    subjects = sorted(subject_counts)
    global_counts = np.sum([subject_counts[subject] for subject in subjects], axis=0)
    global_distribution = global_counts / np.maximum(np.sum(global_counts), 1.0)
    total_epochs = float(np.sum(global_counts))

    rng = np.random.default_rng(seed)
    shuffled_subjects = subjects[:]
    rng.shuffle(shuffled_subjects)
    shuffled_subjects = sorted(
        shuffled_subjects,
        key=lambda subject: float(np.sum(subject_counts[subject])),
        reverse=True,
    )

    splits = {"train": [], "val": [], "test": []}
    split_counts = {
        "train": np.zeros_like(global_counts, dtype=np.float64),
        "val": np.zeros_like(global_counts, dtype=np.float64),
        "test": np.zeros_like(global_counts, dtype=np.float64),
    }
    split_order = ("test", "val", "train")
    target_epoch_counts = {
        name: total_epochs * (target_subject_counts[name] / max(len(subjects), 1))
        for name in split_order
    }

    for subject in shuffled_subjects:
        counts = subject_counts[subject]
        best_name = None
        best_score = np.inf
        for split_name in split_order:
            if len(splits[split_name]) >= target_subject_counts[split_name]:
                continue

            candidate_counts = split_counts[split_name] + counts
            candidate_distribution = candidate_counts / np.maximum(np.sum(candidate_counts), 1.0)
            distribution_error = float(np.sum(np.abs(candidate_distribution - global_distribution)))
            subject_fill = (len(splits[split_name]) + 1) / max(target_subject_counts[split_name], 1)
            epoch_fill = float(np.sum(candidate_counts)) / max(target_epoch_counts[split_name], 1.0)
            size_error = abs(epoch_fill - subject_fill)
            score = distribution_error + 0.25 * size_error
            if score < best_score:
                best_name = split_name
                best_score = score

        if best_name is None:
            best_name = "train"
        splits[best_name].append(subject)
        split_counts[best_name] += counts

    return sorted(splits["train"]), sorted(splits["val"]), sorted(splits["test"])


def _format_label_distribution(labels, class_names=None):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if class_names is None:
        n_classes = int(labels.max()) + 1 if labels.size else 0
        class_names = [str(idx) for idx in range(n_classes)]
    counts = np.bincount(labels, minlength=len(class_names))
    total = int(counts.sum())
    parts = [
        f"{name}:{int(count)}({(count / max(total, 1)):.1%})"
        for name, count in zip(class_names, counts)
    ]
    return ", ".join(parts), counts


def log_npz_split_summary(folder_path, split_map, class_names=None, logger=None):
    for split_name, files in split_map.items():
        subjects = sorted({infer_subject_id_from_npz(filename) for filename in files})
        _emit(logger, f"{split_name} subjects ({len(subjects)}): {subjects}")
        _emit(logger, f"{split_name} files ({len(files)}): {list(files)}")

        split_labels = []
        for filename in files:
            with np.load(os.path.join(folder_path, filename), mmap_mode="r") as data:
                labels = np.asarray(data["y"], dtype=np.int32).reshape(-1)
            split_labels.append(labels)
            file_distribution, _ = _format_label_distribution(labels, class_names)
            _emit(logger, f"  {split_name} file distribution | {filename}: {file_distribution}")

        if split_labels:
            split_distribution, _ = _format_label_distribution(np.concatenate(split_labels), class_names)
        else:
            split_distribution, _ = _format_label_distribution(np.array([], dtype=np.int32), class_names)
        _emit(logger, f"{split_name} distribution: {split_distribution}")


def split_npz_files(
    files,
    test_size=0.15,
    val_size=0.15,
    seed=42,
    folder_path=None,
    class_names=None,
    logger=None,
    log_summary=False,
    stratify_by_labels=False,
):
    subject_groups = _subject_groups_from_files(files)
    if stratify_by_labels and folder_path is not None:
        train_subjects, val_subjects, test_subjects = _split_subject_ids_stratified(
            subject_groups,
            folder_path,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            class_names=class_names,
        )
    else:
        train_subjects, val_subjects, test_subjects = _split_subject_ids(
            subject_groups.keys(),
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )

    def files_for_subjects(subjects):
        split_files = []
        for subject in subjects:
            split_files.extend(subject_groups[subject])
        return sorted(split_files)

    train_files = files_for_subjects(train_subjects)
    val_files = files_for_subjects(val_subjects)
    test_files = files_for_subjects(test_subjects)

    if log_summary:
        split_mode = "stratified subject-level" if stratify_by_labels and folder_path is not None else "subject-level"
        _emit(logger, f"Split mode: {split_mode}")
        if folder_path is not None:
            log_npz_split_summary(
                folder_path,
                {"Train": train_files, "Val": val_files, "Test": test_files},
                class_names=class_names,
                logger=logger,
            )
        else:
            _emit(logger, f"Train subjects ({len(train_subjects)}): {train_subjects}")
            _emit(logger, f"Val subjects ({len(val_subjects)}): {val_subjects}")
            _emit(logger, f"Test subjects ({len(test_subjects)}): {test_subjects}")

    return train_files, val_files, test_files


def get_npz_file_list(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    return sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".npz")])


def load_npz_data(
    folder_path,
    max_files=None,
    file_subset=None,
    preprocess=False,
    apply_filters=True,
    normalize=True,
    normalization_mode="recording_robust_zscore",
    filter_profile="clinical_canonical",
    filter_backend="scipy",
    clip_sigma=None,
    enable_amplitude_clipping=False,
    return_quality=False,
    return_channel_quality=False,
    enable_quality_downweighting=False,
):
    """
    Load NPZ recordings with preallocation.

    When return_quality=True, the third return value is the aligned quality
    confidence sequence Q_t in [0.1, 1.0]. It is not a training sample weight.
    """
    _ = enable_quality_downweighting
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if file_subset is not None:
        files = [filename for filename in file_subset if filename.endswith(".npz")]
    else:
        files = [filename for filename in os.listdir(folder_path) if filename.endswith(".npz")]

    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError("No .npz files found.")

    print(f"Loading {len(files)} files from {folder_path}...")

    total_samples = 0
    for filename in files:
        with np.load(os.path.join(folder_path, filename), mmap_mode="r") as data:
            total_samples += int(data["x"].shape[0])

    print(f"Scan complete, total samples: {total_samples}. Preallocating memory...")

    with np.load(os.path.join(folder_path, files[0]), mmap_mode="r") as first_data:
        first_shape = first_data["x"].shape

    sample_shape = (first_shape[1], 1) if len(first_shape) == 2 else first_shape[1:]
    X = np.empty((total_samples, *sample_shape), dtype=np.float32)
    y = np.empty((total_samples,), dtype=np.int32)
    quality_scores = np.empty((total_samples,), dtype=np.float32) if return_quality else None
    channel_quality_scores = (
        np.empty((total_samples, sample_shape[-1]), dtype=np.float32)
        if return_channel_quality
        else None
    )
    quality_reports = []

    current_idx = 0
    for file_idx, filename in enumerate(files):
        with np.load(os.path.join(folder_path, filename)) as data:
            x_data = data["x"].astype(np.float32)
            y_data = data["y"].astype(np.int32)
            fs = float(data["fs"]) if "fs" in data.files else 100.0

        x_data = _as_3d_float(x_data)
        quality_source = x_data
        n_samples = x_data.shape[0]

        if preprocess:
            x_data = preprocess_multichannel_epochs(
                x_data,
                fs=fs,
                apply_filters=apply_filters,
                normalize=normalize,
                normalization_mode=normalization_mode,
                filter_profile=filter_profile,
                filter_backend=filter_backend,
                clip_sigma=clip_sigma,
                enable_amplitude_clipping=enable_amplitude_clipping,
            )
        else:
            x_data = sanitize_signal_epochs(x_data)

        if return_quality:
            file_quality, _, file_report = compute_epoch_quality_scores(
                quality_source,
                fs=fs,
                return_flags=True,
                return_report=True,
            )
            quality_scores[current_idx: current_idx + n_samples] = file_quality
            quality_reports.append(file_report)

        if return_channel_quality:
            file_channel_quality = compute_epoch_channel_quality_scores(quality_source, fs=fs)
            channel_quality_scores[current_idx: current_idx + n_samples] = file_channel_quality

        X[current_idx: current_idx + n_samples] = x_data
        y[current_idx: current_idx + n_samples] = y_data
        current_idx += n_samples

        if (file_idx + 1) % 5 == 0 or (file_idx + 1) == len(files):
            print(f"Progress: loaded {file_idx + 1}/{len(files)} files.")

    if return_quality:
        report = summarize_quality_reports(
            quality_reports,
            quality_scores,
            report_name=f"{os.path.basename(folder_path)} ({len(files)} files)",
        )
        log_quality_report(report)
        if return_channel_quality:
            return X, y, quality_scores, channel_quality_scores
        return X, y, quality_scores

    if return_channel_quality:
        return X, y, channel_quality_scores

    return X, y


def format_for_model(X, y, num_classes=5):
    X = _as_3d_float(X)
    y_one_hot = to_categorical(y, num_classes=num_classes)
    return X, y_one_hot


def load_edf_epoch(psg_path, hypno_path):
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hypno_path)
    raw.set_annotations(annot, emit_warning=False)
    event_id = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.0)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=30.0 - 1 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs.get_data(), epochs.events[:, 2]


