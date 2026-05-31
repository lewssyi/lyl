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


def _robust_positive_z(metric):
    metric = np.asarray(metric, dtype=np.float32)
    center = np.median(metric, axis=0, keepdims=True)
    mad = np.median(np.abs(metric - center), axis=0, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.std(metric, axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, fallback + 1e-6, scale)
    return (metric - center) / scale


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


def preprocess_multichannel_epochs(
    X,
    fs=100,
    apply_filters=True,
    normalize=True,
    clip_sigma=None,
    enable_amplitude_clipping=False,
):
    """
    Apply the baseline physiological preprocessing used by the best run.

    No global amplitude clipping is applied. clip_sigma and
    enable_amplitude_clipping are kept only so run_main.py remains compatible.
    """
    _ = clip_sigma, enable_amplitude_clipping
    X = sanitize_signal_epochs(X)

    if apply_filters and X.shape[1] > 15:
        fs = float(fs) if fs is not None else 100.0
        nyquist = fs / 2.0
        processed = X.copy()

        eeg_high = min(30.0, nyquist * 0.95)
        if eeg_high > 0.5 and X.shape[-1] >= 1:
            sos_eeg = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
            processed[:, :, 0] = signal.sosfiltfilt(sos_eeg, processed[:, :, 0], axis=1).astype(np.float32)

        if eeg_high > 0.5 and X.shape[-1] >= 2:
            sos_eog = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
            processed[:, :, 1] = signal.sosfiltfilt(sos_eog, processed[:, :, 1], axis=1).astype(np.float32)

        X = processed

    if normalize:
        mean = np.mean(X, axis=(0, 1), keepdims=True)
        std = np.std(X, axis=(0, 1), keepdims=True)
        X = (X - mean) / (std + 1e-6)
        X = sanitize_signal_epochs(X)

    return X.astype(np.float32)


def split_npz_files(files, test_size=0.15, val_size=0.15, seed=42):
    rng = np.random.default_rng(seed)
    files = np.array(sorted(files))
    shuffled = files[rng.permutation(len(files))]

    n_test = max(1, int(round(len(shuffled) * test_size)))
    test_files = sorted(shuffled[:n_test].tolist())
    train_val_files = shuffled[n_test:]

    n_val = max(1, int(round(len(train_val_files) * val_size)))
    val_files = sorted(train_val_files[:n_val].tolist())
    train_files = sorted(train_val_files[n_val:].tolist())

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
    clip_sigma=None,
    enable_amplitude_clipping=False,
    return_quality=False,
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
        return X, y, quality_scores

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


#########0.8213##########
#
#import numpy as np
#import os
#from tensorflow.keras.utils import to_categorical
#from scipy import signal
#
#
#def sanitize_signal_epochs(X, clip_value=None):
#    """Replace non-finite values and optionally clip extreme normalized amplitudes."""
#    X = np.asarray(X, dtype=np.float32)
#    if not np.all(np.isfinite(X)):
#        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#    if clip_value is not None and clip_value > 0:
#        X = np.clip(X, -float(clip_value), float(clip_value))
#    return X.astype(np.float32)
#
#
#def compute_epoch_quality_weights(
#    X,
#    flat_std_threshold=1e-4,
#    extreme_z_threshold=8.0,
#    extreme_ratio_threshold=0.02,
#    jump_z_threshold=10.0,
#    min_weight=0.35,
#    return_flags=False,
#    enable_quality_downweighting=False,
#    channel_names=None,
#    return_report=False,
#    report_name=None,
#    logger=None,
#):
#    """
#    Compute a multichannel quality report without changing the signal stream.
#
#    By default this is report-only: returned weights are all ones so EEG/EOG/EMG
#    epochs are not downweighted. Set enable_quality_downweighting=True only for
#    ablation runs that intentionally restore the old weighting behavior.
#    """
#    X = np.asarray(X, dtype=np.float32)
#    if X.ndim == 2:
#        X = X[..., np.newaxis]
#    if X.ndim != 3:
#        raise ValueError(f"Expected X with shape (epochs, samples, channels), got {X.shape}")
#
#    n_epochs, _, n_channels = X.shape
#    if channel_names is None:
#        default_names = ["EEG", "EOG", "EMG"]
#        channel_names = [
#            default_names[idx] if idx < len(default_names) else f"CH{idx + 1}"
#            for idx in range(n_channels)
#        ]
#    else:
#        channel_names = list(channel_names)
#        if len(channel_names) < n_channels:
#            channel_names.extend(f"CH{idx + 1}" for idx in range(len(channel_names), n_channels))
#
#    finite_channel = np.all(np.isfinite(X), axis=1)
#    finite_epoch = np.all(finite_channel, axis=1)
#    X_safe = np.nan_to_num(X, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
#
#    channel_std = np.std(X_safe, axis=1)
#    flat_channel = channel_std < flat_std_threshold
#    flat_epoch = np.any(flat_channel, axis=1)
#
#    abs_values = np.abs(X_safe)
#    extreme_ratio_channel = np.mean(abs_values >= extreme_z_threshold, axis=1)
#    max_abs_channel = np.max(abs_values, axis=1)
#    extreme_channel = (
#        (extreme_ratio_channel > extreme_ratio_threshold) |
#        (max_abs_channel >= extreme_z_threshold)
#    )
#    extreme_epoch = np.any(extreme_channel, axis=1)
#
#    if X_safe.shape[1] > 1:
#        jump_metric = np.mean(np.abs(np.diff(X_safe, axis=1)), axis=1)
#        median_jump = np.median(jump_metric, axis=0, keepdims=True)
#        mad_jump = np.median(np.abs(jump_metric - median_jump), axis=0, keepdims=True)
#        robust_jump_z = (jump_metric - median_jump) / (1.4826 * mad_jump + 1e-6)
#        jump_channel = robust_jump_z > jump_z_threshold
#        jump_epoch = np.any(jump_channel, axis=1)
#    else:
#        jump_channel = np.zeros((X_safe.shape[0], n_channels), dtype=bool)
#        jump_epoch = np.zeros(X_safe.shape[0], dtype=bool)
#
#    weights = np.ones(X_safe.shape[0], dtype=np.float32)
#    if enable_quality_downweighting:
#        weights[~finite_epoch] *= 0.35
#        weights[flat_epoch] *= 0.50
#        weights[extreme_epoch] *= 0.65
#        weights[jump_epoch] *= 0.75
#        weights = np.clip(weights, min_weight, 1.0).astype(np.float32)
#
#    flags = {
#        "nonfinite": ~finite_epoch,
#        "flat": flat_epoch,
#        "extreme": extreme_epoch,
#        "jump": jump_epoch,
#        "nonfinite_channel": ~finite_channel,
#        "flat_channel": flat_channel,
#        "extreme_channel": extreme_channel,
#        "jump_channel": jump_channel,
#    }
#
#    flagged_epoch = (~finite_epoch) | flat_epoch | extreme_epoch | jump_epoch
#    report = {
#        "epochs": int(n_epochs),
#        "channels": int(n_channels),
#        "downweighting_enabled": bool(enable_quality_downweighting),
#        "flagged_epochs": int(np.sum(flagged_epoch)),
#        "flagged_ratio": float(np.mean(flagged_epoch)) if n_epochs else 0.0,
#        "nonfinite_epochs": int(np.sum(~finite_epoch)),
#        "flat_epochs": int(np.sum(flat_epoch)),
#        "extreme_epochs": int(np.sum(extreme_epoch)),
#        "jump_epochs": int(np.sum(jump_epoch)),
#        "channel_reports": [],
#    }
#    for channel_idx in range(n_channels):
#        report["channel_reports"].append(
#            {
#                "channel": channel_names[channel_idx],
#                "nonfinite_epochs": int(np.sum(~finite_channel[:, channel_idx])),
#                "flat_epochs": int(np.sum(flat_channel[:, channel_idx])),
#                "extreme_epochs": int(np.sum(extreme_channel[:, channel_idx])),
#                "jump_epochs": int(np.sum(jump_channel[:, channel_idx])),
#                "median_std": float(np.median(channel_std[:, channel_idx])),
#                "max_abs": float(np.max(max_abs_channel[:, channel_idx])) if n_epochs else 0.0,
#            }
#        )
#
#    if logger is not None or report_name is not None:
#        if hasattr(logger, "info"):
#            emit = logger.info
#        elif hasattr(logger, "write"):
#            emit = lambda message: logger.write(f"{message}\n")
#        elif callable(logger):
#            emit = logger
#        else:
#            emit = print
#        prefix = f"{report_name} " if report_name else ""
#        emit(
#            f"{prefix}quality report: epochs={report['epochs']}, "
#            f"flagged={report['flagged_epochs']} ({report['flagged_ratio']:.1%}), "
#            f"downweighting={'on' if enable_quality_downweighting else 'off'}, "
#            f"returned_weight_mean={np.mean(weights):.3f}"
#        )
#        for channel_report in report["channel_reports"]:
#            emit(
#                "  "
#                f"{channel_report['channel']}: "
#                f"nonfinite={channel_report['nonfinite_epochs']}, "
#                f"flat={channel_report['flat_epochs']}, "
#                f"extreme={channel_report['extreme_epochs']}, "
#                f"jump={channel_report['jump_epochs']}, "
#                f"median_std={channel_report['median_std']:.4g}, "
#                f"max_abs={channel_report['max_abs']:.4g}"
#            )
#
#    if return_flags and return_report:
#        return weights, flags, report
#    if return_report:
#        return weights, report
#    if return_flags:
#        return weights, flags
#    return weights
#
#
#def preprocess_multichannel_epochs(
#    X,
#    fs=100,
#    apply_filters=True,
#    normalize=True,
#    clip_sigma=None,
#    enable_amplitude_clipping=False,
#):
#    """
#    Apply lightweight physiology-aware preprocessing to one recording.
#
#    X is expected to be (epochs, samples, channels). EEG/EOG are band-passed
#    to 0.5-30 Hz. EMG is left unfiltered to preserve tonic muscle tone,
#    then each channel is normalized
#    per recording. Recording-level normalization keeps relative epoch amplitude
#    cues while reducing subject/device scale drift.
#    """
#    X = sanitize_signal_epochs(X)
#
#    if X.ndim == 2:
#        X = X[..., np.newaxis]
#
#    if apply_filters and X.shape[1] > 15:
#        fs = float(fs) if fs is not None else 100.0
#        nyquist = fs / 2.0
#        processed = X.copy()
#
#        eeg_high = min(30.0, nyquist * 0.95)
#        if eeg_high > 0.5 and X.shape[-1] >= 1:
#            sos_eeg = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
#            processed[:, :, 0] = signal.sosfiltfilt(sos_eeg, processed[:, :, 0], axis=1).astype(np.float32)
#
#        if eeg_high > 0.5 and X.shape[-1] >= 2:
#            sos_eog = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
#            processed[:, :, 1] = signal.sosfiltfilt(sos_eog, processed[:, :, 1], axis=1).astype(np.float32)
#
#        X = processed
#
#    if normalize:
#        mean = np.mean(X, axis=(0, 1), keepdims=True)
#        std = np.std(X, axis=(0, 1), keepdims=True)
#        X = (X - mean) / (std + 1e-6)
#        clip_value = clip_sigma if enable_amplitude_clipping else None
#        X = sanitize_signal_epochs(X, clip_value=clip_value)
#
#    return X.astype(np.float32)
#
#
#def split_npz_files(files, test_size=0.15, val_size=0.15, seed=42):
#    """
#    Deterministic file-level split. This avoids mixing epochs from the same
#    recording across train/validation/test sets.
#    """
#    rng = np.random.default_rng(seed)
#    files = np.array(sorted(files))
#    shuffled = files[rng.permutation(len(files))]
#
#    n_test = max(1, int(round(len(shuffled) * test_size)))
#    test_files = sorted(shuffled[:n_test].tolist())
#    train_val_files = shuffled[n_test:]
#
#    n_val = max(1, int(round(len(train_val_files) * val_size)))
#    val_files = sorted(train_val_files[:n_val].tolist())
#    train_files = sorted(train_val_files[n_val:].tolist())
#
#    return train_files, val_files, test_files
#
## --- 获取文件列表 ---
#def get_npz_file_list(folder_path):
#    """获取指定文件夹下所有 .npz 文件的文件名列表（排序保证可复现）"""
#    if not os.path.exists(folder_path):
#        raise FileNotFoundError(f"找不到目录: {folder_path}")
#    return sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
#
#
## --- 处理 NPZ 文件 (预分配内存 + 支持按文件子集加载) ---
#def load_npz_data(
#    folder_path,
#    max_files=None,
#    file_subset=None,
#    preprocess=False,
#    apply_filters=True,
#    normalize=True,
#    clip_sigma=None,
#    enable_amplitude_clipping=False,
#    return_quality=False,
#    enable_quality_downweighting=False,
#):
#    """
#    加载指定文件夹下所有的 npz 文件并合并。
#    采用预分配内存策略，彻底解决 concatenate 导致的 OOM 内存爆炸问题。
#
#    参数:
#        folder_path: 数据文件夹路径
#        max_files:   (调试用) 只加载前 N 个文件
#        file_subset: (按受试者切分) 只加载指定的文件名列表
#    """
#    if not os.path.exists(folder_path):
#        raise FileNotFoundError(f"找不到目录: {folder_path}")
#
#    if file_subset is not None:
#        files = [f for f in file_subset if f.endswith('.npz')]
#    else:
#        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
#
#    if max_files is not None:
#        files = files[:max_files]
#
#    if len(files) == 0:
#        raise ValueError("没有找到任何 .npz 文件！")
#
#    print(f"正在从 {folder_path} 加载 {len(files)} 个文件...")
#
#    # 第一遍扫描：只读取 shape 算出总样本数，绝不把数据拉入内存
#    total_samples = 0
#    for f in files:
#        data = np.load(os.path.join(folder_path, f), mmap_mode='r')
#        total_samples += data['x'].shape[0]
#
#    print(f"扫描完毕，总样本数: {total_samples}。正在预分配连续内存...")
#
#    with np.load(os.path.join(folder_path, files[0]), mmap_mode='r') as first_data:
#        first_shape = first_data['x'].shape
#
#    if len(first_shape) == 2:
#        sample_shape = (first_shape[1], 1)
#    else:
#        sample_shape = first_shape[1:]
#
#    X = np.empty((total_samples, *sample_shape), dtype=np.float32)
#    y = np.empty((total_samples,), dtype=np.int32)
#    quality_weights = np.empty((total_samples,), dtype=np.float32) if return_quality else None
#
#    # 第二遍扫描：像拼图一样把数据填进预先挖好的坑里
#    current_idx = 0
#    for i, f in enumerate(files):
#        with np.load(os.path.join(folder_path, f)) as data:
#            x_data = data['x'].astype(np.float32)
#            y_data = data['y']
#            fs = float(data['fs']) if 'fs' in data.files else 100.0
#            n_samples = x_data.shape[0]
#
#            # 适配 3 通道格式
#            if len(x_data.shape) == 2:
#                x_data = np.expand_dims(x_data, axis=-1)
#
#            if preprocess:
#                x_data = preprocess_multichannel_epochs(
#                    x_data,
#                    fs=fs,
#                    apply_filters=apply_filters,
#                    normalize=normalize,
#                    clip_sigma=clip_sigma,
#                    enable_amplitude_clipping=enable_amplitude_clipping,
#                )
#            else:
#                clip_value = clip_sigma if enable_amplitude_clipping else None
#                x_data = sanitize_signal_epochs(x_data, clip_value=clip_value)
#
#            X[current_idx: current_idx + n_samples] = x_data
#            y[current_idx: current_idx + n_samples] = y_data
#
#            current_idx += n_samples
#
#            if (i + 1) % 5 == 0 or (i + 1) == len(files):
#                print(f"进度: 已加载 {i + 1}/{len(files)}...")
#
#    if return_quality:
#        report_name = f"{os.path.basename(folder_path)} ({len(files)} files)"
#        quality_weights[:] = compute_epoch_quality_weights(
#            X,
#            enable_quality_downweighting=enable_quality_downweighting,
#            report_name=report_name,
#        )
#        return X, y, quality_weights
#    return X, y
#
#
## --- 格式化函数 (核心：适配 Conv1D) ---
#def format_for_model(X, y, num_classes=5):
#    """
#    确保 X 是 (Samples, 3000, 1) 且 y 是 One-hot 编码。
#    """
#    if len(X.shape) == 2:
#        X = np.expand_dims(X, axis=-1)
#    y_one_hot = to_categorical(y, num_classes=num_classes)
#    return X, y_one_hot
#
#
## --- 处理 EDF 文件 (可选备份) ---
#def load_edf_epoch(psg_path, hypno_path):
#    import mne
#
#    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
#    annot = mne.read_annotations(hypno_path)
#    raw.set_annotations(annot, emit_warning=False)
#    event_id = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
#                'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
#    events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.)
#    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
#                        tmin=0., tmax=30. - 1 / raw.info['sfreq'],
#                        baseline=None, preload=True, verbose=False)
#    return epochs.get_data(), epochs.events[:, 2]
