import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from scipy import signal


def preprocess_multichannel_epochs(X, fs=100, apply_filters=True, normalize=True):
    """
    Apply lightweight physiology-aware preprocessing to one recording.

    X is expected to be (epochs, samples, channels). EEG/EOG are band-passed
    to 0.5-30 Hz. EMG is left unfiltered to preserve tonic muscle tone,
    then each channel is normalized
    per recording. Recording-level normalization keeps relative epoch amplitude
    cues while reducing subject/device scale drift.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2:
        X = X[..., np.newaxis]

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

    return X.astype(np.float32)


def split_npz_files(files, test_size=0.15, val_size=0.15, seed=42):
    """
    Deterministic file-level split. This avoids mixing epochs from the same
    recording across train/validation/test sets.
    """
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

# --- 获取文件列表 ---
def get_npz_file_list(folder_path):
    """获取指定文件夹下所有 .npz 文件的文件名列表（排序保证可复现）"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"找不到目录: {folder_path}")
    return sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])


# --- 处理 NPZ 文件 (预分配内存 + 支持按文件子集加载) ---
def load_npz_data(
    folder_path,
    max_files=None,
    file_subset=None,
    preprocess=False,
    apply_filters=True,
    normalize=True,
):
    """
    加载指定文件夹下所有的 npz 文件并合并。
    采用预分配内存策略，彻底解决 concatenate 导致的 OOM 内存爆炸问题。

    参数:
        folder_path: 数据文件夹路径
        max_files:   (调试用) 只加载前 N 个文件
        file_subset: (按受试者切分) 只加载指定的文件名列表
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"找不到目录: {folder_path}")

    if file_subset is not None:
        files = [f for f in file_subset if f.endswith('.npz')]
    else:
        files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise ValueError("没有找到任何 .npz 文件！")

    print(f"正在从 {folder_path} 加载 {len(files)} 个文件...")

    # 第一遍扫描：只读取 shape 算出总样本数，绝不把数据拉入内存
    total_samples = 0
    for f in files:
        data = np.load(os.path.join(folder_path, f), mmap_mode='r')
        total_samples += data['x'].shape[0]

    print(f"扫描完毕，总样本数: {total_samples}。正在预分配连续内存...")

    with np.load(os.path.join(folder_path, files[0]), mmap_mode='r') as first_data:
        first_shape = first_data['x'].shape

    if len(first_shape) == 2:
        sample_shape = (first_shape[1], 1)
    else:
        sample_shape = first_shape[1:]

    X = np.empty((total_samples, *sample_shape), dtype=np.float32)
    y = np.empty((total_samples,), dtype=np.int32)

    # 第二遍扫描：像拼图一样把数据填进预先挖好的坑里
    current_idx = 0
    for i, f in enumerate(files):
        with np.load(os.path.join(folder_path, f)) as data:
            x_data = data['x'].astype(np.float32)
            y_data = data['y']
            fs = float(data['fs']) if 'fs' in data.files else 100.0
            n_samples = x_data.shape[0]

            # 适配 3 通道格式
            if len(x_data.shape) == 2:
                x_data = np.expand_dims(x_data, axis=-1)

            if preprocess:
                x_data = preprocess_multichannel_epochs(
                    x_data,
                    fs=fs,
                    apply_filters=apply_filters,
                    normalize=normalize,
                )

            X[current_idx: current_idx + n_samples] = x_data
            y[current_idx: current_idx + n_samples] = y_data

            current_idx += n_samples

            if (i + 1) % 5 == 0 or (i + 1) == len(files):
                print(f"进度: 已加载 {i + 1}/{len(files)}...")

    return X, y


# --- 格式化函数 (核心：适配 Conv1D) ---
def format_for_model(X, y, num_classes=5):
    """
    确保 X 是 (Samples, 3000, 1) 且 y 是 One-hot 编码。
    """
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
    y_one_hot = to_categorical(y, num_classes=num_classes)
    return X, y_one_hot


# --- 处理 EDF 文件 (可选备份) ---
def load_edf_epoch(psg_path, hypno_path):
    import mne

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hypno_path)
    raw.set_annotations(annot, emit_warning=False)
    event_id = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.)
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
                        tmin=0., tmax=30. - 1 / raw.info['sfreq'],
                        baseline=None, preload=True, verbose=False)
    return epochs.get_data(), epochs.events[:, 2]


# import numpy as np
# import os
# from tensorflow.keras.utils import to_categorical
# from scipy import signal


# def sanitize_signal_epochs(X, clip_value=None):
#     """Replace non-finite values and optionally clip extreme normalized amplitudes."""
#     X = np.asarray(X, dtype=np.float32)
#     if not np.all(np.isfinite(X)):
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
#     if clip_value is not None and clip_value > 0:
#         X = np.clip(X, -float(clip_value), float(clip_value))
#     return X.astype(np.float32)


# def compute_epoch_quality_weights(
#     X,
#     flat_std_threshold=1e-4,
#     extreme_z_threshold=8.0,
#     extreme_ratio_threshold=0.02,
#     jump_z_threshold=10.0,
#     min_weight=0.35,
#     return_flags=False,
# ):
#     """
#     Estimate conservative per-epoch quality weights after preprocessing.

#     The function does not delete epochs. It downweights epochs with non-finite
#     values, flat channels, extreme normalized amplitudes, or abrupt jumps.
#     """
#     X = np.asarray(X, dtype=np.float32)
#     if X.ndim == 2:
#         X = X[..., np.newaxis]

#     finite_epoch = np.all(np.isfinite(X), axis=(1, 2))
#     X_safe = sanitize_signal_epochs(X)

#     epoch_std = np.std(X_safe, axis=1)
#     flat_epoch = np.any(epoch_std < flat_std_threshold, axis=1)

#     abs_values = np.abs(X_safe)
#     extreme_ratio = np.mean(abs_values >= extreme_z_threshold, axis=(1, 2))
#     max_abs = np.max(abs_values, axis=(1, 2))
#     extreme_epoch = (extreme_ratio > extreme_ratio_threshold) | (max_abs >= extreme_z_threshold)

#     if X_safe.shape[1] > 1:
#         jump_metric = np.mean(np.abs(np.diff(X_safe, axis=1)), axis=1)
#         median_jump = np.median(jump_metric, axis=0, keepdims=True)
#         mad_jump = np.median(np.abs(jump_metric - median_jump), axis=0, keepdims=True)
#         robust_jump_z = (jump_metric - median_jump) / (1.4826 * mad_jump + 1e-6)
#         jump_epoch = np.any(robust_jump_z > jump_z_threshold, axis=1)
#     else:
#         jump_epoch = np.zeros(X_safe.shape[0], dtype=bool)

#     weights = np.ones(X_safe.shape[0], dtype=np.float32)
#     weights[~finite_epoch] *= 0.35
#     weights[flat_epoch] *= 0.50
#     weights[extreme_epoch] *= 0.65
#     weights[jump_epoch] *= 0.75
#     weights = np.clip(weights, min_weight, 1.0).astype(np.float32)

#     flags = {
#         "nonfinite": ~finite_epoch,
#         "flat": flat_epoch,
#         "extreme": extreme_epoch,
#         "jump": jump_epoch,
#     }
#     if return_flags:
#         return weights, flags
#     return weights


# def preprocess_multichannel_epochs(X, fs=100, apply_filters=True, normalize=True, clip_sigma=8.0):
#     """
#     Apply lightweight physiology-aware preprocessing to one recording.

#     X is expected to be (epochs, samples, channels). EEG/EOG are band-passed
#     to 0.5-30 Hz. EMG is left unfiltered to preserve tonic muscle tone,
#     then each channel is normalized
#     per recording. Recording-level normalization keeps relative epoch amplitude
#     cues while reducing subject/device scale drift.
#     """
#     X = sanitize_signal_epochs(X)

#     if X.ndim == 2:
#         X = X[..., np.newaxis]

#     if apply_filters and X.shape[1] > 15:
#         fs = float(fs) if fs is not None else 100.0
#         nyquist = fs / 2.0
#         processed = X.copy()

#         eeg_high = min(30.0, nyquist * 0.95)
#         if eeg_high > 0.5 and X.shape[-1] >= 1:
#             sos_eeg = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
#             processed[:, :, 0] = signal.sosfiltfilt(sos_eeg, processed[:, :, 0], axis=1).astype(np.float32)

#         if eeg_high > 0.5 and X.shape[-1] >= 2:
#             sos_eog = signal.butter(4, [0.5, eeg_high], btype="bandpass", fs=fs, output="sos")
#             processed[:, :, 1] = signal.sosfiltfilt(sos_eog, processed[:, :, 1], axis=1).astype(np.float32)

#         X = processed

#     if normalize:
#         mean = np.mean(X, axis=(0, 1), keepdims=True)
#         std = np.std(X, axis=(0, 1), keepdims=True)
#         X = (X - mean) / (std + 1e-6)
#         X = sanitize_signal_epochs(X, clip_value=clip_sigma)

#     return X.astype(np.float32)


# def split_npz_files(files, test_size=0.15, val_size=0.15, seed=42):
#     """
#     Deterministic file-level split. This avoids mixing epochs from the same
#     recording across train/validation/test sets.
#     """
#     rng = np.random.default_rng(seed)
#     files = np.array(sorted(files))
#     shuffled = files[rng.permutation(len(files))]

#     n_test = max(1, int(round(len(shuffled) * test_size)))
#     test_files = sorted(shuffled[:n_test].tolist())
#     train_val_files = shuffled[n_test:]

#     n_val = max(1, int(round(len(train_val_files) * val_size)))
#     val_files = sorted(train_val_files[:n_val].tolist())
#     train_files = sorted(train_val_files[n_val:].tolist())

#     return train_files, val_files, test_files

# # --- 获取文件列表 ---
# def get_npz_file_list(folder_path):
#     """获取指定文件夹下所有 .npz 文件的文件名列表（排序保证可复现）"""
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"找不到目录: {folder_path}")
#     return sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])


# # --- 处理 NPZ 文件 (预分配内存 + 支持按文件子集加载) ---
# def load_npz_data(
#     folder_path,
#     max_files=None,
#     file_subset=None,
#     preprocess=False,
#     apply_filters=True,
#     normalize=True,
#     clip_sigma=8.0,
#     return_quality=False,
# ):
#     """
#     加载指定文件夹下所有的 npz 文件并合并。
#     采用预分配内存策略，彻底解决 concatenate 导致的 OOM 内存爆炸问题。

#     参数:
#         folder_path: 数据文件夹路径
#         max_files:   (调试用) 只加载前 N 个文件
#         file_subset: (按受试者切分) 只加载指定的文件名列表
#     """
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"找不到目录: {folder_path}")

#     if file_subset is not None:
#         files = [f for f in file_subset if f.endswith('.npz')]
#     else:
#         files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

#     if max_files is not None:
#         files = files[:max_files]

#     if len(files) == 0:
#         raise ValueError("没有找到任何 .npz 文件！")

#     print(f"正在从 {folder_path} 加载 {len(files)} 个文件...")

#     # 第一遍扫描：只读取 shape 算出总样本数，绝不把数据拉入内存
#     total_samples = 0
#     for f in files:
#         data = np.load(os.path.join(folder_path, f), mmap_mode='r')
#         total_samples += data['x'].shape[0]

#     print(f"扫描完毕，总样本数: {total_samples}。正在预分配连续内存...")

#     with np.load(os.path.join(folder_path, files[0]), mmap_mode='r') as first_data:
#         first_shape = first_data['x'].shape

#     if len(first_shape) == 2:
#         sample_shape = (first_shape[1], 1)
#     else:
#         sample_shape = first_shape[1:]

#     X = np.empty((total_samples, *sample_shape), dtype=np.float32)
#     y = np.empty((total_samples,), dtype=np.int32)
#     quality_weights = np.empty((total_samples,), dtype=np.float32) if return_quality else None

#     # 第二遍扫描：像拼图一样把数据填进预先挖好的坑里
#     current_idx = 0
#     for i, f in enumerate(files):
#         with np.load(os.path.join(folder_path, f)) as data:
#             x_data = data['x'].astype(np.float32)
#             y_data = data['y']
#             fs = float(data['fs']) if 'fs' in data.files else 100.0
#             n_samples = x_data.shape[0]

#             # 适配 3 通道格式
#             if len(x_data.shape) == 2:
#                 x_data = np.expand_dims(x_data, axis=-1)

#             if preprocess:
#                 x_data = preprocess_multichannel_epochs(
#                     x_data,
#                     fs=fs,
#                     apply_filters=apply_filters,
#                     normalize=normalize,
#                     clip_sigma=clip_sigma,
#                 )
#             else:
#                 x_data = sanitize_signal_epochs(x_data)

#             if return_quality:
#                 quality_weights[current_idx: current_idx + n_samples] = compute_epoch_quality_weights(x_data)

#             X[current_idx: current_idx + n_samples] = x_data
#             y[current_idx: current_idx + n_samples] = y_data

#             current_idx += n_samples

#             if (i + 1) % 5 == 0 or (i + 1) == len(files):
#                 print(f"进度: 已加载 {i + 1}/{len(files)}...")

#     if return_quality:
#         return X, y, quality_weights
#     return X, y


# # --- 格式化函数 (核心：适配 Conv1D) ---
# def format_for_model(X, y, num_classes=5):
#     """
#     确保 X 是 (Samples, 3000, 1) 且 y 是 One-hot 编码。
#     """
#     if len(X.shape) == 2:
#         X = np.expand_dims(X, axis=-1)
#     y_one_hot = to_categorical(y, num_classes=num_classes)
#     return X, y_one_hot


# # --- 处理 EDF 文件 (可选备份) ---
# def load_edf_epoch(psg_path, hypno_path):
#     import mne

#     raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
#     annot = mne.read_annotations(hypno_path)
#     raw.set_annotations(annot, emit_warning=False)
#     event_id = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
#                 'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
#     events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=30.)
#     epochs = mne.Epochs(raw=raw, events=events, event_id=event_id,
#                         tmin=0., tmax=30. - 1 / raw.info['sfreq'],
#                         baseline=None, preload=True, verbose=False)
#     return epochs.get_data(), epochs.events[:, 2]
