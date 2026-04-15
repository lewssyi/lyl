import numpy as np
from scipy import signal

def load_eeg_npz(file_path):
    npz_data = np.load(file_path, allow_pickle=True)
    print(f"✅ {file_path} 包含的键：{npz_data.files}")

    # 关键修改：用"x"替代"data"
    raw_eeg = npz_data["x"]  # 提取EEG信号（实际键是"x"）
    sfreq = npz_data["fs"]  # 提取采样率（后续预处理用）

    # 调整维度
    if raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    print(f"✅ EEG数据维度：{raw_eeg.shape} (时间点, 导联数)")
    print(f"✅ 采样率：{sfreq} Hz")

    return raw_eeg, sfreq  # 返回信号和采样率


def preprocess_eeg_data(raw_eeg, sfreq=100, normalize=True):
    """
    多通道专属预处理：为 EEG/EOG 和 EMG 分别设置不同的滤波器
    """
    processed_eeg = np.zeros_like(raw_eeg)

    # 通道 0: EEG Fpz-Cz (带通 0.5 ~ 30 Hz)
    # 通道 1: EOG horizontal (带通 0.5 ~ 30 Hz)
    b_eeg, a_eeg = signal.butter(4, [0.5, 30], btype='bandpass', fs=sfreq)
    processed_eeg[:, 0] = signal.filtfilt(b_eeg, a_eeg, raw_eeg[:, 0])
    processed_eeg[:, 1] = signal.filtfilt(b_eeg, a_eeg, raw_eeg[:, 1])

    # 通道 2: EMG submental (高通 > 10 Hz，过滤掉心跳低频干扰)
    b_emg, a_emg = signal.butter(4, 10, btype='highpass', fs=sfreq)
    processed_eeg[:, 2] = signal.filtfilt(b_emg, a_emg, raw_eeg[:, 2])

    # 独立通道标准化
    if normalize:
        mean = np.mean(processed_eeg, axis=0, keepdims=True)
        std = np.std(processed_eeg, axis=0, keepdims=True)
        processed_eeg = (processed_eeg - mean) / (std + 1e-8)

    print(f"✅ 多通道预处理完成，维度：{processed_eeg.shape}")
    return processed_eeg