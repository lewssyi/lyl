import numpy as np
import os
import mne
from tensorflow.keras.utils import to_categorical

# --- 处理 NPZ 文件 ---
def load_npz_data(folder_path):
    """
    加载指定文件夹下所有的 npz 文件并合并。
    适配你的路径: EEG_data/eeg_fpz_cz/
    """
    all_x = []
    all_y = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"找不到目录: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"正在从 {folder_path} 加载 {len(files)} 个文件...")

    for f in files:
        with np.load(os.path.join(folder_path, f)) as data:
            all_x.append(data['x']) # EEG 信号
            all_y.append(data['y']) # 标签

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y

# --- 格式化函数 (核心：适配 Conv1D) ---
def format_for_model(X, y, num_classes=5):
    """
    确保 X 是 (Samples, 3000, 1) 且 y 是 One-hot 编码。
    """
    # 如果原始数据是 (Samples, 3000)，增加通道维
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)

    # 睡眠分期 5 类: 0:W, 1:N1, 2:N2, 3:N3, 4:REM
    y_one_hot = to_categorical(y, num_classes=num_classes)

    return X, y_one_hot

# --- 处理 EDF 文件 (可选备份) ---
def load_edf_epoch(psg_path, hypno_path):
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