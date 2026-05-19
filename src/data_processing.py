import numpy as np
import os
import mne
from tensorflow.keras.utils import to_categorical

# --- 获取文件列表 ---
def get_npz_file_list(folder_path):
    """获取指定文件夹下所有 .npz 文件的文件名列表（排序保证可复现）"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"找不到目录: {folder_path}")
    return sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])


# --- 处理 NPZ 文件 (预分配内存 + 支持按文件子集加载) ---
def load_npz_data(folder_path, max_files=None, file_subset=None):
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

    X = np.empty((total_samples, 3000, 3), dtype=np.float32)
    y = np.empty((total_samples,), dtype=np.int32)

    # 第二遍扫描：像拼图一样把数据填进预先挖好的坑里
    current_idx = 0
    for i, f in enumerate(files):
        with np.load(os.path.join(folder_path, f)) as data:
            x_data = data['x']
            y_data = data['y']
            n_samples = x_data.shape[0]

            # 适配 3 通道格式
            if len(x_data.shape) == 2:
                x_data = np.expand_dims(x_data, axis=-1)

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