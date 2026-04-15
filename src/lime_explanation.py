import numpy as np
import copy
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
import scipy.signal as signal

# ==========================================
# 1. 基础扰动函数 (统一改为就地修改 in-place)
# ==========================================
def perturb_total_mean(signal, start_idx, end_idx):
    """用全局均值替换片段"""
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.mean(signal)

def perturb_mean(signal, start_idx, end_idx):
    """用局部均值替换片段"""
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.mean(signal[start_idx:end_idx])

def perturb_noise(signal, start_idx, end_idx):
    """用随机噪声替换片段"""
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.random.uniform(
        np.min(signal), np.max(signal), end_idx - start_idx
    )

# ==========================================
# 2. 多模态 LIME 核心引擎 (全面升级适配动态 segments)
# ==========================================
def generate_random_perturbations(num_perturbations, num_features):
    """
    生成二进制扰动矩阵 (1=保留, 0=遮盖)
    注意：现在的 num_features = 通道数(3) * 动态切片数量
    """
    random_perturbations = np.random.binomial(1, 0.5, size=(num_perturbations, num_features))
    return random_perturbations


def calculate_cosine_distances(random_perturbations, num_features):
    """计算扰动向量与原始信号表示（全1向量）之间的余弦距离"""
    original_rep = np.ones((1, num_features))
    cosine_distances = pairwise_distances(random_perturbations, original_rep, metric='cosine').ravel()
    return cosine_distances


def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    """使用核函数将距离转化为权重"""
    weights = np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))
    return weights


def analyze_prediction(probability_vector, class_labels):
    """分析概率向量，返回排序后的类别和概率最大的类别"""
    if probability_vector.ndim == 1:
        probability_vector = probability_vector.reshape(1, -1)

    predicted_class_index = int(np.argmax(probability_vector, axis=1)[0])
    top_indices = probability_vector[0].argsort()[::-1]
    top_pred_classes = [(class_labels[i], probability_vector[0][i]) for i in top_indices]

    return top_pred_classes, predicted_class_index


# ==========================================
# 3. 多通道扰动应用层 (关键重构)
# ==========================================
def apply_perturbation_to_multimodal(signal, perturbation, segments, perturb_function=perturb_mean):
    """
    将二进制扰动应用到多通道信号的动态切片上。

    参数:
    - signal: (3000, 3) 原始生理信号
    - perturbation: 长度为 (3 * len(segments)) 的一维数组 (0和1)
    - segments: 动态切片列表，如 [(0, 450), (450, 1200), ...]
    """
    perturbed_signal = copy.deepcopy(signal)
    num_segments = len(segments)
    num_channels = signal.shape[1] if len(signal.shape) == 2 else 1

    for c in range(num_channels):
        # 截取属于当前通道的扰动遮罩
        # 例如: 通道0(EEG)拿前一段，通道1(EOG)拿中段，通道2(EMG)拿后段
        channel_mask = perturbation[c * num_segments: (c + 1) * num_segments]

        for i, active in enumerate(channel_mask):
            # 如果该片段被设置为 0 (inactive)，则执行扰动抹平
            if not active:
                start_idx, end_idx = segments[i]

                # 对当前通道的特定时间段应用扰动
                if num_channels > 1:
                    perturb_function(perturbed_signal[:, c], start_idx, end_idx)
                else:
                    perturb_function(perturbed_signal, start_idx, end_idx)

    return perturbed_signal


def predict_perturbations(model, instance_multimodal, random_perturbations, segments, perturb_function):
    """更新：支持双流输入的批量预测"""
    perturbation_predictions = []

    for perturbation in random_perturbations:
        # 1. 生成打码后的多通道波形
        perturbed_signal = apply_perturbation_to_multimodal(
            instance_multimodal, perturbation, segments, perturb_function
        )
        # 2. 关键！为打码后的波形，重新提取切片特征
        perturbed_seg_features = extract_segment_features(perturbed_signal, segments, max_segments=30)

        # 3. 双流预测
        sig_reshaped = perturbed_signal[np.newaxis, :, :]
        seg_reshaped = perturbed_seg_features[np.newaxis, :, :]

        model_prediction = model.predict([sig_reshaped, seg_reshaped], verbose=0)
        perturbation_predictions.append(model_prediction[0])

    return np.array(perturbation_predictions)


def run_multimodal_lime(model, instance_multimodal, segments, num_perturbations=500, perturb_func=perturb_mean):
    """基准预测支持双流"""
    num_segments = len(segments)
    num_channels = instance_multimodal.shape[1]
    num_features = num_segments * num_channels

    print(
        f"LIME Engine: 正在为 {num_channels} 个通道，共 {num_segments} 个动态切片生成 {num_perturbations} 次随机扰动...")

    random_perturbations = generate_random_perturbations(num_perturbations, num_features)
    predictions = predict_perturbations(model, instance_multimodal, random_perturbations, segments, perturb_func)

    distances = calculate_cosine_distances(random_perturbations, num_features)
    weights = calculate_weights_from_distances(distances)

    # 提取原始切片特征进行基准预测
    orig_seg_features = extract_segment_features(instance_multimodal, segments, max_segments=30)
    original_pred = \
    model.predict([instance_multimodal[np.newaxis, :, :], orig_seg_features[np.newaxis, :, :]], verbose=0)[0]

    target_class = int(np.argmax(original_pred))
    segment_importance = fit_explainable_model(predictions, random_perturbations, weights, target_class)

    return target_class, original_pred, segment_importance

def fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class):
    """拟合带权重的岭回归模型，提取特征重要性系数"""
    if len(perturbation_predictions.shape) == 1:
        y_target = perturbation_predictions
    elif perturbation_predictions.shape[1] == 1:
        y_target = perturbation_predictions.ravel()
    else:
        y_target = perturbation_predictions[:, target_class]

    clf = Ridge(alpha=1.0)
    clf.fit(random_perturbations, y_target, sample_weight=weights)

    return clf.coef_


def semantic_segmentation(instance_multimodal, target_segments=30, min_len=50):
    """
    【升级版】基于多模态能量融合的动态切片算法 (EEG + EOG + EMG)
    它会综合考量大脑活动、眼动和肌肉张力，寻找三个系统共同的“平静期”进行切片。
    """
    total_length = instance_multimodal.shape[0]
    num_channels = instance_multimodal.shape[1]

    # 1. 提取各通道并计算能量包络线
    smoothed_energies = []
    window_size = 100  # 约 1 秒的平滑窗口

    for c in range(num_channels):
        channel_signal = instance_multimodal[:, c]
        # 瞬时能量
        energy = channel_signal ** 2
        # 滑动平均平滑
        smoothed = np.convolve(energy, np.ones(window_size) / window_size, mode='same')

        # 关键步：归一化 (Min-Max Scaling)
        # 防止某个通道（比如波动剧烈的 EEG）掩盖了 EMG 的细微变化
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        if max_val - min_val > 1e-6:
            smoothed_norm = (smoothed - min_val) / (max_val - min_val)
        else:
            smoothed_norm = smoothed  # 防止除以 0

        smoothed_energies.append(smoothed_norm)

    # 2. 多模态能量加权融合
    # 权重可以根据临床经验调整。比如 EEG 是主导，权重设高点；EOG 和 EMG 辅助捕捉突变
    weights = [0.5, 0.25, 0.25]

    total_energy = np.zeros(total_length)
    for c in range(num_channels):
        total_energy += smoothed_energies[c] * weights[c]

    # 3. 寻找全局总能量的波谷 (共同的平静期)
    inverted_energy = -total_energy

    # 根据加权后的尺度微调 prominence（突出度阈值）
    valleys, _ = signal.find_peaks(inverted_energy, distance=min_len, prominence=0.005)

    # 4. 构建切片坐标
    segments = []
    start_idx = 0

    for valley_idx in valleys:
        if valley_idx - start_idx >= min_len:
            segments.append((start_idx, valley_idx))
            start_idx = valley_idx

    # 收尾
    if start_idx < total_length:
        segments.append((start_idx, total_length))

    # 退化保护：如果综合后信号依然极其平滑导致切不出几段，则退化为均分
    if len(segments) < target_segments // 2:
        print("⚠️ 多模态联合特征过于平缓，部分退化为均匀切片以保证 LIME 运行...")
        slice_width = total_length // target_segments
        segments = [(i * slice_width, min((i + 1) * slice_width, total_length)) for i in range(target_segments)]

    return segments


def extract_segment_features(signal, segments, max_segments=30):
    """
    将动态切片提取为统计特征矩阵，喂给 LSTM。
    每个切片提取 13 个特征：片段长度(1) + 均值(3) + 标准差(3) + 最大值(3) + 能量(3)
    """
    features = []
    for start, end in segments:
        seg_len = end - start
        if seg_len == 0: continue

        seg_data = signal[start:end, :]  # 取出这一段的三个通道信号

        mean_val = np.mean(seg_data, axis=0)
        std_val = np.std(seg_data, axis=0)
        max_val = np.max(seg_data, axis=0)
        energy = np.sum(seg_data ** 2, axis=0) / seg_len

        # 拼接特征
        feat = np.concatenate(([seg_len], mean_val, std_val, max_val, energy))
        features.append(feat)

    features = np.array(features)
    num_features = features.shape[1] if len(features) > 0 else 13

    # 补齐或截断到固定长度 (max_segments)
    if len(features) < max_segments:
        pad_width = max_segments - len(features)
        features = np.vstack((features, np.zeros((pad_width, num_features))))
    else:
        features = features[:max_segments]

    return features
