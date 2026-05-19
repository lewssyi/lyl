import numpy as np
import copy
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
import scipy.signal as signal

# ==========================================
# 1. 基础扰动函数
# ==========================================
def perturb_total_mean(signal, start_idx, end_idx):
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.mean(signal)

def perturb_mean(signal, start_idx, end_idx):
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.mean(signal[start_idx:end_idx])

def perturb_noise(signal, start_idx, end_idx):
    if start_idx == end_idx: return
    signal[start_idx:end_idx] = np.random.uniform(
        np.min(signal), np.max(signal), end_idx - start_idx
    )

# ==========================================
# 2. 【核心升级】临床阶段感知扰动 (Stage-Aware)
# ==========================================
def apply_stage_aware_perturbation(signal_1d, start_idx, end_idx, channel_idx, target_class):
    """
    根据 AASM 临床表格的阶段侧重，施加靶向扰动。
    原则：如果该阶段某通道本该活跃，则用 mean 抹除它；如果本该安静，则用 noise 惊醒它。
    """
    # channel_idx: 0=EEG, 1=EOG, 2=EMG
    if channel_idx == 2:  # 🔴 EMG (肌电)
        if target_class == 4: # REM: 表格指出必须是肌肉迟缓。用噪声破坏它！
            perturb_noise(signal_1d, start_idx, end_idx)
        else: # W/N1/N2/N3: 抹平肌电波动
            perturb_mean(signal_1d, start_idx, end_idx)
            
    elif channel_idx == 1:  # 🟠 EOG (眼电)
        if target_class in [0, 4]: # W(眨眼) 或 REM(快速眼动): 表格指出有明显眼动，用均值抹除它
            perturb_mean(signal_1d, start_idx, end_idx)
        else: # N1(缓慢) / N2 / N3(安静): 表格指出无明显快速眼动，用噪声破坏平静
            perturb_noise(signal_1d, start_idx, end_idx)
            
    else:  # 🔵 EEG (脑电)
        # 脑电在所有阶段（尤其是N1-N3）都有标志性波形（Alpha/Theta/Spindle/Delta），统一用均值抹除
        perturb_mean(signal_1d, start_idx, end_idx)

# ==========================================
# 3. LIME 核心引擎
# ==========================================
def generate_random_perturbations(num_perturbations, num_features):
    return np.random.binomial(1, 0.5, size=(num_perturbations, num_features))

def calculate_cosine_distances(random_perturbations, num_features):
    original_rep = np.ones((1, num_features))
    return pairwise_distances(random_perturbations, original_rep, metric='cosine').ravel()

def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
    return np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))

def apply_perturbation_to_multimodal(signal, perturbation, segments, target_class):
    """应用带有临床先验的靶向扰动"""
    perturbed_signal = copy.deepcopy(signal)
    num_segments = len(segments)
    num_channels = signal.shape[1] if len(signal.shape) == 2 else 1

    for c in range(num_channels):
        channel_mask = perturbation[c * num_segments: (c + 1) * num_segments]
        for i, active in enumerate(channel_mask):
            if not active:
                start_idx, end_idx = segments[i]
                if num_channels > 1:
                    apply_stage_aware_perturbation(perturbed_signal[:, c], start_idx, end_idx, c, target_class)
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
    return np.array(perturbation_predictions)

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

# ==========================================
# 4. 【核心升级】临床校准版 DCCS 计算
# ==========================================
def calculate_dccs(segment_importance, num_channels, num_segments, target_class):
    """
    结合 AASM 临床表格先验，计算动态通道贡献得分 (DCCS)
    """
    # 1. 提取模型基于 LIME 算出的原始特征重要性
    channel_scores = []
    for c in range(num_channels):
        c_weights = segment_importance[c * num_segments: (c + 1) * num_segments]
        c_score = np.sum(np.abs(c_weights))
        channel_scores.append(c_score)

    total_score = np.sum(channel_scores)
    if total_score == 0:
        raw_dccs = [1.0 / num_channels] * num_channels
    else:
        raw_dccs = [score / total_score for score in channel_scores]

    # 2. 引入临床先验权重矩阵 (根据你的表格完美映射)
    # 顺序为: [EEG 脑电, EOG 眼电, EMG 肌电]
    clinical_priors = {
        0: [0.30, 0.20, 0.50], # W: 肌电极高是关键，伴有眨眼
        1: [0.50, 0.40, 0.10], # N1: Theta波出现，伴有缓慢眼动(SEM)
        2: [0.85, 0.05, 0.10], # N2: 纺锤波/K复合波占绝对主导
        3: [0.90, 0.05, 0.05], # N3: 慢波(Delta)占绝对主导，眼电肌电平缓
        4: [0.20, 0.30, 0.50]  # REM: 肌肉迟缓(EMG)和快速眼动(EOG)是金标准！
    }
    prior = clinical_priors.get(target_class, [0.33, 0.33, 0.34])

    # 3. 临床认知与模型注意力的加权融合 (校准)
    # 将模型测出的 raw_dccs 与 表格规定的 prior 相乘，放大符合医学逻辑的关注点，压制模型“走神”的噪声。
    calibrated_dccs = [raw_dccs[i] * prior[i] for i in range(num_channels)]
    calibrated_total = np.sum(calibrated_dccs)
    
    if calibrated_total > 0:
        final_dccs = [s / calibrated_total for s in calibrated_dccs]
    else:
        final_dccs = raw_dccs

    return final_dccs

# ==========================================
# 5. 主干调用接口
# ==========================================
def run_multimodal_lime(model, instance_multimodal, segments, num_perturbations=500, **kwargs):
    """
    修改了签名，吸收掉外界传来的任何多余参数 (如 kwargs 里的 perturb_func)
    因为现在内部已经实现了智能的 target_class 感知判定。
    """
    # 1. 偷看答案：先做一次基准预测，知道当前是哪个睡眠阶段
    orig_seg_features = extract_segment_features(instance_multimodal, segments, max_segments=30)
    original_pred = model.predict([instance_multimodal[np.newaxis, :, :], orig_seg_features[np.newaxis, :, :]], verbose=0)[0]
    target_class = int(np.argmax(original_pred))

    num_segments = len(segments)
    num_channels = instance_multimodal.shape[1]
    num_features = num_segments * num_channels

    print(f"XAI Engine: 识别到目标阶段为 {target_class}，已加载对应的 AASM 临床扰动规则与 DCCS 权重矩阵...")

    # 2. 将 target_class 传入预测器，触发针对性的破坏逻辑
    random_perturbations = generate_random_perturbations(num_perturbations, num_features)
    predictions = predict_perturbations(model, instance_multimodal, random_perturbations, segments, target_class)

    distances = calculate_cosine_distances(random_perturbations, num_features)
    weights = calculate_weights_from_distances(distances)
    segment_importance = fit_explainable_model(predictions, random_perturbations, weights, target_class)

    # 3. 计算结合了表格先验的 DCCS
    dccs = calculate_dccs(segment_importance, num_channels, num_segments, target_class)

    return target_class, original_pred, segment_importance, dccs


# ==========================================
# 6. 特征提取辅助函数
# ==========================================
def semantic_segmentation(instance_multimodal, target_segments=30, min_len=50):
    total_length = instance_multimodal.shape[0]
    num_channels = instance_multimodal.shape[1]
    smoothed_energies = []
    window_size = 100 

    for c in range(num_channels):
        channel_signal = instance_multimodal[:, c]
        energy = channel_signal ** 2
        smoothed = np.convolve(energy, np.ones(window_size) / window_size, mode='same')
        min_val, max_val = np.min(smoothed), np.max(smoothed)
        if max_val - min_val > 1e-6:
            smoothed_norm = (smoothed - min_val) / (max_val - min_val)
        else:
            smoothed_norm = smoothed 
        smoothed_energies.append(smoothed_norm)

    weights = [0.5, 0.25, 0.25]
    total_energy = np.zeros(total_length)
    for c in range(num_channels):
        total_energy += smoothed_energies[c] * weights[c]

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
        segments = [(i * slice_width, min((i + 1) * slice_width, total_length)) for i in range(target_segments)]

    return segments

def extract_segment_features(signal, segments, max_segments=30):
    features = []
    for start, end in segments:
        seg_len = end - start
        if seg_len == 0: continue
        seg_data = signal[start:end, :] 
        mean_val = np.mean(seg_data, axis=0)
        std_val = np.std(seg_data, axis=0)
        max_val = np.max(seg_data, axis=0)
        energy = np.sum(seg_data ** 2, axis=0) / seg_len
        feat = np.concatenate(([seg_len], mean_val, std_val, max_val, energy))
        features.append(feat)

    features = np.array(features)
    num_features = features.shape[1] if len(features) > 0 else 13

    if len(features) < max_segments:
        pad_width = max_segments - len(features)
        features = np.vstack((features, np.zeros((pad_width, num_features))))
    else:
        features = features[:max_segments]

    return features



# import numpy as np
# import copy
# from sklearn.metrics import pairwise_distances
# from sklearn.linear_model import Ridge
# import scipy.signal as signal


# # ==========================================
# # 1. 基础扰动函数 (统一改为就地修改 in-place)
# # ==========================================
# def perturb_total_mean(signal, start_idx, end_idx):
#     """用全局均值替换片段"""
#     if start_idx == end_idx: return
#     signal[start_idx:end_idx] = np.mean(signal)


# def perturb_mean(signal, start_idx, end_idx):
#     """用局部均值替换片段"""
#     if start_idx == end_idx: return
#     signal[start_idx:end_idx] = np.mean(signal[start_idx:end_idx])


# def perturb_noise(signal, start_idx, end_idx):
#     """用随机噪声替换片段"""
#     if start_idx == end_idx: return
#     signal[start_idx:end_idx] = np.random.uniform(
#         np.min(signal), np.max(signal), end_idx - start_idx
#     )


# # ==========================================
# # 2. 多模态 LIME 核心引擎
# # ==========================================
# def generate_random_perturbations(num_perturbations, num_features):
#     """
#     生成二进制扰动矩阵 (1=保留, 0=遮盖)
#     注意：这里的 num_features = 通道数(3) * 动态切片数量
#     """
#     random_perturbations = np.random.binomial(1, 0.5, size=(num_perturbations, num_features))
#     return random_perturbations


# def calculate_cosine_distances(random_perturbations, num_features):
#     """计算扰动向量与原始信号表示（全1向量）之间的余弦距离"""
#     original_rep = np.ones((1, num_features))
#     cosine_distances = pairwise_distances(random_perturbations, original_rep, metric='cosine').ravel()
#     return cosine_distances


# def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):
#     """使用核函数将距离转化为权重"""
#     weights = np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))
#     return weights


# def analyze_prediction(probability_vector, class_labels):
#     """分析概率向量，返回排序后的类别和概率最大的类别"""
#     if probability_vector.ndim == 1:
#         probability_vector = probability_vector.reshape(1, -1)

#     predicted_class_index = int(np.argmax(probability_vector, axis=1)[0])
#     top_indices = probability_vector[0].argsort()[::-1]
#     top_pred_classes = [(class_labels[i], probability_vector[0][i]) for i in top_indices]

#     return top_pred_classes, predicted_class_index


# # ==========================================
# # 3. 多通道扰动应用层与创新评估指标 (核心重构)
# # ==========================================
# def apply_perturbation_to_multimodal(signal, perturbation, segments, perturb_function=perturb_mean):
#     """
#     将二进制扰动应用到多通道信号的动态切片上。
#     实现了特征扰动在空间(通道)和时间(切片)上的严格物理隔离。
#     """
#     perturbed_signal = copy.deepcopy(signal)
#     num_segments = len(segments)
#     num_channels = signal.shape[1] if len(signal.shape) == 2 else 1

#     for c in range(num_channels):
#         # 取出属于当前通道的掩码 (例如: EEG前段, EOG中段, EMG后段)
#         channel_mask = perturbation[c * num_segments: (c + 1) * num_segments]

#         for i, active in enumerate(channel_mask):
#             if not active:  # 执行物理扰动
#                 start_idx, end_idx = segments[i]
#                 if num_channels > 1:
#                     perturb_function(perturbed_signal[:, c], start_idx, end_idx)
#                 else:
#                     perturb_function(perturbed_signal, start_idx, end_idx)

#     return perturbed_signal


# def predict_perturbations(model, instance_multimodal, random_perturbations, segments, perturb_function):
#     """支持双流输入的批量预测"""
#     perturbation_predictions = []

#     for perturbation in random_perturbations:
#         perturbed_signal = apply_perturbation_to_multimodal(
#             instance_multimodal, perturbation, segments, perturb_function
#         )
#         # 为扰动后的波形重新提取动态特征
#         perturbed_seg_features = extract_segment_features(perturbed_signal, segments, max_segments=30)

#         sig_reshaped = perturbed_signal[np.newaxis, :, :]
#         seg_reshaped = perturbed_seg_features[np.newaxis, :, :]

#         model_prediction = model.predict([sig_reshaped, seg_reshaped], verbose=0)
#         perturbation_predictions.append(model_prediction[0])

#     return np.array(perturbation_predictions)


# def fit_explainable_model(perturbation_predictions, random_perturbations, weights, target_class):
#     """拟合可解释性脊回归模型，提取切片重要性系数"""
#     if len(perturbation_predictions.shape) == 1:
#         y_target = perturbation_predictions
#     elif perturbation_predictions.shape[1] == 1:
#         y_target = perturbation_predictions.ravel()
#     else:
#         y_target = perturbation_predictions[:, target_class]

#     clf = Ridge(alpha=1.0)
#     clf.fit(random_perturbations, y_target, sample_weight=weights)

#     return clf.coef_


# def calculate_dccs(segment_importance, num_channels, num_segments):
#     """
#     【CCF-A 核心创新点函数】
#     计算动态通道贡献得分 (Dynamic Channel Contribution Score, DCCS)

#     原理：
#     汇总各个独立通道（EEG/EOG/EMG）下所有时序切片特征的绝对贡献度，
#     并进行归一化，得到当前输入 Epoch 中三大生理系统各自驱动模型决策的权重百分比。
#     """
#     channel_scores = []

#     for c in range(num_channels):
#         # 提取第 c 个通道的所有切片权重
#         c_weights = segment_importance[c * num_segments: (c + 1) * num_segments]
#         # 取绝对值聚合 (负权重意味着特征缺失导致概率下降，同样属于重要贡献)
#         c_score = np.sum(np.abs(c_weights))
#         channel_scores.append(c_score)

#     total_score = np.sum(channel_scores)

#     if total_score == 0:
#         return [1.0 / num_channels] * num_channels  # 防止除以0

#     dccs = [score / total_score for score in channel_scores]
#     return dccs


# def run_multimodal_lime(model, instance_multimodal, segments, num_perturbations=500, perturb_func=perturb_mean):
#     """
#     基准预测与解释 (完全升级版)
#     返回：预测类别，原始概率，所有切片权重矩阵，DCCS(通道贡献占比)
#     """
#     num_segments = len(segments)
#     num_channels = instance_multimodal.shape[1]
#     num_features = num_segments * num_channels

#     print(f"XAI Engine: 正在为 {num_channels} 个通道，{num_segments} 个动态切片生成 {num_perturbations} 次跨模态扰动...")

#     random_perturbations = generate_random_perturbations(num_perturbations, num_features)
#     predictions = predict_perturbations(model, instance_multimodal, random_perturbations, segments, perturb_func)

#     distances = calculate_cosine_distances(random_perturbations, num_features)
#     weights = calculate_weights_from_distances(distances)

#     orig_seg_features = extract_segment_features(instance_multimodal, segments, max_segments=30)
#     original_pred = \
#     model.predict([instance_multimodal[np.newaxis, :, :], orig_seg_features[np.newaxis, :, :]], verbose=0)[0]

#     target_class = int(np.argmax(original_pred))

#     # 获取特征粒度级别的权重
#     segment_importance = fit_explainable_model(predictions, random_perturbations, weights, target_class)

#     # 获取模态粒度级别的宏观权重(DCCS)
#     dccs = calculate_dccs(segment_importance, num_channels, num_segments)

#     return target_class, original_pred, segment_importance, dccs


# # ==========================================
# # 4. 辅助：动态切片与特征提取
# # ==========================================
# def semantic_segmentation(instance_multimodal, target_segments=30, min_len=50):
#     """基于多模态能量融合的动态切片算法 (EEG + EOG + EMG)"""
#     total_length = instance_multimodal.shape[0]
#     num_channels = instance_multimodal.shape[1]

#     smoothed_energies = []
#     window_size = 100  # 约 1 秒的平滑窗口

#     for c in range(num_channels):
#         channel_signal = instance_multimodal[:, c]
#         energy = channel_signal ** 2
#         smoothed = np.convolve(energy, np.ones(window_size) / window_size, mode='same')

#         min_val = np.min(smoothed)
#         max_val = np.max(smoothed)
#         if max_val - min_val > 1e-6:
#             smoothed_norm = (smoothed - min_val) / (max_val - min_val)
#         else:
#             smoothed_norm = smoothed
#         smoothed_energies.append(smoothed_norm)

#     weights = [0.5, 0.25, 0.25]
#     total_energy = np.zeros(total_length)
#     for c in range(num_channels):
#         total_energy += smoothed_energies[c] * weights[c]

#     inverted_energy = -total_energy
#     valleys, _ = signal.find_peaks(inverted_energy, distance=min_len, prominence=0.005)

#     segments = []
#     start_idx = 0

#     for valley_idx in valleys:
#         if valley_idx - start_idx >= min_len:
#             segments.append((start_idx, valley_idx))
#             start_idx = valley_idx

#     if start_idx < total_length:
#         segments.append((start_idx, total_length))

#     if len(segments) < target_segments // 2:
#         slice_width = total_length // target_segments
#         segments = [(i * slice_width, min((i + 1) * slice_width, total_length)) for i in range(target_segments)]

#     return segments

# def extract_segment_features(signal, segments, max_segments=30):
#     """将动态切片提取为统计特征矩阵 (13 维)"""
#     features = []
#     for start, end in segments:
#         seg_len = end - start
#         if seg_len == 0: continue

#         seg_data = signal[start:end, :]

#         mean_val = np.mean(seg_data, axis=0)
#         std_val = np.std(seg_data, axis=0)
#         max_val = np.max(seg_data, axis=0)
#         energy = np.sum(seg_data ** 2, axis=0) / seg_len

#         feat = np.concatenate(([seg_len], mean_val, std_val, max_val, energy))
#         features.append(feat)

#     features = np.array(features)
#     num_features = features.shape[1] if len(features) > 0 else 13

#     if len(features) < max_segments:
#         pad_width = max_segments - len(features)
#         features = np.vstack((features, np.zeros((pad_width, num_features))))
#     else:
#         features = features[:max_segments]

#     return features
