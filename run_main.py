# filepath: run_main.py
import sys
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
# ==========================================
# 0. 环境路径与随机种子设置
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from src.data_processing import load_npz_data
    from src.model_training import create_hybrid_model
    from src.lime_explanation import (
        run_multimodal_lime, perturb_mean,
        semantic_segmentation, extract_segment_features
    )
    from src.visualization import visualize_lime_explanation, visualize_gradcam_explanation
    from src.grad_cam import run_hybrid_grad_cam  # 导入我们新建的模块
    print("✅ 核心模块及增强版模型加载成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit()

# ==========================================
# 1. 全局配置与数据加载
# ==========================================
NUM_CLASSES = 5
CLASS_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']
MULTI_CHANNEL_PATH = r"E:\Python\LIME-for-Time-Series\EEG_data\eeg_eog_emg"

print("\n1. 正在加载多通道 (EEG/EOG/EMG) 原始数据...")
X_raw, y_raw = load_npz_data(MULTI_CHANNEL_PATH)
y = to_categorical(y_raw, num_classes=NUM_CLASSES)

# ==========================================
# 2. 提取生理基序 (Motifs) 特征
# ==========================================
print("\n2. 正在提取特征流数据 (用于双流网络的逻辑分支)...")
X_segments = []
for instance in tqdm(X_raw, desc="特征提取进度"):
    segs = semantic_segmentation(instance, target_segments=30)
    feats = extract_segment_features(instance, segs, max_segments=30)
    X_segments.append(feats)

X_segments = np.array(X_segments)
print(f"✅ 双流数据构建完成！波形: {X_raw.shape}, 特征: {X_segments.shape}")

split = int(len(X_raw) * 0.85)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
X_train_seg, X_test_seg = X_segments[:split], X_segments[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================
# 3. 训练增强版双流模型
# ==========================================
print("\n3. 正在构建【注意力机制驱动的多尺度融合网络】...")
model = create_hybrid_model(
    raw_shape=(3000, 3),
    segment_shape=(30, 13),
    num_classes=NUM_CLASSES
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

print("🚀 开始训练...")
model.fit(
    [X_train_raw, X_train_seg], y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks
)

# ==========================================
# 3.5 模型在测试集上的全面评估 (引入阈值移动 Threshold Moving)
# ==========================================
print("\n==========================================")
print("3.5 正在测试集上进行全面评估 (混淆矩阵与分类报告)...")
print("==========================================")

y_pred_probs = model.predict([X_test_raw, X_test_seg])
y_pred_classes = np.zeros(len(y_pred_probs), dtype=int)
y_true_classes = np.argmax(y_test, axis=1)

N1_THRESHOLD = 0.22

for i in range(len(y_pred_probs)):
    probs = y_pred_probs[i]
    if probs[1] >= N1_THRESHOLD:
        y_pred_classes[i] = 1
    else:
        probs_copy = np.copy(probs)
        probs_copy[1] = -1.0
        y_pred_classes[i] = np.argmax(probs_copy)

print(f"\n📊 分类报告 (应用 N1 专属阈值: {N1_THRESHOLD}):")
print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_LABELS, digits=4))

# ==========================================
# 4. 可解释性对比分析：LIME vs Grad-CAM
# ==========================================
print("\n==========================================")
print("4. 正在对同一测试集样本进行 LIME 与 Grad-CAM 对比分析...")
print("==========================================")

# 随机挑一个样本观察 (确保 LIME 和 Grad-CAM 用的是同一个数据)
id_sample = np.random.randint(0, len(X_test_raw))
instance_raw = X_test_raw[id_sample]
instance_seg = X_test_seg[id_sample] # 对应的特征流数据
actual_label = CLASS_LABELS[np.argmax(y_test[id_sample])]

print(f"📌 选定样本 ID: {id_sample} | 真实标签: {actual_label}")

# ----------------- A. 运行 LIME -----------------
print("\n[A] 启动 LIME 解释器 (分析片段特征重要性)...")
segments = semantic_segmentation(instance_raw, target_segments=30)
lime_class, lime_pred_probs, lime_importance = run_multimodal_lime(
    model=model,
    instance_multimodal=instance_raw,
    segments=segments,
    num_perturbations=300,
    perturb_func=perturb_mean
)

print(f"🎯 LIME 侧预测结果: {CLASS_LABELS[lime_class]} (置信度: {lime_pred_probs[lime_class]:.2%})")
visualize_lime_explanation(
    instance_multimodal=instance_raw,
    segment_importance=lime_importance,
    segments=segments,
    predicted_class=lime_class,
    prob=lime_pred_probs[lime_class],
    class_labels=CLASS_LABELS
)
plt.show()
# ----------------- B. 运行 Grad-CAM -----------------
print("\n[B] 启动 Grad-CAM 解释器 (分析原始波形空间显著性)...")
heatmap_resized, gradcam_class, gradcam_prob = run_hybrid_grad_cam(
    model=model,
    instance_raw=instance_raw,
    instance_seg=instance_seg
)

print(f"🎯 Grad-CAM 侧预测结果: {CLASS_LABELS[gradcam_class]} (置信度: {gradcam_prob:.2%})")
visualize_gradcam_explanation(
    instance_raw=instance_raw,
    heatmap=heatmap_resized,
    predicted_class=gradcam_class,
    prob=gradcam_prob,
    class_labels=CLASS_LABELS
)
plt.show()
print("\n🎉 分析完成！你可以对比两次弹出的图像，LIME 解释了'哪个片段最重要'，而 Grad-CAM 解释了'模型在盯着哪个波峰/波谷'。")

# import sys
# import os
# import numpy as np
# import random
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.utils import class_weight
# from tqdm import tqdm
#
# # ==========================================
# # 0. 环境路径与随机种子设置
# # ==========================================
# np.random.seed(42)
# tf.random.set_seed(42)
# random.seed(42)
#
# current_dir = os.getcwd()
# src_path = os.path.join(current_dir, 'src')
# if src_path not in sys.path:
#     sys.path.append(src_path)
#
# try:
#     from src.data_processing import load_npz_data
#     # 导入我们重写后的增强版模型
#     from src.model_training import create_hybrid_model
#     from src.lime_explanation import (
#         run_multimodal_lime, perturb_mean,
#         semantic_segmentation, extract_segment_features
#     )
#     from src.visualization import visualize_lime_explanation
#     print("✅ 核心模块及增强版模型加载成功！")
# except ImportError as e:
#     print(f"❌ 导入失败: {e}")
#     sys.exit()
#
# # ==========================================
# # 1. 全局配置与数据加载
# # ==========================================
# NUM_CLASSES = 5
# CLASS_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']
# MULTI_CHANNEL_PATH = r"E:\Python\LIME-for-Time-Series\EEG_data\eeg_eog_emg"
#
# print("\n1. 正在加载多通道 (EEG/EOG/EMG) 原始数据...")
# X_raw, y_raw = load_npz_data(MULTI_CHANNEL_PATH)
#
# # --- 核心改进：计算类别权重 ---
# # 这能让模型“重视”样本量少的类别（如 N1），显著提升整体准确率
# weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_raw),
#     y=y_raw
# )
# class_weight_dict = dict(enumerate(weights))
# print(f"📊 自动计算的类别权重: {class_weight_dict}")
#
# y = to_categorical(y_raw, num_classes=NUM_CLASSES)
#
# # ==========================================
# # 2. 提取生理基序 (Motifs) 特征
# # ==========================================
# print("\n2. 正在提取特征流数据 (用于双流网络的逻辑分支)...")
# X_segments = []
# for instance in tqdm(X_raw, desc="特征提取进度"):
#     segs = semantic_segmentation(instance, target_segments=30)
#     feats = extract_segment_features(instance, segs, max_segments=30)
#     X_segments.append(feats)
#
# X_segments = np.array(X_segments)
# print(f"✅ 双流数据构建完成！波形: {X_raw.shape}, 特征: {X_segments.shape}")
#
# # 同步划分训练集和测试集
# split = int(len(X_raw) * 0.85) # 预留 15% 做测试
# X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
# X_train_seg, X_test_seg = X_segments[:split], X_segments[split:]
# y_train, y_test = y[:split], y[split:]
#
# # ==========================================
# # 3. 训练增强版双流模型
# # ==========================================
# print("\n3. 正在构建【注意力机制驱动的多尺度融合网络】...")
# model = create_hybrid_model(
#     raw_shape=(3000, 3),
#     segment_shape=(30, 13),
#     num_classes=NUM_CLASSES
# )
#
# # 训练策略：早停 + 学习率衰减
# callbacks = [
#     EarlyStopping(
#         monitor='val_accuracy',
#         patience=5,
#         restore_best_weights=True,
#         verbose=1
#     ),
#
#
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         min_lr=1e-6,
#         verbose=1
#     )
# ]
#
# print("开始训练...")
# model.fit(
#     [X_train_raw, X_train_seg], y_train,
#     epochs=30, # 结构变复杂了，建议适当增加 Epochs
#     batch_size=32,
#     validation_split=0.15,
#     class_weight=class_weight_dict, # 应用类别权重
#     callbacks=callbacks
# )
#
# # ==========================================
# # 4. LIME 解释与可视化
# # ==========================================
# print("\n4. 正在对测试集样本进行 LIME 可解释性分析...")
# # 随机挑一个样本观察
# id_sample = np.random.randint(0, len(X_test_raw))
# instance_raw = X_test_raw[id_sample]
#
# segments = semantic_segmentation(instance_raw, target_segments=30)
#
# target_class, original_pred, importance = run_multimodal_lime(
#     model=model,
#     instance_multimodal=instance_raw,
#     segments=segments,
#     num_perturbations=300,
#     perturb_func=perturb_mean
# )
#
# print(f"🎯 模型预测结果: {CLASS_LABELS[target_class]} (置信度: {original_pred[target_class]:.2%})")
# print(f"实际标签: {CLASS_LABELS[np.argmax(y_test[id_sample])]}")
#
# visualize_lime_explanation(
#     instance_multimodal=instance_raw,
#     segment_importance=importance,
#     segments=segments,
#     predicted_class=target_class,
#     prob=original_pred[target_class],
#     class_labels=CLASS_LABELS
# )
