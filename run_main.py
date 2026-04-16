import sys
import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd  # 新增：用于处理报告表格数据
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# ==========================================
# 0. 环境路径与随机种子设置 (保留原版 [cite: 20])
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
    from src.visualization import plot_xai_multimodal_dashboard
    from src.grad_cam import run_hybrid_grad_cam

    print("✅ 核心模块及增强版模型加载成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit()


# ==========================================
# 辅助绘图函数 (新增：分类报告图片化 [cite: 22])
# ==========================================
def save_confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Sleep Stage Confusion Matrix', fontweight='bold', fontsize=14)
    plt.ylabel('True Stage', fontweight='bold')
    plt.xlabel('Predicted Stage', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report_visual(y_true, y_pred, class_names, output_path):
    """将分类报告转换为带有热力颜色映射的表格图片"""
    # 获取字典格式的报告
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 转换为 DataFrame 并剔除辅助行，只保留核心指标
    # 我们过滤掉 'accuracy' 因为它的结构与 precision/recall 不同
    report_df = pd.DataFrame(report_dict).iloc[:-1, :].T

    plt.figure(figsize=(12, 7))
    # 使用 RdYlGn (红-黄-绿) 色偏，指标越高颜色越绿
    sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt=".4f", cbar=True, annot_kws={"size": 12})
    plt.title('Classification Report: Precision, Recall, F1-Score', fontweight='bold', fontsize=14)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_grad_cam_heatmap(signal, heatmap, channel_names, target_class_name, save_path):
    num_channels = signal.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)
    if num_channels == 1: axes = [axes]
    heatmap_2d = heatmap[np.newaxis, :]
    time_axis = np.arange(signal.shape[0])
    for c in range(num_channels):
        ax = axes[c]
        ax.plot(time_axis, signal[:, c], color='black', linewidth=0.8, alpha=0.9)
        ymin, ymax = ax.get_ylim()
        im = ax.imshow(heatmap_2d, aspect='auto', cmap='jet', alpha=0.4,
                       extent=[0, signal.shape[0], ymin, ymax], origin='lower')
        ax.set_ylabel(f'{channel_names[c]}', fontweight='bold')
    axes[0].set_title(f'Grad-CAM: {target_class_name}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 1. 全局配置与数据加载 (保留原版 [cite: 20])
# ==========================================
NUM_CLASSES = 5
CLASS_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']
CHANNEL_NAMES = ['EEG', 'EOG', 'EMG']
MULTI_CHANNEL_PATH = r"E:\Python\LIME-for-Time-Series\EEG_data\eeg_eog_emg"
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n1. 正在加载多通道 (EEG/EOG/EMG) 原始数据...")
X_raw, y_raw = load_npz_data(MULTI_CHANNEL_PATH)
y = to_categorical(y_raw, num_classes=NUM_CLASSES)

# ==========================================
# 2. 提取生理基序 (Motifs) 特征 (保留原版 [cite: 20])
# ==========================================
print("\n2. 正在提取特征流数据 (用于双流网络的逻辑分支)...")
X_segments = []
for instance in tqdm(X_raw, desc="特征提取进度"):
    segs = semantic_segmentation(instance, target_segments=30)
    feats = extract_segment_features(instance, segs, max_segments=30)
    X_segments.append(feats)

X_segments = np.array(X_segments)
split = int(len(X_raw) * 0.85)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
X_train_seg, X_test_seg = X_segments[:split], X_segments[split:]
y_train, y_test = y[:split], y[split:]

# ==========================================
# 3. 训练增强版双流模型 (保留 Focal Loss 架构 [cite: 20])
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
# 4. 全面评估与双视觉报告出图 (升级版 )
# ==========================================
print("\n4. 正在测试集上进行全面评估与多维报告生成...")
y_pred_probs = model.predict([X_test_raw, X_test_seg])
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 终端输出 (保留你的习惯)
print("\n📊 分类报告 (Classification Report):")
print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_LABELS, digits=4))

# 4.1 生成混淆矩阵图片
cm_img_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
save_confusion_matrix_plot(y_true_classes, y_pred_classes, CLASS_LABELS, cm_img_path)

# 4.2 生成分类报告热力图图片 (新增需求)
report_img_path = os.path.join(OUTPUT_DIR, 'classification_report_visual.png')
save_classification_report_visual(y_true_classes, y_pred_classes, CLASS_LABELS, report_img_path)

print(f"✅ 评估视觉件已生成: \n   - 混淆矩阵: {cm_img_path}\n   - 分类报告图: {report_img_path}")

# ==========================================
# 5. DCCS 协同解释与对比 (核心需求升级 [cite: 21, 27])
# ==========================================
print("\n5. 执行多通道 XAI 协同分析 (DCCS 机制)...")
id_sample = np.random.randint(0, len(X_test_raw))
instance_raw = X_test_raw[id_sample]
instance_seg = X_test_seg[id_sample]

segments = semantic_segmentation(instance_raw, target_segments=30)
target_class, original_pred, importance, dccs = run_multimodal_lime(
    model=model, instance_multimodal=instance_raw, segments=segments, num_perturbations=300
)

lime_path = os.path.join(OUTPUT_DIR, f'LIME_DCCS_Dashboard.png')
plot_xai_multimodal_dashboard(
    signal=instance_raw, segments=segments, segment_importance=importance,
    dccs=dccs, channel_names=CHANNEL_NAMES, target_class_name=CLASS_LABELS[target_class],
    save_path=lime_path
)

try:
    heatmap, g_class, _ = run_hybrid_grad_cam(model, instance_raw, instance_seg)
    gcam_path = os.path.join(OUTPUT_DIR, f'GradCAM_Comparison.png')
    plot_grad_cam_heatmap(instance_raw, heatmap, CHANNEL_NAMES, CLASS_LABELS[g_class], gcam_path)
    print(f"✅ XAI 对比图 (LIME/Grad-CAM) 已保存至 {OUTPUT_DIR}")
except Exception as e:
    print(f"⚠️ Grad-CAM 运行跳过: {e}")

print("\n🎉 任务圆满完成！所有指标图表已在 outputs 文件夹待命。")


# # filepath: run_main.py
# import sys
# import os
# import numpy as np
# import random
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import confusion_matrix, classification_report
# from tqdm import tqdm
# import matplotlib.pyplot as plt
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
#     from src.model_training import create_hybrid_model
#     from src.lime_explanation import (
#         run_multimodal_lime, perturb_mean,
#         semantic_segmentation, extract_segment_features
#     )
#     from src.visualization import visualize_lime_explanation, visualize_gradcam_explanation
#     from src.grad_cam import run_hybrid_grad_cam  # 导入我们新建的模块
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
# split = int(len(X_raw) * 0.85)
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
# callbacks = [
#     EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
# ]
#
# print("🚀 开始训练...")
# model.fit(
#     [X_train_raw, X_train_seg], y_train,
#     epochs=30,
#     batch_size=32,
#     validation_split=0.15,
#     callbacks=callbacks
# )
#
# # ==========================================
# # 3.5 模型在测试集上的全面评估 (引入阈值移动 Threshold Moving)
# # ==========================================
# print("\n==========================================")
# print("3.5 正在测试集上进行全面评估 (混淆矩阵与分类报告)...")
# print("==========================================")
#
# y_pred_probs = model.predict([X_test_raw, X_test_seg])
# y_pred_classes = np.zeros(len(y_pred_probs), dtype=int)
# y_true_classes = np.argmax(y_test, axis=1)
#
# N1_THRESHOLD = 0.22
#
# for i in range(len(y_pred_probs)):
#     probs = y_pred_probs[i]
#     if probs[1] >= N1_THRESHOLD:
#         y_pred_classes[i] = 1
#     else:
#         probs_copy = np.copy(probs)
#         probs_copy[1] = -1.0
#         y_pred_classes[i] = np.argmax(probs_copy)
#
# print(f"\n📊 分类报告 (应用 N1 专属阈值: {N1_THRESHOLD}):")
# print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_LABELS, digits=4))
#
# # ==========================================
# # 4. 可解释性对比分析：LIME vs Grad-CAM
# # ==========================================
# print("\n==========================================")
# print("4. 正在对同一测试集样本进行 LIME 与 Grad-CAM 对比分析...")
# print("==========================================")
#
# # 随机挑一个样本观察 (确保 LIME 和 Grad-CAM 用的是同一个数据)
# id_sample = np.random.randint(0, len(X_test_raw))
# instance_raw = X_test_raw[id_sample]
# instance_seg = X_test_seg[id_sample] # 对应的特征流数据
# actual_label = CLASS_LABELS[np.argmax(y_test[id_sample])]
#
# print(f"📌 选定样本 ID: {id_sample} | 真实标签: {actual_label}")
#
# # ----------------- A. 运行 LIME -----------------
# print("\n[A] 启动 LIME 解释器 (分析片段特征重要性)...")
# segments = semantic_segmentation(instance_raw, target_segments=30)
# lime_class, lime_pred_probs, lime_importance = run_multimodal_lime(
#     model=model,
#     instance_multimodal=instance_raw,
#     segments=segments,
#     num_perturbations=300,
#     perturb_func=perturb_mean
# )
#
# print(f"🎯 LIME 侧预测结果: {CLASS_LABELS[lime_class]} (置信度: {lime_pred_probs[lime_class]:.2%})")
# visualize_lime_explanation(
#     instance_multimodal=instance_raw,
#     segment_importance=lime_importance,
#     segments=segments,
#     predicted_class=lime_class,
#     prob=lime_pred_probs[lime_class],
#     class_labels=CLASS_LABELS
# )
# plt.show()
# # ----------------- B. 运行 Grad-CAM -----------------
# print("\n[B] 启动 Grad-CAM 解释器 (分析原始波形空间显著性)...")
# heatmap_resized, gradcam_class, gradcam_prob = run_hybrid_grad_cam(
#     model=model,
#     instance_raw=instance_raw,
#     instance_seg=instance_seg
# )
#
# print(f"🎯 Grad-CAM 侧预测结果: {CLASS_LABELS[gradcam_class]} (置信度: {gradcam_prob:.2%})")
# visualize_gradcam_explanation(
#     instance_raw=instance_raw,
#     heatmap=heatmap_resized,
#     predicted_class=gradcam_class,
#     prob=gradcam_prob,
#     class_labels=CLASS_LABELS
# )
# plt.show()
# print("\n🎉 分析完成！你可以对比两次弹出的图像，LIME 解释了'哪个片段最重要'，而 Grad-CAM 解释了'模型在盯着哪个波峰/波谷'。")