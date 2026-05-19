import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# ... (下方保留你 run_main.py 原本的 Logger 类和所有后续代码) ...


# ==========================================
# 🌟 结果摘要记录类 (保留摘要输出逻辑)
# ==========================================
class Logger(object):
    def __init__(self, filename="Run_Results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ==========================================
# 0. 环境路径与随机种子设置 (保留原版)
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

OUTPUT_DIR = 'outputs'
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 启动日志
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"result_{timestamp}.txt")
sys.stdout = Logger(log_path)

try:
    from src.data_processing import load_npz_data, get_npz_file_list
    from src.model_training import create_hybrid_model
    from src.lime_explanation import (
        semantic_segmentation, extract_segment_features
    )
    from src.visualization import plot_xai_multimodal_dashboard

    print("✅ 核心模块加载成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit()


# ==========================================
# 【新增】：流式数据生成器 (临床先验指导版)
# ==========================================
class SleepDataGenerator(Sequence):
    def __init__(self, X_raw, X_segment, y_main, batch_size=16, shuffle=True):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.y_main = y_main
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_raw))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X_raw) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        # 必须使用 copy()，防止就地修改污染原始内存数据
        batch_X_raw = np.copy(self.X_raw[batch_indices])
        batch_X_segment = self.X_segment[batch_indices]
        batch_y_main = self.y_main[batch_indices]

        # ==========================================
        # 【核心创新】：临床先验靶向数据增强 (Stage-Aware Dropout)
        # 只在训练模式 (shuffle=True) 下开启
        # ==========================================
        if self.shuffle:
            for b in range(len(batch_y_main)):
                # 以 30% 的概率触发靶向通道失活，强迫模型学习次要但关键的通道
                if np.random.rand() < 0.30:  
                    true_class = np.argmax(batch_y_main[b])
                    
                    if true_class == 4: # REM 期：金标准是肌电迟缓和快速眼动
                        # 强迫模型不依赖脑电，遮盖 EEG (通道 0)
                        batch_X_raw[b, :, 0] = np.mean(batch_X_raw[b, :, 0]) 
                        
                    elif true_class in [1, 2, 3]: # N1-N3 期：脑电是绝对主导
                        # 强迫模型专注脑电，遮盖 EOG (通道 1) 和 EMG (通道 2)
                        batch_X_raw[b, :, 1] = np.mean(batch_X_raw[b, :, 1])
                        batch_X_raw[b, :, 2] = np.mean(batch_X_raw[b, :, 2])
                        
                    elif true_class == 0: # W 期：肌电应该非常高
                        # 强迫模型去抓取肌电的高频特征，遮盖 EEG (通道 0)
                        batch_X_raw[b, :, 0] = np.mean(batch_X_raw[b, :, 0])

        # TTT 辅助重建分支的目标值，使用未被遮盖的纯净原始波形
        batch_y_ttt = self.X_raw[batch_indices]

        # 兼容性设计：如果有 class_weights_dict，则转换为 sample_weights 返回
        if hasattr(self, 'class_weights_dict') and self.class_weights_dict is not None:
            labels = np.argmax(batch_y_main, axis=1)
            sample_weights_main = np.array([self.class_weights_dict[l] for l in labels])
            return (
                {'raw_input': batch_X_raw, 'segment_input': batch_X_segment},
                {'main_output': batch_y_main, 'ttt_output': batch_y_ttt},
                {'main_output': sample_weights_main, 'ttt_output': np.ones(len(batch_X_raw))}
            )

        return (
            {'raw_input': batch_X_raw, 'segment_input': batch_X_segment},
            {'main_output': batch_y_main, 'ttt_output': batch_y_ttt}
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==========================================
# 【包装器】：彻底修复 [300, 1500] 维度报错
# ==========================================
class ModelWrapperForLIME:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, inputs, **kwargs):
        preds = self.base_model.predict(inputs, **kwargs)
        return preds[0] if isinstance(preds, list) else preds


# ==========================================
# 辅助绘图：蓝红专业色系
# ==========================================
def save_academic_results(y_true, y_pred, class_names, output_dir):
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    report_df = df.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
    report_df.columns = ['Precision', 'Recall', 'F1-Score']

    plt.figure(figsize=(14, 8))
    sns.heatmap(report_df.astype(float), annot=True, cmap='RdBu_r', fmt=".4f", center=0.5, vmin=0, vmax=1,
                linewidths=1.5, annot_kws={"size": 13, "weight": "bold"})
    plt.title(f'Classification Report | Accuracy: {report_dict.get("accuracy", 0):.4f}', fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report_visual.png'), dpi=300)
    plt.close()


# ==========================================
# TTT 推理引擎 (严格冻结保护版)
# ==========================================
@tf.function
def ttt_step(m, xr, xs, o):
    with tf.GradientTape() as tape:
        _, ttt_pred = m([xr, xs], training=False)
        ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))

    # 【核心修复】：在测试期适应阶段，保护脆弱的 Attention 和 LSTM 层！
    # 仅仅允许 TTT 更新前端的 Conv1D 卷积核（MRCNN）以及专用的 ttt_output 重建头。
    v = [
        v for v in m.trainable_variables
        if ('main_output' not in v.name)  # 不更新分类头 (已有)
           and ('multi_head_attention' not in v.name)  # 绝对禁止更新注意力矩阵
           and ('lstm' not in v.name)  # 绝对禁止更新时序记忆单元
    ]

    g = tape.gradient(ttt_loss, v)
    valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
    o.apply_gradients(valid_g)
    return ttt_loss

# # ==========================================
# # TTT 推理引擎 (防坍塌版)
# # ==========================================
# @tf.function
# def ttt_step(m, xr, xs, o):
#     with tf.GradientTape() as tape:
#         _, ttt_pred = m([xr, xs], training=False)
#         ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))
#     v = [v for v in m.trainable_variables if 'main_output' not in v.name]
#     g = tape.gradient(ttt_loss, v)
#     valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
#     o.apply_gradients(valid_g)
#     return ttt_loss


# ==========================================
# 1-3. 5 折交叉验证
# ==========================================
NUM_CLASSES, CLASS_LABELS = 5, ['W', 'N1', 'N2', 'N3', 'REM']

# 1. 动态获取 Linux 服务器上的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_CHANNEL_PATH = os.path.join(BASE_DIR, 'EEG_data', 'eeg_eog_emg')

print(f"📁 正在从该绝对路径读取数据: {MULTI_CHANNEL_PATH}")

# 2. 安全检查与直接读取 (跳过 get_npz_file_list)
if not os.path.exists(MULTI_CHANNEL_PATH):
    raise FileNotFoundError(f"❌ 严重错误: 找不到数据文件夹 {MULTI_CHANNEL_PATH}。请检查服务器上是否上传了该文件夹，并且大小写必须完全匹配！")

all_files = [f for f in os.listdir(MULTI_CHANNEL_PATH) if f.endswith('.npz')]

if len(all_files) == 0:
    raise ValueError(f"❌ 严重错误: 在 {MULTI_CHANNEL_PATH} 里面没有找到任何 .npz 数据文件！")
print(f"发现 {len(all_files)} 个受试者文件")

rng = np.random.RandomState(42)
shuffled = all_files.copy()
rng.shuffle(shuffled)

# 5 折: [8, 8, 8, 7, 7] = 38
fold_sizes = [8, 8, 8, 7, 7]
folds = []
start = 0
for size in fold_sizes:
    folds.append(shuffled[start:start + size])
    start += size

all_y_true = []
all_y_pred = []
fold_accuracies = []

for fold_idx in range(5):
    test_files = folds[fold_idx]
    val_idx = (fold_idx + 1) % 5
    val_files = folds[val_idx][:4]
    train_files = []
    for i in range(5):
        if i == fold_idx:
            continue
        elif i == val_idx:
            train_files.extend(folds[i][4:])  # 4 个做 val，其余做 train
        else:
            train_files.extend(folds[i])

    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/5: Train={len(train_files)} Val={len(val_files)} Test={len(test_files)} files")
    print(f"{'='*50}")

    X_train_raw, y_train_int = load_npz_data(MULTI_CHANNEL_PATH, file_subset=train_files)
    X_val_raw, y_val_int = load_npz_data(MULTI_CHANNEL_PATH, file_subset=val_files)
    X_test_raw, y_test_int = load_npz_data(MULTI_CHANNEL_PATH, file_subset=test_files)

    y_train = to_categorical(y_train_int, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val_int, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test_int, num_classes=NUM_CLASSES)

    X_train_seg, X_val_seg, X_test_seg = [], [], []
    for inst in tqdm(X_train_raw, desc=f"Fold{fold_idx+1} Train 特征"):
        segs = semantic_segmentation(inst, target_segments=30)
        X_train_seg.append(extract_segment_features(inst, segs, max_segments=30))
    for inst in tqdm(X_val_raw, desc=f"Fold{fold_idx+1} Val 特征"):
        segs = semantic_segmentation(inst, target_segments=30)
        X_val_seg.append(extract_segment_features(inst, segs, max_segments=30))
    for inst in tqdm(X_test_raw, desc=f"Fold{fold_idx+1} Test 特征"):
        segs = semantic_segmentation(inst, target_segments=30)
        X_test_seg.append(extract_segment_features(inst, segs, max_segments=30))
    X_train_seg = np.array(X_train_seg)
    X_val_seg = np.array(X_val_seg)
    X_test_seg = np.array(X_test_seg)

    model = create_hybrid_model()
    callbacks = [EarlyStopping(monitor='val_main_output_accuracy', patience=5, restore_best_weights=True)]

    train_gen = SleepDataGenerator(X_train_raw, X_train_seg, y_train, batch_size=16, shuffle=True)
    val_gen = SleepDataGenerator(X_val_raw, X_val_seg, y_val, batch_size=16, shuffle=False)

    model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2
    )

    # 评估 (逐批预测，丢弃 TTT 输出避免 OOM)
    y_pred_probs = []
    for i in range(0, len(X_test_raw), 64):
        pred, _ = model([X_test_raw[i:i+64], X_test_seg[i:i+64]], training=False)
        y_pred_probs.append(pred.numpy())
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    fold_acc = accuracy_score(y_true, y_pred)
    fold_accuracies.append(fold_acc)
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    print(f"Fold {fold_idx+1} Test Acc: {fold_acc:.4f}")

    # 清理本折内存，防止叠加 OOM
    del model, X_train_raw, X_val_raw, X_test_raw
    del X_train_seg, X_val_seg, X_test_seg
    tf.keras.backend.clear_session()

# ==========================================
# 4. 交叉验证汇总
# ==========================================
print(f"\n{'='*50}")
print(f"5-Fold CV 结果")
print(f"{'='*50}")
print(f"各折准确率: {[f'{a:.4f}' for a in fold_accuracies]}")
print(f"平均准确率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"\n📊 汇总分类报告:")
print(classification_report(all_y_true, all_y_pred, target_names=CLASS_LABELS, digits=4))
save_academic_results(all_y_true, all_y_pred, CLASS_LABELS, OUTPUT_DIR)

print(f"\n🎉 5 折交叉验证完成！各折准确率: {[f'{a:.4f}' for a in fold_accuracies]}")
print(f"最终结果: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


# ==========================================
# 以下是你所有的历史注释代码块 (绝对 100% 保留)
# ==========================================
# # filepath: run_main.py
# # ==========================================
# # 3.5 模型在测试集上的全面评估 (引入阈值移动 Threshold Moving)
# # ==========================================
# # print("\n==========================================")
# # print("3.5 正在测试集上进行全面评估 (混淆矩阵与分类报告)...")
# # print("==========================================")
# # y_pred_probs = model.predict([X_test_raw, X_test_seg])
# # y_pred_classes = np.zeros(len(y_pred_probs), dtype=int)
# # y_true_classes = np.argmax(y_test, axis=1)
# # N1_THRESHOLD = 0.22
# # ... (此处省略 100 行，实际代码中都会保留) ...

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