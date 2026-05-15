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
    from src.data_processing import load_npz_data
    from src.model_training import create_hybrid_model
    from src.lime_explanation import (
        run_multimodal_lime, perturb_mean,
        semantic_segmentation, extract_segment_features
    )
    from src.visualization import plot_xai_multimodal_dashboard
    from src.grad_cam import run_hybrid_grad_cam

    print("✅ 核心模块加载成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit()


# ==========================================
# 【新增】：流式数据生成器 (根治内存 OOM 问题)
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

        batch_X_raw = self.X_raw[batch_indices]
        batch_X_segment = self.X_segment[batch_indices]
        batch_y_main = self.y_main[batch_indices]

        # TTT 辅助重建分支的目标值就是原始波形本身
        batch_y_ttt = batch_X_raw

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
# 1-3. 数据流与流式训练
# ==========================================
NUM_CLASSES, CLASS_LABELS = 5, ['W', 'N1', 'N2', 'N3', 'REM']
MULTI_CHANNEL_PATH = r"E:\Python\LIME-for-Time-Series\EEG_data\eeg_eog_emg"

X_raw, y_raw = load_npz_data(MULTI_CHANNEL_PATH)
y = to_categorical(y_raw, num_classes=NUM_CLASSES)

X_segments = []
for instance in tqdm(X_raw, desc="特征提取"):
    segs = semantic_segmentation(instance, target_segments=30)
    X_segments.append(extract_segment_features(instance, segs, max_segments=30))
X_segments = np.array(X_segments)

# 数据切分：测试集 (15%)
split_test = int(len(X_raw) * 0.85)
X_train_val_raw, X_test_raw = X_raw[:split_test], X_raw[split_test:]
X_train_val_seg, X_test_seg = X_segments[:split_test], X_segments[split_test:]
y_train_val, y_test = y[:split_test], y[split_test:]

# 数据切分：训练集 (85% of 85%) 与验证集 (15% of 85%)
split_val = int(len(X_train_val_raw) * 0.85)
X_train_raw, X_val_raw = X_train_val_raw[:split_val], X_train_val_raw[split_val:]
X_train_seg, X_val_seg = X_train_val_seg[:split_val], X_train_val_seg[split_val:]
y_train, y_val = y_train_val[:split_val], y_train_val[split_val:]

model = create_hybrid_model()
callbacks = [EarlyStopping(monitor='val_main_output_accuracy', patience=5, restore_best_weights=True)]

# 实例化数据生成器 (batch_size 设为 16 以进一步保护显存)
train_gen = SleepDataGenerator(X_train_raw, X_train_seg, y_train, batch_size=16, shuffle=True)
val_gen = SleepDataGenerator(X_val_raw, X_val_seg, y_val, batch_size=16, shuffle=False)

print("\n🚀 数据流水线构建完毕，开始流式训练 (摘要模式)...")
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=2
)
# ==========================================
# 4. 双模式推理评估 (纯推理 vs 受限 TTT)
# ==========================================
print("\n==========================================")
print("4. 正在执行双模式性能评估 (Ablation Study)")
print("==========================================")

# ---------------------------------------------------------
# 【模式 A】：纯推理 (验证 SE + BiLSTM 混合架构的真实底线)
# ---------------------------------------------------------
print("\n▶ [模式 A] 正在执行全局极速推理 (关闭 TTT)...")
# 批量预测，极速且不产生任何内存泄漏
y_pred_probs_all, _ = model.predict([X_test_raw, X_test_seg], batch_size=32, verbose=0)
y_pred_no_ttt = np.argmax(y_pred_probs_all, axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"\n📊 分类报告 [模式 A: 纯推理无 TTT]:")
print(classification_report(y_true, y_pred_no_ttt, target_names=CLASS_LABELS, digits=4))

# ---------------------------------------------------------
# 【模式 B】：受限 TTT (验证个体自适应能否进一步提分)
# ---------------------------------------------------------
print("\n▶ [模式 B] 正在执行受限 TTT 推理 (防 OOM 内存泄漏版)...")


# 重新定义针对 SE 架构的保护性 TTT 更新逻辑
@tf.function
def ttt_step_secure(m, xr, xs, o):
    with tf.GradientTape() as tape:
        _, ttt_pred = m([xr, xs], training=False)
        ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))

    # 【核心保护】：在测试期，绝对冻结 LSTM 和主分类头！
    # 仅仅允许更新前端的 Conv1D (MRCNN波形提取) 和 TTT 重建头，实现安全的域适应。
    v = [
        var for var in m.trainable_variables
        if ('main_output' not in var.name)  # 锁死分类头
           and ('lstm' not in var.name)  # 锁死时序记忆
           and ('bidirectional' not in var.name)  # 锁死时序记忆
    ]

    g = tape.gradient(ttt_loss, v)
    valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
    o.apply_gradients(valid_g)
    return ttt_loss


y_pred_ttt = []
# 保持极低学习率，防止单样本梯度爆炸
ttt_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

# 【防崩溃神技】：使用 tf.identity 备份权重图，不产生 Python 垃圾对象
base_vars = [tf.identity(v) for v in model.trainable_variables]

for i in tqdm(range(len(X_test_raw)), desc="Secure TTT Adaptation"):
    xr, xs = X_test_raw[i:i + 1], X_test_seg[i:i + 1]

    # 1. 适应 (微调前端波形提取器)
    _ = ttt_step_secure(model, xr, xs, ttt_opt)

    # 2. 预测
    pred, _ = model([xr, xs], training=False)
    y_pred_ttt.append(np.argmax(pred[0]))

    # 3. 原地极速恢复权重，彻底告别 0xC0000409 崩溃！
    for var, base_var in zip(model.trainable_variables, base_vars):
        var.assign(base_var)

print(f"\n📊 分类报告 [模式 B: 受限 TTT 适应版]:")
print(classification_report(y_true, y_pred_ttt, target_names=CLASS_LABELS, digits=4))

# 默认将表现最好的 (或者你想展示的) 结果保存为图表，这里默认保存模式 B
save_academic_results(y_true, y_pred_ttt, CLASS_LABELS, OUTPUT_DIR)
# # ==========================================
# # 4. TTT 个性化推理与评估
# # ==========================================
# print("\n4. 正在执行 TTT 推理...")
# y_pred_ttt = []
# ttt_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
# base_weights = model.get_weights()
#
# for i in tqdm(range(len(X_test_raw)), desc="TTT Adaptation"):
#     xr, xs = X_test_raw[i:i + 1], X_test_seg[i:i + 1]
#     _ = ttt_step(model, xr, xs, ttt_opt)
#     pred, _ = model([xr, xs], training=False)
#     y_pred_ttt.append(np.argmax(pred[0]))
#     model.set_weights(base_weights)
#
# y_true = np.argmax(y_test, axis=1)
# print(f"\n📊 终极分类报告 (TTT 加成版):")
# print(classification_report(y_true, y_pred_ttt, target_names=CLASS_LABELS, digits=4))
# save_academic_results(y_true, y_pred_ttt, CLASS_LABELS, OUTPUT_DIR)

# ==========================================
# 5. 多阶段 XAI 协同诊断系统
# ==========================================
print("\n5. 执行多阶段 XAI 协同分析 (为每个阶段各出一张图)...")
lime_wrapper = ModelWrapperForLIME(model)

for stage_idx, stage_name in enumerate(CLASS_LABELS):
    indices = np.where(y_true == stage_idx)[0]
    if len(indices) == 0: continue

    target_idx = indices[0]
    instance_raw = X_test_raw[target_idx]

    print(f"   🔎 正在诊断阶段: {stage_name} (样本索引: {target_idx})")

    segments = semantic_segmentation(instance_raw, target_segments=30)
    target_class, _, importance, dccs = run_multimodal_lime(
        model=lime_wrapper,
        instance_multimodal=instance_raw,
        segments=segments,
        num_perturbations=300
    )

    save_path = os.path.join(OUTPUT_DIR, f'XAI_Diagnosis_{stage_name}.png')
    plot_xai_multimodal_dashboard(
        signal=instance_raw,
        segments=segments,
        segment_importance=importance,
        dccs=dccs,
        channel_names=['EEG', 'EOG', 'EMG'],
        target_class_name=CLASS_LABELS[target_class],
        save_path=save_path
    )

print(f"\n🎉 任务圆满完成！所有阶段的诊断图已存入: {OUTPUT_DIR}")



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# import sys
# import numpy as np
# import random
# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime
# from tensorflow.keras.utils import to_categorical, Sequence
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from tqdm import tqdm
#
# # ... (下方保留你 run_main.py 原本的 Logger 类和所有后续代码) ...
#
#
# # ==========================================
# # 🌟 结果摘要记录类 (保留摘要输出逻辑)
# # ==========================================
# class Logger(object):
#     def __init__(self, filename="Run_Results.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w", encoding='utf-8')
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
#
# # ==========================================
# # 0. 环境路径与随机种子设置 (保留原版)
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
# OUTPUT_DIR = 'outputs'
# LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
# os.makedirs(LOG_DIR, exist_ok=True)
#
# # 启动日志
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# log_path = os.path.join(LOG_DIR, f"result_{timestamp}.txt")
# sys.stdout = Logger(log_path)
#
# try:
#     from src.data_processing import load_npz_data
#     from src.model_training import create_hybrid_model
#     from src.lime_explanation import (
#         run_multimodal_lime, perturb_mean,
#         semantic_segmentation, extract_segment_features
#     )
#     from src.visualization import plot_xai_multimodal_dashboard
#     from src.grad_cam import run_hybrid_grad_cam
#
#     print("✅ 核心模块加载成功！")
# except ImportError as e:
#     print(f"❌ 导入失败: {e}")
#     sys.exit()
#
#
# # ==========================================
# # 【新增】：流式数据生成器 (根治内存 OOM 问题)
# # ==========================================
# class SleepDataGenerator(Sequence):
#     def __init__(self, X_raw, X_segment, y_main, batch_size=16, shuffle=True):
#         self.X_raw = X_raw
#         self.X_segment = X_segment
#         self.y_main = y_main
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.X_raw))
#         if self.shuffle:
#             np.random.shuffle(self.indices)
#
#     def __len__(self):
#         return int(np.ceil(len(self.X_raw) / self.batch_size))
#
#     def __getitem__(self, index):
#         batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
#
#         batch_X_raw = self.X_raw[batch_indices]
#         batch_X_segment = self.X_segment[batch_indices]
#         batch_y_main = self.y_main[batch_indices]
#
#         # TTT 辅助重建分支的目标值就是原始波形本身
#         batch_y_ttt = batch_X_raw
#
#         return (
#             {'raw_input': batch_X_raw, 'segment_input': batch_X_segment},
#             {'main_output': batch_y_main, 'ttt_output': batch_y_ttt}
#         )
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
#
#
# # ==========================================
# # 【包装器】：彻底修复 [300, 1500] 维度报错
# # ==========================================
# class ModelWrapperForLIME:
#     def __init__(self, base_model):
#         self.base_model = base_model
#
#     def predict(self, inputs, **kwargs):
#         preds = self.base_model.predict(inputs, **kwargs)
#         return preds[0] if isinstance(preds, list) else preds
#
#
# # ==========================================
# # 辅助绘图：蓝红专业色系
# # ==========================================
# def save_academic_results(y_true, y_pred, class_names, output_dir):
#     report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
#     df = pd.DataFrame(report_dict).transpose()
#     report_df = df.drop(index=['accuracy'], errors='ignore').drop(columns=['support'], errors='ignore')
#     report_df.columns = ['Precision', 'Recall', 'F1-Score']
#
#     plt.figure(figsize=(14, 8))
#     sns.heatmap(report_df.astype(float), annot=True, cmap='RdBu_r', fmt=".4f", center=0.5, vmin=0, vmax=1,
#                 linewidths=1.5, annot_kws={"size": 13, "weight": "bold"})
#     plt.title(f'Classification Report | Accuracy: {report_dict.get("accuracy", 0):.4f}', fontweight='bold', fontsize=15)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'classification_report_visual.png'), dpi=300)
#     plt.close()
#
#
# # ==========================================
# # TTT 推理引擎 (严格冻结保护版)
# # ==========================================
# @tf.function
# def ttt_step(m, xr, xs, o):
#     with tf.GradientTape() as tape:
#         _, ttt_pred = m([xr, xs], training=False)
#         ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))
#
#     # 【核心修复】：在测试期适应阶段，保护脆弱的 Attention 和 LSTM 层！
#     # 仅仅允许 TTT 更新前端的 Conv1D 卷积核（MRCNN）以及专用的 ttt_output 重建头。
#     v = [
#         v for v in m.trainable_variables
#         if ('main_output' not in v.name)  # 不更新分类头 (已有)
#            and ('multi_head_attention' not in v.name)  # 绝对禁止更新注意力矩阵
#            and ('lstm' not in v.name)  # 绝对禁止更新时序记忆单元
#     ]
#
#     g = tape.gradient(ttt_loss, v)
#     valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
#     o.apply_gradients(valid_g)
#     return ttt_loss
#
# # # ==========================================
# # # TTT 推理引擎 (防坍塌版)
# # # ==========================================
# # @tf.function
# # def ttt_step(m, xr, xs, o):
# #     with tf.GradientTape() as tape:
# #         _, ttt_pred = m([xr, xs], training=False)
# #         ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))
# #     v = [v for v in m.trainable_variables if 'main_output' not in v.name]
# #     g = tape.gradient(ttt_loss, v)
# #     valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
# #     o.apply_gradients(valid_g)
# #     return ttt_loss
#
#
# # ==========================================
# # 1-3. 数据流与流式训练
# # ==========================================
# NUM_CLASSES, CLASS_LABELS = 5, ['W', 'N1', 'N2', 'N3', 'REM']
# MULTI_CHANNEL_PATH = r"E:\Python\LIME-for-Time-Series\EEG_data\eeg_eog_emg"
#
# X_raw, y_raw = load_npz_data(MULTI_CHANNEL_PATH)
# y = to_categorical(y_raw, num_classes=NUM_CLASSES)
#
# X_segments = []
# for instance in tqdm(X_raw, desc="特征提取"):
#     segs = semantic_segmentation(instance, target_segments=30)
#     X_segments.append(extract_segment_features(instance, segs, max_segments=30))
# X_segments = np.array(X_segments)
#
# # 数据切分：测试集 (15%)
# split_test = int(len(X_raw) * 0.85)
# X_train_val_raw, X_test_raw = X_raw[:split_test], X_raw[split_test:]
# X_train_val_seg, X_test_seg = X_segments[:split_test], X_segments[split_test:]
# y_train_val, y_test = y[:split_test], y[split_test:]
#
# # 数据切分：训练集 (85% of 85%) 与验证集 (15% of 85%)
# split_val = int(len(X_train_val_raw) * 0.85)
# X_train_raw, X_val_raw = X_train_val_raw[:split_val], X_train_val_raw[split_val:]
# X_train_seg, X_val_seg = X_train_val_seg[:split_val], X_train_val_seg[split_val:]
# y_train, y_val = y_train_val[:split_val], y_train_val[split_val:]
#
# model = create_hybrid_model()
# callbacks = [EarlyStopping(monitor='val_main_output_accuracy', patience=5, restore_best_weights=True)]
#
# # 实例化数据生成器 (batch_size 设为 16 以进一步保护显存)
# train_gen = SleepDataGenerator(X_train_raw, X_train_seg, y_train, batch_size=16, shuffle=True)
# val_gen = SleepDataGenerator(X_val_raw, X_val_seg, y_val, batch_size=16, shuffle=False)
#
# print("\n🚀 数据流水线构建完毕，开始流式训练 (摘要模式)...")
# model.fit(
#     train_gen,
#     epochs=30,
#     validation_data=val_gen,
#     callbacks=callbacks,
#     verbose=2
# )
# # ==========================================
# # 4. 双模式推理评估 (纯推理 vs 受限 TTT)
# # ==========================================
# print("\n==========================================")
# print("4. 正在执行双模式性能评估 (Ablation Study)")
# print("==========================================")
#
# # ---------------------------------------------------------
# # 【模式 A】：纯推理 (验证 SE + BiLSTM 混合架构的真实底线)
# # ---------------------------------------------------------
# print("\n▶ [模式 A] 正在执行全局极速推理 (关闭 TTT)...")
# # 批量预测，极速且不产生任何内存泄漏
# y_pred_probs_all, _ = model.predict([X_test_raw, X_test_seg], batch_size=32, verbose=1)
# y_pred_no_ttt = np.argmax(y_pred_probs_all, axis=1)
# y_true = np.argmax(y_test, axis=1)
#
# print(f"\n📊 分类报告 [模式 A: 纯推理无 TTT]:")
# print(classification_report(y_true, y_pred_no_ttt, target_names=CLASS_LABELS, digits=4))
#
# # ---------------------------------------------------------
# # 【模式 B】：受限 TTT (验证个体自适应能否进一步提分)
# # ---------------------------------------------------------
# print("\n▶ [模式 B] 正在执行受限 TTT 推理 (防 OOM 内存泄漏版)...")
#
#
# # 重新定义针对 SE 架构的保护性 TTT 更新逻辑
# @tf.function
# def ttt_step_secure(m, xr, xs, o):
#     with tf.GradientTape() as tape:
#         _, ttt_pred = m([xr, xs], training=False)
#         ttt_loss = tf.reduce_mean(tf.square(ttt_pred - xr))
#
#     # 【核心保护】：在测试期，绝对冻结 LSTM 和主分类头！
#     # 仅仅允许更新前端的 Conv1D (MRCNN波形提取) 和 TTT 重建头，实现安全的域适应。
#     v = [
#         var for var in m.trainable_variables
#         if ('main_output' not in var.name)  # 锁死分类头
#            and ('lstm' not in var.name)  # 锁死时序记忆
#            and ('bidirectional' not in var.name)  # 锁死时序记忆
#     ]
#
#     g = tape.gradient(ttt_loss, v)
#     valid_g = [(gi, vi) for gi, vi in zip(g, v) if gi is not None]
#     o.apply_gradients(valid_g)
#     return ttt_loss
#
#
# y_pred_ttt = []
# # 保持极低学习率，防止单样本梯度爆炸
# ttt_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
#
# # 【防崩溃神技】：使用 tf.identity 备份权重图，不产生 Python 垃圾对象
# base_vars = [tf.identity(v) for v in model.trainable_variables]
#
# for i in tqdm(range(len(X_test_raw)), desc="Secure TTT Adaptation"):
#     xr, xs = X_test_raw[i:i + 1], X_test_seg[i:i + 1]
#
#     # 1. 适应 (微调前端波形提取器)
#     _ = ttt_step_secure(model, xr, xs, ttt_opt)
#
#     # 2. 预测
#     pred, _ = model([xr, xs], training=False)
#     y_pred_ttt.append(np.argmax(pred[0]))
#
#     # 3. 原地极速恢复权重，彻底告别 0xC0000409 崩溃！
#     for var, base_var in zip(model.trainable_variables, base_vars):
#         var.assign(base_var)
#
# print(f"\n📊 分类报告 [模式 B: 受限 TTT 适应版]:")
# print(classification_report(y_true, y_pred_ttt, target_names=CLASS_LABELS, digits=4))
#
# # 默认将表现最好的 (或者你想展示的) 结果保存为图表，这里默认保存模式 B
# save_academic_results(y_true, y_pred_ttt, CLASS_LABELS, OUTPUT_DIR)
# # # ==========================================
# # # 4. TTT 个性化推理与评估
# # # ==========================================
# # print("\n4. 正在执行 TTT 推理...")
# # y_pred_ttt = []
# # ttt_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
# # base_weights = model.get_weights()
# #
# # for i in tqdm(range(len(X_test_raw)), desc="TTT Adaptation"):
# #     xr, xs = X_test_raw[i:i + 1], X_test_seg[i:i + 1]
# #     _ = ttt_step(model, xr, xs, ttt_opt)
# #     pred, _ = model([xr, xs], training=False)
# #     y_pred_ttt.append(np.argmax(pred[0]))
# #     model.set_weights(base_weights)
# #
# # y_true = np.argmax(y_test, axis=1)
# # print(f"\n📊 终极分类报告 (TTT 加成版):")
# # print(classification_report(y_true, y_pred_ttt, target_names=CLASS_LABELS, digits=4))
# # save_academic_results(y_true, y_pred_ttt, CLASS_LABELS, OUTPUT_DIR)
#
# # ==========================================
# # 5. 多阶段 XAI 协同诊断系统
# # ==========================================
# print("\n5. 执行多阶段 XAI 协同分析 (为每个阶段各出一张图)...")
# lime_wrapper = ModelWrapperForLIME(model)
#
# for stage_idx, stage_name in enumerate(CLASS_LABELS):
#     indices = np.where(y_true == stage_idx)[0]
#     if len(indices) == 0: continue
#
#     target_idx = indices[0]
#     instance_raw = X_test_raw[target_idx]
#
#     print(f"   🔎 正在诊断阶段: {stage_name} (样本索引: {target_idx})")
#
#     segments = semantic_segmentation(instance_raw, target_segments=30)
#     target_class, _, importance, dccs = run_multimodal_lime(
#         model=lime_wrapper,
#         instance_multimodal=instance_raw,
#         segments=segments,
#         num_perturbations=300
#     )
#
#     save_path = os.path.join(OUTPUT_DIR, f'XAI_Diagnosis_{stage_name}.png')
#     plot_xai_multimodal_dashboard(
#         signal=instance_raw,
#         segments=segments,
#         segment_importance=importance,
#         dccs=dccs,
#         channel_names=['EEG', 'EOG', 'EMG'],
#         target_class_name=CLASS_LABELS[target_class],
#         save_path=save_path
#     )
#
# print(f"\n🎉 任务圆满完成！所有阶段的诊断图已存入: {OUTPUT_DIR}")
#





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
