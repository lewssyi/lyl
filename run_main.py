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


# ==========================================
# 🌟 结果摘要记录类
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
# 0. 环境路径与随机种子设置
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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"result_{timestamp}.txt")
sys.stdout = Logger(log_path)

try:
    from src.data_processing import load_npz_data
    # 🌟 导入我们最成功的 V2 架构 (BiLSTM + 先验路由)
    from src.model_training import create_hybrid_model_v2 
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
# 流式数据生成器
# ==========================================
class SleepDataGenerator(Sequence):
    def __init__(self, X_raw, X_segment, y_main, batch_size=32, shuffle=True):
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

        # 辅助目标
        batch_y_ttt = batch_X_raw
        batch_y_att = batch_y_main 

        return (
            {'raw_input': batch_X_raw, 'segment_input': batch_X_segment},
            {
                'main_output': batch_y_main, 
                'ttt_output': batch_y_ttt, 
                'channel_attention_weights': batch_y_att 
            }
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==========================================
# 模型包装器 (修复 LIME 维度报错)
# ==========================================
class ModelWrapperForLIME:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, inputs, **kwargs):
        preds = self.base_model.predict(inputs, **kwargs)
        return preds[0] if isinstance(preds, list) else preds


# ==========================================
# 辅助绘图
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
# 1-3. 数据流与模型训练
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

# 切分：测试集 (15%)
split_test = int(len(X_raw) * 0.85)
X_train_val_raw, X_test_raw = X_raw[:split_test], X_raw[split_test:]
X_train_val_seg, X_test_seg = X_segments[:split_test], X_segments[split_test:]
y_train_val, y_test = y[:split_test], y[split_test:]

# 切分：训练集 (85% of 85%) 与验证集
split_val = int(len(X_train_val_raw) * 0.85)
X_train_raw, X_val_raw = X_train_val_raw[:split_val], X_train_val_raw[split_val:]
X_train_seg, X_val_seg = X_train_val_seg[:split_val], X_train_val_seg[split_val:]
y_train, y_val = y_train_val[:split_val], y_train_val[split_val:]

# 🌟 实例化最终的 V3 模型
model = create_hybrid_model_v2()
callbacks = [EarlyStopping(monitor='val_main_output_accuracy', patience=5, restore_best_weights=True)]

train_gen = SleepDataGenerator(X_train_raw, X_train_seg, y_train, batch_size=32, shuffle=True)
val_gen = SleepDataGenerator(X_val_raw, X_val_seg, y_val, batch_size=32, shuffle=False)

print("\n🚀 数据流水线构建完毕，开始流式训练 (最终 V3 架构)...")
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=2
)

# ==========================================
# 4. 全局极速推理评估
# ==========================================
print("\n==========================================")
print("4. 正在执行全局极速推理...")
print("==========================================")

y_pred_probs_all, _, _ = model.predict([X_test_raw, X_test_seg], batch_size=64, verbose=1)
y_pred_final = np.argmax(y_pred_probs_all, axis=1)
y_true = np.argmax(y_test, axis=1)

print(f"\n📊 最终分类报告:")
print(classification_report(y_true, y_pred_final, target_names=CLASS_LABELS, digits=4))

save_academic_results(y_true, y_pred_final, CLASS_LABELS, OUTPUT_DIR)


# ==========================================
# 5. 多阶段 XAI 协同诊断系统 (LIME)
# ==========================================
print("\n5. 执行多阶段 XAI 协同分析 (生成可解释图)...")
lime_wrapper = ModelWrapperForLIME(model)

for stage_idx, stage_name in enumerate(CLASS_LABELS):
    # 🌟 精准抓取：只挑选真实标签和预测标签都正确的样本进行解释
    correct_indices = np.where((y_true == stage_idx) & (y_pred_final == stage_idx))[0]
    
    if len(correct_indices) == 0: 
        print(f"⚠️ 阶段 {stage_name} 没有预测正确的样本，跳过解释。")
        continue

    target_idx = correct_indices[0] 
    instance_raw = X_test_raw[target_idx]

    print(f"   🔎 正在诊断阶段: {stage_name} (样本索引: {target_idx})")

    segments = semantic_segmentation(instance_raw, target_segments=30)
    target_class, _, importance, raw_dccs, calibrated_dccs = run_multimodal_lime(
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
        raw_dccs=raw_dccs,
        calibrated_dccs=calibrated_dccs,
        channel_names=['EEG', 'EOG', 'EMG'],
        target_class_name=CLASS_LABELS[target_class],
        save_path=save_path
    )

print(f"\n🎉 任务圆满完成！诊断图和分类报告已存入: {OUTPUT_DIR}")