import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os


# ==========================================
# 1. 核心创新可视化：多通道 XAI 动态贡献仪表盘
# ==========================================
def plot_xai_multimodal_dashboard(signal, segments, segment_importance, dccs,
                                  channel_names=['EEG', 'EOG', 'EMG'],
                                  target_class_name="REM",
                                  save_path=None):
    """
    绘制 CCF-A 级别的多通道 XAI 解释仪表盘
    包含：原始信号波形、LIME 切片热力图、DCCS 通道贡献度环形图
    """
    # 设置学术级字体与风格
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    num_channels = signal.shape[1]
    num_segments = len(segments)

    # 创建画布与网格布局
    fig = plt.figure(figsize=(16, 10))
    # 设定网格：左边占 3 列画波形，右边占 1 列画环形图
    gs = gridspec.GridSpec(num_channels, 4, figure=fig, width_ratios=[1, 1, 1, 0.8])

    # 颜色映射：正贡献为红(促进该分类)，负贡献为蓝(抑制该分类)
    cmap = plt.get_cmap('coolwarm')

    # 获取特征权重的极值用于归一化颜色
    max_abs_weight = np.max(np.abs(segment_importance))
    if max_abs_weight == 0:
        max_abs_weight = 1.0

    # ------------------------------------------
    # 左侧模块：绘制原始波形与 LIME 切片热力图
    # ------------------------------------------
    ax_signals = []
    for c in range(num_channels):
        ax = fig.add_subplot(gs[c, :3])
        ax_signals.append(ax)

        # 画原始波形
        time_axis = np.arange(signal.shape[0])
        ax.plot(time_axis, signal[:, c], color='black', linewidth=0.8, alpha=0.8)

        # 叠加 LIME 切片热力背景
        c_weights = segment_importance[c * num_segments: (c + 1) * num_segments]

        for i, (start_idx, end_idx) in enumerate(segments):
            weight = c_weights[i]
            # 归一化权重到 0-1 之间用于取色 (0.5 为中性白)
            norm_weight = 0.5 + (weight / (2 * max_abs_weight))
            color = cmap(norm_weight)

            # 绘制背景色块
            ax.axvspan(start_idx, end_idx, facecolor=color, alpha=0.5, edgecolor='none')

            # 画切片分割线
            ax.axvline(x=end_idx, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        channel_label = channel_names[c] if c < len(channel_names) else f'CH {c}'
        ax.set_ylabel(f'{channel_label}\nAmplitude', fontweight='bold')
        ax.set_xlim(0, signal.shape[0])

        # 隐藏上方通道的 X 轴刻度，使其更紧凑
        if c < num_channels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (Samples)', fontweight='bold')

    ax_signals[0].set_title(f'XAI Explanation for Predicted Stage: {target_class_name}',
                            fontweight='bold', fontsize=16, loc='left')

    # ------------------------------------------
    # 右侧模块：绘制 DCCS 动态通道贡献环形图
    # ------------------------------------------
    # 将右侧合并为一个大子图
    ax_dccs = fig.add_subplot(gs[:, 3])

    # 环形图配色 (根据通道数量动态调整，优先使用预设的温和学术色)
    base_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    colors = base_colors[:num_channels]
    explode = [0.05] * num_channels  # 让环形图切片稍微裂开，增加立体感

    # 过滤可能存在的全0情况
    plot_dccs = dccs if sum(dccs) > 0 else [1.0 / num_channels] * num_channels

    wedges, texts, autotexts = ax_dccs.pie(
        plot_dccs,
        labels=channel_names[:num_channels],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops=dict(color="w", weight="bold", fontsize=12),
        wedgeprops=dict(width=0.4, edgecolor='w')  # width 参数控制环的厚度
    )

    # 优化标签颜色使其与区块对应
    for text, color in zip(texts, colors):
        text.set_color(color)
        text.set_fontsize(14)
        text.set_fontweight('bold')

    ax_dccs.set_title("Dynamic Channel\nContribution Score (DCCS)", fontweight='bold', fontsize=14)

    # ------------------------------------------
    # 全局模块：排版与 Colorbar
    # ------------------------------------------
    plt.tight_layout()

    # 添加底部的全局 Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight))
    sm.set_array([])
    # 位置：[左, 下, 宽, 高]
    cbar_ax = fig.add_axes([0.05, -0.05, 0.65, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('LIME Feature Importance (Red: Supports Prediction, Blue: Contradicts Prediction)',
                   fontweight='bold', fontsize=12)

    # 确保保存目录存在
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ XAI 学术仪表盘已保存至: {save_path}")
    else:
        plt.show()

    plt.close()


# ==========================================
# 2. 基础调试工具：动态切片结果可视化
# ==========================================
def plot_segmented_signals(signal, segments, channel_names=['EEG', 'EOG', 'EMG'], save_path=None):
    """
    用于调试 semantic_segmentation 函数，查看多模态能量融合切片是否合理
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    num_channels = signal.shape[1]

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    time_axis = np.arange(signal.shape[0])

    for c in range(num_channels):
        ax = axes[c]
        ax.plot(time_axis, signal[:, c], color='black', linewidth=0.8)

        # 绘制切片边界
        for start_idx, end_idx in segments:
            ax.axvline(x=start_idx, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            # 交替填充背景色以便区分切片
            if (segments.index((start_idx, end_idx)) % 2 == 0):
                ax.axvspan(start_idx, end_idx, facecolor='gray', alpha=0.1)

        channel_label = channel_names[c] if c < len(channel_names) else f'CH {c}'
        ax.set_ylabel(channel_label, fontweight='bold')

    axes[0].set_title('Physiological Smoothness Segmentation Results', fontweight='bold', fontsize=14)
    axes[-1].set_xlabel('Time (Samples)', fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 切片调试图已保存至: {save_path}")
    else:
        plt.show()

    plt.close()


# ==========================================
# 3. 基础调试工具：简单的原始信号对比图
# ==========================================
def plot_raw_signals(signal, title="Multichannel Physiological Signal", save_path=None):
    """
    基础的信号绘图函数，用于快速查看模型输入
    """
    plt.figure(figsize=(12, 4))
    num_channels = signal.shape[1] if len(signal.shape) > 1 else 1

    if num_channels > 1:
        for c in range(num_channels):
            # 将多通道信号错开显示
            offset = np.max(np.abs(signal)) * 2 * c
            plt.plot(signal[:, c] - offset, linewidth=0.8, label=f'Channel {c}')
        plt.yticks([])
    else:
        plt.plot(signal, color='black', linewidth=0.8)

    plt.title(title, fontweight='bold')
    plt.xlabel('Time (Samples)', fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def plot_class_distribution(labels, title="Sleep Stage Distribution"):
#     """
#     绘制睡眠阶段分布。适配 W, N1, N2, N3, REM 五类。
#     """
#     class_colors = {0: "gray", 1: "skyblue", 2: "royalblue", 3: "purple", 4: "red"}
#     plt.style.use('default')
#     plt.figure(figsize=(8, 5), facecolor='white')
#
#     ax = sns.countplot(x=labels, palette=class_colors)
#     ax.set_facecolor('white')
#     ax.set_title(title, fontweight='bold')
#
#     plt.xlabel("Sleep Stage (0:W, 1:N1, 2:N2, 3:N3, 4:REM)")
#     plt.ylabel("Count")
#     plt.show()
#
#
# def plot_segmented_eeg(instance_multimodal, segments):
#     """
#     查看 30 秒信号是如何被“生理平滑算法”动态切分的。
#     展示 EEG, EOG, EMG 三个通道的切分线。
#     """
#     num_channels = instance_multimodal.shape[1]
#     fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3 * num_channels), sharex=True, facecolor='white')
#     channel_names = ['EEG (Brain)', 'EOG (Eyes)', 'EMG (Muscle)']
#
#     for c in range(num_channels):
#         ax = axes[c]
#         ax.plot(instance_multimodal[:, c], color='black', linewidth=0.8)
#         ax.set_ylabel(channel_names[c], fontweight='bold')
#
#         # 使用动态 segments 绘制切割线
#         for (start_idx, end_idx) in segments:
#             ax.axvline(x=start_idx, color='blue', linestyle='--', alpha=0.3, linewidth=1)
#             # 画出结束线（如果是连续切片，结束线和下一段的开始线会重合）
#             ax.axvline(x=end_idx, color='blue', linestyle='--', alpha=0.3, linewidth=1)
#
#     axes[0].set_title('Dynamically Segmented Physiological Signals', fontweight='bold')
#     axes[-1].set_xlabel('Time Points (3000 points = 30s)')
#     plt.tight_layout()
#     plt.show()
#
#
# def visualize_lime_explanation(instance_multimodal, segment_importance, segments, predicted_class, prob, class_labels):
#     """
#     【核心更新】支持动态切片 (segments) 和多模态三通道独立解释。
#     """
#     if segment_importance is None:
#         print("❌ 错误: segment_importance 为 None。")
#         return
#
#     num_segments = len(segments)
#     num_channels = instance_multimodal.shape[1]
#
#     # 防御性检查：确保 LIME 权重数量 = 通道数 * 切片数
#     if len(segment_importance) != num_channels * num_segments:
#         print(
#             f"❌ 权重长度错误！期望 {num_channels * num_segments} (3通道 x {num_segments}切片)，但收到 {len(segment_importance)}")
#         return
#
#     # 拆分三个通道的 LIME 权重
#     channel_importances = [
#         segment_importance[0: num_segments],  # EEG 权重
#         segment_importance[num_segments: 2 * num_segments],  # EOG 权重
#         segment_importance[2 * num_segments: 3 * num_segments]  # EMG 权重
#     ]
#
#     max_imp = np.max(np.abs(segment_importance))
#     if max_imp == 0: max_imp = 1e-8
#
#     fig, axes = plt.subplots(num_channels, 1, figsize=(16, 4 * num_channels), sharex=True, facecolor='white')
#     channel_names = ['1. EEG - Brain Activity', '2. EOG - Eye Movement', '3. EMG - Muscle Tone']
#     line_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
#
#     for c in range(num_channels):
#         ax = axes[c]
#         signal = instance_multimodal[:, c]
#         imp_array = channel_importances[c]
#
#         ax.set_facecolor('#fafafa')
#         ax.plot(signal, color=line_colors[c], linewidth=1.0, alpha=0.9)
#         ax.set_ylabel("Amplitude", fontweight='bold')
#         ax.set_title(channel_names[c], fontsize=12, fontweight='bold', loc='left')
#
#         # 遍历动态片段
#         for i, (start_idx, end_idx) in enumerate(segments):
#             importance = imp_array[i]
#
#             if abs(importance) > 0:
#                 color = 'green' if importance > 0 else 'red'
#                 # 透明度根据全局最大权重归一化
#                 alpha = (abs(importance) / max_imp) * 0.4
#                 ax.axvspan(start_idx, end_idx, color=color, alpha=alpha)
#
#             # 画出智能切割的边界线
#             ax.axvline(start_idx, color='gray', linestyle=':', alpha=0.3, linewidth=1)
#
#     fig.suptitle(f"Motif-Aware LIME Interpretation | Pred: {class_labels[predicted_class]} ({prob:.2%})",
#                  fontsize=16, fontweight='bold', color='#333333')
#     axes[-1].set_xlabel("Time Index (3000 points = 30 seconds)", fontsize=12, fontweight='bold')
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)
#     plt.show()
#
#
# def plot_perturbed_ecg(original_multimodal, perturbed_multimodal, perturbation, segments, channel_idx=0):
#     """
#     对比原始信号和被“遮蔽”后的信号。适配动态切片。
#     为了不让图表太乱，默认只画出被扰动的一个通道 (channel_idx, 0=EEG, 1=EOG, 2=EMG) 进行对比。
#     """
#     plt.figure(figsize=(15, 7), facecolor='white')
#     num_segments = len(segments)
#
#     # 提取该通道对应的扰动遮罩 (0 或 1)
#     channel_perturbation = perturbation[channel_idx * num_segments: (channel_idx + 1) * num_segments]
#
#     # 子图1：原始信号及其被遮蔽的区域
#     plt.subplot(2, 1, 1)
#     plt.plot(original_multimodal[:, channel_idx], color='black', label='Original')
#
#     for i, (start_idx, end_idx) in enumerate(segments):
#         active = channel_perturbation[i]
#         if not active:  # 如果被遮蔽 (值为0)
#             plt.axvspan(start_idx, end_idx, color='red', alpha=0.2)
#
#     channel_name = ['EEG', 'EOG', 'EMG'][channel_idx]
#     plt.title(f"{channel_name} - Original Signal with Perturbed Segments (Red areas are masked)", fontweight='bold')
#
#     # 子图2：实际喂给模型的扰动信号
#     plt.subplot(2, 1, 2)
#     plt.plot(perturbed_multimodal[:, channel_idx], color='darkgreen', label='Perturbed')
#     plt.title(f"{channel_name} - Actual Signal Input to Model after Perturbation", fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # filepath: src/visualization.py (追加到末尾)
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def visualize_gradcam_explanation(instance_raw, heatmap, predicted_class, prob, class_labels):
#     """
#     可视化 Grad-CAM 结果，将热力图叠加在三个通道上
#     """
#     channels = ['EEG', 'EOG', 'EMG']
#     fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
#     fig.suptitle(f"Grad-CAM 空间显著性分析 - 预测: {class_labels[predicted_class]} (置信度: {prob:.2%})", fontsize=16)
#
#     time_steps = np.arange(len(instance_raw))
#
#     for i in range(3):
#         ax = axes[i]
#         signal = instance_raw[:, i]
#
#         # 绘制原始波形
#         ax.plot(time_steps, signal, color='black', alpha=0.7, linewidth=1, label=f'Raw {channels[i]}')
#
#         # 叠加 Grad-CAM 热力图
#         y_min, y_max = np.min(signal), np.max(signal)
#         # 使用 extent 将 1D 热力图拉伸覆盖到波形的整个高度
#         im = ax.imshow(heatmap[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.4,
#                        extent=[0, len(instance_raw), y_min, y_max], origin='lower')
#
#         ax.set_ylabel(f'{channels[i]} 信号值')
#         ax.legend(loc='upper right')
#
#     axes[-1].set_xlabel("Time Samples (采样点)")
#
#     # 增加颜色条
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     fig.colorbar(im, cax=cbar_ax, label='Grad-CAM 重要性权重 (对分类的贡献)')
#
#     plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出右侧空间给 colorbar
#     save_path = f"GradCAM_Result_{class_labels[predicted_class]}.png"
#     plt.savefig(save_path, dpi=150)
#     print(f"✅ Grad-CAM 可视化图像已保存至: {save_path}")
#     plt.show()