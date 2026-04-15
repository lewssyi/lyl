import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_class_distribution(labels, title="Sleep Stage Distribution"):
    """
    绘制睡眠阶段分布。适配 W, N1, N2, N3, REM 五类。
    """
    class_colors = {0: "gray", 1: "skyblue", 2: "royalblue", 3: "purple", 4: "red"}
    plt.style.use('default')
    plt.figure(figsize=(8, 5), facecolor='white')

    ax = sns.countplot(x=labels, palette=class_colors)
    ax.set_facecolor('white')
    ax.set_title(title, fontweight='bold')

    plt.xlabel("Sleep Stage (0:W, 1:N1, 2:N2, 3:N3, 4:REM)")
    plt.ylabel("Count")
    plt.show()


def plot_segmented_eeg(instance_multimodal, segments):
    """
    查看 30 秒信号是如何被“生理平滑算法”动态切分的。
    展示 EEG, EOG, EMG 三个通道的切分线。
    """
    num_channels = instance_multimodal.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3 * num_channels), sharex=True, facecolor='white')
    channel_names = ['EEG (Brain)', 'EOG (Eyes)', 'EMG (Muscle)']

    for c in range(num_channels):
        ax = axes[c]
        ax.plot(instance_multimodal[:, c], color='black', linewidth=0.8)
        ax.set_ylabel(channel_names[c], fontweight='bold')

        # 使用动态 segments 绘制切割线
        for (start_idx, end_idx) in segments:
            ax.axvline(x=start_idx, color='blue', linestyle='--', alpha=0.3, linewidth=1)
            # 画出结束线（如果是连续切片，结束线和下一段的开始线会重合）
            ax.axvline(x=end_idx, color='blue', linestyle='--', alpha=0.3, linewidth=1)

    axes[0].set_title('Dynamically Segmented Physiological Signals', fontweight='bold')
    axes[-1].set_xlabel('Time Points (3000 points = 30s)')
    plt.tight_layout()
    plt.show()


def visualize_lime_explanation(instance_multimodal, segment_importance, segments, predicted_class, prob, class_labels):
    """
    【核心更新】支持动态切片 (segments) 和多模态三通道独立解释。
    """
    if segment_importance is None:
        print("❌ 错误: segment_importance 为 None。")
        return

    num_segments = len(segments)
    num_channels = instance_multimodal.shape[1]

    # 防御性检查：确保 LIME 权重数量 = 通道数 * 切片数
    if len(segment_importance) != num_channels * num_segments:
        print(
            f"❌ 权重长度错误！期望 {num_channels * num_segments} (3通道 x {num_segments}切片)，但收到 {len(segment_importance)}")
        return

    # 拆分三个通道的 LIME 权重
    channel_importances = [
        segment_importance[0: num_segments],  # EEG 权重
        segment_importance[num_segments: 2 * num_segments],  # EOG 权重
        segment_importance[2 * num_segments: 3 * num_segments]  # EMG 权重
    ]

    max_imp = np.max(np.abs(segment_importance))
    if max_imp == 0: max_imp = 1e-8

    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 4 * num_channels), sharex=True, facecolor='white')
    channel_names = ['1. EEG - Brain Activity', '2. EOG - Eye Movement', '3. EMG - Muscle Tone']
    line_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    for c in range(num_channels):
        ax = axes[c]
        signal = instance_multimodal[:, c]
        imp_array = channel_importances[c]

        ax.set_facecolor('#fafafa')
        ax.plot(signal, color=line_colors[c], linewidth=1.0, alpha=0.9)
        ax.set_ylabel("Amplitude", fontweight='bold')
        ax.set_title(channel_names[c], fontsize=12, fontweight='bold', loc='left')

        # 遍历动态片段
        for i, (start_idx, end_idx) in enumerate(segments):
            importance = imp_array[i]

            if abs(importance) > 0:
                color = 'green' if importance > 0 else 'red'
                # 透明度根据全局最大权重归一化
                alpha = (abs(importance) / max_imp) * 0.4
                ax.axvspan(start_idx, end_idx, color=color, alpha=alpha)

            # 画出智能切割的边界线
            ax.axvline(start_idx, color='gray', linestyle=':', alpha=0.3, linewidth=1)

    fig.suptitle(f"Motif-Aware LIME Interpretation | Pred: {class_labels[predicted_class]} ({prob:.2%})",
                 fontsize=16, fontweight='bold', color='#333333')
    axes[-1].set_xlabel("Time Index (3000 points = 30 seconds)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_perturbed_ecg(original_multimodal, perturbed_multimodal, perturbation, segments, channel_idx=0):
    """
    对比原始信号和被“遮蔽”后的信号。适配动态切片。
    为了不让图表太乱，默认只画出被扰动的一个通道 (channel_idx, 0=EEG, 1=EOG, 2=EMG) 进行对比。
    """
    plt.figure(figsize=(15, 7), facecolor='white')
    num_segments = len(segments)

    # 提取该通道对应的扰动遮罩 (0 或 1)
    channel_perturbation = perturbation[channel_idx * num_segments: (channel_idx + 1) * num_segments]

    # 子图1：原始信号及其被遮蔽的区域
    plt.subplot(2, 1, 1)
    plt.plot(original_multimodal[:, channel_idx], color='black', label='Original')

    for i, (start_idx, end_idx) in enumerate(segments):
        active = channel_perturbation[i]
        if not active:  # 如果被遮蔽 (值为0)
            plt.axvspan(start_idx, end_idx, color='red', alpha=0.2)

    channel_name = ['EEG', 'EOG', 'EMG'][channel_idx]
    plt.title(f"{channel_name} - Original Signal with Perturbed Segments (Red areas are masked)", fontweight='bold')

    # 子图2：实际喂给模型的扰动信号
    plt.subplot(2, 1, 2)
    plt.plot(perturbed_multimodal[:, channel_idx], color='darkgreen', label='Perturbed')
    plt.title(f"{channel_name} - Actual Signal Input to Model after Perturbation", fontweight='bold')

    plt.tight_layout()
    plt.show()


# filepath: src/visualization.py (追加到末尾)
import matplotlib.pyplot as plt
import numpy as np


def visualize_gradcam_explanation(instance_raw, heatmap, predicted_class, prob, class_labels):
    """
    可视化 Grad-CAM 结果，将热力图叠加在三个通道上
    """
    channels = ['EEG', 'EOG', 'EMG']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Grad-CAM 空间显著性分析 - 预测: {class_labels[predicted_class]} (置信度: {prob:.2%})", fontsize=16)

    time_steps = np.arange(len(instance_raw))

    for i in range(3):
        ax = axes[i]
        signal = instance_raw[:, i]

        # 绘制原始波形
        ax.plot(time_steps, signal, color='black', alpha=0.7, linewidth=1, label=f'Raw {channels[i]}')

        # 叠加 Grad-CAM 热力图
        y_min, y_max = np.min(signal), np.max(signal)
        # 使用 extent 将 1D 热力图拉伸覆盖到波形的整个高度
        im = ax.imshow(heatmap[np.newaxis, :], aspect='auto', cmap='jet', alpha=0.4,
                       extent=[0, len(instance_raw), y_min, y_max], origin='lower')

        ax.set_ylabel(f'{channels[i]} 信号值')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time Samples (采样点)")

    # 增加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Grad-CAM 重要性权重 (对分类的贡献)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出右侧空间给 colorbar
    save_path = f"GradCAM_Result_{class_labels[predicted_class]}.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ Grad-CAM 可视化图像已保存至: {save_path}")
    plt.show()