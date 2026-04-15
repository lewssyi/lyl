# filepath: src/grad_cam_explanation.py
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom


def run_hybrid_grad_cam(model, instance_raw, instance_seg, layer_name=None):
    """
    针对双流模型定制的 Grad-CAM 解释器
    :param model: 训练好的双流模型
    :param instance_raw: 原始波形数据，形状 (3000, 3)
    :param instance_seg: 提取的片段特征数据，形状 (30, 13)
    :param layer_name: 想要可视化的卷积层名称，默认自动寻找最后一个波形卷积层
    :return: (调整尺寸后的热力图, 预测类别索引, 预测置信度)
    """
    # 1. 扩充维度以匹配模型输入 (Batch size = 1)
    raw_input = tf.expand_dims(instance_raw, axis=0)  # (1, 3000, 3)
    seg_input = tf.expand_dims(instance_seg, axis=0)  # (1, 30, 13)

    # 2. 自动定位最后一个卷积层（如果未指定）
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("未在模型中找到包含 'conv' 的卷积层！")

    print(f"🔍 Grad-CAM 正在监听卷积层: {layer_name}")

    # 3. 构建梯度模型
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # 4. 计算梯度
    with tf.GradientTape() as tape:
        # 双流模型需要同时输入 raw 和 seg
        conv_outputs, predictions = grad_model([raw_input, seg_input])
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # 获取预测概率
    prob = predictions[0, pred_index].numpy()

    # 计算目标层对预测类别的梯度
    grads = tape.gradient(loss, conv_outputs)

    # 在时间轴维度上做全局平均池化 (Conv1D 输出形状通常是 [batch, time_steps, filters])
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # 加权合成热力图
    conv_outputs = conv_outputs[0]  # 去掉 batch 维度
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU 激活（只保留对分类有正向贡献的特征）并归一化
    heatmap = tf.nn.relu(heatmap).numpy()
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # 5. 使用 scipy 插值，将热力图拉伸到与原始波形一样长 (3000)
    zoom_factor = len(instance_raw) / len(heatmap)
    heatmap_resized = zoom(heatmap, zoom_factor)

    return heatmap_resized, pred_index.numpy(), prob