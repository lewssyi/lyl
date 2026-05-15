import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense,
    BatchNormalization, Concatenate, GlobalAveragePooling1D,
    Reshape, Multiply, Add, Activation, Bidirectional, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalFocalCrossentropy


def se_block_1d(input_tensor, reduction=4):
    """
    【你的创新强化】：标准 1D SE 模块
    用于在时间序列上对不同特征通道（如 13 个切片特征、数百个卷积核通道）进行动态重要性重标定
    """
    channels = int(input_tensor.shape[-1])
    # 空间维度压缩 (Squeeze)
    squeeze = GlobalAveragePooling1D()(input_tensor)
    # 通道权重激励 (Excitation)
    excitation = Dense(max(1, channels // reduction), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    # 维度对齐与相乘
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([input_tensor, excitation])


def se_block_dense(input_tensor, reduction=4):
    """
    【你的创新强化】：2D 稠密 SE 模块
    专用于特征融合后（Concatenate）的全局权重分配，让网络在波形和切片间做权衡
    """
    channels = int(input_tensor.shape[-1])
    excitation = Dense(max(1, channels // reduction), activation='relu')(input_tensor)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    return Multiply()([input_tensor, excitation])


def se_multi_channel_extractor(input_layer):
    """
    【SE 赋能的多模态特征提取器】
    在 MRCNN 的提取过程中，逐级嵌入 SE 模块进行通道提纯
    """
    # 分支 1：高频微事件 (EMG/纺锤波)
    b1 = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=4, strides=2, padding='same')(b1)
    b1 = se_block_1d(b1, reduction=4)  # 第 1 层提纯

    b1 = Conv1D(128, kernel_size=5, strides=1, padding='same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=4, strides=5, padding='same')(b1)
    b1 = se_block_1d(b1, reduction=8)  # 第 2 层提纯

    # 分支 2：低频宏观波形 (EOG/慢波)
    b2 = Conv1D(64, kernel_size=31, strides=4, padding='same')(input_layer)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = MaxPooling1D(pool_size=4, strides=5, padding='same')(b2)
    b2 = se_block_1d(b2, reduction=4)

    b2 = Conv1D(128, kernel_size=15, strides=1, padding='same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = se_block_1d(b2, reduction=8)

    # 合并双频带并做最后一次通道重标定
    out = Concatenate(axis=-1)([b1, b2])
    out = se_block_1d(out, reduction=16)
    out = Dropout(0.4)(out)
    return out


def create_hybrid_model(raw_shape=(3000, 3), segment_shape=(30, 13), num_classes=5):
    """
    【终极版本：全链路 SE 增强双向循环网络】
    """
    # ==========================================
    # 分支 1：感知流 (波形 -> 嵌套SE的MRCNN -> BiLSTM)
    # ==========================================
    input_raw = Input(shape=raw_shape, name='raw_input')

    x1 = se_multi_channel_extractor(input_raw)

    # 深度时序记忆
    x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
    x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)

    cnn_features = Dense(128, activation='relu')(x1)

    # ==========================================
    # 分支 2：逻辑流 (生理切片 -> 切片级SE -> BiLSTM)
    # ==========================================
    input_segments = Input(shape=segment_shape, name='segment_input')

    # 【核心放大】：在时序建模前，直接对 13 个专家特征进行 SE 加权！
    # 彻底释放生理切片在不同睡眠阶段的区分潜力
    x2 = se_block_1d(input_segments, reduction=2)

    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
    x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)

    lstm_features = Dense(64, activation='relu')(x2)

    # ==========================================
    # 融合层与多任务输出
    # ==========================================
    merged = Concatenate(name='feature_fusion')([cnn_features, lstm_features])

    # 【核心放大】：融合后进行全局 SE 权重分配
    merged_se = se_block_dense(merged, reduction=4)

    # 主分类头
    dense = Dense(256, activation='relu')(merged_se)
    dense = Dropout(0.5)(dense)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(dense)

    # 辅助重建头 (为后续若需要 TTT 保留接口)
    flat_dim = int(np.prod(raw_shape))
    dense_ttt = Dense(512, activation='relu')(merged)
    ttt_flat = Dense(flat_dim, activation='linear')(dense_ttt)
    ttt_output = Reshape(raw_shape, name='ttt_output')(ttt_flat)

    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, ttt_output])

    # 优化器
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'main_output': CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
            'ttt_output': 'mse'
        },
        loss_weights={'main_output': 1.0, 'ttt_output': 1e-6},
        metrics={'main_output': 'accuracy'}
    )

    return model


    return model
# import tensorflow as tf
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense,
#     BatchNormalization, LSTM, Concatenate, GlobalAveragePooling1D,
#     Reshape, multiply, Add, Activation
# )
# from tensorflow.keras.models import Model
# # 引入 Focal Loss (解决极度不平衡分类的终极武器)
# from tensorflow.keras.losses import CategoricalFocalCrossentropy
#
#
# def squeeze_excitation_layer(input_x, out_dim, ratio=4):
#     """
#     【通道注意力模块】
#     根据输入特征自动分配权重，识别 EEG/EOG/EMG 哪个通道对当前样本更重要
#     """
#     # Squeeze: 全局信息压缩
#     squeeze = GlobalAveragePooling1D()(input_x)
#
#     # Excitation: 学习通道间的非线性依赖关系
#     excitation = Dense(out_dim // ratio, activation='relu')(squeeze)
#     excitation = Dense(out_dim, activation='sigmoid')(excitation)
#     excitation = Reshape((1, out_dim))(excitation)
#
#     # Scale: 将权重乘回原特征图
#     return multiply([input_x, excitation])
#
#
# def multi_scale_block(input_layer, filters=48):
#     """
#     【多尺度卷积模块】
#     并行使用长、中、短卷积核，全方位覆盖睡眠波形特征
#     """
#     # 长核：捕捉宏观慢波 (如 Delta 波)
#     b1 = Conv1D(filters, kernel_size=64, padding='same', activation='relu')(input_layer)
#     # 中核：捕捉典型特征 (如 Spindles)
#     b2 = Conv1D(filters, kernel_size=32, padding='same', activation='relu')(input_layer)
#     # 短核：捕捉高频噪声或微小事件 (如 EMG 爆发)
#     b3 = Conv1D(filters, kernel_size=8, padding='same', activation='relu')(input_layer)
#
#     out = Concatenate()([b1, b2, b3])
#     out = BatchNormalization()(out)
#     return out
#
#
# def create_hybrid_model(raw_shape=(3000, 3), segment_shape=(30, 13), num_classes=5):
#     """
#     【增强版双流网络】引入注意力机制与多尺度融合 + Focal Loss 优化
#     """
#
#     # ==========================================
#     # 分支 1：感知流 (改进版 CNN)
#     # ==========================================
#     input_raw = Input(shape=raw_shape, name='raw_input')
#
#     # 第一层：多尺度特征提取
#     x1 = multi_scale_block(input_raw, filters=32)  # 输出 32*3=96 个特征图
#
#     # 核心：引入通道注意力，让模型自己决定关注哪个通道
#     x1 = squeeze_excitation_layer(x1, out_dim=96)
#     x1 = MaxPooling1D(pool_size=8)(x1)
#
#     # 第二层：深度特征抽象
#     x1 = Conv1D(filters=128, kernel_size=10, padding='same', activation='relu')(x1)
#     x1 = BatchNormalization()(x1)
#     x1 = squeeze_excitation_layer(x1, out_dim=128)
#     x1 = MaxPooling1D(pool_size=4)(x1)
#     x1 = Dropout(0.3)(x1)
#
#     # 第三层：压缩特征空间
#     x1 = Conv1D(filters=128, kernel_size=5, activation='relu')(x1)
#     x1 = MaxPooling1D(pool_size=2)(x1)
#
#     cnn_features = Flatten()(x1)
#
#     # ==========================================
#     # 分支 2：逻辑流 (LSTM) 保持对生理事件的捕捉
#     # ==========================================
#     input_segments = Input(shape=segment_shape, name='segment_input')
#
#     x2 = LSTM(64, return_sequences=False)(input_segments)
#     x2 = BatchNormalization()(x2)
#     lstm_features = Dropout(0.3)(x2)
#
#     # ==========================================
#     # 融合层：多模态汇合
#     # ==========================================
#     merged = Concatenate(name='feature_fusion')([cnn_features, lstm_features])
#
#     # 深度全连接层进行非线性映射
#     dense = Dense(256, activation='relu')(merged)
#     dense = Dropout(0.5)(dense)
#     dense = Dense(128, activation='relu')(dense)
#
#     output = Dense(num_classes, activation='softmax', name='main_output')(dense)
#
#     # 构建并编译模型
#     model = Model(inputs=[input_raw, input_segments], outputs=output)
#
#     # 【核心修改】：使用 Focal Loss 替换普通的交叉熵
#     # alpha 调节整体类别不平衡，gamma=2.0 会呈指数级放大困难样本（如N1/N3）的 loss，压制简单样本（如大量N2）
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
#         metrics=['accuracy']
#     )
#
#     return model