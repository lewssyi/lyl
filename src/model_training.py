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

# 🚀 重构版辅助重建头：使用 Conv1D 替代超大全连接层，彻底告别 [512, 9000] OOM 崩溃
    dense_ttt = Dense(256, activation='relu')(merged)
    # 先将维度平稳映射到时间步长度（3000）
    ttt_mid = Dense(raw_shape[0], activation='relu')(dense_ttt) 
    # 增加一个伪通道轴，变为 (3000, 1)
    ttt_mid = Reshape((raw_shape[0], 1))(ttt_mid)
    # 最终通过 1x1 卷积无损膨胀至目标通道数 (3000, 3)
    ttt_output = Conv1D(raw_shape[1], kernel_size=1, padding='same', name='ttt_output')(ttt_mid)

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

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense,
#     BatchNormalization, Concatenate, GlobalAveragePooling1D,
#     Reshape, Multiply, Add, Activation, Bidirectional, LSTM
# )
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import CategoricalFocalCrossentropy


# def se_block_1d(input_tensor, reduction=4):
#     """
#     【你的创新强化】：标准 1D SE 模块
#     用于在时间序列上对不同特征通道（如 13 个切片特征、数百个卷积核通道）进行动态重要性重标定
#     """
#     channels = int(input_tensor.shape[-1])
#     # 空间维度压缩 (Squeeze)
#     squeeze = GlobalAveragePooling1D()(input_tensor)
#     # 通道权重激励 (Excitation)
#     excitation = Dense(max(1, channels // reduction), activation='relu')(squeeze)
#     excitation = Dense(channels, activation='sigmoid')(excitation)
#     # 维度对齐与相乘
#     excitation = Reshape((1, channels))(excitation)
#     return Multiply()([input_tensor, excitation])


# def se_block_dense(input_tensor, reduction=4):
#     """
#     【你的创新强化】：2D 稠密 SE 模块
#     专用于特征融合后（Concatenate）的全局权重分配，让网络在波形和切片间做权衡
#     """
#     channels = int(input_tensor.shape[-1])
#     excitation = Dense(max(1, channels // reduction), activation='relu')(input_tensor)
#     excitation = Dense(channels, activation='sigmoid')(excitation)
#     return Multiply()([input_tensor, excitation])


# def se_multi_channel_extractor(input_layer):
#     """
#     【SE 赋能的多模态特征提取器】
#     在 MRCNN 的提取过程中，逐级嵌入 SE 模块进行通道提纯
#     """
#     # 分支 1：高频微事件 (EMG/纺锤波)
#     b1 = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
#     b1 = BatchNormalization()(b1)
#     b1 = Activation('relu')(b1)
#     b1 = MaxPooling1D(pool_size=4, strides=2, padding='same')(b1)
#     b1 = se_block_1d(b1, reduction=4)  # 第 1 层提纯

#     b1 = Conv1D(128, kernel_size=5, strides=1, padding='same')(b1)
#     b1 = BatchNormalization()(b1)
#     b1 = Activation('relu')(b1)
#     b1 = MaxPooling1D(pool_size=4, strides=5, padding='same')(b1)
#     b1 = se_block_1d(b1, reduction=8)  # 第 2 层提纯

#     # 分支 2：低频宏观波形 (EOG/慢波)
#     b2 = Conv1D(64, kernel_size=31, strides=4, padding='same')(input_layer)
#     b2 = BatchNormalization()(b2)
#     b2 = Activation('relu')(b2)
#     b2 = MaxPooling1D(pool_size=4, strides=5, padding='same')(b2)
#     b2 = se_block_1d(b2, reduction=4)

#     b2 = Conv1D(128, kernel_size=15, strides=1, padding='same')(b2)
#     b2 = BatchNormalization()(b2)
#     b2 = Activation('relu')(b2)
#     b2 = se_block_1d(b2, reduction=8)

#     # 合并双频带并做最后一次通道重标定
#     out = Concatenate(axis=-1)([b1, b2])
#     out = se_block_1d(out, reduction=16)
#     out = Dropout(0.4)(out)
#     return out


# def create_hybrid_model(raw_shape=(3000, 3), segment_shape=(30, 13), num_classes=5):
#     """
#     【终极版本：全链路 SE 增强双向循环网络】
#     """
#     # ==========================================
#     # 分支 1：感知流 (波形 -> 嵌套SE的MRCNN -> BiLSTM)
#     # ==========================================
#     input_raw = Input(shape=raw_shape, name='raw_input')

#     x1 = se_multi_channel_extractor(input_raw)

#     # 深度时序记忆
#     x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
#     x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)

#     cnn_features = Dense(128, activation='relu')(x1)

#     # ==========================================
#     # 分支 2：逻辑流 (生理切片 -> 切片级SE -> BiLSTM)
#     # ==========================================
#     input_segments = Input(shape=segment_shape, name='segment_input')

#     # 【核心放大】：在时序建模前，直接对 13 个专家特征进行 SE 加权！
#     # 彻底释放生理切片在不同睡眠阶段的区分潜力
#     x2 = se_block_1d(input_segments, reduction=2)

#     x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
#     x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)

#     lstm_features = Dense(64, activation='relu')(x2)

#     # ==========================================
#     # 融合层与多任务输出
#     # ==========================================
#     merged = Concatenate(name='feature_fusion')([cnn_features, lstm_features])

#     # 【核心放大】：融合后进行全局 SE 权重分配
#     merged_se = se_block_dense(merged, reduction=4)

#     # 主分类头
#     dense = Dense(256, activation='relu')(merged_se)
#     dense = Dropout(0.5)(dense)
#     main_output = Dense(num_classes, activation='softmax', name='main_output')(dense)

# # 🚀 重构版辅助重建头：使用 Conv1D 替代超大全连接层，彻底告别 [512, 9000] OOM 崩溃
#     dense_ttt = Dense(256, activation='relu')(merged)
#     # 先将维度平稳映射到时间步长度（3000）
#     ttt_mid = Dense(raw_shape[0], activation='relu')(dense_ttt) 
#     # 增加一个伪通道轴，变为 (3000, 1)
#     ttt_mid = Reshape((raw_shape[0], 1))(ttt_mid)
#     # 最终通过 1x1 卷积无损膨胀至目标通道数 (3000, 3)
#     ttt_output = Conv1D(raw_shape[1], kernel_size=1, padding='same', name='ttt_output')(ttt_mid)

#     model = Model(inputs=[input_raw, input_segments], outputs=[main_output, ttt_output])

#     # 优化器
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss={
#             'main_output': CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
#             'ttt_output': 'mse'
#         },
#         loss_weights={'main_output': 1.0, 'ttt_output': 1e-6},
#         metrics={'main_output': 'accuracy'}
#     )

#     return model
