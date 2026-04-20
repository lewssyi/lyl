import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense,
    BatchNormalization, LSTM, Concatenate, GlobalAveragePooling1D,
    Reshape, multiply, Add, Activation
)
from tensorflow.keras.models import Model
# 引入 Focal Loss (解决极度不平衡分类的终极武器)
from tensorflow.keras.losses import CategoricalFocalCrossentropy


def squeeze_excitation_layer(input_x, out_dim, ratio=4):
    """
    【通道注意力模块】
    根据输入特征自动分配权重，识别 EEG/EOG/EMG 哪个通道对当前样本更重要
    """
    # Squeeze: 全局信息压缩
    squeeze = GlobalAveragePooling1D()(input_x)

    # Excitation: 学习通道间的非线性依赖关系
    excitation = Dense(out_dim // ratio, activation='relu')(squeeze)
    excitation = Dense(out_dim, activation='sigmoid')(excitation)
    excitation = Reshape((1, out_dim))(excitation)

    # Scale: 将权重乘回原特征图
    return multiply([input_x, excitation])


def multi_scale_block(input_layer, filters=48):
    """
    【多尺度卷积模块】
    并行使用长、中、短卷积核，全方位覆盖睡眠波形特征
    """
    # 长核：捕捉宏观慢波 (如 Delta 波)
    b1 = Conv1D(filters, kernel_size=64, padding='same', activation='relu')(input_layer)
    # 中核：捕捉典型特征 (如 Spindles)
    b2 = Conv1D(filters, kernel_size=32, padding='same', activation='relu')(input_layer)
    # 短核：捕捉高频噪声或微小事件 (如 EMG 爆发)
    b3 = Conv1D(filters, kernel_size=8, padding='same', activation='relu')(input_layer)

    out = Concatenate()([b1, b2, b3])
    out = BatchNormalization()(out)
    return out


def create_hybrid_model(raw_shape=(3000, 3), segment_shape=(30, 13), num_classes=5):
    """
    【增强版双流网络】引入注意力机制与多尺度融合 + Focal Loss 优化 + TTT辅助重建分支
    """

    # ==========================================
    # 分支 1：感知流 (改进版 CNN)
    # ==========================================
    input_raw = Input(shape=raw_shape, name='raw_input')

    # 第一层：多尺度特征提取
    x1 = multi_scale_block(input_raw, filters=32)  # 输出 32*3=96 个特征图

    # 核心：引入通道注意力，让模型自己决定关注哪个通道
    x1 = squeeze_excitation_layer(x1, out_dim=96)
    x1 = MaxPooling1D(pool_size=8)(x1)

    # 第二层：深度特征抽象
    x1 = Conv1D(filters=128, kernel_size=10, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = squeeze_excitation_layer(x1, out_dim=128)
    x1 = MaxPooling1D(pool_size=4)(x1)
    x1 = Dropout(0.3)(x1)

    # 第三层：压缩特征空间
    x1 = Conv1D(filters=128, kernel_size=5, activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    cnn_features = Flatten()(x1)

    # ==========================================
    # 分支 2：逻辑流 (LSTM) 保持对生理事件的捕捉
    # ==========================================
    input_segments = Input(shape=segment_shape, name='segment_input')

    x2 = LSTM(64, return_sequences=False)(input_segments)
    x2 = BatchNormalization()(x2)
    lstm_features = Dropout(0.3)(x2)

    # ==========================================
    # 融合层：多模态汇合
    # ==========================================
    merged = Concatenate(name='feature_fusion')([cnn_features, lstm_features])

    # 深度全连接层进行非线性映射 (用于任务 A: 主分类头)
    dense = Dense(256, activation='relu')(merged)
    dense = Dropout(0.5)(dense)
    dense = Dense(128, activation='relu')(dense)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(dense)

    # 深度全连接层进行非线性映射 (用于任务 B: TTT 重建辅助头)
    dense_ttt = Dense(512, activation='relu')(merged)
    ttt_flat = Dense(3000 * 3, activation='linear')(dense_ttt)
    ttt_output = Reshape((3000, 3), name='ttt_output')(ttt_flat)

    # 构建模型
    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, ttt_output])

    # 【核心修改】：使用 Focal Loss 替换普通的交叉熵
    # alpha 调节整体类别不平衡，gamma=2.0 会呈指数级放大困难样本（如N1/N3）的 loss，压制简单样本（如大量N2）
    # 【TTT降权保护】：ttt_output 权重设定为 1e-6，确保不会发生梯度淹没
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