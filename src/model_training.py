import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense,
    BatchNormalization, LayerNormalization, Concatenate, GlobalAveragePooling1D,
    Reshape, Multiply, Add, Activation, Bidirectional, LSTM, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalFocalCrossentropy


# ==========================================
# 1. 基础注意力模块 (完全保留)
# ==========================================
def se_block_1d(input_tensor, reduction=4):
    channels = int(input_tensor.shape[-1])
    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(max(1, channels // reduction), activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([input_tensor, excitation])

def se_block_dense(input_tensor, reduction=4):
    channels = int(input_tensor.shape[-1])
    excitation = Dense(max(1, channels // reduction), activation='relu')(input_tensor)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    return Multiply()([input_tensor, excitation])


# ==========================================
# 2. 原版多通道提取器 (你的得意之作)
# ==========================================
def se_multi_channel_extractor(input_layer):
    # 分支 1：高频微事件 (EMG/纺锤波)
    b1 = Conv1D(64, kernel_size=7, strides=2, padding='same')(input_layer)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=4, strides=2, padding='same')(b1)
    b1 = se_block_1d(b1, reduction=4)  

    b1 = Conv1D(128, kernel_size=5, strides=1, padding='same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=4, strides=5, padding='same')(b1)
    b1 = se_block_1d(b1, reduction=8)  

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


# ==========================================
# 3. 进阶创新模块 (Transformer)
# ==========================================
def transformer_encoder_block(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs]) 
    x = LayerNormalization(epsilon=1e-6)(x) 

    res = x
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([x, res]) 
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


# ==========================================
# 4. 【核心创新】生理学先验引导的自定义损失函数
# ==========================================
def prior_guided_attention_loss(y_true, y_pred):
    """
    y_true: (batch, 5) ['W', 'N1', 'N2', 'N3', 'REM']
    y_pred: (batch, 3) ['EEG', 'EOG', 'EMG'] 通道权重
    """
    w_mask = y_true[:, 0]        
    n2_n3_mask = y_true[:, 2] + y_true[:, 3] 
    rem_mask = y_true[:, 4]      

    eeg_weight = y_pred[:, 0]
    eog_weight = y_pred[:, 1]
    emg_weight = y_pred[:, 2]

    # 惩罚机制：W期必须重看EMG，REM期重看EOG，深睡重看EEG
    w_penalty = w_mask * tf.square(0.4 - emg_weight)
    rem_penalty = rem_mask * tf.square(0.4 - eog_weight)
    deep_penalty = n2_n3_mask * tf.square(0.4 - eeg_weight)

    return tf.reduce_mean(w_penalty + rem_penalty + deep_penalty)

def dynamic_channel_routing(input_tensor):
    eeg = tf.expand_dims(input_tensor[:, :, 0], axis=-1)
    eog = tf.expand_dims(input_tensor[:, :, 1], axis=-1)
    emg = tf.expand_dims(input_tensor[:, :, 2], axis=-1)
    
    eeg_feat = Conv1D(16, kernel_size=7, padding='same', activation='relu')(eeg)
    eog_feat = Conv1D(16, kernel_size=7, padding='same', activation='relu')(eog)
    emg_feat = Conv1D(16, kernel_size=7, padding='same', activation='relu')(emg)
    
    stacked_feats = tf.stack([eeg_feat, eog_feat, emg_feat], axis=-1)
    
    pool = GlobalAveragePooling1D()(input_tensor) 
    channel_weights = Dense(16, activation='relu')(pool)
    channel_weights = Dense(3, activation='softmax', name='channel_attention_weights')(channel_weights) 
    
    channel_weights_reshaped = Reshape((1, 1, 3))(channel_weights) 
    weighted_feats = stacked_feats * channel_weights_reshaped
    
    routed_output = tf.reduce_sum(weighted_feats, axis=-1) 
    return routed_output, channel_weights


# ==========================================
# 5. 终极版模型构建主函数
# ==========================================
# ==========================================
# 5. 终极混血版模型 (通道路由先验 + BiLSTM 时序)
# ==========================================
def create_hybrid_model_v2(raw_shape=(3000, 3), segment_shape=(30, 13), num_classes=5):
    
    # --- 分支 1：感知流 ---
    input_raw = Input(shape=raw_shape, name='raw_input')
    
    # 🌟 创新点 1：保留动态通道路由
    routed_raw, attention_weights = dynamic_channel_routing(input_raw)

    x1 = Conv1D(raw_shape[-1], kernel_size=1, padding='same')(routed_raw) 
    x1 = se_multi_channel_extractor(x1)

    # 🌟 核心回归：换回你原版强大的 BiLSTM 捕捉时间先后顺序！
    x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
    x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)
    
    cnn_features = Dense(128, activation='relu')(x1)
    cnn_features = LayerNormalization()(cnn_features) 

    # --- 分支 2：逻辑流 ---
    input_segments = Input(shape=segment_shape, name='segment_input')

    x2 = se_block_1d(input_segments, reduction=2)
    
    # 🌟 核心回归：切片级也换回 BiLSTM
    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
    x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)

    lstm_features = Dense(64, activation='relu')(x2)
    lstm_features = LayerNormalization()(lstm_features)

    # --- 融合层与多任务输出 ---
    merged = Concatenate(name='feature_fusion')([cnn_features, lstm_features])
    merged_se = se_block_dense(merged, reduction=4)

    dense = Dense(256, activation='relu')(merged_se)
    dense = LayerNormalization()(dense) 
    dense = Dropout(0.5)(dense)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(dense)

    flat_dim = int(np.prod(raw_shape))
    dense_ttt = Dense(512, activation='relu')(merged)
    ttt_flat = Dense(flat_dim, activation='linear')(dense_ttt)
    ttt_output = Reshape(raw_shape, name='ttt_output')(ttt_flat)

    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, ttt_output, attention_weights])

    # 🌟 创新点 2：保留先验 Loss，保持 0.01 的温和约束，学习率回调到 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'main_output': CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
            'ttt_output': 'mse',
            'channel_attention_weights': prior_guided_attention_loss
        },
        loss_weights={
            'main_output': 1.0, 
            'ttt_output': 1e-5,
            'channel_attention_weights': 0.01 
        },
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

#     # 辅助重建头 (TTT 保留接口)
#     flat_dim = int(np.prod(raw_shape))
#     dense_ttt = Dense(512, activation='relu')(merged)
#     ttt_flat = Dense(flat_dim, activation='linear')(dense_ttt)
#     ttt_output = Reshape(raw_shape, name='ttt_output')(ttt_flat)

#     # 恢复为你最稳定的纯双输出模式 (分类 + TTT)
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