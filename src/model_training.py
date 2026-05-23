import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Lambda,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    Reshape,
)
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.models import Model


def se_block_1d(input_tensor, reduction=4):
    channels = int(input_tensor.shape[-1])
    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(max(1, channels // reduction), activation="relu")(squeeze)
    excitation = Dense(channels, activation="sigmoid")(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([input_tensor, excitation])


def se_block_dense(input_tensor, reduction=4):
    channels = int(input_tensor.shape[-1])
    excitation = Dense(max(1, channels // reduction), activation="relu")(input_tensor)
    excitation = Dense(channels, activation="sigmoid")(excitation)
    return Multiply()([input_tensor, excitation])


def se_multi_channel_extractor(input_layer):
    b1 = Conv1D(64, kernel_size=7, strides=2, padding="same")(input_layer)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = MaxPooling1D(pool_size=4, strides=2, padding="same")(b1)
    b1 = se_block_1d(b1, reduction=4)

    b1 = Conv1D(128, kernel_size=5, strides=1, padding="same")(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = MaxPooling1D(pool_size=4, strides=5, padding="same")(b1)
    b1 = se_block_1d(b1, reduction=8)

    b2 = Conv1D(64, kernel_size=31, strides=4, padding="same")(input_layer)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = MaxPooling1D(pool_size=4, strides=5, padding="same")(b2)
    b2 = se_block_1d(b2, reduction=4)

    b2 = Conv1D(128, kernel_size=15, strides=1, padding="same")(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = se_block_1d(b2, reduction=8)

    out = Concatenate(axis=-1)([b1, b2])
    out = se_block_1d(out, reduction=16)
    out = Dropout(0.4)(out)
    return out


def transformer_encoder_block(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    residual = x
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([x, residual])
    return LayerNormalization(epsilon=1e-6)(x)


def prior_guided_attention_loss(y_true, y_pred):
    """Soft physiological priors for [EEG, EOG, EMG] routing weights."""
    w_mask = y_true[:, 0]
    n1_mask = y_true[:, 1]
    n2_n3_mask = y_true[:, 2] + y_true[:, 3]
    rem_mask = y_true[:, 4]

    eeg_weight = y_pred[:, 0]
    eog_weight = y_pred[:, 1]
    emg_weight = y_pred[:, 2]

    def floor_penalty(weight, floor):
        return tf.square(tf.nn.relu(floor - weight))

    w_penalty = w_mask * floor_penalty(emg_weight, 0.40)
    n1_penalty = n1_mask * (
        floor_penalty(eeg_weight, 0.35) + 0.5 * floor_penalty(eog_weight, 0.25)
    )
    rem_penalty = rem_mask * (
        floor_penalty(eog_weight, 0.35) + 0.5 * floor_penalty(emg_weight, 0.30)
    )
    deep_penalty = n2_n3_mask * floor_penalty(eeg_weight, 0.40)

    return w_penalty + n1_penalty + rem_penalty + deep_penalty


def dynamic_channel_routing(input_tensor):
    eeg = Lambda(lambda x: x[:, :, 0:1], name="slice_eeg")(input_tensor)
    eog = Lambda(lambda x: x[:, :, 1:2], name="slice_eog")(input_tensor)
    emg = Lambda(lambda x: x[:, :, 2:3], name="slice_emg")(input_tensor)

    eeg_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(eeg)
    eog_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(eog)
    emg_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(emg)

    stacked_feats = Lambda(lambda xs: tf.stack(xs, axis=-1), name="stack_channel_features")(
        [eeg_feat, eog_feat, emg_feat]
    )

    pool = GlobalAveragePooling1D()(input_tensor)
    channel_weights = Dense(16, activation="relu")(pool)
    channel_weights = Dense(3, activation="softmax", name="channel_attention_weights")(channel_weights)

    channel_weights_reshaped = Reshape((1, 1, 3))(channel_weights)
    weighted_feats = Multiply()([stacked_feats, channel_weights_reshaped])
    routed_output = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name="routed_channel_sum")(weighted_feats)
    return routed_output, channel_weights


def create_hybrid_model_v2(
    raw_shape=(3000, 3),
    segment_shape=(30, 28),
    num_classes=5,
    learning_rate=0.001,
    attention_loss_weight=0.01,
):
    input_raw = tf.keras.Input(shape=raw_shape, name="raw_input")
    routed_raw, attention_weights = dynamic_channel_routing(input_raw)

    x1 = Conv1D(raw_shape[-1], kernel_size=1, padding="same")(routed_raw)
    x1 = se_multi_channel_extractor(x1)
    x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
    x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)
    cnn_features = Dense(128, activation="relu")(x1)
    cnn_features = LayerNormalization()(cnn_features)

    input_segments = tf.keras.Input(shape=segment_shape, name="segment_input")
    x2 = se_block_1d(input_segments, reduction=2)
    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
    x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)
    segment_features = Dense(64, activation="relu")(x2)
    segment_features = LayerNormalization()(segment_features)

    merged = Concatenate(name="feature_fusion")([cnn_features, segment_features])
    merged = se_block_dense(merged, reduction=4)

    dense = Dense(256, activation="relu")(merged)
    dense = LayerNormalization()(dense)
    dense = Dropout(0.5)(dense)
    main_output = Dense(num_classes, activation="softmax", name="main_output")(dense)

    classification_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=1.5)
    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, attention_weights])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "main_output": classification_loss,
            "channel_attention_weights": prior_guided_attention_loss,
        },
        loss_weights={
            "main_output": 1.0,
            "channel_attention_weights": attention_loss_weight,
        },
        metrics={"main_output": "accuracy"},
    )
    return model
