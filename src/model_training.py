

import numpy as np
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
from tensorflow.keras.initializers import Constant


CLINICAL_CONCEPT_NAMES = [
    "EEG Delta",
    "EEG Theta",
    "EEG Alpha",
    "EEG Sigma",
    "EEG Beta",
    "EOG Slow Eye",
    "EOG Rapid Eye",
    "EMG Low Tone",
    "EMG Muscle Activation",
]

STAGE_CONCEPT_PRIOR = np.array(
    [
        [0.04, 0.12, 0.18, 0.62, 0.04],  # EEG Delta
        [0.08, 0.32, 0.20, 0.16, 0.18],  # EEG Theta
        [0.34, 0.18, 0.04, 0.02, 0.08],  # EEG Alpha
        [0.03, 0.04, 0.36, 0.10, 0.02],  # EEG Sigma
        [0.18, 0.04, 0.02, 0.02, 0.10],  # EEG Beta
        [0.10, 0.26, 0.12, 0.05, 0.04],  # EOG Slow Eye
        [0.02, 0.02, 0.02, 0.01, 0.34],  # EOG Rapid Eye
        [0.02, 0.02, 0.02, 0.07, 0.30],  # EMG Low Tone
        [0.30, 0.12, 0.14, 0.05, 0.02],  # EMG Muscle Activation
    ],
    dtype=np.float32,
)

STAGE_CONCEPT_PRIOR = STAGE_CONCEPT_PRIOR / np.maximum(
    np.sum(STAGE_CONCEPT_PRIOR, axis=0, keepdims=True),
    1e-8,
)


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


def extract_clinical_concepts_from_segments(segment_tensor):
    eeg_delta = segment_tensor[:, :, 13:14]
    eog_delta = segment_tensor[:, :, 14:15]
    emg_delta = segment_tensor[:, :, 15:16]
    eeg_theta = segment_tensor[:, :, 16:17]
    eog_theta = segment_tensor[:, :, 17:18]
    eeg_alpha = segment_tensor[:, :, 19:20]
    eeg_sigma = segment_tensor[:, :, 22:23]
    eeg_beta = segment_tensor[:, :, 25:26]
    eog_beta = segment_tensor[:, :, 26:27]
    emg_beta = segment_tensor[:, :, 27:28]
    eog_energy = segment_tensor[:, :, 11:12]
    emg_energy = segment_tensor[:, :, 12:13]

    eog_slow = 0.70 * eog_delta + 0.30 * eog_theta
    eog_rapid = 0.55 * eog_energy + 0.30 * eog_theta + 0.15 * eog_beta
    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
    emg_low_tone = 1.0 / (1.0 + tf.nn.relu(emg_activation))

    return tf.concat(
        [
            eeg_delta,
            eeg_theta,
            eeg_alpha,
            eeg_sigma,
            eeg_beta,
            eog_slow,
            eog_rapid,
            emg_low_tone,
            emg_activation,
        ],
        axis=-1,
    )


def clinical_concept_branch(input_segments, num_classes):
    concepts = Lambda(
        extract_clinical_concepts_from_segments,
        name="clinical_concepts",
    )(input_segments)
    concepts = LayerNormalization(name="clinical_concept_norm")(concepts)

    concept_gate = GlobalAveragePooling1D(name="clinical_concept_pool")(concepts)
    concept_gate = Dense(32, activation="relu", name="clinical_concept_gate_hidden")(concept_gate)
    concept_gate = Dense(len(CLINICAL_CONCEPT_NAMES), activation="sigmoid", name="clinical_concept_gate")(concept_gate)
    concept_weighted = Multiply(name="clinical_concept_weighted")(
        [concepts, Reshape((1, len(CLINICAL_CONCEPT_NAMES)))(concept_gate)]
    )

    concept_sequence = Bidirectional(
        LSTM(16, return_sequences=True, dropout=0.1),
        name="clinical_concept_bilstm_1",
    )(concept_weighted)
    concept_sequence = Bidirectional(
        LSTM(8, return_sequences=False, dropout=0.1),
        name="clinical_concept_bilstm_2",
    )(concept_sequence)
    concept_features = Dense(64, activation="relu", name="clinical_concept_features")(concept_sequence)
    concept_features = LayerNormalization(name="clinical_concept_features_norm")(concept_features)

    concept_summary = GlobalAveragePooling1D(name="clinical_concept_summary")(concepts)
    concept_logits = Dense(
        num_classes,
        use_bias=False,
        kernel_initializer=Constant(STAGE_CONCEPT_PRIOR),
        name="clinical_concept_prior_logits",
    )(concept_summary)
    concept_output = Activation("softmax", name="concept_output")(concept_logits)

    return concept_features, concept_output


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
    concept_loss_weight=0.15,
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

    concept_features, concept_output = clinical_concept_branch(input_segments, num_classes)

    merged = Concatenate(name="feature_fusion")([cnn_features, segment_features, concept_features])
    merged = se_block_dense(merged, reduction=4)

    dense = Dense(256, activation="relu")(merged)
    dense = LayerNormalization()(dense)
    dense = Dropout(0.5)(dense)
    main_output = Dense(num_classes, activation="softmax", name="main_output")(dense)

    classification_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=1.5)
    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, attention_weights, concept_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "main_output": classification_loss,
            "channel_attention_weights": prior_guided_attention_loss,
            "concept_output": classification_loss,
        },
        loss_weights={
            "main_output": 1.0,
            "channel_attention_weights": attention_loss_weight,
            "concept_output": concept_loss_weight,
        },
        metrics={"main_output": "accuracy"},
    )
    return model



#########0.8213##########

#import numpy as np
#import tensorflow as tf
#from tensorflow.keras.layers import (
#    Activation,
#    Add,
#    BatchNormalization,
#    Bidirectional,
#    Concatenate,
#    Conv1D,
#    Dense,
#    Dropout,
#    GlobalAveragePooling1D,
#    Lambda,
#    LayerNormalization,
#    LSTM,
#    MaxPooling1D,
#    MultiHeadAttention,
#    Multiply,
#    Reshape,
#)
#from tensorflow.keras.losses import CategoricalFocalCrossentropy
#from tensorflow.keras.models import Model
#from tensorflow.keras.initializers import Constant
#
#
#CLINICAL_CONCEPT_NAMES = [
#    "EEG Delta",
#    "EEG Theta",
#    "EEG Alpha",
#    "EEG Sigma",
#    "EEG Beta",
#    "EOG Slow Eye",
#    "EOG Rapid Eye",
#    "EMG Low Tone",
#    "EMG Muscle Activation",
#]
#
#STAGE_CONCEPT_PRIOR = np.array(
#    [
#        [0.04, 0.12, 0.18, 0.62, 0.04],  # EEG Delta
#        [0.08, 0.32, 0.20, 0.16, 0.18],  # EEG Theta
#        [0.34, 0.18, 0.04, 0.02, 0.08],  # EEG Alpha
#        [0.03, 0.04, 0.36, 0.10, 0.02],  # EEG Sigma
#        [0.18, 0.04, 0.02, 0.02, 0.10],  # EEG Beta
#        [0.10, 0.26, 0.12, 0.05, 0.04],  # EOG Slow Eye
#        [0.02, 0.02, 0.02, 0.01, 0.34],  # EOG Rapid Eye
#        [0.02, 0.02, 0.02, 0.07, 0.30],  # EMG Low Tone
#        [0.30, 0.12, 0.14, 0.05, 0.02],  # EMG Muscle Activation
#    ],
#    dtype=np.float32,
#)
#
#STAGE_CONCEPT_PRIOR = STAGE_CONCEPT_PRIOR / np.maximum(
#    np.sum(STAGE_CONCEPT_PRIOR, axis=0, keepdims=True),
#    1e-8,
#)
#
#
#def se_block_1d(input_tensor, reduction=4):
#    channels = int(input_tensor.shape[-1])
#    squeeze = GlobalAveragePooling1D()(input_tensor)
#    excitation = Dense(max(1, channels // reduction), activation="relu")(squeeze)
#    excitation = Dense(channels, activation="sigmoid")(excitation)
#    excitation = Reshape((1, channels))(excitation)
#    return Multiply()([input_tensor, excitation])
#
#
#def se_block_dense(input_tensor, reduction=4):
#    channels = int(input_tensor.shape[-1])
#    excitation = Dense(max(1, channels // reduction), activation="relu")(input_tensor)
#    excitation = Dense(channels, activation="sigmoid")(excitation)
#    return Multiply()([input_tensor, excitation])
#
#
#def se_multi_channel_extractor(input_layer):
#    b1 = Conv1D(64, kernel_size=7, strides=2, padding="same")(input_layer)
#    b1 = BatchNormalization()(b1)
#    b1 = Activation("relu")(b1)
#    b1 = MaxPooling1D(pool_size=4, strides=2, padding="same")(b1)
#    b1 = se_block_1d(b1, reduction=4)
#
#    b1 = Conv1D(128, kernel_size=5, strides=1, padding="same")(b1)
#    b1 = BatchNormalization()(b1)
#    b1 = Activation("relu")(b1)
#    b1 = MaxPooling1D(pool_size=4, strides=5, padding="same")(b1)
#    b1 = se_block_1d(b1, reduction=8)
#
#    b2 = Conv1D(64, kernel_size=31, strides=4, padding="same")(input_layer)
#    b2 = BatchNormalization()(b2)
#    b2 = Activation("relu")(b2)
#    b2 = MaxPooling1D(pool_size=4, strides=5, padding="same")(b2)
#    b2 = se_block_1d(b2, reduction=4)
#
#    b2 = Conv1D(128, kernel_size=15, strides=1, padding="same")(b2)
#    b2 = BatchNormalization()(b2)
#    b2 = Activation("relu")(b2)
#    b2 = se_block_1d(b2, reduction=8)
#
#    out = Concatenate(axis=-1)([b1, b2])
#    out = se_block_1d(out, reduction=16)
#    out = Dropout(0.4)(out)
#    return out
#
#
#def transformer_encoder_block(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
#    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
#    x = Dropout(dropout)(x)
#    x = Add()([x, inputs])
#    x = LayerNormalization(epsilon=1e-6)(x)
#
#    residual = x
#    x = Dense(ff_dim, activation="relu")(x)
#    x = Dropout(dropout)(x)
#    x = Dense(inputs.shape[-1])(x)
#    x = Add()([x, residual])
#    return LayerNormalization(epsilon=1e-6)(x)
#
#
#def extract_clinical_concepts_from_segments(segment_tensor):
#    eeg_delta = segment_tensor[:, :, 13:14]
#    eog_delta = segment_tensor[:, :, 14:15]
#    emg_delta = segment_tensor[:, :, 15:16]
#    eeg_theta = segment_tensor[:, :, 16:17]
#    eog_theta = segment_tensor[:, :, 17:18]
#    eeg_alpha = segment_tensor[:, :, 19:20]
#    eeg_sigma = segment_tensor[:, :, 22:23]
#    eeg_beta = segment_tensor[:, :, 25:26]
#    eog_beta = segment_tensor[:, :, 26:27]
#    emg_beta = segment_tensor[:, :, 27:28]
#    eog_energy = segment_tensor[:, :, 11:12]
#    emg_energy = segment_tensor[:, :, 12:13]
#
#    eog_slow = 0.70 * eog_delta + 0.30 * eog_theta
#    eog_rapid = 0.55 * eog_energy + 0.30 * eog_theta + 0.15 * eog_beta
#    emg_activation = 0.70 * emg_energy + 0.30 * emg_beta
#    emg_low_tone = 1.0 / (1.0 + tf.nn.relu(emg_activation))
#
#    return tf.concat(
#        [
#            eeg_delta,
#            eeg_theta,
#            eeg_alpha,
#            eeg_sigma,
#            eeg_beta,
#            eog_slow,
#            eog_rapid,
#            emg_low_tone,
#            emg_activation,
#        ],
#        axis=-1,
#    )
#
#
#def clinical_concept_branch(input_segments, num_classes):
#    concepts = Lambda(
#        extract_clinical_concepts_from_segments,
#        name="clinical_concepts",
#    )(input_segments)
#    concepts = LayerNormalization(name="clinical_concept_norm")(concepts)
#
#    concept_gate = GlobalAveragePooling1D(name="clinical_concept_pool")(concepts)
#    concept_gate = Dense(32, activation="relu", name="clinical_concept_gate_hidden")(concept_gate)
#    concept_gate = Dense(len(CLINICAL_CONCEPT_NAMES), activation="sigmoid", name="clinical_concept_gate")(concept_gate)
#    concept_weighted = Multiply(name="clinical_concept_weighted")(
#        [concepts, Reshape((1, len(CLINICAL_CONCEPT_NAMES)))(concept_gate)]
#    )
#
#    concept_sequence = Bidirectional(
#        LSTM(16, return_sequences=True, dropout=0.1),
#        name="clinical_concept_bilstm_1",
#    )(concept_weighted)
#    concept_sequence = Bidirectional(
#        LSTM(8, return_sequences=False, dropout=0.1),
#        name="clinical_concept_bilstm_2",
#    )(concept_sequence)
#    concept_features = Dense(64, activation="relu", name="clinical_concept_features")(concept_sequence)
#    concept_features = LayerNormalization(name="clinical_concept_features_norm")(concept_features)
#
#    concept_summary = GlobalAveragePooling1D(name="clinical_concept_summary")(concepts)
#    concept_logits = Dense(
#        num_classes,
#        use_bias=False,
#        kernel_initializer=Constant(STAGE_CONCEPT_PRIOR),
#        name="clinical_concept_prior_logits",
#    )(concept_summary)
#    concept_output = Activation("softmax", name="concept_output")(concept_logits)
#
#    return concept_features, concept_output
#
#
#def prior_guided_attention_loss(y_true, y_pred):
#    """Soft physiological priors for [EEG, EOG, EMG] routing weights."""
#    w_mask = y_true[:, 0]
#    n1_mask = y_true[:, 1]
#    n2_n3_mask = y_true[:, 2] + y_true[:, 3]
#    rem_mask = y_true[:, 4]
#
#    eeg_weight = y_pred[:, 0]
#    eog_weight = y_pred[:, 1]
#    emg_weight = y_pred[:, 2]
#
#    def floor_penalty(weight, floor):
#        return tf.square(tf.nn.relu(floor - weight))
#
#    w_penalty = w_mask * floor_penalty(emg_weight, 0.40)
#    n1_penalty = n1_mask * (
#        floor_penalty(eeg_weight, 0.35) + 0.5 * floor_penalty(eog_weight, 0.25)
#    )
#    rem_penalty = rem_mask * (
#        floor_penalty(eog_weight, 0.35) + 0.5 * floor_penalty(emg_weight, 0.30)
#    )
#    deep_penalty = n2_n3_mask * floor_penalty(eeg_weight, 0.40)
#
#    return w_penalty + n1_penalty + rem_penalty + deep_penalty
#
#
#def dynamic_channel_routing(input_tensor):
#    eeg = Lambda(lambda x: x[:, :, 0:1], name="slice_eeg")(input_tensor)
#    eog = Lambda(lambda x: x[:, :, 1:2], name="slice_eog")(input_tensor)
#    emg = Lambda(lambda x: x[:, :, 2:3], name="slice_emg")(input_tensor)
#
#    eeg_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(eeg)
#    eog_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(eog)
#    emg_feat = Conv1D(16, kernel_size=7, padding="same", activation="relu")(emg)
#
#    stacked_feats = Lambda(lambda xs: tf.stack(xs, axis=-1), name="stack_channel_features")(
#        [eeg_feat, eog_feat, emg_feat]
#    )
#
#    pool = GlobalAveragePooling1D()(input_tensor)
#    channel_weights = Dense(16, activation="relu")(pool)
#    channel_weights = Dense(3, activation="softmax", name="channel_attention_weights")(channel_weights)
#
#    channel_weights_reshaped = Reshape((1, 1, 3))(channel_weights)
#    weighted_feats = Multiply()([stacked_feats, channel_weights_reshaped])
#    routed_output = Lambda(lambda x: tf.reduce_sum(x, axis=-1), name="routed_channel_sum")(weighted_feats)
#    return routed_output, channel_weights
#
#
#def create_hybrid_model_v2(
#    raw_shape=(3000, 3),
#    segment_shape=(30, 28),
#    num_classes=5,
#    learning_rate=0.001,
#    attention_loss_weight=0.01,
#    concept_loss_weight=0.15,
#):
#    input_raw = tf.keras.Input(shape=raw_shape, name="raw_input")
#    routed_raw, attention_weights = dynamic_channel_routing(input_raw)
#
#    x1 = Conv1D(raw_shape[-1], kernel_size=1, padding="same")(routed_raw)
#    x1 = se_multi_channel_extractor(x1)
#    x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
#    x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)
#    cnn_features = Dense(128, activation="relu")(x1)
#    cnn_features = LayerNormalization()(cnn_features)
#
#    input_segments = tf.keras.Input(shape=segment_shape, name="segment_input")
#    x2 = se_block_1d(input_segments, reduction=2)
#    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
#    x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)
#    segment_features = Dense(64, activation="relu")(x2)
#    segment_features = LayerNormalization()(segment_features)
#
#    concept_features, concept_output = clinical_concept_branch(input_segments, num_classes)
#
#    merged = Concatenate(name="feature_fusion")([cnn_features, segment_features, concept_features])
#    merged = se_block_dense(merged, reduction=4)
#
#    dense = Dense(256, activation="relu")(merged)
#    dense = LayerNormalization()(dense)
#    dense = Dropout(0.5)(dense)
#    main_output = Dense(num_classes, activation="softmax", name="main_output")(dense)
#
#    classification_loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=1.5)
#    model = Model(inputs=[input_raw, input_segments], outputs=[main_output, attention_weights, concept_output])
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#        loss={
#            "main_output": classification_loss,
#            "channel_attention_weights": prior_guided_attention_loss,
#            "concept_output": classification_loss,
#        },
#        loss_weights={
#            "main_output": 1.0,
#            "channel_attention_weights": attention_loss_weight,
#            "concept_output": concept_loss_weight,
#        },
#        metrics={"main_output": "accuracy"},
#    )
#    return model
