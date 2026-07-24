
import os

import numpy as np
import tensorflow as tf
try:
    from .xai.posthoc_freeze import assert_model_xai_loss_is_zero
except ImportError:  # run_main.py also supports importing src modules by path
    from xai.posthoc_freeze import assert_model_xai_loss_is_zero
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
    TimeDistributed,
)
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalFocalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant

try:
    from .clinical_concepts import (
        BASE_STAGE_CHANNEL_PRIOR,
        BASE_STAGE_CONCEPT_PRIOR,
        CONCEPT_CHANNEL_INDEX,
        CONCEPT_NAMES_V1,
        CONCEPT_NAMES_V2,
        concepts_from_segment_features,
        concepts_from_segment_tensor,
        normalize_concept_version,
    )
except ImportError:  # run_main.py also supports importing src modules by path
    from clinical_concepts import (
        BASE_STAGE_CHANNEL_PRIOR,
        BASE_STAGE_CONCEPT_PRIOR,
        CONCEPT_CHANNEL_INDEX,
        CONCEPT_NAMES_V1,
        CONCEPT_NAMES_V2,
        concepts_from_segment_features,
        concepts_from_segment_tensor,
        normalize_concept_version,
    )

CLINICAL_CONCEPT_NAMES = list(CONCEPT_NAMES_V1)
CLINICAL_CONCEPT_NAMES_V2 = list(CONCEPT_NAMES_V2)
STAGE_CONCEPT_PRIOR = BASE_STAGE_CONCEPT_PRIOR.copy()
STAGE_CHANNEL_PRIOR = BASE_STAGE_CHANNEL_PRIOR.copy()


def _normalize_clinical_concept_version(concept_version):
    return "v2" if normalize_concept_version(concept_version).endswith("_v2") else "v1"


def get_clinical_concept_names(concept_version="v1"):
    if _normalize_clinical_concept_version(concept_version) == "v2":
        return list(CLINICAL_CONCEPT_NAMES_V2)
    return list(CLINICAL_CONCEPT_NAMES)


class MixStyle(tf.keras.layers.Layer):
    """Training-only feature statistic mixing for domain-style augmentation."""

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "alpha": self.alpha, "eps": self.eps})
        return config

    def call(self, inputs, training=None):
        if self.p <= 0.0 or self.alpha <= 0.0:
            return inputs

        rank = inputs.shape.rank
        if rank is None or rank < 3:
            return inputs

        if training is False:
            return inputs
        if training is None:
            training = tf.keras.backend.learning_phase()

        def apply_if_sampled():
            should_mix = tf.random.uniform(()) < self.p
            return tf.cond(should_mix, self._apply_mixstyle(inputs), lambda: inputs)

        if isinstance(training, bool):
            return apply_if_sampled() if training else inputs
        return tf.cond(tf.cast(training, tf.bool), apply_if_sampled, lambda: inputs)

    def _apply_mixstyle(self, inputs):
        def apply():
            rank = inputs.shape.rank
            axes = list(range(1, rank - 1))
            mean = tf.reduce_mean(inputs, axis=axes, keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=axes, keepdims=True)
            std = tf.sqrt(variance + self.eps)
            normalized = (inputs - mean) / std

            batch_size = tf.shape(inputs)[0]
            permutation = tf.random.shuffle(tf.range(batch_size))
            shuffled_mean = tf.gather(mean, permutation)
            shuffled_std = tf.gather(std, permutation)

            gamma_1 = tf.random.gamma([batch_size], self.alpha, 1.0)
            gamma_2 = tf.random.gamma([batch_size], self.alpha, 1.0)
            lam = gamma_1 / tf.maximum(gamma_1 + gamma_2, self.eps)
            lam_shape = tf.concat(
                [[batch_size], tf.ones([rank - 1], dtype=tf.int32)],
                axis=0,
            )
            lam = tf.reshape(lam, lam_shape)

            mixed_mean = lam * mean + (1.0 - lam) * shuffled_mean
            mixed_std = lam * std + (1.0 - lam) * shuffled_std
            return normalized * mixed_std + mixed_mean

        return lambda: tf.cond(tf.shape(inputs)[0] > 1, apply, lambda: inputs)


class ClinicalConceptAdapter(tf.keras.layers.Layer):
    """Dataset-adaptive calibration layer for weak clinical concept tokens."""

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

    def build(self, input_shape):
        num_concepts = int(input_shape[-1])
        self.scale_delta = self.add_weight(
            name="scale_delta",
            shape=(num_concepts,),
            initializer="zeros",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(num_concepts,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        concepts = tf.cast(inputs, tf.float32)
        concepts = tf.nn.relu(concepts)
        concepts = concepts / tf.maximum(
            tf.reduce_max(concepts, axis=-1, keepdims=True),
            self.eps,
        )
        concepts = tf.clip_by_value(concepts, self.eps, 1.0 - self.eps)
        logits = tf.math.log(concepts) - tf.math.log1p(-concepts)
        scale = 1.0 + 0.50 * tf.tanh(self.scale_delta)
        return tf.sigmoid(logits * scale + self.bias)


class ClinicalConceptPrototypeClassifier(tf.keras.layers.Layer):
    """Classify stages by OT-aligning concept distributions to clinical prototypes."""

    def __init__(
            self,
            prototype_initializer,
            initial_temperature=8.0,
            trainable_temperature=True,
            trainable_prototypes=True,
            sinkhorn_epsilon=0.08,
            sinkhorn_iterations=8,
            cosine_weight=1.0,
            trainable_cosine_weight=True,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.prototype_initializer = np.asarray(prototype_initializer, dtype=np.float32)
        self.initial_temperature = float(initial_temperature)
        self.trainable_temperature = bool(trainable_temperature)
        self.trainable_prototypes = bool(trainable_prototypes)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.cosine_weight = float(cosine_weight)
        self.trainable_cosine_weight = bool(trainable_cosine_weight)
        self.eps = float(eps)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "prototype_initializer": self.prototype_initializer.tolist(),
                "initial_temperature": self.initial_temperature,
                "trainable_temperature": self.trainable_temperature,
                "trainable_prototypes": self.trainable_prototypes,
                "sinkhorn_epsilon": self.sinkhorn_epsilon,
                "sinkhorn_iterations": self.sinkhorn_iterations,
                "cosine_weight": self.cosine_weight,
                "trainable_cosine_weight": self.trainable_cosine_weight,
                "eps": self.eps,
            }
        )
        return config

    def build(self, input_shape):
        num_concepts = int(input_shape[-1])
        if self.prototype_initializer.ndim != 2 or self.prototype_initializer.shape[1] != num_concepts:
            raise ValueError(
                "prototype_initializer must have shape (num_classes, num_concepts), "
                f"got {self.prototype_initializer.shape} for {num_concepts} concepts"
            )
        prototype_distribution = np.asarray(self.prototype_initializer, dtype=np.float32)
        prototype_distribution = np.maximum(prototype_distribution, self.eps)
        prototype_distribution = prototype_distribution / np.maximum(
            np.sum(prototype_distribution, axis=1, keepdims=True),
            self.eps,
        )
        self.num_classes = int(prototype_distribution.shape[0])
        self.num_concepts = int(prototype_distribution.shape[1])
        self.prototype_logits = self.add_weight(
            name="prototype_logits",
            shape=self.prototype_initializer.shape,
            initializer=Constant(np.log(prototype_distribution)),
            trainable=self.trainable_prototypes,
        )
        self.log_temperature = self.add_weight(
            name="log_temperature",
            shape=(),
            initializer=Constant(np.log(max(self.initial_temperature, 1e-3))),
            trainable=self.trainable_temperature,
        )
        self.log_cosine_weight = self.add_weight(
            name="log_cosine_weight",
            shape=(),
            initializer=Constant(np.log(max(self.cosine_weight, 1e-3))),
            trainable=self.trainable_cosine_weight,
        )
        self.transport_cost = tf.constant(
            self._build_concept_transport_cost(self.num_concepts),
            dtype=tf.float32,
        )
        super().build(input_shape)

    @staticmethod
    def _build_concept_transport_cost(num_concepts):
        positions = np.arange(int(num_concepts), dtype=np.float32)
        ordinal_cost = np.abs(positions[:, None] - positions[None, :]) / max(int(num_concepts) - 1, 1)
        channel_index = np.asarray(CONCEPT_CHANNEL_INDEX[: int(num_concepts)], dtype=np.int32)
        if len(channel_index) < int(num_concepts):
            channel_index = np.arange(int(num_concepts), dtype=np.int32)
        channel_cost = (channel_index[:, None] != channel_index[None, :]).astype(np.float32)
        cost = 0.35 * ordinal_cost + 0.65 * channel_cost
        np.fill_diagonal(cost, 0.0)
        return (cost / max(float(np.max(cost)), 1e-6)).astype(np.float32)

    def _normalize_distribution(self, values):
        values = tf.nn.relu(tf.cast(values, tf.float32)) + self.eps
        return values / tf.maximum(tf.reduce_sum(values, axis=-1, keepdims=True), self.eps)

    def _sinkhorn_distance(self, q_distribution, prototype_distribution):
        cost = tf.cast(self.transport_cost, tf.float32)
        epsilon = max(float(self.sinkhorn_epsilon), self.eps)
        kernel = tf.exp(-cost / epsilon)
        q = q_distribution[:, None, :]
        p = prototype_distribution[None, :, :]
        u = tf.ones_like(q)
        v = tf.ones_like(p)
        for _ in range(max(1, int(self.sinkhorn_iterations))):
            kv = tf.einsum("ij,nkj->nki", kernel, v)
            u = tf.identity(q / tf.maximum(kv, self.eps))
            ktu = tf.einsum("ji,nkj->nki", kernel, u)
            v = tf.identity(p / tf.maximum(ktu, self.eps))

        transport_plan = (
            u[:, :, :, None]
            * kernel[None, None, :, :]
            * v[:, :, None, :]
        )
        return tf.reduce_sum(transport_plan * cost[None, None, :, :], axis=[2, 3])

    def call(self, inputs):
        q_distribution = self._normalize_distribution(inputs)
        leading_shape = tf.shape(q_distribution)[:-1]
        flat_q = tf.reshape(q_distribution, [-1, self.num_concepts])
        prototype_distribution = tf.nn.softmax(self.prototype_logits, axis=-1)

        sinkhorn_distance = self._sinkhorn_distance(flat_q, prototype_distribution)
        cosine = tf.linalg.matmul(
            tf.math.l2_normalize(flat_q, axis=-1),
            tf.math.l2_normalize(prototype_distribution, axis=-1),
            transpose_b=True,
        )
        transport_scale = tf.clip_by_value(tf.exp(self.log_temperature), 1.0, 30.0)
        cosine_scale = tf.clip_by_value(tf.exp(self.log_cosine_weight), 0.0, 30.0)
        logits = -transport_scale * sinkhorn_distance + cosine_scale * cosine
        return tf.reshape(logits, tf.concat([leading_shape, [self.num_classes]], axis=0))


class DynamicPhysiologicalConceptMasking(tf.keras.layers.Layer):
    """Use clinical concept activations as a multiplicative physiological mask."""

    def __init__(
            self,
            num_concepts=9,
            concept_dim=16,
            concept_prior_strength=None,
            mask_amplitude=1.0,
            prior_init_scale=0.05,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.mask_amplitude = float(np.clip(mask_amplitude, 0.05, 1.0))
        self.prior_init_scale = float(np.clip(prior_init_scale, 1e-4, 0.50))
        self.eps = float(eps)
        if concept_prior_strength is None:
            concept_prior_strength = np.ones(self.num_concepts, dtype=np.float32)
        concept_prior_strength = np.asarray(concept_prior_strength, dtype=np.float32).reshape(-1)
        if len(concept_prior_strength) != self.num_concepts:
            raise ValueError(
                "concept_prior_strength must align with num_concepts: "
                f"got {len(concept_prior_strength)} for {self.num_concepts}"
            )
        concept_prior_strength = concept_prior_strength / max(float(np.mean(concept_prior_strength)), self.eps)
        self.concept_prior_strength = np.clip(concept_prior_strength, 0.50, 1.75).astype(np.float32)
        self.intensity_heads = []
        for index, concept_name in enumerate(CLINICAL_CONCEPT_NAMES[:self.num_concepts]):
            safe_name = concept_name.lower().replace(" ", "_").replace("/", "_")
            self.intensity_heads.append(
                tf.keras.Sequential(
                    [
                        Dense(max(self.concept_dim * 2, 16), activation="swish"),
                        LayerNormalization(epsilon=1e-6),
                        Dense(1, activation="sigmoid"),
                    ],
                    name=f"dpcm_{index}_{safe_name}_concept_intensity",
                )
            )
        self.output_norm = LayerNormalization(epsilon=1e-6, name="dpcm_masked_feature_norm")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_concepts": self.num_concepts,
                "concept_dim": self.concept_dim,
                "concept_prior_strength": self.concept_prior_strength.tolist(),
                "mask_amplitude": self.mask_amplitude,
                "prior_init_scale": self.prior_init_scale,
                "eps": self.eps,
            }
        )
        return config

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        rng = np.random.default_rng(17)
        random_basis = rng.normal(0.0, 1.0, size=(self.num_concepts, feature_dim)).astype(np.float32)
        random_basis = random_basis / np.maximum(
            np.std(random_basis, axis=1, keepdims=True),
            self.eps,
        )
        prior_init = (
            self.prior_init_scale
            * self.concept_prior_strength[:, None]
            * random_basis
        ).astype(np.float32)
        self.concept_prior_bank = self.add_weight(
            name="concept_prior_bank",
            shape=(self.num_concepts, feature_dim),
            initializer=Constant(prior_init),
            trainable=True,
        )
        self.mask_bias = self.add_weight(
            name="mask_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        fused_features = tf.cast(inputs, tf.float32)
        intensities = []

        for index in range(self.num_concepts):
            intensity = self.intensity_heads[index](fused_features)
            intensities.append(intensity)

        concept_intensity = tf.concat(intensities, axis=-1)
        mask_logits = (
            tf.linalg.matmul(concept_intensity, self.concept_prior_bank)
            + self.mask_bias
        )
        mask_profile = tf.tanh(mask_logits)
        mask_multiplier = 1.0 + self.mask_amplitude * mask_profile
        masked_features = self.output_norm(fused_features * mask_multiplier)

        if training is not False:
            self.add_metric(
                tf.reduce_mean(concept_intensity),
                name="dpcm_mean_concept_intensity",
                aggregation="mean",
            )
            self.add_metric(
                tf.reduce_mean(mask_multiplier),
                name="dpcm_mask_multiplier_mean",
                aggregation="mean",
            )
            self.add_metric(
                tf.math.reduce_std(mask_multiplier),
                name="dpcm_mask_multiplier_std",
                aggregation="mean",
            )
        return masked_features, concept_intensity, mask_profile


class TransitionBoundaryMaskContrast(tf.keras.layers.Layer):
    """Encourage DPCM mask profiles to change when the boundary head is confident."""

    def __init__(
            self,
            loss_weight=0.015,
            margin=0.35,
            activation_threshold=0.35,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_weight = float(loss_weight)
        self.margin = float(margin)
        self.activation_threshold = float(np.clip(activation_threshold, 0.0, 0.95))
        self.eps = float(eps)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "loss_weight": self.loss_weight,
                "margin": self.margin,
                "activation_threshold": self.activation_threshold,
                "eps": self.eps,
            }
        )
        return config

    def call(self, inputs, training=None):
        mask_profile, boundary_output = inputs
        if self.loss_weight <= 0.0 or training is False:
            return boundary_output

        left = tf.math.l2_normalize(mask_profile[:, :-1, :], axis=-1)
        right = tf.math.l2_normalize(mask_profile[:, 1:, :], axis=-1)
        cosine = tf.reduce_sum(left * right, axis=-1)

        if boundary_output.shape[-1] is not None and int(boundary_output.shape[-1]) > 1:
            boundary_prob = tf.reduce_max(boundary_output[..., 1:], axis=-1)
        else:
            boundary_prob = tf.squeeze(boundary_output, axis=-1)
        boundary_prob = tf.cast(boundary_prob[:, 1:], tf.float32)
        boundary_weight = tf.stop_gradient(
            tf.nn.relu(boundary_prob - self.activation_threshold)
            / max(1.0 - self.activation_threshold, self.eps)
        )

        contrast_loss = boundary_weight * tf.nn.relu(cosine - self.margin)
        contrast_loss = tf.reduce_sum(contrast_loss) / tf.maximum(
            tf.reduce_sum(boundary_weight),
            self.eps,
        )
        self.add_loss(self.loss_weight * contrast_loss)
        self.add_metric(
            contrast_loss,
            name="dpcm_boundary_mask_contrast_loss",
            aggregation="mean",
        )
        self.add_metric(
            tf.reduce_sum(boundary_weight) / tf.cast(tf.size(boundary_weight), tf.float32),
            name="dpcm_boundary_contrast_active_rate",
            aggregation="mean",
        )
        return boundary_output


def get_stage_concept_prior(num_concepts=None, num_classes=None):
    prior = np.asarray(STAGE_CONCEPT_PRIOR, dtype=np.float32)
    if num_concepts is not None:
        prior = prior[: int(num_concepts)]
    if num_classes is not None:
        num_classes = int(num_classes)
        if num_classes > prior.shape[1]:
            extra = np.ones((prior.shape[0], num_classes - prior.shape[1]), dtype=np.float32)
            extra = extra / np.maximum(np.sum(extra, axis=0, keepdims=True), 1e-8)
            prior = np.concatenate([prior, extra], axis=1)
        prior = prior[:, :num_classes]
    return prior / np.maximum(np.sum(prior, axis=0, keepdims=True), 1e-8)


def get_stage_channel_prior(num_classes=None, num_channels=None):
    prior = np.asarray(STAGE_CHANNEL_PRIOR, dtype=np.float32)
    if num_classes is not None:
        num_classes = int(num_classes)
        if num_classes > prior.shape[0]:
            extra = np.ones((num_classes - prior.shape[0], prior.shape[1]), dtype=np.float32)
            extra = extra / np.maximum(np.sum(extra, axis=1, keepdims=True), 1e-8)
            prior = np.concatenate([prior, extra], axis=0)
        prior = prior[:num_classes]
    if num_channels is not None:
        prior = prior[:, : int(num_channels)]
    return prior / np.maximum(np.sum(prior, axis=1, keepdims=True), 1e-8)


def set_adaptive_priors(stage_concept_prior=None, stage_channel_prior=None):
    global STAGE_CONCEPT_PRIOR, STAGE_CHANNEL_PRIOR
    if stage_concept_prior is not None:
        concept_prior = np.asarray(stage_concept_prior, dtype=np.float32)
        if concept_prior.ndim != 2:
            raise ValueError(f"stage_concept_prior must be 2D, got {concept_prior.shape}")
        STAGE_CONCEPT_PRIOR = concept_prior / np.maximum(
            np.sum(concept_prior, axis=0, keepdims=True),
            1e-8,
        )
    if stage_channel_prior is not None:
        channel_prior = np.asarray(stage_channel_prior, dtype=np.float32)
        if channel_prior.ndim != 2:
            raise ValueError(f"stage_channel_prior must be 2D, got {channel_prior.shape}")
        STAGE_CHANNEL_PRIOR = channel_prior / np.maximum(
            np.sum(channel_prior, axis=1, keepdims=True),
            1e-8,
        )


def reset_adaptive_priors():
    set_adaptive_priors(BASE_STAGE_CONCEPT_PRIOR, BASE_STAGE_CHANNEL_PRIOR)


def _labels_to_indices(labels):
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return np.argmax(labels, axis=1).astype(np.int32)
    return labels.astype(np.int32).ravel()


def _normalize_rows(values, fallback=None):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    row_sums = np.sum(values, axis=1, keepdims=True)
    normalized = values / np.maximum(row_sums, 1e-8)
    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float32)
        empty_rows = (row_sums[:, 0] <= 1e-8)
        normalized[empty_rows] = fallback[empty_rows]
    return normalized.astype(np.float32)


def _coerce_channel_targets(values, n_samples, n_channels=3, name="channel targets"):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        if len(values) != n_channels:
            raise ValueError(f"{name} must have {n_channels} channels, got shape {values.shape}")
        if n_samples != 1:
            values = np.tile(values[np.newaxis, :], (n_samples, 1))
        else:
            values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape != (n_samples, n_channels):
        raise ValueError(
            f"{name} must have shape ({n_samples}, {n_channels}), got {values.shape}"
        )
    return _normalize_rows(values)


def _apply_channel_quality_to_targets(channel_targets, channel_quality=None, quality_floor=0.10, fallback=None):
    channel_targets = np.asarray(channel_targets, dtype=np.float32)
    if channel_quality is None:
        return _normalize_rows(channel_targets, fallback=fallback)

    channel_quality = np.asarray(channel_quality, dtype=np.float32)
    if channel_quality.ndim == 1:
        channel_quality = channel_quality[np.newaxis, :]
    if channel_quality.shape != channel_targets.shape:
        raise ValueError(
            "channel_quality must match channel target shape "
            f"{channel_targets.shape}, got {channel_quality.shape}"
        )
    quality_floor = float(np.clip(quality_floor, 0.0, 1.0))
    gated_targets = channel_targets * np.clip(channel_quality, quality_floor, 1.0)
    return _normalize_rows(gated_targets, fallback=fallback)


def _normalize_component_weights(weights, expected_size, name):
    weights = np.asarray(weights, dtype=np.float32).ravel()
    if weights.shape[0] != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values, got {weights.shape[0]}")
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-8:
        raise ValueError(f"{name} must contain at least one positive value")
    return (weights / total).astype(np.float32)


def normalize_feedback_mode(mode):
    mode = str(mode or "clinical").strip().lower().replace("-", "_")
    aliases = {
        "none": "baseline",
        "off": "baseline",
        "no_feedback": "baseline",
        "baseline": "baseline",
        "clinical": "clinical",
        "clinical_feedback": "clinical",
        "current": "clinical",
        "self_lime": "self_lime",
        "self_lime_dccs": "self_lime",
        "lime": "self_lime",
        "lime_dccs": "self_lime",
        "clinical_calibrated_lime": "clinical_calibrated_lime_dccs",
        "clinically_calibrated_lime": "clinical_calibrated_lime_dccs",
        "clinical_calibrated_lime_dccs": "clinical_calibrated_lime_dccs",
        "clinically_calibrated_lime_dccs": "clinical_calibrated_lime_dccs",
    }
    if mode not in aliases:
        valid_modes = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown explanation feedback mode {mode!r}. Use one of: {valid_modes}")
    return aliases[mode]


def _scale_rows_to_unit(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    row_max = np.max(values, axis=1, keepdims=True)
    return (values / np.maximum(row_max, 1e-8)).astype(np.float32)


def _segment_feature(segment_features, feature_idx):
    if feature_idx >= segment_features.shape[-1]:
        return np.zeros(segment_features.shape[:2], dtype=np.float32)
    return segment_features[:, :, feature_idx].astype(np.float32)


def extract_clinical_concept_targets_from_segment_features(
    segment_features,
    labels=None,
    concept_prior_weight=0.0,
    concept_version="v1",
):
    if _normalize_clinical_concept_version(concept_version) == "v2":
        return extract_clinical_concept_targets_v2_from_segment_features(
            segment_features,
            labels=labels,
            concept_prior_weight=concept_prior_weight,
        )
    return extract_clinical_concept_targets_v1_from_segment_features(
        segment_features,
        labels=labels,
        concept_prior_weight=concept_prior_weight,
    )


def extract_clinical_concept_targets_v1_from_segment_features(
    segment_features,
    labels=None,
    concept_prior_weight=0.0,
):
    """Build segment-proxy targets; label conditioning is an explicit ablation."""
    concept_profile = concepts_from_segment_features(
        segment_features,
        version="segment_proxy_v1",
        reduce_segments=True,
    )

    if labels is not None and concept_prior_weight > 0:
        stage_concept_prior = get_stage_concept_prior(num_concepts=concept_profile.shape[1])
        label_indices = np.clip(_labels_to_indices(labels), 0, stage_concept_prior.shape[1] - 1)
        stage_prior = stage_concept_prior[:, label_indices].T
        stage_prior = _scale_rows_to_unit(stage_prior)
        weight = float(np.clip(concept_prior_weight, 0.0, 1.0))
        concept_profile = (1.0 - weight) * concept_profile + weight * stage_prior

    return np.clip(concept_profile, 0.0, 1.0).astype(np.float32)


def extract_clinical_concept_targets_v2_from_segment_features(
    segment_features,
    labels=None,
    concept_prior_weight=0.0,
):
    """Build the versioned V2 segment proxy through the canonical module."""
    concept_profile = concepts_from_segment_features(
        segment_features,
        version="segment_proxy_v2",
        reduce_segments=True,
    )

    if labels is not None and concept_prior_weight > 0:
        stage_concept_prior = get_stage_concept_prior(num_concepts=concept_profile.shape[1])
        label_indices = np.clip(_labels_to_indices(labels), 0, stage_concept_prior.shape[1] - 1)
        stage_prior = stage_concept_prior[:, label_indices].T
        stage_prior = _scale_rows_to_unit(stage_prior)
        weight = float(np.clip(concept_prior_weight, 0.0, 1.0))
        concept_profile = (1.0 - weight) * concept_profile + weight * stage_prior

    return np.clip(concept_profile, 0.0, 1.0).astype(np.float32)


def build_adaptive_clinical_priors(
    segment_features,
    labels,
    num_classes=5,
    alpha=0.35,
    min_class_samples=100,
    contrast_power=0.50,
    reliability_power=0.50,
    concept_version="v1",
):
    """
    Estimate dataset-specific clinical priors from the training split only.

    The returned matrices keep the hand-written clinical prior as the anchor
    and blend in train-set evidence. This lets SleepEDF/ISRUC/HMC keep their
    own montage and scoring style without using validation or test labels.
    """
    labels = np.clip(_labels_to_indices(labels), 0, int(num_classes) - 1)
    concept_profile = extract_clinical_concept_targets_from_segment_features(
        segment_features,
        labels=None,
        concept_prior_weight=0.0,
        concept_version=concept_version,
    )
    n_concepts = concept_profile.shape[1]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    contrast_power = float(np.clip(contrast_power, 0.0, 2.0))
    reliability_power = float(np.clip(reliability_power, 0.0, 2.0))
    min_class_samples = max(1, int(min_class_samples))

    base_concept_prior = get_stage_concept_prior(
        num_concepts=n_concepts,
        num_classes=num_classes,
    )
    base_channel_prior = get_stage_channel_prior(num_classes=num_classes)

    estimated_concept_prior = base_concept_prior.copy()
    estimated_channel_prior = base_channel_prior.copy()
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.int32)
    class_adaptive_alpha = np.zeros(num_classes, dtype=np.float32)

    concept_channel_scores = np.zeros((len(labels), base_channel_prior.shape[1]), dtype=np.float32)
    for channel_idx in range(concept_channel_scores.shape[1]):
        concept_channel_scores[:, channel_idx] = np.sum(
            concept_profile[:, CONCEPT_CHANNEL_INDEX == channel_idx],
            axis=1,
        )
    global_concept_mean = np.mean(concept_profile, axis=0)
    global_channel_mean = np.mean(concept_channel_scores, axis=0)

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_count = int(np.sum(class_mask))
        if class_count < min_class_samples:
            continue

        concept_mean = np.mean(concept_profile[class_mask], axis=0)
        concept_mean = np.clip(concept_mean, 0.0, None)
        concept_contrast = concept_mean / np.maximum(global_concept_mean, 1e-6)
        concept_contrast = np.power(np.clip(concept_contrast, 0.25, 4.0), contrast_power)
        concept_column = concept_mean * concept_contrast
        concept_column = concept_column / np.maximum(np.sum(concept_column), 1e-8)

        sample_reliability = min(1.0, class_count / float(max(min_class_samples * 4, 1)))
        sample_reliability = float(np.power(sample_reliability, reliability_power))
        class_alpha = alpha * sample_reliability
        class_adaptive_alpha[class_idx] = class_alpha
        estimated_concept_prior[:, class_idx] = (
            (1.0 - class_alpha) * base_concept_prior[:, class_idx]
            + class_alpha * concept_column
        )

        channel_mean = np.mean(concept_channel_scores[class_mask], axis=0)
        channel_mean = np.clip(channel_mean, 0.0, None)
        channel_contrast = channel_mean / np.maximum(global_channel_mean, 1e-6)
        channel_contrast = np.power(np.clip(channel_contrast, 0.25, 4.0), contrast_power)
        channel_row = channel_mean * channel_contrast
        channel_row = channel_row / np.maximum(np.sum(channel_row), 1e-8)
        estimated_channel_prior[class_idx] = (
            (1.0 - class_alpha) * base_channel_prior[class_idx]
            + class_alpha * channel_row
        )

    estimated_concept_prior = estimated_concept_prior / np.maximum(
        np.sum(estimated_concept_prior, axis=0, keepdims=True),
        1e-8,
    )
    estimated_channel_prior = estimated_channel_prior / np.maximum(
        np.sum(estimated_channel_prior, axis=1, keepdims=True),
        1e-8,
    )
    return {
        "stage_concept_prior": estimated_concept_prior.astype(np.float32),
        "stage_channel_prior": estimated_channel_prior.astype(np.float32),
        "class_counts": class_counts,
        "class_adaptive_alpha": class_adaptive_alpha,
        "concept_prior_shift_l1": np.sum(
            np.abs(estimated_concept_prior - base_concept_prior),
            axis=0,
        ).astype(np.float32),
        "channel_prior_shift_l1": np.sum(
            np.abs(estimated_channel_prior - base_channel_prior),
            axis=1,
        ).astype(np.float32),
        "alpha": alpha,
        "contrast_power": contrast_power,
        "reliability_power": reliability_power,
        "min_class_samples": min_class_samples,
        "concept_version": _normalize_clinical_concept_version(concept_version),
    }


def build_concept_dccs_attention_targets(
    segment_features,
    labels,
    concept_prior_weight=0.25,
    channel_prior_weight=0.25,
    channel_quality=None,
    quality_floor=0.10,
    concept_version="v1",
):
    """Approximate DCCS targets from clinical concepts for attention feedback."""
    stage_channel_prior_matrix = get_stage_channel_prior()
    stage_concept_prior_matrix = get_stage_concept_prior()
    label_indices = np.clip(_labels_to_indices(labels), 0, stage_channel_prior_matrix.shape[0] - 1)
    concept_targets = extract_clinical_concept_targets_from_segment_features(
        segment_features,
        labels=label_indices,
        concept_prior_weight=concept_prior_weight,
        concept_version=concept_version,
    )
    concept_prior = stage_concept_prior_matrix[:, label_indices].T
    concept_scores = concept_targets * concept_prior

    channel_scores = np.zeros((len(label_indices), stage_channel_prior_matrix.shape[1]), dtype=np.float32)
    for channel_idx in range(channel_scores.shape[1]):
        channel_scores[:, channel_idx] = np.sum(
            concept_scores[:, CONCEPT_CHANNEL_INDEX == channel_idx],
            axis=1,
        )

    stage_channel_prior = stage_channel_prior_matrix[label_indices]
    channel_scores = _normalize_rows(channel_scores, fallback=stage_channel_prior)
    weight = float(np.clip(channel_prior_weight, 0.0, 1.0))
    channel_targets = (1.0 - weight) * channel_scores + weight * stage_channel_prior
    channel_targets = _apply_channel_quality_to_targets(
        channel_targets,
        channel_quality=channel_quality,
        quality_floor=quality_floor,
        fallback=stage_channel_prior,
    )
    return _normalize_rows(channel_targets, fallback=stage_channel_prior)


def build_clinically_calibrated_lime_dccs_targets(
    segment_features,
    labels,
    lime_dccs_targets,
    alpha=0.50,
    beta=0.25,
    gamma=0.25,
    concept_prior_weight=0.25,
    channel_quality=None,
    quality_floor=0.10,
    concept_version="v1",
):
    """Blend self LIME-DCCS, clinical channel prior, and concept-DCCS targets."""
    stage_channel_prior_matrix = get_stage_channel_prior()
    label_indices = np.clip(_labels_to_indices(labels), 0, stage_channel_prior_matrix.shape[0] - 1)
    n_samples = len(label_indices)
    clinical_channel_prior = stage_channel_prior_matrix[label_indices]
    lime_channel_targets = _coerce_channel_targets(
        lime_dccs_targets,
        n_samples=n_samples,
        n_channels=stage_channel_prior_matrix.shape[1],
        name="lime_dccs_targets",
    )
    concept_channel_targets = build_concept_dccs_attention_targets(
        segment_features,
        label_indices,
        concept_prior_weight=concept_prior_weight,
        channel_prior_weight=0.0,
        concept_version=concept_version,
    )
    weights = _normalize_component_weights(
        (alpha, beta, gamma),
        expected_size=3,
        name="clinical calibrated LIME-DCCS weights",
    )
    channel_targets = (
        weights[0] * lime_channel_targets
        + weights[1] * clinical_channel_prior
        + weights[2] * concept_channel_targets
    )
    channel_targets = _apply_channel_quality_to_targets(
        channel_targets,
        channel_quality=channel_quality,
        quality_floor=quality_floor,
        fallback=clinical_channel_prior,
    )
    return _normalize_rows(channel_targets, fallback=clinical_channel_prior)


def build_explanation_feedback_targets(
    segment_features,
    labels,
    num_classes=5,
    concept_prior_weight=0.25,
    channel_prior_weight=0.25,
    feedback_mode=None,
    lime_dccs_targets=None,
    feedback_weights=None,
    channel_quality=None,
    quality_floor=0.10,
    concept_version="v1",
):
    """Return targets for concept bottleneck and DCCS-style attention feedback."""
    include_metadata = (
        feedback_mode is not None
        or lime_dccs_targets is not None
        or feedback_weights is not None
        or channel_quality is not None
    )
    feedback_mode = normalize_feedback_mode(feedback_mode or "clinical")
    label_indices = np.clip(_labels_to_indices(labels), 0, num_classes - 1)
    stage_channel_prior_matrix = get_stage_channel_prior(num_classes=num_classes)
    clinical_channel_prior = stage_channel_prior_matrix[
        np.clip(label_indices, 0, stage_channel_prior_matrix.shape[0] - 1)
    ]
    concept_targets = extract_clinical_concept_targets_from_segment_features(
        segment_features,
        labels=label_indices,
        concept_prior_weight=concept_prior_weight,
        concept_version=concept_version,
    )
    concept_channel_targets = build_concept_dccs_attention_targets(
        segment_features,
        label_indices,
        concept_prior_weight=concept_prior_weight,
        channel_prior_weight=channel_prior_weight,
        channel_quality=channel_quality,
        quality_floor=quality_floor,
        concept_version=concept_version,
    )
    lime_channel_targets = None
    component_weights = np.array([0.0, channel_prior_weight, 1.0 - channel_prior_weight], dtype=np.float32)

    if feedback_mode == "baseline":
        channel_targets = clinical_channel_prior
    elif feedback_mode == "clinical":
        channel_targets = concept_channel_targets
    elif feedback_mode == "self_lime":
        if lime_dccs_targets is None:
            raise ValueError("feedback_mode='self_lime' requires lime_dccs_targets")
        lime_channel_targets = _coerce_channel_targets(
            lime_dccs_targets,
            n_samples=len(label_indices),
            n_channels=stage_channel_prior_matrix.shape[1],
            name="lime_dccs_targets",
        )
        channel_targets = _apply_channel_quality_to_targets(
            lime_channel_targets,
            channel_quality=channel_quality,
            quality_floor=quality_floor,
            fallback=clinical_channel_prior,
        )
        component_weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif feedback_mode == "clinical_calibrated_lime_dccs":
        if lime_dccs_targets is None:
            raise ValueError(
                "feedback_mode='clinical_calibrated_lime_dccs' requires lime_dccs_targets"
            )
        if feedback_weights is None:
            feedback_weights = (0.50, 0.25, 0.25)
        component_weights = _normalize_component_weights(
            feedback_weights,
            expected_size=3,
            name="feedback_weights",
        )
        lime_channel_targets = _coerce_channel_targets(
            lime_dccs_targets,
            n_samples=len(label_indices),
            n_channels=stage_channel_prior_matrix.shape[1],
            name="lime_dccs_targets",
        )
        channel_targets = build_clinically_calibrated_lime_dccs_targets(
            segment_features,
            label_indices,
            lime_channel_targets,
            alpha=component_weights[0],
            beta=component_weights[1],
            gamma=component_weights[2],
            concept_prior_weight=concept_prior_weight,
            channel_quality=channel_quality,
            quality_floor=quality_floor,
            concept_version=concept_version,
        )
    else:
        raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")

    stage_targets = np.eye(num_classes, dtype=np.float32)[label_indices]
    attention_feedback_targets = np.concatenate([channel_targets, stage_targets], axis=1)
    result = {
        "concept_targets": concept_targets.astype(np.float32),
        "channel_targets": channel_targets.astype(np.float32),
        "clinical_channel_prior": clinical_channel_prior.astype(np.float32),
        "concept_channel_targets": concept_channel_targets.astype(np.float32),
        "lime_channel_targets": (
            lime_channel_targets.astype(np.float32)
            if lime_channel_targets is not None
            else np.zeros_like(channel_targets, dtype=np.float32)
        ),
        "attention_feedback_targets": attention_feedback_targets.astype(np.float32),
    }
    if include_metadata:
        result["feedback_mode"] = feedback_mode
        result["feedback_component_weights"] = component_weights.astype(np.float32)
    return result


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


def _risk_class_weight_vector(default=None):
    raw = os.environ.get("LFTS_RISK_CLASS_WEIGHTS", "").strip()
    if raw:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = list(default or [1.0, 1.0, 1.0, 1.0, 1.0])
    if not values:
        values = [1.0]
    return tf.constant(values, dtype=tf.float32)


def _gather_risk_class_weights(y_true, y_pred, default=None):
    class_weights = _risk_class_weight_vector(default=default)
    n_classes = tf.shape(y_pred)[-1]
    class_weights = class_weights[:n_classes]
    pad_len = tf.maximum(0, n_classes - tf.shape(class_weights)[0])
    class_weights = tf.concat([class_weights, tf.ones([pad_len], dtype=tf.float32)], axis=0)
    return tf.reduce_sum(tf.cast(y_true, tf.float32) * class_weights, axis=-1)


def risk_weighted_categorical_crossentropy_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    y_pred = y_pred / tf.maximum(tf.reduce_sum(y_pred, axis=-1, keepdims=True), 1e-6)
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return ce * _gather_risk_class_weights(y_true, y_pred)


def make_classification_loss(loss_name="focal"):
    loss_key = str(loss_name or "focal").strip().lower().replace("-", "_")
    if loss_key in {"risk_weighted_ce", "risk_weighted_crossentropy", "risk_weighted_categorical_crossentropy"}:
        return risk_weighted_categorical_crossentropy_loss
    if loss_key in {"risk_weighted_n1_margin_focal", "risk_n1_margin_focal", "hard_example_n1_margin_focal"}:
        return risk_weighted_n1_margin_focal_loss
    if loss_key in {
        "n1_margin_focal",
        "n1_aware_focal",
        "n1_hard_negative_focal",
        "clinical_margin_focal",
    }:
        return n1_margin_focal_loss
    if loss_key in {"focal", "categorical_focal", "categorical_focal_crossentropy"}:
        return CategoricalFocalCrossentropy(alpha=0.25, gamma=1.5)
    if loss_key in {"ce", "crossentropy", "categorical_crossentropy", "weighted_ce"}:
        return CategoricalCrossentropy()
    raise ValueError(
        "loss_name must be 'focal' or 'categorical_crossentropy', "
        f"got {loss_name!r}"
    )


def make_xai_alignment_loss(loss_name="normalized_mse"):
    """Build a loss over [B,T,30,3] local-importance logits."""
    key = str(loss_name or "normalized_mse").strip().lower().replace("-", "_")

    def distributions(y_true, logits):
        target = tf.nn.relu(tf.cast(y_true, tf.float32))
        shape = tf.shape(target)
        target_flat = tf.reshape(target, (-1, shape[-2] * shape[-1]))
        target_flat /= tf.maximum(tf.reduce_sum(target_flat, axis=-1, keepdims=True), 1e-8)
        logit_flat = tf.reshape(tf.cast(logits, tf.float32), (-1, shape[-2] * shape[-1]))
        return target_flat, tf.nn.softmax(logit_flat, axis=-1), logit_flat, shape

    def normalized_mse(y_true, y_pred):
        target, predicted, _logits, shape = distributions(y_true, y_pred)
        return tf.reshape(tf.reduce_mean(tf.square(target - predicted), axis=-1), shape[:-2])

    def kl_divergence(y_true, y_pred):
        target, predicted, _logits, shape = distributions(y_true, y_pred)
        loss = tf.reduce_sum(target * tf.math.log((target + 1e-8) / (predicted + 1e-8)), axis=-1)
        return tf.reshape(loss, shape[:-2])

    def js_divergence(y_true, y_pred):
        target, predicted, _logits, shape = distributions(y_true, y_pred)
        middle = 0.5 * (target + predicted)
        loss = 0.5 * tf.reduce_sum(target * tf.math.log((target + 1e-8) / (middle + 1e-8)), axis=-1)
        loss += 0.5 * tf.reduce_sum(predicted * tf.math.log((predicted + 1e-8) / (middle + 1e-8)), axis=-1)
        return tf.reshape(loss, shape[:-2])

    def rank_correlation_surrogate(y_true, y_pred):
        target, predicted, _logits, shape = distributions(y_true, y_pred)
        target -= tf.reduce_mean(target, axis=-1, keepdims=True)
        predicted -= tf.reduce_mean(predicted, axis=-1, keepdims=True)
        correlation = tf.reduce_sum(target * predicted, axis=-1) / tf.maximum(
            tf.norm(target, axis=-1) * tf.norm(predicted, axis=-1), 1e-8
        )
        return tf.reshape(1.0 - correlation, shape[:-2])

    def top_k_bce(y_true, y_pred):
        target, _predicted, logits, shape = distributions(y_true, y_pred)
        top_indices = tf.math.top_k(target, k=tf.minimum(10, tf.shape(target)[-1])).indices
        labels = tf.reduce_sum(tf.one_hot(top_indices, tf.shape(target)[-1]), axis=1)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=-1
        )
        return tf.reshape(loss, shape[:-2])

    losses = {
        "normalized_mse": normalized_mse,
        "mse": normalized_mse,
        "kl": kl_divergence,
        "kl_divergence": kl_divergence,
        "js": js_divergence,
        "js_divergence": js_divergence,
        "rank_correlation_surrogate": rank_correlation_surrogate,
        "top_k_bce": top_k_bce,
    }
    if key not in losses:
        raise ValueError(f"Unknown XAI alignment loss: {loss_name!r}")
    return losses[key]


def scheduled_xai_loss_weight(epoch, target_weight, warmup_epochs=3, ramp_epochs=5):
    """Return the prescribed zero-warmup then linear XAI-loss ramp."""
    epoch = int(epoch)
    target_weight = max(0.0, float(target_weight))
    warmup_epochs = max(0, int(warmup_epochs))
    ramp_epochs = max(1, int(ramp_epochs))
    if epoch < warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs + 1) / float(ramp_epochs))
    return target_weight * progress


def n1_margin_focal_loss(
        y_true,
        y_pred,
        n1_margin=0.08,
        gamma=1.75,
        n1_alpha=1.65,
        hard_negative_weight=0.35,
):
    """Focal CE plus explicit N1-vs-neighbor margin pressure.

    The model's weak point is usually not all minority stages equally; it is the
    W/N1/N2/REM confusion band.  This loss keeps the standard focal objective but
    adds a small differentiable penalty when true N1 loses to a neighboring
    class, or when non-N1 samples are pulled too close to N1.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    y_pred = y_pred / tf.maximum(tf.reduce_sum(y_pred, axis=-1, keepdims=True), 1e-6)

    pt = tf.reduce_sum(y_true * y_pred, axis=-1)
    class_alpha = tf.constant([1.00, n1_alpha, 1.05, 1.15, 1.15], dtype=tf.float32)
    class_alpha = class_alpha[:tf.shape(y_pred)[-1]]
    alpha_t = tf.reduce_sum(y_true * class_alpha, axis=-1)
    focal = -alpha_t * tf.pow(1.0 - pt, float(gamma)) * tf.math.log(pt)

    n1_true = y_true[..., 1]
    n1_pred = y_pred[..., 1]
    non_n1_pred = tf.concat([y_pred[..., :1], y_pred[..., 2:]], axis=-1)
    strongest_non_n1 = tf.reduce_max(non_n1_pred, axis=-1)
    n1_false_negative_margin = tf.square(
        tf.nn.relu(strongest_non_n1 + float(n1_margin) - n1_pred)
    )

    true_stage_prob = tf.reduce_sum(y_true * y_pred, axis=-1)
    non_n1_true = 1.0 - n1_true
    n1_false_positive_margin = tf.square(
        tf.nn.relu(n1_pred + float(n1_margin) - true_stage_prob)
    )
    margin_loss = (
        n1_true * n1_false_negative_margin
        + 0.60 * non_n1_true * n1_false_positive_margin
    )
    return focal + float(hard_negative_weight) * margin_loss


def risk_weighted_n1_margin_focal_loss(y_true, y_pred):
    class_weighted = _gather_risk_class_weights(
        y_true,
        y_pred,
        default=[1.0, 1.25, 1.10, 1.05, 1.15],
    )
    base = n1_margin_focal_loss(y_true, y_pred)
    return base * class_weighted


def n1_supervised_contrastive_loss(
        y_true,
        y_pred,
        temperature=0.12,
        eps=1e-8,
):
    """N1-focused supervised contrastive loss on sequence embeddings.

    Anchors are true N1 epochs.  Positives are other N1 epochs in the same
    batch/window pool; hard negatives are W, N2, and REM.  N3 is ignored for the
    denominator because it is not the main N1 confusion band.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    output_shape = tf.shape(y_true)[:-1]
    labels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    valid = tf.reduce_sum(y_true, axis=-1) > 0.5

    flat_labels = tf.reshape(labels, [-1])
    flat_valid = tf.reshape(valid, [-1])
    features = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    features = tf.math.l2_normalize(features, axis=-1)

    n_items = tf.shape(features)[0]
    n1_mask = tf.logical_and(tf.equal(flat_labels, 1), flat_valid)
    hard_negative_mask = tf.logical_and(
        tf.logical_or(
            tf.equal(flat_labels, 0),
            tf.logical_or(tf.equal(flat_labels, 2), tf.equal(flat_labels, 4)),
        ),
        flat_valid,
    )

    anchor_mask = n1_mask[:, tf.newaxis]
    positive_mask = tf.logical_and(anchor_mask, n1_mask[tf.newaxis, :])
    not_self = tf.logical_not(tf.eye(n_items, dtype=tf.bool))
    positive_mask = tf.logical_and(positive_mask, not_self)
    candidate_mask = tf.logical_and(
        anchor_mask,
        tf.logical_or(n1_mask[tf.newaxis, :], hard_negative_mask[tf.newaxis, :]),
    )
    candidate_mask = tf.logical_and(candidate_mask, not_self)

    logits = tf.matmul(features, features, transpose_b=True) / float(temperature)
    row_max = tf.reduce_max(tf.where(candidate_mask, logits, -1e9), axis=1, keepdims=True)
    logits = logits - tf.stop_gradient(row_max)
    masked_logits = tf.where(candidate_mask, logits, -1e9)
    log_denominator = tf.reduce_logsumexp(masked_logits, axis=1, keepdims=True)
    log_prob = logits - log_denominator

    positive_mask_f = tf.cast(positive_mask, tf.float32)
    positive_count = tf.reduce_sum(positive_mask_f, axis=1)
    anchor_loss = -tf.reduce_sum(log_prob * positive_mask_f, axis=1) / tf.maximum(positive_count, float(eps))
    anchor_loss = tf.where(
        tf.logical_and(n1_mask, positive_count > 0.0),
        anchor_loss,
        tf.zeros_like(anchor_loss),
    )
    return tf.reshape(anchor_loss, output_shape)


def n1_n2_pair_supervised_contrastive_loss(
        y_true,
        y_pred,
        temperature=0.12,
        eps=1e-8,
):
    """Pair-only supervised contrastive loss on the 128-d temporal representation.

    Only true N1/N2 epochs are eligible anchors. Same-stage samples are
    positives and the other target stage is the negative class. The returned
    tensor keeps the sequence shape so the existing per-epoch sample-weight
    path can mask padded epochs and disable the loss by phase.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    output_shape = tf.shape(y_true)[:-1]
    labels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    valid = tf.reduce_sum(y_true, axis=-1) > 0.5

    flat_labels = tf.reshape(labels, [-1])
    flat_valid = tf.reshape(valid, [-1])
    features = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    features = tf.math.l2_normalize(tf.cast(features, tf.float32), axis=-1)

    target_mask = tf.logical_and(
        flat_valid,
        tf.logical_or(tf.equal(flat_labels, 1), tf.equal(flat_labels, 2)),
    )
    n_items = tf.shape(features)[0]
    not_self = tf.logical_not(tf.eye(n_items, dtype=tf.bool))
    anchor_mask = target_mask[:, tf.newaxis]
    candidate_mask = tf.logical_and(anchor_mask, target_mask[tf.newaxis, :])
    candidate_mask = tf.logical_and(candidate_mask, not_self)
    positive_mask = tf.logical_and(
        candidate_mask,
        tf.equal(flat_labels[:, tf.newaxis], flat_labels[tf.newaxis, :]),
    )

    logits = tf.matmul(features, features, transpose_b=True) / float(temperature)
    row_max = tf.reduce_max(tf.where(candidate_mask, logits, -1e9), axis=1, keepdims=True)
    logits = logits - tf.stop_gradient(row_max)
    masked_logits = tf.where(candidate_mask, logits, -1e9)
    log_denominator = tf.reduce_logsumexp(masked_logits, axis=1, keepdims=True)
    log_prob = logits - log_denominator

    positive_mask_f = tf.cast(positive_mask, tf.float32)
    positive_count = tf.reduce_sum(positive_mask_f, axis=1)
    anchor_loss = -tf.reduce_sum(log_prob * positive_mask_f, axis=1) / tf.maximum(positive_count, float(eps))
    anchor_loss = tf.where(
        tf.logical_and(target_mask, positive_count > 0.0),
        anchor_loss,
        tf.zeros_like(anchor_loss),
    )
    return tf.reshape(anchor_loss, output_shape)


def n1_context_subtype_masks(y_true):
    """Return train-time N1 boundary subtype masks from within-window labels."""
    y_true = tf.cast(y_true, tf.float32)
    labels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    valid = tf.reduce_sum(y_true, axis=-1) > 0.5
    seq_len = tf.shape(labels)[1]
    center_valid = tf.logical_and(
        tf.greater(seq_len, 2),
        tf.ones_like(labels, dtype=tf.bool),
    )
    internal = tf.concat(
        [
            tf.zeros_like(labels[:, :1], dtype=tf.bool),
            tf.ones_like(labels[:, 1:-1], dtype=tf.bool),
            tf.zeros_like(labels[:, -1:], dtype=tf.bool),
        ],
        axis=1,
    )
    prev_labels = tf.concat([labels[:, :1], labels[:, :-1]], axis=1)
    next_labels = tf.concat([labels[:, 1:], labels[:, -1:]], axis=1)
    prev_valid = tf.concat([tf.zeros_like(valid[:, :1], dtype=tf.bool), valid[:, :-1]], axis=1)
    next_valid = tf.concat([valid[:, 1:], tf.zeros_like(valid[:, -1:], dtype=tf.bool)], axis=1)
    context_valid = tf.logical_and(tf.logical_and(valid, prev_valid), tf.logical_and(next_valid, internal))
    context_valid = tf.logical_and(context_valid, center_valid)
    is_n1 = tf.equal(labels, 1)

    prev_w = tf.equal(prev_labels, 0)
    next_w = tf.equal(next_labels, 0)
    prev_n1 = tf.equal(prev_labels, 1)
    next_n1 = tf.equal(next_labels, 1)
    prev_n2 = tf.equal(prev_labels, 2)
    next_n2 = tf.equal(next_labels, 2)
    prev_rem = tf.equal(prev_labels, 4)
    next_rem = tf.equal(next_labels, 4)

    w_adjacent = tf.logical_and(
        tf.logical_and(context_valid, is_n1),
        tf.logical_and(
            tf.logical_or(prev_w, next_w),
            tf.logical_and(tf.logical_or(prev_w, prev_n1), tf.logical_or(next_w, next_n1)),
        ),
    )
    n2_adjacent = tf.logical_and(
        tf.logical_and(context_valid, is_n1),
        tf.logical_and(
            tf.logical_or(prev_n2, next_n2),
            tf.logical_and(tf.logical_or(prev_n2, prev_n1), tf.logical_or(next_n2, next_n1)),
        ),
    )
    rem_adjacent = tf.logical_and(
        tf.logical_and(context_valid, is_n1),
        tf.logical_and(
            tf.logical_or(prev_rem, next_rem),
            tf.logical_and(tf.logical_or(prev_rem, prev_n1), tf.logical_or(next_rem, next_n1)),
        ),
    )
    return {
        "labels": labels,
        "valid": valid,
        "W_ADJACENT_N1": w_adjacent,
        "N2_ADJACENT_N1": n2_adjacent,
        "REM_ADJACENT_N1": rem_adjacent,
    }


def n1_context_subtype_local_contrastive_components(
        y_true,
        y_pred,
        temperature=0.12,
        eps=1e-8,
):
    """Subtype-balanced local InfoNCE for directed N1 boundary subtypes."""
    masks = n1_context_subtype_masks(y_true)
    labels = masks["labels"]
    valid = masks["valid"]
    features = tf.reshape(tf.cast(y_pred, tf.float32), [-1, tf.shape(y_pred)[-1]])
    features = tf.math.l2_normalize(features, axis=-1)
    flat_labels = tf.reshape(labels, [-1])
    flat_valid = tf.reshape(valid, [-1])
    n_items = tf.shape(features)[0]
    not_self = tf.logical_not(tf.eye(n_items, dtype=tf.bool))

    logits_pos = tf.matmul(features, features, transpose_b=True) / float(temperature)
    logits_neg = tf.matmul(features, tf.stop_gradient(features), transpose_b=True) / float(temperature)

    def subtype_loss(subtype_mask, negative_stage):
        anchor = tf.reshape(subtype_mask, [-1])
        negative = tf.logical_and(flat_valid, tf.equal(flat_labels, int(negative_stage)))
        anchor_col = anchor[:, tf.newaxis]
        positive_mask = tf.logical_and(anchor_col, anchor[tf.newaxis, :])
        positive_mask = tf.logical_and(positive_mask, not_self)
        negative_mask = tf.logical_and(anchor_col, negative[tf.newaxis, :])
        candidate_mask = tf.logical_or(positive_mask, negative_mask)
        candidate_logits = tf.where(
            positive_mask,
            logits_pos,
            tf.where(negative_mask, logits_neg, tf.fill(tf.shape(logits_pos), tf.constant(-1e9, tf.float32))),
        )
        row_max = tf.reduce_max(candidate_logits, axis=1, keepdims=True)
        candidate_logits = candidate_logits - tf.stop_gradient(row_max)
        positive_logits = tf.where(
            positive_mask,
            logits_pos - tf.stop_gradient(row_max),
            tf.fill(tf.shape(logits_pos), tf.constant(-1e9, tf.float32)),
        )
        log_num = tf.reduce_logsumexp(positive_logits, axis=1)
        log_den = tf.reduce_logsumexp(candidate_logits, axis=1)
        positive_count = tf.reduce_sum(tf.cast(positive_mask, tf.float32), axis=1)
        negative_count = tf.reduce_sum(tf.cast(negative_mask, tf.float32), axis=1)
        valid_anchor = tf.logical_and(anchor, tf.logical_and(positive_count >= 1.0, negative_count >= 1.0))
        per_anchor = -(log_num - log_den)
        per_anchor = tf.where(valid_anchor, per_anchor, tf.zeros_like(per_anchor))
        valid_count = tf.reduce_sum(tf.cast(valid_anchor, tf.float32))
        loss = tf.reduce_sum(per_anchor) / tf.maximum(valid_count, float(eps))
        loss = tf.where(valid_count > 0.0, loss, tf.zeros_like(loss))
        return loss, valid_count, tf.reduce_sum(tf.cast(anchor, tf.float32)), tf.reduce_sum(tf.cast(negative, tf.float32))

    w_loss, w_valid, w_anchor, w_negative = subtype_loss(masks["W_ADJACENT_N1"], 0)
    n2_loss, n2_valid, n2_anchor, n2_negative = subtype_loss(masks["N2_ADJACENT_N1"], 2)
    rem_loss, rem_valid, rem_anchor, rem_negative = subtype_loss(masks["REM_ADJACENT_N1"], 4)
    subtype_losses = tf.stack([w_loss, n2_loss, rem_loss])
    subtype_valid = tf.cast(tf.stack([w_valid, n2_valid, rem_valid]) > 0.0, tf.float32)
    valid_subtype_count = tf.reduce_sum(subtype_valid)
    context_loss = tf.reduce_sum(subtype_losses * subtype_valid) / tf.maximum(valid_subtype_count, 1.0)
    context_loss = tf.where(valid_subtype_count > 0.0, context_loss, tf.zeros_like(context_loss))
    return {
        "L_W": w_loss,
        "L_N2": n2_loss,
        "L_REM": rem_loss,
        "L_context": context_loss,
        "valid_subtype_count": valid_subtype_count,
        "W_valid_anchor_count": w_valid,
        "N2_valid_anchor_count": n2_valid,
        "REM_valid_anchor_count": rem_valid,
        "W_anchor_count": w_anchor,
        "N2_anchor_count": n2_anchor,
        "REM_anchor_count": rem_anchor,
        "W_negative_count": w_negative,
        "N2_negative_count": n2_negative,
        "REM_negative_count": rem_negative,
    }


def n1_context_subtype_local_contrastive_loss(
        y_true,
        y_pred,
        temperature=0.12,
        eps=1e-8,
):
    """Auxiliary loss preserving directed N1 context subtypes.

    W/N2/REM adjacent N1 anchors use same-subtype N1 positives and only the
    corresponding adjacent stage as stop-gradient negatives. Other N1
    subtypes, other N1 boundary modes, N3-adjacent N1, and padding are ignored.
    """
    output_shape = tf.shape(y_true)[:-1]
    components = n1_context_subtype_local_contrastive_components(
        y_true,
        y_pred,
        temperature=temperature,
        eps=eps,
    )
    return tf.fill(output_shape, components["L_context"])


def make_domain_alignment_loss(mode="coral"):
    mode_key = str(mode or "coral").strip().lower().replace("-", "_")

    def _prepare_features(y_pred):
        features = tf.cast(y_pred, tf.float32)
        features = tf.reshape(features, [-1, tf.shape(features)[-1]])
        return features

    def _prepare_domain_labels(y_true):
        target = tf.cast(y_true, tf.float32)
        if target.shape.rank is not None and target.shape.rank > 1 and target.shape[-1] == 2:
            target = target[..., 0]
        return tf.cast(tf.reshape(target, [-1]), tf.int32)

    def _prepare_domain_stage_labels(y_true):
        target = tf.cast(y_true, tf.float32)
        if target.shape.rank is not None and target.shape.rank > 1 and target.shape[-1] == 2:
            flat = tf.reshape(target, [-1, 2])
            return tf.cast(flat[:, 0], tf.int32), tf.cast(flat[:, 1], tf.int32), True
        labels = tf.cast(tf.reshape(target, [-1]), tf.int32)
        stages = tf.fill(tf.shape(labels), -1)
        return labels, stages, False

    def _coral_from_labels(labels, features):
        unique_labels, _ = tf.unique(labels)
        num_domains = tf.size(unique_labels)

        def compute():
            global_mean = tf.reduce_mean(features, axis=0, keepdims=True)
            centered = features - global_mean
            n_total = tf.cast(tf.maximum(tf.shape(features)[0] - 1, 1), tf.float32)
            global_cov = tf.matmul(centered, centered, transpose_a=True) / n_total

            def per_domain(label):
                mask = tf.equal(labels, label)
                domain_features = tf.boolean_mask(features, mask)
                count = tf.shape(domain_features)[0]

                def domain_loss():
                    domain_mean = tf.reduce_mean(domain_features, axis=0, keepdims=True)
                    domain_centered = domain_features - domain_mean
                    denom = tf.cast(tf.maximum(count - 1, 1), tf.float32)
                    domain_cov = tf.matmul(
                        domain_centered,
                        domain_centered,
                        transpose_a=True,
                    ) / denom
                    mean_loss = tf.reduce_mean(tf.square(domain_mean - global_mean))
                    cov_loss = tf.reduce_mean(tf.square(domain_cov - global_cov))
                    return mean_loss + cov_loss

                return tf.cond(
                    count > 1,
                    domain_loss,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            losses = tf.map_fn(per_domain, unique_labels, fn_output_signature=tf.float32)
            return tf.reduce_mean(losses)

        return tf.cond(
            num_domains > 1,
            compute,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def _mmd_from_labels(labels, features):
        unique_labels, _ = tf.unique(labels)
        num_domains = tf.size(unique_labels)

        def rbf_kernel(x, y):
            x_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
            y_sq = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
            distances = x_sq - 2.0 * tf.matmul(x, y, transpose_b=True) + tf.transpose(y_sq)
            feature_dim = tf.cast(tf.maximum(tf.shape(x)[-1], 1), tf.float32)
            return tf.exp(-tf.nn.relu(distances) / feature_dim)

        def compute():
            def per_domain(label):
                mask = tf.equal(labels, label)
                domain_features = tf.boolean_mask(features, mask)
                count = tf.shape(domain_features)[0]

                def domain_loss():
                    k_xx = tf.reduce_mean(rbf_kernel(domain_features, domain_features))
                    k_yy = tf.reduce_mean(rbf_kernel(features, features))
                    k_xy = tf.reduce_mean(rbf_kernel(domain_features, features))
                    return tf.nn.relu(k_xx + k_yy - 2.0 * k_xy)

                return tf.cond(
                    count > 0,
                    domain_loss,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            losses = tf.map_fn(per_domain, unique_labels, fn_output_signature=tf.float32)
            return tf.reduce_mean(losses)

        return tf.cond(
            num_domains > 1,
            compute,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def _coral_loss(y_true, y_pred):
        labels = _prepare_domain_labels(y_true)
        features = _prepare_features(y_pred)
        valid = labels >= 0
        labels = tf.boolean_mask(labels, valid)
        features = tf.boolean_mask(features, valid)
        return _coral_from_labels(labels, features)

    def _mmd_loss(y_true, y_pred):
        labels = _prepare_domain_labels(y_true)
        features = _prepare_features(y_pred)
        valid = labels >= 0
        labels = tf.boolean_mask(labels, valid)
        features = tf.boolean_mask(features, valid)
        return _mmd_from_labels(labels, features)

    def _stage_conditional_loss(base_loss_fn):
        def loss(y_true, y_pred):
            domain_labels, stage_labels, has_stage = _prepare_domain_stage_labels(y_true)
            features = _prepare_features(y_pred)

            def global_fallback():
                return base_loss_fn(domain_labels, features)

            def conditional():
                valid = tf.logical_and(domain_labels >= 0, stage_labels >= 0)
                unique_stages, _ = tf.unique(tf.boolean_mask(stage_labels, valid))

                def per_stage(stage_id):
                    mask = tf.logical_and(valid, tf.equal(stage_labels, stage_id))
                    stage_features = tf.boolean_mask(features, mask)
                    stage_domains = tf.boolean_mask(domain_labels, mask)
                    return tf.cond(
                        tf.shape(stage_features)[0] > 1,
                        lambda: base_loss_fn(stage_domains, stage_features),
                        lambda: tf.constant(0.0, dtype=tf.float32),
                    )

                def compute():
                    losses = tf.map_fn(per_stage, unique_stages, fn_output_signature=tf.float32)
                    return tf.reduce_mean(losses)

                return tf.cond(
                    tf.size(unique_stages) > 0,
                    compute,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            return tf.cond(tf.cast(has_stage, tf.bool), conditional, global_fallback)

        return loss

    if mode_key in {"coral", "deep_coral"}:
        return _coral_loss
    if mode_key in {"mmd", "rbf_mmd"}:
        return _mmd_loss
    if mode_key in {"stage_coral", "stage_conditional_coral", "conditional_coral"}:
        return _stage_conditional_loss(_coral_from_labels)
    if mode_key in {"stage_mmd", "stage_conditional_mmd", "conditional_mmd"}:
        return _stage_conditional_loss(_mmd_from_labels)
    raise ValueError(
        "domain alignment mode must be 'coral', 'mmd', 'stage_coral', or 'stage_mmd', "
        f"got {mode!r}"
    )


def _time_distributed_raw_features(
    raw_sequence,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
):
    x = TimeDistributed(
        Conv1D(64, kernel_size=7, strides=2, padding="same"),
        name="td_raw_conv1",
    )(raw_sequence)
    x = TimeDistributed(BatchNormalization(), name="td_raw_bn1")(x)
    x = Activation("relu", name="td_raw_relu1")(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4, strides=2, padding="same"), name="td_raw_pool1")(x)
    x = TimeDistributed(
        Conv1D(128, kernel_size=5, strides=1, padding="same"),
        name="td_raw_conv2",
    )(x)
    x = TimeDistributed(BatchNormalization(), name="td_raw_bn2")(x)
    x = Activation("relu", name="td_raw_relu2")(x)
    x = TimeDistributed(MaxPooling1D(pool_size=4, strides=2, padding="same"), name="td_raw_pool2")(x)
    x = TimeDistributed(
        Conv1D(128, kernel_size=3, padding="same"),
        name="td_raw_conv3",
    )(x)
    x = TimeDistributed(BatchNormalization(), name="td_raw_bn3")(x)
    x = Activation("relu", name="td_raw_relu3")(x)
    x = TimeDistributed(
        Bidirectional(LSTM(48, return_sequences=False, dropout=0.10)),
        name="td_raw_epoch_bilstm",
    )(x)
    if use_domain_mixstyle:
        x = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="sequence_raw_domain_mixstyle")(x)
    x = TimeDistributed(Dense(128, activation="relu"), name="td_raw_dense")(x)
    x = TimeDistributed(LayerNormalization(), name="td_raw_norm")(x)
    return Dropout(dropout_rate, name="raw_feature_dropout")(x)


def _time_distributed_segment_features(
    segment_sequence,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
):
    x = TimeDistributed(LayerNormalization(), name="td_segment_input_norm")(segment_sequence)
    x = TimeDistributed(
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.10)),
        name="td_segment_bilstm_1",
    )(x)
    x = TimeDistributed(
        Bidirectional(LSTM(16, return_sequences=False, dropout=0.10)),
        name="td_segment_bilstm_2",
    )(x)
    if use_domain_mixstyle:
        x = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="sequence_segment_domain_mixstyle")(x)
    x = TimeDistributed(Dense(64, activation="relu"), name="td_segment_dense")(x)
    x = TimeDistributed(LayerNormalization(), name="td_segment_norm")(x)
    return Dropout(dropout_rate, name="segment_feature_dropout")(x)


def _time_distributed_raw_multiscale_features(
    raw_sequence,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
):
    """Multi-scale epoch encoder without recurrent layers inside each epoch."""
    x = TimeDistributed(
        Conv1D(48, kernel_size=9, strides=2, padding="same"),
        name="td_v3_raw_stem_conv",
    )(raw_sequence)
    x = TimeDistributed(BatchNormalization(), name="td_v3_raw_stem_bn")(x)
    x = Activation("relu", name="td_v3_raw_stem_relu")(x)
    x = TimeDistributed(
        MaxPooling1D(pool_size=4, strides=2, padding="same"),
        name="td_v3_raw_stem_pool",
    )(x)

    branches = []
    for kernel_size in (5, 11, 21):
        branch = TimeDistributed(
            Conv1D(64, kernel_size=kernel_size, padding="same"),
            name=f"td_v3_raw_k{kernel_size}_conv1",
        )(x)
        branch = TimeDistributed(
            BatchNormalization(),
            name=f"td_v3_raw_k{kernel_size}_bn1",
        )(branch)
        branch = Activation("relu", name=f"td_v3_raw_k{kernel_size}_relu1")(branch)
        branch = TimeDistributed(
            Conv1D(64, kernel_size=3, strides=2, padding="same"),
            name=f"td_v3_raw_k{kernel_size}_conv2",
        )(branch)
        branch = TimeDistributed(
            BatchNormalization(),
            name=f"td_v3_raw_k{kernel_size}_bn2",
        )(branch)
        branch = Activation("relu", name=f"td_v3_raw_k{kernel_size}_relu2")(branch)
        branches.append(branch)

    x = Concatenate(name="td_v3_raw_multiscale_concat")(branches)
    x = TimeDistributed(
        Conv1D(160, kernel_size=5, strides=2, padding="same"),
        name="td_v3_raw_fuse_conv",
    )(x)
    x = TimeDistributed(BatchNormalization(), name="td_v3_raw_fuse_bn")(x)
    x = Activation("relu", name="td_v3_raw_fuse_relu")(x)
    x = TimeDistributed(
        GlobalAveragePooling1D(),
        name="td_v3_raw_global_pool",
    )(x)
    if use_domain_mixstyle:
        x = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="sequence_v3_raw_domain_mixstyle")(x)
    x = TimeDistributed(Dense(160, activation="relu"), name="td_v3_raw_dense")(x)
    x = TimeDistributed(LayerNormalization(), name="td_v3_raw_norm")(x)
    return Dropout(dropout_rate, name="v3_raw_feature_dropout")(x)



def _epoch_channel_domain_stats(values):
    mean = tf.reduce_mean(values, axis=2)
    centered = values - tf.expand_dims(mean, axis=2)
    std = tf.sqrt(tf.reduce_mean(tf.square(centered), axis=2) + 1e-5)
    mean_abs = tf.reduce_mean(tf.abs(values), axis=2)
    max_abs = tf.reduce_max(tf.abs(values), axis=2)
    return tf.concat([mean, tf.math.log(std + 1e-5), mean_abs, max_abs], axis=-1)


def _time_distributed_channel_residual_expert(
    raw_sequence,
    channel_index,
    channel_name,
    stem_kernel,
    branch_kernels,
    stem_filters=12,
    branch_filters=16,
    output_units=32,
):
    channel = Lambda(
        lambda x, idx=channel_index: x[:, :, :, idx:idx + 1],
        name=f"td_chan_resid_{channel_name}_slice",
    )(raw_sequence)
    x = TimeDistributed(
        Conv1D(stem_filters, kernel_size=stem_kernel, strides=2, padding="same"),
        name=f"td_chan_resid_{channel_name}_stem_conv",
    )(channel)
    x = TimeDistributed(BatchNormalization(), name=f"td_chan_resid_{channel_name}_stem_bn")(x)
    x = Activation("relu", name=f"td_chan_resid_{channel_name}_stem_relu")(x)
    x = TimeDistributed(
        MaxPooling1D(pool_size=4, strides=2, padding="same"),
        name=f"td_chan_resid_{channel_name}_stem_pool",
    )(x)

    branches = []
    for kernel_size in branch_kernels:
        branch = TimeDistributed(
            Conv1D(branch_filters, kernel_size=kernel_size, padding="same"),
            name=f"td_chan_resid_{channel_name}_k{kernel_size}_conv1",
        )(x)
        branch = TimeDistributed(
            BatchNormalization(),
            name=f"td_chan_resid_{channel_name}_k{kernel_size}_bn1",
        )(branch)
        branch = Activation("relu", name=f"td_chan_resid_{channel_name}_k{kernel_size}_relu1")(branch)
        branch = TimeDistributed(
            Conv1D(branch_filters, kernel_size=3, strides=2, padding="same"),
            name=f"td_chan_resid_{channel_name}_k{kernel_size}_conv2",
        )(branch)
        branch = TimeDistributed(
            BatchNormalization(),
            name=f"td_chan_resid_{channel_name}_k{kernel_size}_bn2",
        )(branch)
        branch = Activation("relu", name=f"td_chan_resid_{channel_name}_k{kernel_size}_relu2")(branch)
        branches.append(branch)

    x = Concatenate(name=f"td_chan_resid_{channel_name}_multiscale_concat")(branches)
    x = TimeDistributed(
        Conv1D(24, kernel_size=5, strides=2, padding="same"),
        name=f"td_chan_resid_{channel_name}_fuse_conv",
    )(x)
    x = TimeDistributed(BatchNormalization(), name=f"td_chan_resid_{channel_name}_fuse_bn")(x)
    x = Activation("relu", name=f"td_chan_resid_{channel_name}_fuse_relu")(x)
    x = TimeDistributed(
        GlobalAveragePooling1D(),
        name=f"td_chan_resid_{channel_name}_global_pool",
    )(x)

    stats = Lambda(
        _epoch_channel_domain_stats,
        name=f"td_chan_resid_{channel_name}_domain_stats",
    )(channel)
    stats = TimeDistributed(
        Dense(4, activation="relu"),
        name=f"td_chan_resid_{channel_name}_domain_stats_dense",
    )(stats)
    x = Concatenate(name=f"td_chan_resid_{channel_name}_feature_stats_concat")([x, stats])
    x = TimeDistributed(
        Dense(output_units, activation="relu"),
        name=f"td_chan_resid_{channel_name}_dense",
    )(x)
    return TimeDistributed(
        LayerNormalization(),
        name=f"td_chan_resid_{channel_name}_norm",
    )(x)


def _scale_channel_residual(values):
    return 0.10 * values


def _time_distributed_channel_specialized_residual_features(
    raw_sequence,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
):
    """Standard v3 raw encoder plus a lightweight zero-initialized channel-specialist residual."""
    base_features = _time_distributed_raw_multiscale_features(
        raw_sequence,
        dropout_rate=dropout_rate,
        use_domain_mixstyle=use_domain_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
    )
    channel_specs = [
        (0, "eeg", 9, (7, 21)),
        (1, "eog", 15, (15, 31)),
        (2, "emg", 5, (3, 7)),
    ]
    channel_count = raw_sequence.shape[-1]
    if channel_count is None:
        channel_count = 3
    experts = []
    for channel_index, channel_name, stem_kernel, branch_kernels in channel_specs[: int(channel_count)]:
        experts.append(
            _time_distributed_channel_residual_expert(
                raw_sequence,
                channel_index,
                channel_name,
                stem_kernel,
                branch_kernels,
            )
        )

    if len(experts) == 1:
        expert_features = experts[0]
    else:
        concat_features = Concatenate(name="td_chan_resid_feature_concat")(experts)
        gate_hidden = TimeDistributed(
            Dense(24, activation="relu"),
            name="td_chan_resid_gate_hidden",
        )(concat_features)
        gate_logits = TimeDistributed(
            Dense(len(experts)),
            name="td_chan_resid_gate_logits",
        )(gate_hidden)
        gate_probs = Activation("softmax", name="td_chan_resid_gate_probs")(gate_logits)
        weighted_features = []
        for idx, expert in enumerate(experts):
            gate_slice = Lambda(
                lambda x, i=idx: x[:, :, i:i + 1],
                name=f"td_chan_resid_gate_{idx}_slice",
            )(gate_probs)
            weighted_features.append(
                Multiply(name=f"td_chan_resid_weighted_{idx}")([expert, gate_slice])
            )
        weighted_concat = Concatenate(name="td_chan_resid_weighted_concat")(weighted_features)
        expert_features = Concatenate(name="td_chan_resid_fusion_context")(
            [concat_features, weighted_concat, gate_probs]
        )

    if use_domain_mixstyle:
        expert_features = MixStyle(
            p=mixstyle_p,
            alpha=mixstyle_alpha,
            name="sequence_chan_resid_domain_mixstyle",
        )(expert_features)
    expert_features = TimeDistributed(
        Dense(48, activation="relu"),
        name="td_chan_resid_fuse_hidden",
    )(expert_features)
    expert_features = TimeDistributed(
        LayerNormalization(),
        name="td_chan_resid_fuse_norm",
    )(expert_features)
    residual = TimeDistributed(
        Dense(160, kernel_initializer="zeros", bias_initializer="zeros"),
        name="td_chan_resid_zero_projection",
    )(expert_features)
    residual = Lambda(_scale_channel_residual, name="td_chan_resid_scale")(residual)
    return Add(name="td_chan_resid_raw_feature_add")([base_features, residual])

def _time_distributed_segment_multiscale_features(
    segment_sequence,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
):
    """Segment-feature encoder aligned with clinical time-frequency summaries."""
    x = TimeDistributed(LayerNormalization(), name="td_v3_segment_input_norm")(segment_sequence)
    branches = []
    for kernel_size in (3, 5, 9):
        branch = TimeDistributed(
            Conv1D(48, kernel_size=kernel_size, padding="same", activation="relu"),
            name=f"td_v3_segment_k{kernel_size}_conv1",
        )(x)
        branch = TimeDistributed(
            Conv1D(48, kernel_size=3, padding="same", activation="relu"),
            name=f"td_v3_segment_k{kernel_size}_conv2",
        )(branch)
        branches.append(branch)
    x = Concatenate(name="td_v3_segment_multiscale_concat")(branches)
    x = TimeDistributed(
        GlobalAveragePooling1D(),
        name="td_v3_segment_global_pool",
    )(x)
    if use_domain_mixstyle:
        x = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="sequence_v3_segment_domain_mixstyle")(x)
    x = TimeDistributed(Dense(96, activation="relu"), name="td_v3_segment_dense")(x)
    x = TimeDistributed(LayerNormalization(), name="td_v3_segment_norm")(x)
    return Dropout(dropout_rate, name="v3_segment_feature_dropout")(x)


class EpochLevelTFInteraction(tf.keras.layers.Layer):
    """Lightweight epoch-level interaction between raw and segment features."""

    def __init__(
        self,
        interaction_dim=128,
        num_heads=4,
        dropout_rate=0.10,
        seed=2024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.interaction_dim = int(interaction_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.seed = int(seed)
        self.raw_projection = Dense(
            self.interaction_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 1),
            name="raw_epoch_token_projection",
        )
        self.segment_projection = Dense(
            self.interaction_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 2),
            name="segment_epoch_token_projection",
        )
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.interaction_dim // max(1, self.num_heads)),
            dropout=self.dropout_rate,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 3),
            bias_initializer="zeros",
            name="epoch_domain_self_attention",
        )
        self.attention_dropout = Dropout(self.dropout_rate, name="epoch_domain_attention_dropout")
        self.attention_norm = LayerNormalization(name="epoch_domain_attention_norm")
        self.raw_delta_projection = None
        self.segment_delta_projection = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("EpochLevelTFInteraction expects [raw_features, segment_features].")
        raw_dim = int(input_shape[0][-1])
        segment_dim = int(input_shape[1][-1])
        self.raw_delta_projection = Dense(
            raw_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 4),
            name="raw_epoch_delta_projection",
        )
        self.segment_delta_projection = Dense(
            segment_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed + 5),
            name="segment_epoch_delta_projection",
        )
        self.domain_embedding = self.add_weight(
            name="epoch_domain_embedding",
            shape=(2, self.interaction_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=self.seed + 6),
            trainable=True,
        )
        self.gate_raw = self.add_weight(
            name="gate_raw",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        self.gate_segment = self.add_weight(
            name="gate_segment",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        raw_features, segment_features = inputs
        raw_token = self.raw_projection(raw_features)
        segment_token = self.segment_projection(segment_features)
        tokens = tf.stack([raw_token, segment_token], axis=2)
        tokens = tokens + tf.reshape(self.domain_embedding, (1, 1, 2, self.interaction_dim))

        token_shape = tf.shape(tokens)
        batch_size = token_shape[0]
        seq_len = token_shape[1]
        flat_tokens = tf.reshape(tokens, (-1, 2, self.interaction_dim))
        attended = self.attention(flat_tokens, flat_tokens, training=training)
        attended = self.attention_dropout(attended, training=training)
        interacted = self.attention_norm(flat_tokens + attended)
        interacted = tf.reshape(interacted, (batch_size, seq_len, 2, self.interaction_dim))

        raw_interacted = interacted[:, :, 0, :]
        segment_interacted = interacted[:, :, 1, :]
        raw_delta = self.raw_delta_projection(raw_interacted)
        segment_delta = self.segment_delta_projection(segment_interacted)
        raw_new = raw_features + tf.tanh(self.gate_raw) * raw_delta
        segment_new = segment_features + tf.tanh(self.gate_segment) * segment_delta
        return raw_new, segment_new

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "interaction_dim": self.interaction_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "seed": self.seed,
            }
        )
        return config


def apply_epoch_utf_lite_interaction(
    raw_features,
    segment_features,
    interaction_dim=128,
    num_heads=4,
    dropout_rate=0.10,
    seed=2024,
):
    return EpochLevelTFInteraction(
        interaction_dim=interaction_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        seed=seed,
        name="epoch_utf_lite_interaction",
    )([raw_features, segment_features])


class StageSpecificConceptRouter(tf.keras.layers.Layer):
    """Add bounded stage-specific clinical concept evidence to stage logits."""

    def __init__(
            self,
            clinical_prior,
            prior_delta_scale=0.10,
            expose_diagnostics=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        clinical_prior = np.asarray(clinical_prior, dtype=np.float32)
        if clinical_prior.ndim != 2:
            raise ValueError(f"clinical_prior must be 2D, got {clinical_prior.shape}")
        self.clinical_prior_value = clinical_prior
        self.num_classes = int(clinical_prior.shape[0])
        self.num_concepts = int(clinical_prior.shape[1])
        self.prior_delta_scale = float(prior_delta_scale)
        self.expose_diagnostics = bool(expose_diagnostics)

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("StageSpecificConceptRouter expects [base_logits, clinical_concepts].")
        concept_dim = int(input_shape[1][-1])
        logit_dim = int(input_shape[0][-1])
        if concept_dim != self.num_concepts:
            raise ValueError(
                "clinical_concepts dimension must match clinical_prior columns: "
                f"got {concept_dim}, expected {self.num_concepts}"
            )
        if logit_dim != self.num_classes:
            raise ValueError(
                "base_logits dimension must match clinical_prior rows: "
                f"got {logit_dim}, expected {self.num_classes}"
            )
        self.clinical_prior = self.add_weight(
            name="clinical_prior",
            shape=(self.num_classes, self.num_concepts),
            initializer=Constant(self.clinical_prior_value),
            trainable=False,
        )
        self.prior_delta = self.add_weight(
            name="prior_delta",
            shape=(self.num_classes, self.num_concepts),
            initializer="zeros",
            trainable=True,
        )
        self.concept_logit_gate = self.add_weight(
            name="concept_logit_gate",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        base_logits, clinical_concepts = inputs
        p_effective = self.clinical_prior + self.prior_delta_scale * tf.tanh(self.prior_delta)
        stage_evidence = tf.linalg.matmul(clinical_concepts, p_effective, transpose_b=True)
        stage_evidence = tf.tanh(stage_evidence)
        router_contribution = tf.tanh(self.concept_logit_gate) * stage_evidence
        final_logits = base_logits + router_contribution
        if self.expose_diagnostics:
            return tf.concat(
                [final_logits, base_logits, stage_evidence, router_contribution],
                axis=-1,
            )
        return final_logits

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "clinical_prior": self.clinical_prior_value.tolist(),
                "prior_delta_scale": self.prior_delta_scale,
                "expose_diagnostics": self.expose_diagnostics,
            }
        )
        return config


def apply_stage_specific_concept_router(
        base_logits,
        segment_sequence,
        num_classes=5,
        clinical_concept_version="v1",
        prior_delta_scale=0.10,
        expose_diagnostics=False,
):
    raw_concepts = TimeDistributed(
        Lambda(
            lambda x: extract_clinical_concepts_from_segments(
                x,
                concept_version=clinical_concept_version,
            )
        ),
        name="scr_td_clinical_concepts",
    )(segment_sequence)
    concepts = LayerNormalization(
        name="scr_td_clinical_concept_norm",
        trainable=False,
    )(raw_concepts)
    concept_summary = TimeDistributed(
        GlobalAveragePooling1D(),
        name="scr_td_clinical_concept_summary",
    )(concepts)
    clinical_prior = get_stage_concept_prior(
        num_concepts=len(CLINICAL_CONCEPT_NAMES),
        num_classes=num_classes,
    ).T
    router_output = StageSpecificConceptRouter(
        clinical_prior=clinical_prior,
        prior_delta_scale=prior_delta_scale,
        expose_diagnostics=expose_diagnostics,
        name="stage_specific_concept_router",
    )([base_logits, concept_summary])
    if not expose_diagnostics:
        return router_output
    final_logits = Lambda(
        lambda x: x[..., :num_classes],
        name="scr_final_logits_probe",
    )(router_output)
    base_logits_probe = Lambda(
        lambda x: x[..., num_classes:2 * num_classes],
        name="scr_base_logits_probe",
    )(router_output)
    stage_evidence_probe = Lambda(
        lambda x: x[..., 2 * num_classes:3 * num_classes],
        name="scr_stage_evidence_probe",
    )(router_output)
    router_contribution_probe = Lambda(
        lambda x: x[..., 3 * num_classes:4 * num_classes],
        name="scr_router_contribution_probe",
    )(router_output)
    return final_logits, {
        "base_logits": base_logits_probe,
        "stage_evidence": stage_evidence_probe,
        "router_contribution": router_contribution_probe,
        "final_logits": final_logits,
    }


def _temporal_residual_tcn_block(
        inputs,
        filters,
        dilation_rate,
        dropout_rate=0.10,
        name="temporal_tcn_block",
):
    shortcut = inputs
    if inputs.shape[-1] is None or int(inputs.shape[-1]) != int(filters):
        shortcut = Conv1D(
            int(filters),
            kernel_size=1,
            padding="same",
            name=f"{name}_shortcut",
        )(shortcut)

    x = Conv1D(
        int(filters),
        kernel_size=3,
        dilation_rate=int(dilation_rate),
        padding="same",
        name=f"{name}_conv1",
    )(inputs)
    x = LayerNormalization(name=f"{name}_norm1")(x)
    x = Activation("relu", name=f"{name}_relu1")(x)
    x = Dropout(dropout_rate, name=f"{name}_dropout1")(x)
    x = Conv1D(
        int(filters),
        kernel_size=3,
        dilation_rate=int(dilation_rate),
        padding="same",
        name=f"{name}_conv2",
    )(x)
    x = LayerNormalization(name=f"{name}_norm2")(x)
    x = Add(name=f"{name}_residual_add")([shortcut, x])
    return Activation("relu", name=f"{name}_out_relu")(x)


def _sequence_state_tcn_decoder(
        features,
        temporal_units=96,
        dropout_rate=0.30,
        use_transformer=False,
):
    filters = max(96, int(temporal_units))
    local = Conv1D(filters, kernel_size=3, padding="same", activation="relu", name="v3_state_local_conv")(features)
    wide = Conv1D(filters, kernel_size=7, padding="same", activation="relu", name="v3_state_wide_conv")(features)
    x = Concatenate(name="v3_state_local_wide_fusion")([features, local, wide])
    x = Dense(filters, activation="relu", name="v3_state_input_projection")(x)
    x = LayerNormalization(name="v3_state_input_norm")(x)

    for block_idx, dilation_rate in enumerate((1, 2, 4, 8, 1, 2), start=1):
        x = _temporal_residual_tcn_block(
            x,
            filters=filters,
            dilation_rate=dilation_rate,
            dropout_rate=0.10,
            name=f"v3_state_tcn_d{dilation_rate}_b{block_idx}",
        )

    if use_transformer:
        x = transformer_encoder_block(
            x,
            head_size=max(16, filters // 4),
            num_heads=4,
            ff_dim=max(128, filters * 2),
            dropout=0.10,
        )

    x = Dense(160, activation="relu", name="v3_state_dense")(x)
    x = LayerNormalization(name="v3_state_norm")(x)
    return Dropout(dropout_rate, name="v3_state_dropout")(x)


def _time_distributed_clinical_concept_features(
    segment_sequence,
    num_classes=5,
    dropout_rate=0.20,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
    clinical_prior_trainable=True,
    use_concept_language=False,
    ot_sinkhorn_epsilon=0.08,
    ot_sinkhorn_iterations=8,
    ot_cosine_weight=1.0,
    concept_bottleneck_layer_name="concept_bottleneck_output",
    concept_output_layer_name="concept_output",
    prototype_logits_layer_name="td_clinical_concept_prototype_logits",
    clinical_concept_version="v1",
):
    raw_concepts = TimeDistributed(
        Lambda(
            lambda x: extract_clinical_concepts_from_segments(
                x,
                concept_version=clinical_concept_version,
            )
        ),
        name="td_clinical_concepts",
    )(segment_sequence)
    concept_bottleneck_output = ClinicalConceptAdapter(
        name=concept_bottleneck_layer_name,
    )(TimeDistributed(
        GlobalAveragePooling1D(),
        name="td_clinical_raw_concept_summary",
    )(raw_concepts))

    concepts = LayerNormalization(name="td_clinical_concept_norm")(raw_concepts)
    if use_domain_mixstyle:
        concepts = MixStyle(
            p=mixstyle_p,
            alpha=mixstyle_alpha,
            name="sequence_clinical_domain_mixstyle",
        )(concepts)

    concept_summary = TimeDistributed(
        GlobalAveragePooling1D(),
        name="td_clinical_concept_summary",
    )(concepts)
    concept_prior = get_stage_concept_prior(
        num_concepts=len(CLINICAL_CONCEPT_NAMES),
        num_classes=num_classes,
    )
    concept_logits = Dense(
        num_classes,
        use_bias=False,
        kernel_initializer=Constant(concept_prior),
        trainable=bool(clinical_prior_trainable),
        name="td_clinical_concept_prior_logits",
    )(concept_summary)
    concept_probs = Activation("softmax", name="td_clinical_concept_prior_probs")(concept_logits)

    prototype_init = concept_prior.T
    concept_language_logits = ClinicalConceptPrototypeClassifier(
        prototype_init,
        trainable_prototypes=bool(clinical_prior_trainable),
        sinkhorn_epsilon=ot_sinkhorn_epsilon,
        sinkhorn_iterations=ot_sinkhorn_iterations,
        cosine_weight=ot_cosine_weight,
        name=prototype_logits_layer_name,
    )(concept_bottleneck_output)
    concept_language_probs = Activation("softmax", name=concept_output_layer_name)(concept_language_logits)

    evidence_blocks = [concept_summary, concept_probs]
    if use_concept_language:
        evidence_blocks.extend([concept_bottleneck_output, concept_language_probs])
    concept_features = Concatenate(name="td_clinical_concept_evidence")(evidence_blocks)
    concept_features = Dense(64, activation="relu", name="td_clinical_concept_dense")(concept_features)
    concept_features = LayerNormalization(name="td_clinical_concept_features_norm")(concept_features)
    concept_features = Dropout(dropout_rate, name="clinical_concept_feature_dropout")(concept_features)
    if use_concept_language:
        return concept_features, concept_language_probs, concept_bottleneck_output
    return concept_features


def _time_distributed_late_clinical_concept_residual_logits(
    segment_sequence,
    num_classes=5,
    clinical_concept_version="v1",
    residual_scale=0.05,
    residual_trainable=True,
    stop_gradient=True,
):
    raw_concepts = TimeDistributed(
        Lambda(
            lambda x: extract_clinical_concepts_from_segments(
                x,
                concept_version=clinical_concept_version,
            )
        ),
        name="late_td_clinical_concepts",
    )(segment_sequence)
    concepts = LayerNormalization(
        name="late_td_clinical_concept_norm",
        trainable=False,
    )(raw_concepts)
    concept_summary = TimeDistributed(
        GlobalAveragePooling1D(),
        name="late_td_clinical_concept_summary",
    )(concepts)
    if stop_gradient:
        concept_summary = Lambda(
            lambda x: tf.stop_gradient(x),
            name="late_td_clinical_concept_stop_gradient",
        )(concept_summary)

    concept_prior = get_stage_concept_prior(
        num_concepts=len(CLINICAL_CONCEPT_NAMES),
        num_classes=num_classes,
    )
    residual_logits = Dense(
        num_classes,
        use_bias=False,
        kernel_initializer=Constant(concept_prior),
        trainable=bool(residual_trainable),
        name="late_td_clinical_concept_residual_logits",
    )(concept_summary)
    residual_scale = float(np.clip(residual_scale, 0.0, 1.0))
    return Lambda(
        lambda x: residual_scale * x,
        name="late_td_clinical_concept_residual_scale",
    )(residual_logits)


def _shuffle_batch_concepts(concept_sequence):
    batch_size = tf.shape(concept_sequence)[0]
    permutation = tf.random.shuffle(tf.range(batch_size))
    return tf.gather(concept_sequence, permutation, axis=0)


class EOGSlowEvidenceResidual(tf.keras.layers.Layer):
    """Per-segment EOG delta/theta residual for the existing FiLM hidden."""

    def __init__(self, conditioning_dim, hidden_dim=8, dropout=0.1, gate_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conditioning_dim = int(conditioning_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.gate_init = float(gate_init)
        self.hidden = Dense(self.hidden_dim, activation=tf.nn.gelu, name="view_hidden")
        self.dropout = Dropout(self.dropout_rate, name="view_dropout")
        self.projection = Dense(self.conditioning_dim, name="view_projection")

    def build(self, input_shape):
        self.gate_parameter = self.add_weight(
            name="gate_parameter", shape=(), initializer=Constant(self.gate_init), trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        evidence, scalar_hidden = inputs
        delta = self.projection(self.dropout(self.hidden(evidence), training=training))
        effective_gate = tf.tanh(self.gate_parameter)
        hidden_segments = tf.expand_dims(scalar_hidden, axis=-2) + effective_gate * delta
        conditioned_hidden = tf.reduce_mean(hidden_segments, axis=-2)
        ratio = tf.norm(effective_gate * delta) / (tf.norm(scalar_hidden) + tf.keras.backend.epsilon())
        if training is not False:
            self.add_metric(self.gate_parameter, name="eog_slow_gate_parameter", aggregation="mean")
            self.add_metric(effective_gate, name="eog_slow_effective_gate", aggregation="mean")
            self.add_metric(tf.reduce_mean(tf.norm(scalar_hidden, axis=-1)), name="eog_slow_scalar_hidden_norm", aggregation="mean")
            self.add_metric(tf.reduce_mean(tf.norm(delta, axis=-1)), name="eog_slow_residual_hidden_norm", aggregation="mean")
            self.add_metric(ratio, name="eog_slow_injection_ratio", aggregation="mean")
        return conditioned_hidden

    def get_config(self):
        config = super().get_config()
        config.update({"conditioning_dim": self.conditioning_dim, "hidden_dim": self.hidden_dim,
                       "dropout": self.dropout_rate, "gate_init": self.gate_init})
        return config


def _time_distributed_concept_film_modulation(
    feature_sequence,
    segment_sequence,
    clinical_concept_version="v1",
    modulation_strength=0.10,
    stop_gradient=True,
    concept_mode="real",
    hidden_units=64,
    enable_eog_slow_evidence_residual=False,
    eog_slow_evidence_source="raw",
    eog_slow_residual_hidden_dim=8,
    eog_slow_residual_dropout=0.1,
    eog_slow_residual_gate_init=0.0,
):
    feature_dim = feature_sequence.shape[-1]
    if feature_dim is None:
        raise ValueError("Concept-FiLM modulation requires a known feature dimension.")
    feature_dim = int(feature_dim)
    mode = str(concept_mode or "real").strip().lower().replace("-", "_")
    mode_aliases = {
        "real": "real",
        "concept": "real",
        "concepts": "real",
        "zero": "zero",
        "zeros": "zero",
        "without": "zero",
        "no_concept": "zero",
        "no_concepts": "zero",
        "shuffled": "shuffled",
        "shuffle": "shuffled",
        "random": "shuffled",
        "random_concepts": "shuffled",
    }
    if mode not in mode_aliases:
        raise ValueError(
            "concept_mode must be one of real, zero, or shuffled; "
            f"got {concept_mode!r}"
        )
    mode = mode_aliases[mode]

    raw_concepts = TimeDistributed(
        Lambda(
            lambda x: extract_clinical_concepts_from_segments(
                x,
                concept_version=clinical_concept_version,
            )
        ),
        name="pfm_td_clinical_concepts",
    )(segment_sequence)
    concepts = LayerNormalization(
        name="pfm_td_clinical_concept_norm",
        trainable=False,
    )(raw_concepts)
    concept_summary = TimeDistributed(
        GlobalAveragePooling1D(),
        name="pfm_td_clinical_concept_summary",
    )(concepts)
    if mode == "zero":
        concept_summary = Lambda(
            lambda x: tf.zeros_like(x),
            name="pfm_td_clinical_concept_zero",
        )(concept_summary)
    elif mode == "shuffled":
        concept_summary = Lambda(
            _shuffle_batch_concepts,
            name="pfm_td_clinical_concept_shuffle",
        )(concept_summary)
    if stop_gradient:
        concept_summary = Lambda(
            lambda x: tf.stop_gradient(x),
            name="pfm_td_clinical_concept_stop_gradient",
        )(concept_summary)

    hidden = Dense(
        int(hidden_units),
        activation="relu",
        name="pfm_concept_modulation_hidden",
    )(concept_summary)
    if enable_eog_slow_evidence_residual:
        source = str(eog_slow_evidence_source or "raw").strip().lower()
        if source == "raw":
            evidence = tf.concat([segment_sequence[..., 14:15], segment_sequence[..., 17:18]], axis=-1)
        elif source == "matched_scalar":
            slow_scalar = raw_concepts[..., 5:6]
            evidence = tf.concat([slow_scalar, slow_scalar], axis=-1)
        else:
            raise ValueError("eog_slow_evidence_source must be raw or matched_scalar")
        hidden = EOGSlowEvidenceResidual(
            conditioning_dim=int(hidden_units),
            hidden_dim=eog_slow_residual_hidden_dim,
            dropout=eog_slow_residual_dropout,
            gate_init=eog_slow_residual_gate_init,
            name="eog_slow_evidence_residual",
        )([evidence, hidden])
    gamma = Dense(
        feature_dim,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="pfm_concept_modulation_gamma",
    )(hidden)
    beta = Dense(
        feature_dim,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="pfm_concept_modulation_beta",
    )(hidden)
    strength = float(np.clip(modulation_strength, 0.0, 1.0))
    return Lambda(
        lambda xs: xs[0] * (1.0 + strength * tf.tanh(xs[1])) + strength * tf.tanh(xs[2]),
        name="concept_film_modulated_features",
    )([feature_sequence, gamma, beta])


def _late_concept_confidence_gate(
    main_logits,
    low_confidence_threshold=0.65,
    high_confidence_threshold=0.85,
    mid_confidence_gate=0.50,
):
    low_threshold = float(np.clip(low_confidence_threshold, 0.0, 1.0))
    high_threshold = float(np.clip(high_confidence_threshold, 0.0, 1.0))
    if high_threshold < low_threshold:
        low_threshold, high_threshold = high_threshold, low_threshold
    mid_gate = float(np.clip(mid_confidence_gate, 0.0, 1.0))

    main_probs = tf.nn.softmax(main_logits, axis=-1)
    confidence = tf.reduce_max(main_probs, axis=-1, keepdims=True)
    zeros = tf.zeros_like(confidence)
    weak_gate = tf.fill(tf.shape(confidence), tf.cast(mid_gate, confidence.dtype))
    full_gate = tf.ones_like(confidence)
    return tf.where(
        confidence >= high_threshold,
        zeros,
        tf.where(confidence >= low_threshold, weak_gate, full_gate),
    )


def _main_output_metrics():
    return {"main_output": "accuracy"}


def _compile_sleep_model(model, learning_rate, losses, loss_weights):
    compile_config = {
        "learning_rate": float(learning_rate),
        "losses": dict(losses),
        "loss_weights": dict(loss_weights),
    }
    strategy = getattr(model, "distribute_strategy", None)
    scope = strategy.scope() if strategy is not None else tf.distribute.get_strategy().scope()
    with scope:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss=compile_config["losses"],
            loss_weights=compile_config["loss_weights"],
            metrics=_main_output_metrics(),
        )
    model._lfts_compile_config = compile_config
    return model

def recompile_sleep_model(model, learning_rate=None, loss_weight_overrides=None):
    """Recompile a sleep-staging model after changing trainable layer flags."""
    if not hasattr(model, "_lfts_compile_config"):
        raise ValueError(
            "The model does not carry an LFTS compile config. "
            "Create it with create_sequence_sleep_model before staged recompilation."
        )
    compile_config = model._lfts_compile_config
    effective_learning_rate = (
        float(compile_config["learning_rate"])
        if learning_rate is None
        else float(learning_rate)
    )
    loss_weights = dict(compile_config["loss_weights"])
    if loss_weight_overrides:
        for output_name, override in loss_weight_overrides.items():
            if output_name in loss_weights:
                loss_weights[output_name] = float(override)
    return _compile_sleep_model(
        model,
        effective_learning_rate,
        compile_config["losses"],
        loss_weights,
    )


def create_sequence_sleep_model(
    raw_shape=(3000, 3),
    segment_shape=(30, 28),
    seq_len=25,
    num_classes=5,
    learning_rate=0.001,
    temporal_units=96,
    dropout_rate=0.30,
    sequence_backbone="v3_tcn",
    epoch_encoder_variant="v3_multiscale",
    use_transformer=False,
    loss_name="categorical_crossentropy",
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
    use_domain_alignment=False,
    domain_alignment_loss_weight=0.0,
    domain_alignment_mode="coral",
    use_clinical_concept_features=True,
    clinical_concept_version="v1",
    clinical_prior_trainable=True,
    use_late_clinical_concept_residual=False,
    late_concept_residual_scale=0.05,
    late_concept_residual_trainable=True,
    late_concept_stop_gradient=True,
    use_late_concept_confidence_gate=False,
    late_concept_gate_low_confidence=0.65,
    late_concept_gate_high_confidence=0.85,
    late_concept_gate_mid_value=0.50,
    use_concept_feature_modulation=False,
    concept_modulation_strength=0.10,
    concept_modulation_stop_gradient=True,
    concept_modulation_mode="real",
    use_concept_language=False,
    concept_loss_weight=0.08,
    concept_bottleneck_loss_weight=0.04,
    use_ot_tacb=False,
    ot_sinkhorn_epsilon=0.08,
    ot_sinkhorn_iterations=8,
    ot_cosine_weight=1.0,
    use_mpdg=False,
    mpdg_concept_dim=16,
    dpcm_mask_amplitude=1.0,
    dpcm_prior_init_scale=0.05,
    dpcm_boundary_contrast_loss_weight=0.015,
    dpcm_boundary_contrast_margin=0.35,
    use_transition_boundary_auxiliary=False,
    transition_boundary_loss_weight=0.0,
    transition_boundary_num_classes=1,
    use_concept_derivative_auxiliary=False,
    concept_derivative_loss_weight=0.0,
    use_n1_supervised_contrastive=False,
    n1_contrastive_loss_weight=0.04,
    n1_contrastive_projection_dim=64,
    enable_n1_n2_pair_supcon=False,
    n1_n2_pair_supcon_loss_weight=0.01,
    enable_n1_context_local_contrast=False,
    n1_context_local_contrast_loss_weight=0.01,
    enable_epoch_utf_lite=False,
    epoch_utf_lite_dim=128,
    epoch_utf_lite_heads=4,
    epoch_utf_lite_dropout=0.10,
    epoch_utf_lite_seed=2024,
    enable_stage_concept_router=False,
    stage_concept_prior_delta_scale=0.10,
    instrument_stage_concept_router=False,
    enable_eog_slow_evidence_residual=False,
    eog_slow_evidence_source="raw",
    eog_slow_residual_hidden_dim=8,
    eog_slow_residual_dropout=0.1,
    eog_slow_residual_gate_init=0.0,
    enable_xai_feedback_head=False,
    xai_feedback_loss_weight=0.0,
    xai_alignment_loss_name="normalized_mse",
    xai_gradient_scope="head_only",
):
    xai_feedback_loss_weight = float(xai_feedback_loss_weight)
    assert_model_xai_loss_is_zero(xai_feedback_loss_weight)
    xai_gradient_scope = str(xai_gradient_scope or "head_only").strip().lower()
    if xai_gradient_scope not in {"head_only", "fusion_and_head", "full_backbone"}:
        raise ValueError(
            "xai_gradient_scope must be head_only, fusion_and_head, or full_backbone"
        )
    if enable_stage_concept_router:
        concept_mode = str(concept_modulation_mode or "real").strip().lower().replace("-", "_")
        if concept_mode not in {"zero", "zeros", "without", "no_concept", "no_concepts"}:
            raise ValueError(
                "enable_stage_concept_router=True requires concept_modulation_mode='zero' "
                f"to preserve the A1-zero backbone; got {concept_modulation_mode!r}."
            )
        if use_late_clinical_concept_residual:
            raise ValueError(
                "enable_stage_concept_router=True cannot be combined with late clinical concept residuals."
            )
    raw_input = tf.keras.Input(shape=(seq_len, *raw_shape), name="raw_input")
    segment_input = tf.keras.Input(shape=(seq_len, *segment_shape), name="segment_input")

    epoch_encoder_key = str(epoch_encoder_variant or "v3_multiscale").strip().lower().replace("-", "_")
    if epoch_encoder_key in {
        "v3_channel_specialized_residual",
        "v3_channel_specialized_residual_lite",
        "channel_specialized_residual",
        "channel_specialized_residual_lite",
        "chan_spec_residual",
        "chan_spec_residual_lite",
    }:
        raw_features = _time_distributed_channel_specialized_residual_features(
            raw_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        segment_features = _time_distributed_segment_multiscale_features(
            segment_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
    elif epoch_encoder_key in {"v3", "v3_multiscale", "multiscale", "multiscale_tcn", "tcn"}:
        raw_features = _time_distributed_raw_multiscale_features(
            raw_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        segment_features = _time_distributed_segment_multiscale_features(
            segment_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
    elif epoch_encoder_key in {"legacy", "bilstm", "legacy_bilstm"}:
        raw_features = _time_distributed_raw_features(
            raw_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
        segment_features = _time_distributed_segment_features(
            segment_input,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
        )
    else:
        raise ValueError(
            "epoch_encoder_variant must be one of 'v3_multiscale', 'v3_channel_specialized_residual_lite', 'v3_channel_specialized_residual', or 'legacy_bilstm', "
            f"got {epoch_encoder_variant!r}"
        )

    if enable_epoch_utf_lite:
        raw_features, segment_features = apply_epoch_utf_lite_interaction(
            raw_features,
            segment_features,
            interaction_dim=epoch_utf_lite_dim,
            num_heads=epoch_utf_lite_heads,
            dropout_rate=epoch_utf_lite_dropout,
            seed=epoch_utf_lite_seed,
        )

    feature_blocks = [raw_features, segment_features]
    concept_output = None
    concept_bottleneck_output = None
    use_mpdg = bool(use_mpdg)
    if use_clinical_concept_features:
        clinical_bottleneck_name = (
            "td_aux_clinical_concept_bottleneck_output"
            if use_mpdg
            else "concept_bottleneck_output"
        )
        clinical_output_name = "td_aux_clinical_concept_output" if use_mpdg else "concept_output"
        clinical_prototype_name = (
            "td_aux_clinical_concept_prototype_logits"
            if use_mpdg
            else "td_clinical_concept_prototype_logits"
        )
        clinical_result = _time_distributed_clinical_concept_features(
            segment_input,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_domain_mixstyle=use_domain_mixstyle,
            mixstyle_p=mixstyle_p,
            mixstyle_alpha=mixstyle_alpha,
            clinical_prior_trainable=clinical_prior_trainable,
            use_concept_language=use_concept_language,
            ot_sinkhorn_epsilon=ot_sinkhorn_epsilon,
            ot_sinkhorn_iterations=ot_sinkhorn_iterations,
            ot_cosine_weight=ot_cosine_weight,
            concept_bottleneck_layer_name=clinical_bottleneck_name,
            concept_output_layer_name=clinical_output_name,
            prototype_logits_layer_name=clinical_prototype_name,
            clinical_concept_version=clinical_concept_version,
        )
        if use_concept_language:
            clinical_features, concept_output, concept_bottleneck_output = clinical_result
        else:
            clinical_features = clinical_result
        feature_blocks.append(clinical_features)
    fused_features = Concatenate(name="epoch_feature_fusion")(feature_blocks)
    fused_features = TimeDistributed(Dense(160, activation="relu"), name="td_epoch_feature_dense")(fused_features)
    fused_features = TimeDistributed(LayerNormalization(), name="td_epoch_feature_norm")(fused_features)

    dpcm_sequence = None
    dpcm_mask_profile = None
    if use_mpdg:
        concept_prior = get_stage_concept_prior(
            num_concepts=len(CLINICAL_CONCEPT_NAMES),
            num_classes=num_classes,
        )
        concept_prior_strength = np.max(concept_prior, axis=1)
        dpcm_sequence, dpcm_concept_intensity, dpcm_mask_profile = DynamicPhysiologicalConceptMasking(
            num_concepts=len(CLINICAL_CONCEPT_NAMES),
            concept_dim=mpdg_concept_dim,
            concept_prior_strength=concept_prior_strength,
            mask_amplitude=dpcm_mask_amplitude,
            prior_init_scale=dpcm_prior_init_scale,
            name="dpcm_dynamic_physiological_masking",
        )(fused_features)
        features = Dropout(dropout_rate, name="dpcm_masked_feature_dropout")(dpcm_sequence)
        if use_concept_language:
            concept_bottleneck_output = Activation(
                "linear",
                name="concept_bottleneck_output",
            )(dpcm_concept_intensity)
            prototype_init = concept_prior.T
            concept_language_logits = ClinicalConceptPrototypeClassifier(
                prototype_init,
                trainable_prototypes=bool(clinical_prior_trainable),
                sinkhorn_epsilon=ot_sinkhorn_epsilon,
                sinkhorn_iterations=ot_sinkhorn_iterations,
                cosine_weight=ot_cosine_weight,
                name="dpcm_concept_prototype_logits",
            )(concept_bottleneck_output)
            concept_output = Activation("softmax", name="concept_output")(concept_language_logits)
    else:
        features = fused_features

    if use_concept_feature_modulation:
        features = _time_distributed_concept_film_modulation(
            features,
            segment_input,
            clinical_concept_version=clinical_concept_version,
            modulation_strength=concept_modulation_strength,
            stop_gradient=concept_modulation_stop_gradient,
            concept_mode=concept_modulation_mode,
            enable_eog_slow_evidence_residual=enable_eog_slow_evidence_residual,
            eog_slow_evidence_source=eog_slow_evidence_source,
            eog_slow_residual_hidden_dim=eog_slow_residual_hidden_dim,
            eog_slow_residual_dropout=eog_slow_residual_dropout,
            eog_slow_residual_gate_init=eog_slow_residual_gate_init,
        )

    sequence_backbone_key = str(sequence_backbone or "v3_tcn").strip().lower().replace("-", "_")
    if sequence_backbone_key in {"v3", "v3_tcn", "tcn", "state_tcn", "temporal_tcn"}:
        temporal = _sequence_state_tcn_decoder(
            features,
            temporal_units=temporal_units,
            dropout_rate=dropout_rate,
            use_transformer=use_transformer,
        )
    elif sequence_backbone_key in {"legacy", "bilstm", "legacy_bilstm"}:
        temporal = Bidirectional(
            LSTM(int(temporal_units), return_sequences=True, dropout=0.10),
            name="epoch_bilstm_1",
        )(features)
        temporal = Bidirectional(
            LSTM(max(16, int(temporal_units) // 2), return_sequences=True, dropout=0.10),
            name="epoch_bilstm_2",
        )(temporal)
        if use_transformer:
            temporal = transformer_encoder_block(
                temporal,
                head_size=max(16, int(temporal_units) // 2),
                num_heads=4,
                ff_dim=max(64, int(temporal_units) * 2),
                dropout=0.10,
            )
    else:
        raise ValueError(
            "sequence_backbone must be one of 'v3_tcn' or 'legacy_bilstm', "
            f"got {sequence_backbone!r}"
        )

    temporal = TimeDistributed(Dense(128, activation="relu"), name="td_temporal_dense")(temporal)
    pair_supcon_features = None
    if enable_n1_n2_pair_supcon:
        pair_supcon_features = Activation(
            "linear",
            name="n1_n2_pair_supcon_features",
        )(temporal)
    n1_context_local_contrast_features = None
    if enable_n1_context_local_contrast:
        n1_context_local_contrast_features = Activation(
            "linear",
            name="n1_context_local_contrast_features",
        )(temporal)
    temporal = Dropout(dropout_rate, name="temporal_dropout")(temporal)
    scr_diagnostic_tensors = None
    if use_late_clinical_concept_residual:
        main_logits = Dense(num_classes, name="main_output_logits")(temporal)
        concept_residual_logits = _time_distributed_late_clinical_concept_residual_logits(
            segment_input,
            num_classes=num_classes,
            clinical_concept_version=clinical_concept_version,
            residual_scale=late_concept_residual_scale,
            residual_trainable=late_concept_residual_trainable,
            stop_gradient=late_concept_stop_gradient,
        )
        if use_late_concept_confidence_gate:
            confidence_gate = Lambda(
                lambda x: _late_concept_confidence_gate(
                    x,
                    low_confidence_threshold=late_concept_gate_low_confidence,
                    high_confidence_threshold=late_concept_gate_high_confidence,
                    mid_confidence_gate=late_concept_gate_mid_value,
                ),
                name="late_td_clinical_concept_confidence_gate",
            )(main_logits)
            concept_residual_logits = Multiply(
                name="late_td_clinical_concept_gated_residual_logits",
            )([concept_residual_logits, confidence_gate])
        main_logits = Add(name="main_output_late_concept_residual_logits")(
            [main_logits, concept_residual_logits]
        )
        main_output = Activation("softmax", name="main_output")(main_logits)
    elif enable_stage_concept_router:
        main_logits = Dense(num_classes, name="main_output_logits")(temporal)
        router_result = apply_stage_specific_concept_router(
            main_logits,
            segment_input,
            num_classes=num_classes,
            clinical_concept_version=clinical_concept_version,
            prior_delta_scale=stage_concept_prior_delta_scale,
            expose_diagnostics=instrument_stage_concept_router,
        )
        if instrument_stage_concept_router:
            main_logits, scr_diagnostic_tensors = router_result
        else:
            main_logits = router_result
        main_output = Activation("softmax", name="main_output")(main_logits)
    else:
        main_output = Dense(num_classes, activation="softmax", name="main_output")(temporal)

    xai_importance_logits = None
    if enable_xai_feedback_head:
        # Construct the optional head only after the canonical classifier path.
        # This preserves main-path RNG/initialization order when comparing E0
        # with a lambda=0 feedback-head control.
        local_source = segment_input
        if xai_gradient_scope in {"head_only", "fusion_and_head"}:
            local_source = Lambda(
                lambda value: tf.stop_gradient(value),
                name="xai_local_stop_gradient",
            )(local_source)
        xai_local = TimeDistributed(
            TimeDistributed(Dense(16, activation="relu")),
            name="xai_local_pre_pool_projection",
        )(local_source)
        if xai_gradient_scope in {"fusion_and_head", "full_backbone"}:
            context_source = fused_features
            if xai_gradient_scope == "fusion_and_head":
                context_source = Lambda(
                    lambda value: tf.stop_gradient(value),
                    name="xai_fusion_stop_gradient",
                )(context_source)
            context = Lambda(
                lambda value: tf.repeat(
                    tf.expand_dims(value, axis=2),
                    repeats=int(segment_shape[0]),
                    axis=2,
                ),
                name="xai_repeat_epoch_context",
            )(context_source)
            xai_local = Concatenate(name="xai_local_context_fusion")(
                [xai_local, context]
            )
        xai_hidden = TimeDistributed(
            TimeDistributed(Dense(16, activation="relu")),
            name="xai_importance_hidden",
        )(xai_local)
        xai_importance_logits = TimeDistributed(
            TimeDistributed(Dense(3)),
            name="xai_importance_logits",
        )(xai_hidden)

    outputs = [main_output]
    main_loss_name = "focal" if (use_ot_tacb or use_mpdg) else loss_name
    losses = {"main_output": make_classification_loss(main_loss_name)}
    loss_weights = {"main_output": 1.0}
    if enable_xai_feedback_head:
        outputs.append(xai_importance_logits)
        losses["xai_importance_logits"] = make_xai_alignment_loss(
            xai_alignment_loss_name
        )
        loss_weights["xai_importance_logits"] = xai_feedback_loss_weight
    if use_concept_language and concept_output is not None and concept_bottleneck_output is not None:
        outputs.append(concept_output)
        losses["concept_output"] = CategoricalCrossentropy(label_smoothing=0.03)
        loss_weights["concept_output"] = float(concept_loss_weight)
        outputs.append(concept_bottleneck_output)
        losses["concept_bottleneck_output"] = concept_bottleneck_loss
        loss_weights["concept_bottleneck_output"] = float(concept_bottleneck_loss_weight)
    if use_concept_derivative_auxiliary:
        if concept_bottleneck_output is None:
            raise ValueError(
                "use_concept_derivative_auxiliary=True requires sequence clinical concept language outputs."
            )
        concept_derivative_output = Activation(
            "linear",
            name="concept_derivative_output",
        )(concept_bottleneck_output)
        outputs.append(concept_derivative_output)
        losses["concept_derivative_output"] = concept_derivative_energy_loss
        loss_weights["concept_derivative_output"] = float(concept_derivative_loss_weight)
    if use_transition_boundary_auxiliary:
        transition_boundary_num_classes = max(1, int(transition_boundary_num_classes))
        dpcm_transition_output_name = (
            "dpcm_transition_boundary_prediction"
            if use_mpdg and dpcm_mask_profile is not None
            else "transition_boundary_output"
        )
        if transition_boundary_num_classes > 1:
            transition_boundary_output = Dense(
                transition_boundary_num_classes,
                activation="softmax",
                name=dpcm_transition_output_name,
            )(temporal)
            transition_boundary_loss = fine_grained_boundary_entropy_loss
        else:
            transition_boundary_output = Dense(
                1,
                activation="sigmoid",
                name=dpcm_transition_output_name,
            )(temporal)
            transition_boundary_loss = fine_grained_boundary_entropy_loss
        if use_mpdg and dpcm_mask_profile is not None:
            transition_boundary_output = TransitionBoundaryMaskContrast(
                loss_weight=dpcm_boundary_contrast_loss_weight,
                margin=dpcm_boundary_contrast_margin,
                name="transition_boundary_output",
            )([dpcm_mask_profile, transition_boundary_output])
        outputs.append(transition_boundary_output)
        losses["transition_boundary_output"] = transition_boundary_loss
        loss_weights["transition_boundary_output"] = float(transition_boundary_loss_weight)
    if use_n1_supervised_contrastive:
        contrastive_features = TimeDistributed(
            Dense(
                max(8, int(n1_contrastive_projection_dim)),
                use_bias=False,
            ),
            name="n1_contrastive_features",
        )(temporal)
        outputs.append(contrastive_features)
        losses["n1_contrastive_features"] = n1_supervised_contrastive_loss
        loss_weights["n1_contrastive_features"] = float(n1_contrastive_loss_weight)
    if enable_n1_n2_pair_supcon:
        outputs.append(pair_supcon_features)
        losses["n1_n2_pair_supcon_features"] = n1_n2_pair_supervised_contrastive_loss
        loss_weights["n1_n2_pair_supcon_features"] = float(n1_n2_pair_supcon_loss_weight)
    if enable_n1_context_local_contrast:
        outputs.append(n1_context_local_contrast_features)
        losses["n1_context_local_contrast_features"] = n1_context_subtype_local_contrastive_loss
        loss_weights["n1_context_local_contrast_features"] = float(n1_context_local_contrast_loss_weight)
    if use_domain_alignment:
        domain_features = Activation("linear", name="domain_alignment_features")(temporal)
        outputs.append(domain_features)
        losses["domain_alignment_features"] = make_domain_alignment_loss(domain_alignment_mode)
        loss_weights["domain_alignment_features"] = float(domain_alignment_loss_weight)

    model = Model(inputs=[raw_input, segment_input], outputs=outputs[0] if len(outputs) == 1 else outputs)
    if enable_stage_concept_router and instrument_stage_concept_router:
        if scr_diagnostic_tensors is None:
            raise RuntimeError("SCR diagnostics requested but diagnostic tensors were not created.")
        model._scr_base_logits_tensor = scr_diagnostic_tensors["base_logits"]
        model._scr_stage_evidence_tensor = scr_diagnostic_tensors["stage_evidence"]
        model._scr_router_contribution_tensor = scr_diagnostic_tensors["router_contribution"]
        model._scr_final_logits_tensor = scr_diagnostic_tensors["final_logits"]
    return _compile_sleep_model(model, learning_rate, losses, loss_weights)


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


def _extract_clinical_concepts_v1_from_segments(segment_tensor):
    return concepts_from_segment_tensor(segment_tensor, version="segment_proxy_v1")


def _extract_clinical_concepts_v2_from_segments(segment_tensor):
    return concepts_from_segment_tensor(segment_tensor, version="segment_proxy_v2")


def extract_clinical_concepts_from_segments(segment_tensor, concept_version="v1"):
    if _normalize_clinical_concept_version(concept_version) == "v2":
        return _extract_clinical_concepts_v2_from_segments(segment_tensor)
    return _extract_clinical_concepts_v1_from_segments(segment_tensor)


def clinical_concept_branch(
    input_segments,
    num_classes,
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
    clinical_prior_trainable=True,
    clinical_concept_version="v1",
):
    concepts = Lambda(
        lambda x: extract_clinical_concepts_from_segments(
            x,
            concept_version=clinical_concept_version,
        ),
        name="clinical_concepts",
    )(input_segments)
    concepts = LayerNormalization(name="clinical_concept_norm")(concepts)
    if use_domain_mixstyle:
        concepts = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="clinical_domain_mixstyle")(concepts)

    concept_gate = GlobalAveragePooling1D(name="clinical_concept_pool")(concepts)
    concept_gate = Dense(32, activation="relu", name="clinical_concept_gate_hidden")(concept_gate)
    concept_gate = Dense(len(CLINICAL_CONCEPT_NAMES), activation="sigmoid", name="clinical_concept_gate")(concept_gate)
    concept_bottleneck_output = Activation("linear", name="concept_bottleneck_output")(concept_gate)
    concept_weighted = Multiply(name="clinical_concept_weighted")(
        [concepts, Reshape((1, len(CLINICAL_CONCEPT_NAMES)))(concept_bottleneck_output)]
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
    concept_prior = get_stage_concept_prior(
        num_concepts=len(CLINICAL_CONCEPT_NAMES),
        num_classes=num_classes,
    )
    concept_logits = Dense(
        num_classes,
        use_bias=False,
        kernel_initializer=Constant(concept_prior),
        trainable=bool(clinical_prior_trainable),
        name="clinical_concept_prior_logits",
    )(concept_summary)
    concept_output = Activation("softmax", name="concept_output")(concept_logits)

    return concept_features, concept_output, concept_bottleneck_output


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


def explanation_feedback_attention_loss(y_true, y_pred):
    """Match attention to concept-DCCS targets while retaining stage priors."""
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    y_pred = y_pred / tf.maximum(tf.reduce_sum(y_pred, axis=1, keepdims=True), 1e-6)

    if y_true.shape[-1] is not None and y_true.shape[-1] <= 5:
        return prior_guided_attention_loss(y_true, y_pred)

    target_channels = tf.clip_by_value(y_true[:, :3], 1e-6, 1.0)
    target_channels = target_channels / tf.maximum(
        tf.reduce_sum(target_channels, axis=1, keepdims=True),
        1e-6,
    )
    stage_targets = y_true[:, 3:]

    dccs_kl = tf.reduce_sum(target_channels * tf.math.log(target_channels / y_pred), axis=1)
    cosine_num = tf.reduce_sum(target_channels * y_pred, axis=1)
    cosine_den = tf.maximum(
        tf.norm(target_channels, axis=1) * tf.norm(y_pred, axis=1),
        1e-6,
    )
    cosine_loss = 1.0 - cosine_num / cosine_den
    stage_prior_loss = prior_guided_attention_loss(stage_targets, y_pred)
    return dccs_kl + 0.25 * cosine_loss + 0.25 * stage_prior_loss


def concept_bottleneck_loss(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return bce + 0.25 * mse


def fine_grained_boundary_entropy_loss(y_true, y_pred):
    """Supervise transition-type boundaries while discouraging diffuse boundary logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_pred.shape[-1] == 1:
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        entropy = -(
            y_pred * tf.math.log(y_pred)
            + (1.0 - y_pred) * tf.math.log(1.0 - y_pred)
        )
        return bce + 0.02 * tf.squeeze(y_true, axis=-1) * tf.squeeze(entropy, axis=-1)

    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
    y_pred = y_pred / tf.maximum(tf.reduce_sum(y_pred, axis=-1, keepdims=True), 1e-6)
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_true = y_true / tf.maximum(tf.reduce_sum(y_true, axis=-1, keepdims=True), 1e-6)
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.02)
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
    boundary_mask = 1.0 - y_true[..., 0]
    return cross_entropy + 0.02 * boundary_mask * entropy


def concept_derivative_energy_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    target_delta = y_true[..., :len(CLINICAL_CONCEPT_NAMES)]
    transition_mask = y_true[..., len(CLINICAL_CONCEPT_NAMES):len(CLINICAL_CONCEPT_NAMES) + 1]

    predicted_delta = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    target_delta = target_delta[:, 1:, :]
    transition_mask = transition_mask[:, 1:, :]

    direction = tf.sign(target_delta)
    magnitude = tf.abs(target_delta)
    directional_margin = 0.03
    signed_progress = direction * predicted_delta
    direction_penalty = tf.square(tf.nn.relu(directional_margin - signed_progress)) * magnitude
    magnitude_penalty = 0.25 * tf.square(predicted_delta - target_delta) * magnitude
    pair_energy = tf.reduce_sum(direction_penalty + magnitude_penalty, axis=-1)
    pair_energy = pair_energy / tf.maximum(tf.reduce_sum(magnitude, axis=-1), 1e-6)
    pair_energy = pair_energy * tf.squeeze(transition_mask, axis=-1)

    first_token_energy = tf.zeros_like(pair_energy[:, :1])
    return tf.concat([first_token_energy, pair_energy], axis=1)


def estimate_raw_channel_quality(input_tensor, min_quality=0.10):
    """Estimate per-channel signal quality for routing without changing inputs."""
    safe_input = tf.where(tf.math.is_finite(input_tensor), input_tensor, tf.zeros_like(input_tensor))
    channel_std = tf.math.reduce_std(safe_input, axis=1)
    std_quality = 1.0 - tf.exp(-4.0 * channel_std)

    diffs = safe_input[:, 1:, :] - safe_input[:, :-1, :]
    roughness = tf.reduce_mean(tf.abs(diffs), axis=1) / tf.maximum(channel_std, 1e-6)
    roughness_quality = tf.exp(-tf.nn.relu(roughness - 1.0))

    finite_quality = tf.reduce_mean(
        tf.cast(tf.math.is_finite(input_tensor), tf.float32),
        axis=1,
    )
    quality = std_quality * roughness_quality * finite_quality
    return tf.clip_by_value(quality, float(min_quality), 1.0)


def dynamic_channel_routing(
    input_tensor,
    use_channel_quality_routing=False,
    channel_quality_bias_strength=0.35,
):
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
    channel_hidden = Dense(16, activation="relu")(pool)
    channel_logits = Dense(3, activation=None, name="channel_attention_logits")(channel_hidden)
    if use_channel_quality_routing:
        quality_scores = Lambda(
            estimate_raw_channel_quality,
            name="raw_channel_quality_scores",
        )(input_tensor)
        quality_bias_strength = float(np.clip(channel_quality_bias_strength, 0.0, 2.0))
        quality_log_bias = Lambda(
            lambda x: quality_bias_strength * tf.math.log(tf.clip_by_value(x, 1e-3, 1.0)),
            name="channel_quality_log_bias",
        )(quality_scores)
        channel_logits = Add(name="quality_aware_channel_attention_logits")(
            [channel_logits, quality_log_bias]
        )
    channel_weights = Activation("softmax", name="channel_attention_weights")(channel_logits)

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
    concept_bottleneck_loss_weight=0.05,
    use_channel_quality_routing=False,
    channel_quality_bias_strength=0.35,
    loss_name="focal",
    use_domain_mixstyle=False,
    mixstyle_p=0.50,
    mixstyle_alpha=0.30,
    use_domain_alignment=False,
    domain_alignment_loss_weight=0.0,
    domain_alignment_mode="coral",
    clinical_concept_version="v1",
    clinical_prior_trainable=True,
):
    input_raw = tf.keras.Input(shape=raw_shape, name="raw_input")
    routed_raw, attention_weights = dynamic_channel_routing(
        input_raw,
        use_channel_quality_routing=use_channel_quality_routing,
        channel_quality_bias_strength=channel_quality_bias_strength,
    )

    x1 = Conv1D(raw_shape[-1], kernel_size=1, padding="same")(routed_raw)
    x1 = se_multi_channel_extractor(x1)
    if use_domain_mixstyle:
        x1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="raw_domain_mixstyle")(x1)
    x1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(x1)
    x1 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.1))(x1)
    cnn_features = Dense(128, activation="relu")(x1)
    cnn_features = LayerNormalization()(cnn_features)

    input_segments = tf.keras.Input(shape=segment_shape, name="segment_input")
    x2 = se_block_1d(input_segments, reduction=2)
    if use_domain_mixstyle:
        x2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, name="segment_domain_mixstyle")(x2)
    x2 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1))(x2)
    x2 = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(x2)
    segment_features = Dense(64, activation="relu")(x2)
    segment_features = LayerNormalization()(segment_features)

    concept_features, concept_output, concept_bottleneck_output = clinical_concept_branch(
        input_segments,
        num_classes,
        use_domain_mixstyle=use_domain_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        clinical_prior_trainable=clinical_prior_trainable,
        clinical_concept_version=clinical_concept_version,
    )

    merged = Concatenate(name="feature_fusion")([cnn_features, segment_features, concept_features])
    merged = se_block_dense(merged, reduction=4)

    dense = Dense(256, activation="relu")(merged)
    dense = LayerNormalization()(dense)
    dense = Dropout(0.5)(dense)
    main_output = Dense(num_classes, activation="softmax", name="main_output")(dense)

    classification_loss = make_classification_loss(loss_name)
    outputs = [main_output, attention_weights, concept_output, concept_bottleneck_output]
    losses = {
        "main_output": classification_loss,
        "channel_attention_weights": explanation_feedback_attention_loss,
        "concept_output": classification_loss,
        "concept_bottleneck_output": concept_bottleneck_loss,
    }
    loss_weights = {
        "main_output": 1.0,
        "channel_attention_weights": attention_loss_weight,
        "concept_output": concept_loss_weight,
        "concept_bottleneck_output": concept_bottleneck_loss_weight,
    }
    if use_domain_alignment:
        domain_features = Activation("linear", name="domain_alignment_features")(dense)
        outputs.append(domain_features)
        losses["domain_alignment_features"] = make_domain_alignment_loss(domain_alignment_mode)
        loss_weights["domain_alignment_features"] = float(domain_alignment_loss_weight)

    model = Model(
        inputs=[input_raw, input_segments],
        outputs=outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics={"main_output": "accuracy"},
    )
    return model
