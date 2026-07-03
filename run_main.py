

#之前加上，跑的较好的版本
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # LIME-for-Time-Series directory
TARGET_GPU_IDS = os.environ.get("LFTS_CUDA_VISIBLE_DEVICES", "7")
REQUESTED_GPU_COUNT = len([gpu_id for gpu_id in TARGET_GPU_IDS.split(",") if gpu_id.strip()])
REQUIRE_REQUESTED_GPUS = os.environ.get("LFTS_REQUIRE_REQUESTED_GPUS", "1") != "0"
RUN_MAIN_PROTOCOL_VERSION = "micro_macro_v3_locked_20260622"
PROTOCOL_PROFILE = os.environ.get("LFTS_PROTOCOL_PROFILE", "micro_macro_v3").strip().lower()
if PROTOCOL_PROFILE not in {"micro_macro_v3", "strong", "custom"}:
    PROTOCOL_PROFILE = "micro_macro_v3"
ALLOW_PROTOCOL_ENV_OVERRIDES = os.environ.get(
    "LFTS_ALLOW_PROTOCOL_OVERRIDES",
    "0",
).strip().lower() not in {"0", "false", "no", "off", ""}
_FORCED_PROTOCOL_ENV = {
    "LFTS_MODEL_INPUT_MODE": "sequence",
    "LFTS_SEQ_LEN": "75",
    "LFTS_SEQUENCE_TRAIN_STRIDE": "10",
    "LFTS_MICRO_MACRO_TRAINING": "1",
    "LFTS_MICRO_PRETRAIN_EPOCHS": "5",
    "LFTS_MACRO_STABILIZATION_EPOCHS": "4",
    "LFTS_MACRO_AUX_RAMP_EPOCHS": "8",
    "LFTS_MICRO_FINETUNE_LR_SCALE": "0.20",
    "LFTS_USE_TRANSITION_BOUNDARY_WEIGHTS": "1",
    "LFTS_TRANSITION_BOUNDARY_RADIUS": "1",
    "LFTS_TRANSITION_BOUNDARY_BASE_BOOST": "1.15",
    "LFTS_TRANSITION_BOUNDARY_N1_BOOST": "1.30",
    "LFTS_TRANSITION_BOUNDARY_TARGET_BOOST": "1.25",
    "LFTS_TRANSITION_BOUNDARY_MAX_WEIGHT": "2.20",
    "LFTS_USE_OT_TACB": "1",
    "LFTS_OT_TACB_SINKHORN_LOSS_WEIGHT": "0.12",
    "LFTS_OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT": "0.08",
    "LFTS_OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT": "0.06",
    "LFTS_CONCEPT_DERIVATIVE_LOSS_WEIGHT": "0.06",
    "LFTS_CLINICAL_PRIOR_TRAINABLE": "1",
}
_IGNORED_PROTOCOL_ENV_OVERRIDES = {}
if PROTOCOL_PROFILE in {"micro_macro_v3", "strong"} and not ALLOW_PROTOCOL_ENV_OVERRIDES:
    for _env_name, _forced_value in _FORCED_PROTOCOL_ENV.items():
        _existing_value = os.environ.get(_env_name)
        if _existing_value is not None and str(_existing_value) != _forced_value:
            _IGNORED_PROTOCOL_ENV_OVERRIDES[_env_name] = _existing_value
        os.environ[_env_name] = _forced_value
MEMORY_PROFILE = os.environ.get("LFTS_MEMORY_PROFILE", "balanced").strip().lower()
if MEMORY_PROFILE not in {"safe", "balanced", "max"}:
    MEMORY_PROFILE = "balanced"
DETERMINISTIC_TF = os.environ.get("LFTS_DETERMINISTIC_OPS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
    "",
}
_CUDNN_WORKSPACE_MB_BY_PROFILE = {
    "safe": "512",
    "balanced": "1024",
    "max": "2048",
}
CU_DNN_WORKSPACE_LIMIT_MB = os.environ.get(
    "LFTS_CUDNN_WORKSPACE_LIMIT_IN_MB",
    _CUDNN_WORKSPACE_MB_BY_PROFILE.get(MEMORY_PROFILE, "1024"),
)
TF_TMP_DIR = os.environ.get("LFTS_TF_TMPDIR", os.path.join(PROJECT_DIR, ".tf_tmp"))
os.makedirs(TF_TMP_DIR, exist_ok=True)
os.environ.setdefault("TMPDIR", TF_TMP_DIR)
os.environ.setdefault("TEMP", TF_TMP_DIR)
os.environ.setdefault("TMP", TF_TMP_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_IDS
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1" if DETERMINISTIC_TF else "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = str(CU_DNN_WORKSPACE_LIMIT_MB)
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import datetime
import inspect
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm

PARENT_DIR = os.path.dirname(PROJECT_DIR)

SRC_PATH = os.path.join(PROJECT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_TAG = os.environ.get("LFTS_RUN_TAG", "").strip()


def sanitize_run_tag(tag):
    tag = str(tag or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)[:80]


def parse_float_tuple_env(name, default_values):
    raw_value = os.environ.get(name)
    if not raw_value:
        return tuple(default_values)
    values = tuple(
        float(item.strip())
        for item in raw_value.split(",")
        if item.strip()
    )
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value")
    return values


def parse_bool_env(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return bool(default)
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def normalize_concept_modulation_mode(mode):
    key = str(mode or "real").strip().lower().replace("-", "_")
    aliases = {
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
    if key not in aliases:
        raise ValueError(
            "LFTS_CONCEPT_MODULATION_MODE must be one of real, zero, or shuffled; "
            f"got {mode!r}"
        )
    return aliases[key]


NUM_CLASSES = 5
CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
RUNTIME_STAGE_CHANNEL_PRIOR = None

MULTI_CHANNEL_PATH = os.environ.get(
    "LFTS_MULTI_CHANNEL_PATH",
    os.path.join(PARENT_DIR, "ISRUC-Sleep", "npz_epoch3"),
)

PREPROCESS_SIGNALS = True
PREPROCESS_FILTER_PROFILE = os.environ.get(
    "LFTS_PREPROCESS_FILTER_PROFILE",
    "clinical_canonical",
)
PREPROCESS_NORMALIZATION_MODE = os.environ.get(
    "LFTS_PREPROCESS_NORMALIZATION_MODE",
    "recording_robust_zscore",
)
INCLUDE_SPECTRAL_FEATURES = True
MODEL_INPUT_MODE = os.environ.get("LFTS_MODEL_INPUT_MODE", "sequence").strip().lower()
USE_SEQUENCE_MODEL = MODEL_INPUT_MODE in {"sequence", "seq", "temporal", "context"}
EXPERIMENT_PROTOCOL = os.environ.get("LFTS_EXPERIMENT_PROTOCOL", "intra_dataset_subject_split")
STRATIFIED_SUBJECT_SPLIT = parse_bool_env("LFTS_STRATIFIED_SUBJECT_SPLIT", True)
_SEQ_LEN_DEFAULT_BY_MEMORY_PROFILE = {
    "safe": "50",
    "balanced": "75",
    "max": "75",
}
_TRANSFORMER_DEFAULT_BY_MEMORY_PROFILE = {
    "safe": False,
    "balanced": True,
    "max": True,
}
_TRAIN_BATCH_DEFAULT_BY_MEMORY_PROFILE = {
    "safe": "4",
    "balanced": "32",
    "max": "32",
}
_PREDICT_BATCH_DEFAULT_BY_MEMORY_PROFILE = {
    "safe": "8",
    "balanced": "32",
    "max": "32",
}
SEQUENCE_LENGTH = int(
    os.environ.get(
        "LFTS_SEQ_LEN",
        _SEQ_LEN_DEFAULT_BY_MEMORY_PROFILE.get(MEMORY_PROFILE, "75"),
    )
)
SEQUENCE_TRAIN_STRIDE = int(os.environ.get("LFTS_SEQUENCE_TRAIN_STRIDE", "10"))
SEQUENCE_PREDICT_STRIDE = int(os.environ.get("LFTS_SEQUENCE_PREDICT_STRIDE", "5"))
SEQUENCE_BACKBONE = os.environ.get("LFTS_SEQUENCE_BACKBONE", "v3_tcn").strip().lower()
EPOCH_ENCODER_VARIANT = os.environ.get("LFTS_EPOCH_ENCODER_VARIANT", "v3_multiscale").strip().lower()
SEQUENCE_USE_TRANSFORMER = parse_bool_env(
    "LFTS_SEQUENCE_USE_TRANSFORMER",
    _TRANSFORMER_DEFAULT_BY_MEMORY_PROFILE.get(MEMORY_PROFILE, True),
)
CLASSIFICATION_LOSS_NAME = os.environ.get("LFTS_LOSS_NAME", "n1_margin_focal")
MICRO_MACRO_TRAINING = parse_bool_env("LFTS_MICRO_MACRO_TRAINING", True)
MICRO_PRETRAIN_EPOCHS = int(os.environ.get("LFTS_MICRO_PRETRAIN_EPOCHS", "5"))
MACRO_STABILIZATION_EPOCHS = int(os.environ.get("LFTS_MACRO_STABILIZATION_EPOCHS", "4"))
MACRO_AUX_RAMP_EPOCHS = int(os.environ.get("LFTS_MACRO_AUX_RAMP_EPOCHS", "8"))
MACRO_AUX_START_SCALE = float(os.environ.get("LFTS_MACRO_AUX_START_SCALE", "0.0"))
MACRO_AUX_FINAL_SCALE = float(os.environ.get("LFTS_MACRO_AUX_FINAL_SCALE", "1.0"))
MICRO_FINETUNE_LR_SCALE = float(os.environ.get("LFTS_MICRO_FINETUNE_LR_SCALE", "0.20"))
VALID_GENERALIZATION_PRESETS = {
    "baseline",
    "mixstyle",
    "mixstyle_coral",
    "mixstyle_mmd",
    "mixstyle_stage_coral",
    "mixstyle_stage_mmd",
    "mixstyle_coral_tent",
    "coral",
    "mmd",
    "stage_coral",
    "stage_mmd",
}
GENERALIZATION_PRESET = os.environ.get("LFTS_GENERALIZATION_PRESET", "mixstyle_stage_coral").strip().lower()
if GENERALIZATION_PRESET not in VALID_GENERALIZATION_PRESETS:
    raise ValueError(
        "LFTS_GENERALIZATION_PRESET must be one of "
        f"{sorted(VALID_GENERALIZATION_PRESETS)}, got {GENERALIZATION_PRESET!r}"
    )
USE_DOMAIN_MIXSTYLE = parse_bool_env(
    "LFTS_USE_DOMAIN_MIXSTYLE",
    GENERALIZATION_PRESET in {
        "mixstyle",
        "mixstyle_coral",
        "mixstyle_mmd",
        "mixstyle_stage_coral",
        "mixstyle_stage_mmd",
        "mixstyle_coral_tent",
    },
)
DOMAIN_MIXSTYLE_P = float(os.environ.get("LFTS_DOMAIN_MIXSTYLE_P", "0.50"))
DOMAIN_MIXSTYLE_ALPHA = float(os.environ.get("LFTS_DOMAIN_MIXSTYLE_ALPHA", "0.30"))
USE_DOMAIN_ALIGNMENT = parse_bool_env(
    "LFTS_USE_DOMAIN_ALIGNMENT",
    GENERALIZATION_PRESET in {
        "mixstyle_coral",
        "mixstyle_mmd",
        "mixstyle_stage_coral",
        "mixstyle_stage_mmd",
        "mixstyle_coral_tent",
        "coral",
        "mmd",
        "stage_coral",
        "stage_mmd",
    },
)
DOMAIN_ALIGNMENT_MODE = os.environ.get(
    "LFTS_DOMAIN_ALIGNMENT_MODE",
    (
        "stage_mmd"
        if GENERALIZATION_PRESET in {"stage_mmd", "mixstyle_stage_mmd"}
        else "stage_coral"
        if GENERALIZATION_PRESET in {"stage_coral", "mixstyle_stage_coral"}
        else "mmd"
        if GENERALIZATION_PRESET in {"mmd", "mixstyle_mmd"}
        else "coral"
    ),
).strip().lower()
DOMAIN_ALIGNMENT_STAGE_CONDITIONED = parse_bool_env(
    "LFTS_DOMAIN_ALIGNMENT_STAGE_CONDITIONED",
    DOMAIN_ALIGNMENT_MODE in {"stage_coral", "stage_conditional_coral", "conditional_coral", "stage_mmd", "stage_conditional_mmd", "conditional_mmd"},
)
DOMAIN_ALIGNMENT_LOSS_WEIGHT = float(
    os.environ.get(
        "LFTS_DOMAIN_ALIGNMENT_LOSS_WEIGHT",
        "0.005" if USE_DOMAIN_ALIGNMENT and DOMAIN_ALIGNMENT_STAGE_CONDITIONED else "0.01" if USE_DOMAIN_ALIGNMENT else "0.0",
    )
)
RUN_XAI_DASHBOARDS = parse_bool_env("LFTS_RUN_XAI_DASHBOARDS", False)
SAVE_PREDICTION_PROBS = parse_bool_env("LFTS_SAVE_PREDICTION_PROBS", True)
SAVE_SPLIT_PREDICTION_PROBS = parse_bool_env("LFTS_SAVE_SPLIT_PREDICTION_PROBS", False)
SPLIT_PREDICT_BATCH_SIZE = int(os.environ.get("LFTS_SPLIT_PREDICT_BATCH_SIZE", "0"))
N1_OVERSAMPLE_FACTOR = 1
N1_WEIGHT_BOOST = float(os.environ.get("LFTS_N1_WEIGHT_BOOST", "1.35"))
USE_N1_SUPCON = parse_bool_env("LFTS_USE_N1_SUPCON", True)
N1_SUPCON_LOSS_WEIGHT = float(os.environ.get("LFTS_N1_SUPCON_LOSS_WEIGHT", "0.040"))
N1_SUPCON_PROJECTION_DIM = int(os.environ.get("LFTS_N1_SUPCON_PROJECTION_DIM", "64"))
USE_BALANCED_SEQUENCE_SAMPLING = parse_bool_env("LFTS_USE_BALANCED_SEQUENCE_SAMPLING", True)
BALANCE_N1_WINDOW_BOOST = float(os.environ.get("LFTS_BALANCE_N1_WINDOW_BOOST", "1.75"))
BALANCE_N3_WINDOW_BOOST = float(os.environ.get("LFTS_BALANCE_N3_WINDOW_BOOST", "1.25"))
BALANCE_TRANSITION_WINDOW_BOOST = float(os.environ.get("LFTS_BALANCE_TRANSITION_WINDOW_BOOST", "1.15"))
USE_TRANSITION_BOUNDARY_WEIGHTS = parse_bool_env("LFTS_USE_TRANSITION_BOUNDARY_WEIGHTS", True)
TRANSITION_BOUNDARY_RADIUS = int(os.environ.get("LFTS_TRANSITION_BOUNDARY_RADIUS", "1"))
TRANSITION_BOUNDARY_BASE_BOOST = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_BASE_BOOST", "1.15"))
TRANSITION_BOUNDARY_N1_BOOST = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_N1_BOOST", "1.30"))
TRANSITION_BOUNDARY_TARGET_BOOST = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_TARGET_BOOST", "1.25"))
TRANSITION_BOUNDARY_MAX_WEIGHT = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_MAX_WEIGHT", "2.20"))
USE_TRANSITION_BOUNDARY_AUX = parse_bool_env("LFTS_USE_TRANSITION_BOUNDARY_AUX", True)
USE_FINE_TRANSITION_AUX = parse_bool_env("LFTS_USE_FINE_TRANSITION_AUX", True)
TRANSITION_BOUNDARY_AUX_RADIUS = int(os.environ.get("LFTS_TRANSITION_BOUNDARY_AUX_RADIUS", "0"))
TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT", "0.08"))
TRANSITION_BOUNDARY_AUX_POS_WEIGHT = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_AUX_POS_WEIGHT", "2.00"))
TRANSITION_BOUNDARY_AUX_N1_WEIGHT = float(os.environ.get("LFTS_TRANSITION_BOUNDARY_AUX_N1_WEIGHT", "1.75"))
PER_REPLICA_BATCH_SIZE = int(
    os.environ.get(
        "LFTS_PER_REPLICA_BATCH_SIZE",
        _TRAIN_BATCH_DEFAULT_BY_MEMORY_PROFILE.get(MEMORY_PROFILE, "32"),
    )
)
PREDICT_PER_REPLICA_BATCH_SIZE = int(
    os.environ.get(
        "LFTS_PREDICT_PER_REPLICA_BATCH_SIZE",
        _PREDICT_BATCH_DEFAULT_BY_MEMORY_PROFILE.get(MEMORY_PROFILE, "32"),
    )
)
BATCH_SIZE = PER_REPLICA_BATCH_SIZE
REFERENCE_BATCH_SIZE = 32
BASE_LEARNING_RATE = 0.001
MAX_MULTI_GPU_LEARNING_RATE = 0.0018
EPOCHS = 40
SEED = int(os.environ.get("LFTS_SEED", "42"))
USE_METRIC_SELECTOR = True
USE_HMM_VITERBI = parse_bool_env("LFTS_USE_HMM_VITERBI", True)
USE_QUALITY_AWARE_HMM = parse_bool_env("LFTS_USE_QUALITY_AWARE_HMM", True)
SELECTOR_MODE = os.environ.get("LFTS_SELECTOR_MODE", "stable_macro_n1_kappa")
USE_CONCEPT_LANGUAGE = parse_bool_env("LFTS_USE_CONCEPT_LANGUAGE", True)
CLINICAL_CONCEPT_VERSION = os.getenv(
    "LFTS_CLINICAL_CONCEPT_VERSION",
    "v1",
).strip().lower()
CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT = float(os.environ.get("LFTS_CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT", "0.05"))
CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT = float(os.environ.get("LFTS_CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT", "0.08"))
USE_OT_TACB = parse_bool_env("LFTS_USE_OT_TACB", True)
OT_TACB_SINKHORN_LOSS_WEIGHT = float(os.environ.get("LFTS_OT_TACB_SINKHORN_LOSS_WEIGHT", "0.12"))
OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT = float(os.environ.get("LFTS_OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT", "0.08"))
OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT = float(os.environ.get("LFTS_OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT", "0.06"))
USE_CONCEPT_DERIVATIVE_AUX = parse_bool_env("LFTS_USE_CONCEPT_DERIVATIVE_AUX", True)
CONCEPT_DERIVATIVE_LOSS_WEIGHT = float(
    os.environ.get(
        "LFTS_CONCEPT_DERIVATIVE_LOSS_WEIGHT",
        os.environ.get("LFTS_OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT", "0.06"),
    )
)
OT_TACB_SINKHORN_EPSILON = float(os.environ.get("LFTS_OT_TACB_SINKHORN_EPSILON", "0.08"))
OT_TACB_SINKHORN_ITERATIONS = int(os.environ.get("LFTS_OT_TACB_SINKHORN_ITERATIONS", "8"))
OT_TACB_COSINE_WEIGHT = float(os.environ.get("LFTS_OT_TACB_COSINE_WEIGHT", "1.00"))
OT_TACB_FINE_TRANSITION_TYPES = 8
OT_TACB_CONCEPT_DERIVATIVE_DIM = 10
USE_MPDG = parse_bool_env("LFTS_USE_MPDG", False)
MPDG_CONCEPT_DIM = int(os.environ.get("LFTS_MPDG_CONCEPT_DIM", "16"))
DPCM_MASK_AMPLITUDE = float(os.environ.get("LFTS_DPCM_MASK_AMPLITUDE", "1.00"))
DPCM_PRIOR_INIT_SCALE = float(os.environ.get("LFTS_DPCM_PRIOR_INIT_SCALE", "0.05"))
DPCM_BOUNDARY_CONTRAST_LOSS_WEIGHT = float(os.environ.get("LFTS_DPCM_BOUNDARY_CONTRAST_LOSS_WEIGHT", "0.015"))
DPCM_BOUNDARY_CONTRAST_MARGIN = float(os.environ.get("LFTS_DPCM_BOUNDARY_CONTRAST_MARGIN", "0.35"))
DPCM_BOUNDARY_LOSS_WEIGHT = float(
    os.environ.get(
        "LFTS_DPCM_BOUNDARY_LOSS_WEIGHT",
        f"{min(OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT, 0.02):.4f}",
    )
)
POSTPROCESS_ACCURACY_GUARD = parse_bool_env("LFTS_POSTPROCESS_ACCURACY_GUARD", True)
POSTPROCESS_SCORE_TIE_EPS = float(os.environ.get("LFTS_POSTPROCESS_SCORE_TIE_EPS", "0.003"))
POSTPROCESS_ACCURACY_GUARD_DROP = float(os.environ.get("LFTS_POSTPROCESS_ACCURACY_GUARD_DROP", "0.002"))
TEMPERATURE_SCALING = True
TEMPERATURE_GRID = np.linspace(0.8, 2.8, 21)
N1_THRESHOLD_GRID = np.linspace(
    float(os.environ.get("LFTS_N1_THRESHOLD_MIN", "0.12")),
    float(os.environ.get("LFTS_N1_THRESHOLD_MAX", "0.55")),
    int(os.environ.get("LFTS_N1_THRESHOLD_STEPS", "23")),
)
N1_THRESHOLD_MAX_ACC_DROP = float(os.environ.get("LFTS_N1_THRESHOLD_MAX_ACC_DROP", "0.025"))
HMM_ILLEGAL_TRANSITION_PENALTY = -7.0
HMM_EMISSION_WEIGHT = 0.85
QUALITY_WEIGHTING = False
ENABLE_AMPLITUDE_CLIPPING = False
QUALITY_CLIP_SIGMA = None
EXTENDED_ABLATION_STUDY = parse_bool_env("LFTS_EXTENDED_ABLATION_STUDY", False)
EXTENDED_CONCEPT_WEIGHTS = (0.00, 0.02, 0.03, 0.05, 0.08)
EXTENDED_CHANNEL_STRENGTHS = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20)
EXTENDED_HMM_EMISSION_WEIGHTS = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
VALIDATION_HMM_EMISSION_WEIGHTS = parse_float_tuple_env(
    "LFTS_VALIDATION_HMM_EMISSION_WEIGHTS",
    (0.85, *EXTENDED_HMM_EMISSION_WEIGHTS),
)
EXTENDED_SELF_LOOP_BOOSTS = (0.45, 0.55, 0.65, 0.75)
EXTENDED_ACCURACY_SELECTOR_MODE = "accuracy_macro"
EXPLANATION_FEEDBACK_MODE = os.environ.get("LFTS_EXPLANATION_FEEDBACK_MODE", "clinical")
EXPLANATION_FEEDBACK_TRAINING = (
    os.environ.get("LFTS_EXPLANATION_FEEDBACK_TRAINING", "0") != "0"
    and EXPLANATION_FEEDBACK_MODE.strip().lower().replace("-", "_") not in {"baseline", "none", "off", "no_feedback"}
)
ADAPTIVE_CLINICAL_PRIORS = parse_bool_env("LFTS_ADAPTIVE_CLINICAL_PRIORS", True)
ADAPTIVE_PRIOR_ALPHA = float(os.environ.get("LFTS_ADAPTIVE_PRIOR_ALPHA", "0.35"))
ADAPTIVE_PRIOR_MIN_CLASS_SAMPLES = int(os.environ.get("LFTS_ADAPTIVE_PRIOR_MIN_CLASS_SAMPLES", "100"))
ADAPTIVE_PRIOR_CONTRAST_POWER = float(os.environ.get("LFTS_ADAPTIVE_PRIOR_CONTRAST_POWER", "0.50"))
ADAPTIVE_PRIOR_RELIABILITY_POWER = float(os.environ.get("LFTS_ADAPTIVE_PRIOR_RELIABILITY_POWER", "0.50"))
USE_SEQUENCE_CLINICAL_CONCEPTS = parse_bool_env("LFTS_USE_SEQUENCE_CLINICAL_CONCEPTS", True)
CLINICAL_PRIOR_TRAINABLE = parse_bool_env("LFTS_CLINICAL_PRIOR_TRAINABLE", True)
USE_LATE_CLINICAL_CONCEPT_RESIDUAL = parse_bool_env("LFTS_USE_LATE_CLINICAL_CONCEPT_RESIDUAL", False)
LATE_CONCEPT_RESIDUAL_SCALE = float(os.environ.get("LFTS_LATE_CONCEPT_RESIDUAL_SCALE", "0.05"))
LATE_CONCEPT_RESIDUAL_TRAINABLE = parse_bool_env("LFTS_LATE_CONCEPT_RESIDUAL_TRAINABLE", True)
LATE_CONCEPT_STOP_GRADIENT = parse_bool_env("LFTS_LATE_CONCEPT_STOP_GRADIENT", True)
USE_LATE_CONCEPT_CONFIDENCE_GATE = parse_bool_env("LFTS_USE_LATE_CONCEPT_CONFIDENCE_GATE", False)
LATE_CONCEPT_GATE_LOW_CONFIDENCE = float(os.environ.get("LFTS_LATE_CONCEPT_GATE_LOW_CONFIDENCE", "0.65"))
LATE_CONCEPT_GATE_HIGH_CONFIDENCE = float(os.environ.get("LFTS_LATE_CONCEPT_GATE_HIGH_CONFIDENCE", "0.85"))
LATE_CONCEPT_GATE_MID_VALUE = float(os.environ.get("LFTS_LATE_CONCEPT_GATE_MID_VALUE", "0.50"))
USE_CONCEPT_FEATURE_MODULATION = parse_bool_env("LFTS_USE_CONCEPT_FEATURE_MODULATION", False)
CONCEPT_MODULATION_STRENGTH = float(os.environ.get("LFTS_CONCEPT_MODULATION_STRENGTH", "0.10"))
CONCEPT_MODULATION_STOP_GRADIENT = parse_bool_env("LFTS_CONCEPT_MODULATION_STOP_GRADIENT", True)
CONCEPT_MODULATION_MODE = normalize_concept_modulation_mode(
    os.environ.get("LFTS_CONCEPT_MODULATION_MODE", "real")
)
CONCEPT_TARGET_PRIOR_WEIGHT = float(os.environ.get("LFTS_CONCEPT_TARGET_PRIOR_WEIGHT", "0.15"))
DCCS_CHANNEL_PRIOR_WEIGHT = float(os.environ.get("LFTS_DCCS_CHANNEL_PRIOR_WEIGHT", "0.15"))
LIME_DCCS_FEEDBACK_ALPHA = float(os.environ.get("LFTS_LIME_DCCS_ALPHA", "0.25"))
LIME_DCCS_FEEDBACK_BETA = float(os.environ.get("LFTS_LIME_DCCS_BETA", "0.35"))
LIME_DCCS_FEEDBACK_GAMMA = float(os.environ.get("LFTS_LIME_DCCS_GAMMA", "0.40"))
LIME_DCCS_TARGET_PATH = os.environ.get("LFTS_LIME_DCCS_TARGET_PATH", "").strip()
LIME_DCCS_FEEDBACK_FILTER = os.environ.get(
    "LFTS_LIME_DCCS_FEEDBACK_FILTER",
    "correct_and_high_confidence",
)
LIME_DCCS_MIN_CONFIDENCE = float(os.environ.get("LFTS_LIME_DCCS_MIN_CONFIDENCE", "0.70"))
LIME_DCCS_USE_VAL_FEEDBACK = os.environ.get("LFTS_LIME_DCCS_USE_VAL_FEEDBACK", "0") != "0"
LIME_DCCS_SOFT_SAMPLE_WEIGHTS = os.environ.get("LFTS_LIME_DCCS_SOFT_SAMPLE_WEIGHTS", "1") != "0"
USE_CHANNEL_QUALITY_FEEDBACK = os.environ.get("LFTS_USE_CHANNEL_QUALITY_FEEDBACK", "0") != "0"
USE_CHANNEL_QUALITY_ROUTING = os.environ.get("LFTS_USE_CHANNEL_QUALITY_ROUTING", "0") != "0"
CONCEPT_SAMPLE_WEIGHT_MODE = os.environ.get("LFTS_CONCEPT_SAMPLE_WEIGHT_MODE", "uniform")
CONCEPT_SAMPLE_WEIGHT_FLOOR = float(os.environ.get("LFTS_CONCEPT_SAMPLE_WEIGHT_FLOOR", "0.10"))
CHANNEL_QUALITY_BIAS_STRENGTH = float(os.environ.get("LFTS_CHANNEL_QUALITY_BIAS_STRENGTH", "0.35"))
CONCEPT_OUTPUT_LOSS_WEIGHT = float(os.environ.get("LFTS_CONCEPT_OUTPUT_LOSS_WEIGHT", "0.0"))
ATTENTION_FEEDBACK_LOSS_WEIGHT = float(os.environ.get("LFTS_ATTENTION_FEEDBACK_LOSS_WEIGHT", "0.0"))
CONCEPT_BOTTLENECK_LOSS_WEIGHT = float(os.environ.get("LFTS_CONCEPT_BOTTLENECK_LOSS_WEIGHT", "0.0"))
AUX_FEEDBACK_WARMUP_EPOCHS = int(os.environ.get("LFTS_AUX_FEEDBACK_WARMUP_EPOCHS", "3"))
AUX_FEEDBACK_RAMP_EPOCHS = int(os.environ.get("LFTS_AUX_FEEDBACK_RAMP_EPOCHS", "5"))
AUX_FEEDBACK_START_SCALE = float(os.environ.get("LFTS_AUX_FEEDBACK_START_SCALE", "0.0"))


def _feedback_mode_key(mode):
    return str(mode or "").strip().lower().replace("-", "_")


def uses_concept_bottleneck_feedback(mode):
    return _feedback_mode_key(mode) not in {"baseline", "none", "off", "no_feedback", "self_lime", "self_lime_dccs", "lime", "lime_dccs"}


def uses_lime_dccs_feedback(mode):
    return _feedback_mode_key(mode) in {
        "self_lime",
        "self_lime_dccs",
        "lime",
        "lime_dccs",
        "clinical_calibrated_lime",
        "clinically_calibrated_lime",
        "clinical_calibrated_lime_dccs",
        "clinically_calibrated_lime_dccs",
    }


OT_TACB_DISABLED_LIME_FEEDBACK_MODE = None
if USE_OT_TACB and uses_lime_dccs_feedback(EXPLANATION_FEEDBACK_MODE):
    OT_TACB_DISABLED_LIME_FEEDBACK_MODE = EXPLANATION_FEEDBACK_MODE
    EXPLANATION_FEEDBACK_MODE = "clinical"
    EXPLANATION_FEEDBACK_TRAINING = False


class Logger:
    def __init__(self, filename="Run_Results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SleepDataGenerator(Sequence):
    def __init__(
            self,
            X_raw,
            X_segment,
            y_main,
            batch_size=32,
            shuffle=True,
            sample_weights=None,
            indices=None,
            attention_targets=None,
            concept_targets=None,
            attention_sample_weights=None,
            concept_bottleneck_sample_weights=None,
            domain_targets=None,
            include_domain_alignment=False,
            include_auxiliary_sample_weights=False,
    ):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.y_main = y_main
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.attention_targets = attention_targets
        self.concept_targets = concept_targets
        self._has_attention_sample_weights = (
            attention_sample_weights is not None or bool(include_auxiliary_sample_weights)
        )
        self._has_concept_bottleneck_sample_weights = (
            concept_bottleneck_sample_weights is not None or bool(include_auxiliary_sample_weights)
        )
        self.attention_sample_weights = attention_sample_weights
        self.concept_bottleneck_sample_weights = concept_bottleneck_sample_weights
        self.include_domain_alignment = bool(include_domain_alignment)
        self.domain_targets = (
            np.asarray(domain_targets, dtype=np.float32).reshape(-1)
            if domain_targets is not None
            else np.zeros(len(self.X_raw), dtype=np.float32)
        )
        self.indices = np.array(indices if indices is not None else np.arange(len(self.X_raw)))
        if self.include_domain_alignment and len(self.domain_targets) != len(self.X_raw):
            raise ValueError(
                "domain_targets must align with X_raw when domain alignment is enabled: "
                f"got {len(self.domain_targets)} targets for {len(self.X_raw)} epochs"
            )
        if self.attention_targets is None or self.concept_targets is None:
            from src.model_training import build_explanation_feedback_targets

            feedback_targets = build_explanation_feedback_targets(
                self.X_segment,
                self.y_main,
                num_classes=self.y_main.shape[1],
                concept_version=CLINICAL_CONCEPT_VERSION,
            )
            self.attention_targets = feedback_targets["attention_feedback_targets"]
            self.concept_targets = feedback_targets["concept_targets"]
        if self.attention_sample_weights is None:
            self.attention_sample_weights = np.ones(len(self.X_raw), dtype=np.float32)
        if self.concept_bottleneck_sample_weights is None:
            self.concept_bottleneck_sample_weights = np.ones(len(self.X_raw), dtype=np.float32)
        self.attention_feedback_scale = 1.0
        self.concept_bottleneck_feedback_scale = 1.0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def set_auxiliary_feedback_scale(self, scale):
        scale = float(np.clip(scale, 0.0, 1.0))
        self.attention_feedback_scale = scale
        self.concept_bottleneck_feedback_scale = scale

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        batch_X_raw = self.X_raw[batch_indices]
        batch_X_segment = self.X_segment[batch_indices]
        batch_y_main = self.y_main[batch_indices]
        batch_y_attention = self.attention_targets[batch_indices]
        batch_y_concepts = self.concept_targets[batch_indices]
        batch_y_domain = self.domain_targets[batch_indices].astype(np.float32)
        batch_weights = (
            self.sample_weights[batch_indices]
            if self.sample_weights is not None
            else np.ones(len(batch_indices), dtype=np.float32)
        )
        batch_attention_weights = (
            self.attention_sample_weights[batch_indices] * self.attention_feedback_scale
        )
        batch_concept_weights = (
            self.concept_bottleneck_sample_weights[batch_indices] * self.concept_bottleneck_feedback_scale
        )

        targets = {
            "main_output": batch_y_main,
            "channel_attention_weights": batch_y_attention,
            "concept_output": batch_y_main,
            "concept_bottleneck_output": batch_y_concepts,
        }
        sample_weight_map = {"main_output": batch_weights}
        if self._has_attention_sample_weights:
            sample_weight_map["channel_attention_weights"] = batch_attention_weights
        if self._has_concept_bottleneck_sample_weights:
            sample_weight_map["concept_bottleneck_output"] = batch_concept_weights
        if self.include_domain_alignment:
            targets["domain_alignment_features"] = batch_y_domain

        return (
            {"raw_input": batch_X_raw, "segment_input": batch_X_segment},
            targets,
            sample_weight_map,
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def _coerce_file_sample_counts(total_len, file_sample_counts=None):
    if file_sample_counts is None:
        return [int(total_len)]
    counts = [int(count) for count in file_sample_counts if int(count) > 0]
    if sum(counts) != int(total_len):
        raise ValueError(
            "file_sample_counts must sum to the number of epochs: "
            f"sum={sum(counts)}, epochs={total_len}"
        )
    return counts


def build_sequence_windows(
        total_len,
        file_sample_counts=None,
        seq_len=25,
        stride=5,
        include_last=True,
        allow_short=True,
):
    seq_len = max(1, int(seq_len))
    stride = max(1, int(stride))
    windows = []
    offset = 0
    for count in _coerce_file_sample_counts(total_len, file_sample_counts):
        record_start = offset
        record_end = offset + count
        if count < seq_len:
            if allow_short:
                windows.append((record_start, count))
            offset = record_end
            continue

        last_start = record_end - seq_len
        starts = list(range(record_start, last_start + 1, stride))
        if include_last and starts[-1] != last_start:
            starts.append(last_start)
        windows.extend((start, seq_len) for start in starts)
        offset = record_end
    return windows


def _window_epoch_indices(start, valid_len, seq_len):
    valid_len = max(1, int(valid_len))
    seq_len = max(1, int(seq_len))
    indices = np.arange(int(start), int(start) + valid_len, dtype=np.int64)
    if valid_len < seq_len:
        pad = np.full(seq_len - valid_len, indices[-1], dtype=np.int64)
        indices = np.concatenate([indices, pad])
    return indices


def _coerce_epoch_target_matrix(targets, total_len, default_width):
    if targets is None:
        return np.zeros((int(total_len), int(default_width)), dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    return target_array


class SleepSequenceDataGenerator(Sequence):
    def __init__(
            self,
            X_raw,
            X_segment,
            y_main,
            file_sample_counts,
            seq_len=25,
            stride=5,
            batch_size=32,
            shuffle=True,
            sample_weights=None,
            domain_targets=None,
            include_domain_alignment=False,
            concept_targets=None,
            include_concept_language=False,
            concept_sample_weights=None,
            transition_boundary_targets=None,
            transition_boundary_sample_weights=None,
            include_transition_boundary_auxiliary=False,
            concept_derivative_targets=None,
            concept_derivative_sample_weights=None,
            include_concept_derivative_auxiliary=False,
            include_n1_contrastive=False,
            domain_alignment_stage_conditioned=False,
            balanced_sequence_sampling=False,
            balance_n1_window_boost=1.75,
            balance_n3_window_boost=1.25,
            balance_transition_window_boost=1.15,
            macro_auxiliary_scale=1.0,
    ):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.y_main = y_main
        self.file_sample_counts = _coerce_file_sample_counts(len(self.X_raw), file_sample_counts)
        self.seq_len = max(1, int(seq_len))
        self.stride = max(1, int(stride))
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32)
            if sample_weights is not None
            else np.ones(len(self.X_raw), dtype=np.float32)
        )
        self.include_domain_alignment = bool(include_domain_alignment)
        self.include_concept_language = bool(include_concept_language)
        self.include_transition_boundary_auxiliary = bool(include_transition_boundary_auxiliary)
        self.include_concept_derivative_auxiliary = bool(include_concept_derivative_auxiliary)
        self.include_n1_contrastive = bool(include_n1_contrastive)
        self.domain_alignment_stage_conditioned = bool(domain_alignment_stage_conditioned)
        self.balanced_sequence_sampling = bool(balanced_sequence_sampling)
        self.balance_n1_window_boost = float(balance_n1_window_boost)
        self.balance_n3_window_boost = float(balance_n3_window_boost)
        self.balance_transition_window_boost = float(balance_transition_window_boost)
        self.macro_auxiliary_scale = float(np.clip(macro_auxiliary_scale, 0.0, 1.0))
        self.domain_targets = (
            np.asarray(domain_targets, dtype=np.float32).reshape(-1)
            if domain_targets is not None
            else np.zeros(len(self.X_raw), dtype=np.float32)
        )
        if self.include_domain_alignment and len(self.domain_targets) != len(self.X_raw):
            raise ValueError(
                "domain_targets must align with X_raw when domain alignment is enabled: "
                f"got {len(self.domain_targets)} targets for {len(self.X_raw)} epochs"
            )
        self.concept_targets = (
            np.asarray(concept_targets, dtype=np.float32)
            if concept_targets is not None
            else np.zeros((len(self.X_raw), 9), dtype=np.float32)
        )
        self.concept_sample_weights = (
            np.asarray(concept_sample_weights, dtype=np.float32).reshape(-1)
            if concept_sample_weights is not None
            else np.ones(len(self.X_raw), dtype=np.float32)
        )
        if self.include_concept_language:
            if len(self.concept_targets) != len(self.X_raw):
                raise ValueError(
                    "concept_targets must align with X_raw when concept language is enabled: "
                    f"got {len(self.concept_targets)} targets for {len(self.X_raw)} epochs"
                )
            if len(self.concept_sample_weights) != len(self.X_raw):
                raise ValueError(
                    "concept_sample_weights must align with X_raw when concept language is enabled: "
                    f"got {len(self.concept_sample_weights)} weights for {len(self.X_raw)} epochs"
                )
        self.transition_boundary_targets = _coerce_epoch_target_matrix(
            transition_boundary_targets,
            len(self.X_raw),
            1,
        )
        self.transition_boundary_sample_weights = (
            np.asarray(transition_boundary_sample_weights, dtype=np.float32).reshape(-1)
            if transition_boundary_sample_weights is not None
            else np.ones(len(self.X_raw), dtype=np.float32)
        )
        if self.include_transition_boundary_auxiliary:
            if len(self.transition_boundary_targets) != len(self.X_raw):
                raise ValueError(
                    "transition_boundary_targets must align with X_raw when boundary auxiliary is enabled: "
                    f"got {len(self.transition_boundary_targets)} targets for {len(self.X_raw)} epochs"
                )
            if len(self.transition_boundary_sample_weights) != len(self.X_raw):
                raise ValueError(
                    "transition_boundary_sample_weights must align with X_raw when boundary auxiliary is enabled: "
                    f"got {len(self.transition_boundary_sample_weights)} weights for {len(self.X_raw)} epochs"
                )
        self.concept_derivative_targets = _coerce_epoch_target_matrix(
            concept_derivative_targets,
            len(self.X_raw),
            OT_TACB_CONCEPT_DERIVATIVE_DIM,
        )
        self.concept_derivative_sample_weights = (
            np.asarray(concept_derivative_sample_weights, dtype=np.float32).reshape(-1)
            if concept_derivative_sample_weights is not None
            else np.ones(len(self.X_raw), dtype=np.float32)
        )
        if self.include_concept_derivative_auxiliary:
            if len(self.concept_derivative_targets) != len(self.X_raw):
                raise ValueError(
                    "concept_derivative_targets must align with X_raw when concept derivative auxiliary is enabled: "
                    f"got {len(self.concept_derivative_targets)} targets for {len(self.X_raw)} epochs"
                )
            if len(self.concept_derivative_sample_weights) != len(self.X_raw):
                raise ValueError(
                    "concept_derivative_sample_weights must align with X_raw when concept derivative auxiliary is enabled: "
                    f"got {len(self.concept_derivative_sample_weights)} weights for {len(self.X_raw)} epochs"
                )
        self.windows = build_sequence_windows(
            len(self.X_raw),
            self.file_sample_counts,
            seq_len=self.seq_len,
            stride=self.stride,
            include_last=True,
            allow_short=True,
        )
        if not self.windows:
            raise ValueError("No sequence windows could be built for training.")
        self.window_sampling_probabilities = self._build_window_sampling_probabilities()
        self.indices = np.arange(len(self.windows))
        if self.shuffle and self.balanced_sequence_sampling:
            self._resample_balanced_indices()
        elif self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def _window_valid_labels(self, start, valid_len):
        valid_len = max(1, int(valid_len))
        labels = np.argmax(self.y_main[int(start): int(start) + valid_len], axis=1).astype(np.int32)
        return labels

    def _build_window_sampling_probabilities(self):
        if not self.balanced_sequence_sampling:
            return None
        epoch_labels = np.argmax(self.y_main, axis=1).astype(np.int32)
        counts = np.bincount(epoch_labels, minlength=self.y_main.shape[-1]).astype(np.float32)
        total = float(np.sum(counts))
        inv_freq = total / np.maximum(counts * max(1, len(counts)), 1.0)
        inv_freq = np.clip(inv_freq, 0.65, 3.50)
        weights = []
        for start, valid_len in self.windows:
            labels = self._window_valid_labels(start, valid_len)
            if len(labels) == 0:
                weights.append(1.0)
                continue
            window_weight = float(np.mean(inv_freq[labels]))
            if np.any(labels == 1):
                window_weight *= self.balance_n1_window_boost
            if np.any(labels == 3):
                window_weight *= self.balance_n3_window_boost
            if np.any(labels[1:] != labels[:-1]):
                window_weight *= self.balance_transition_window_boost
            weights.append(window_weight)
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 1e-6, None)
        return weights / np.sum(weights)

    def _resample_balanced_indices(self):
        self.indices = np.random.choice(
            np.arange(len(self.windows)),
            size=len(self.windows),
            replace=True,
            p=self.window_sampling_probabilities,
        )

    def set_macro_auxiliary_scale(self, scale):
        self.macro_auxiliary_scale = float(np.clip(scale, 0.0, 1.0))

    def __getitem__(self, index):
        window_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        raw_batch = []
        segment_batch = []
        target_batch = []
        weight_batch = []
        domain_batch = []
        concept_target_batch = []
        concept_weight_batch = []
        boundary_target_batch = []
        boundary_weight_batch = []
        derivative_target_batch = []
        derivative_weight_batch = []
        for window_idx in window_indices:
            start, valid_len = self.windows[int(window_idx)]
            epoch_indices = _window_epoch_indices(start, valid_len, self.seq_len)
            raw_batch.append(self.X_raw[epoch_indices])
            segment_batch.append(self.X_segment[epoch_indices])
            target_batch.append(self.y_main[epoch_indices])
            if self.include_domain_alignment:
                domain_values = self.domain_targets[epoch_indices].astype(np.float32)
                if self.domain_alignment_stage_conditioned:
                    stage_values = np.argmax(self.y_main[epoch_indices], axis=1).astype(np.float32)
                    if valid_len < self.seq_len:
                        domain_values[valid_len:] = -1.0
                        stage_values[valid_len:] = -1.0
                    domain_batch.append(np.stack([domain_values, stage_values], axis=-1))
                else:
                    if valid_len < self.seq_len:
                        domain_values[valid_len:] = -1.0
                    domain_batch.append(domain_values)
            if self.include_concept_language:
                concept_target_batch.append(self.concept_targets[epoch_indices].astype(np.float32))
                concept_weights = self.concept_sample_weights[epoch_indices].astype(np.float32).copy()
                if valid_len < self.seq_len:
                    concept_weights[valid_len:] = 0.0
                concept_weight_batch.append(concept_weights)
            if self.include_transition_boundary_auxiliary:
                boundary_target_batch.append(
                    self.transition_boundary_targets[epoch_indices].astype(np.float32)
                )
                boundary_weights = self.transition_boundary_sample_weights[epoch_indices].astype(np.float32).copy()
                if valid_len < self.seq_len:
                    boundary_weights[valid_len:] = 0.0
                boundary_weight_batch.append(boundary_weights)
            if self.include_concept_derivative_auxiliary:
                derivative_target_batch.append(
                    self.concept_derivative_targets[epoch_indices].astype(np.float32)
                )
                derivative_weights = self.concept_derivative_sample_weights[epoch_indices].astype(np.float32).copy()
                if valid_len < self.seq_len:
                    derivative_weights[valid_len:] = 0.0
                derivative_weight_batch.append(derivative_weights)
            weights = self.sample_weights[epoch_indices].astype(np.float32).copy()
            if valid_len < self.seq_len:
                weights[valid_len:] = 0.0
            weight_batch.append(weights)

        targets = {"main_output": np.asarray(target_batch, dtype=np.float32)}
        if self.include_domain_alignment:
            targets["domain_alignment_features"] = np.asarray(domain_batch, dtype=np.float32)
        sample_weight_map = {"main_output": np.asarray(weight_batch, dtype=np.float32)}
        if self.include_concept_language:
            concept_weights = np.asarray(concept_weight_batch, dtype=np.float32)
            targets["concept_output"] = np.asarray(target_batch, dtype=np.float32)
            targets["concept_bottleneck_output"] = np.asarray(concept_target_batch, dtype=np.float32)
            sample_weight_map["concept_output"] = concept_weights * self.macro_auxiliary_scale
            sample_weight_map["concept_bottleneck_output"] = concept_weights * self.macro_auxiliary_scale
        if self.include_transition_boundary_auxiliary:
            targets["transition_boundary_output"] = np.asarray(boundary_target_batch, dtype=np.float32)
            sample_weight_map["transition_boundary_output"] = (
                np.asarray(boundary_weight_batch, dtype=np.float32) * self.macro_auxiliary_scale
            )
        if self.include_concept_derivative_auxiliary:
            targets["concept_derivative_output"] = np.asarray(derivative_target_batch, dtype=np.float32)
            sample_weight_map["concept_derivative_output"] = (
                np.asarray(derivative_weight_batch, dtype=np.float32) * self.macro_auxiliary_scale
            )
        if self.include_n1_contrastive:
            targets["n1_contrastive_features"] = targets["main_output"]
            sample_weight_map["n1_contrastive_features"] = (
                sample_weight_map["main_output"] * self.macro_auxiliary_scale
            )

        return (
            {
                "raw_input": np.asarray(raw_batch, dtype=np.float32),
                "segment_input": np.asarray(segment_batch, dtype=np.float32),
            },
            targets,
            sample_weight_map,
        )

    def on_epoch_end(self):
        if self.shuffle and self.balanced_sequence_sampling:
            self._resample_balanced_indices()
        elif self.shuffle:
            np.random.shuffle(self.indices)


class SleepSequenceInputGenerator(Sequence):
    def __init__(
            self,
            X_raw,
            X_segment,
            file_sample_counts,
            seq_len=25,
            stride=5,
            batch_size=32,
    ):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.file_sample_counts = _coerce_file_sample_counts(len(self.X_raw), file_sample_counts)
        self.seq_len = max(1, int(seq_len))
        self.stride = max(1, int(stride))
        self.batch_size = max(1, int(batch_size))
        self.windows = build_sequence_windows(
            len(self.X_raw),
            self.file_sample_counts,
            seq_len=self.seq_len,
            stride=self.stride,
            include_last=True,
            allow_short=True,
        )
        if not self.windows:
            raise ValueError("No sequence windows could be built for prediction.")

    def __len__(self):
        return int(np.ceil(len(self.windows) / self.batch_size))

    def __getitem__(self, index):
        batch_windows = self.windows[index * self.batch_size: (index + 1) * self.batch_size]
        raw_batch = []
        segment_batch = []
        for start, valid_len in batch_windows:
            epoch_indices = _window_epoch_indices(start, valid_len, self.seq_len)
            raw_batch.append(self.X_raw[epoch_indices])
            segment_batch.append(self.X_segment[epoch_indices])
        return {
            "raw_input": np.asarray(raw_batch, dtype=np.float32),
            "segment_input": np.asarray(segment_batch, dtype=np.float32),
        }


class AuxiliaryFeedbackScheduler(Callback):
    def __init__(
            self,
            generators,
            warmup_epochs=3,
            ramp_epochs=5,
            start_scale=0.0,
            final_scale=1.0,
    ):
        super().__init__()
        self.generators = list(generators)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.start_scale = float(np.clip(start_scale, 0.0, 1.0))
        self.final_scale = float(np.clip(final_scale, 0.0, 1.0))

    def _scale_for_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            return self.start_scale
        progress = min((epoch - self.warmup_epochs + 1) / float(self.ramp_epochs), 1.0)
        return self.start_scale + progress * (self.final_scale - self.start_scale)

    def on_epoch_begin(self, epoch, logs=None):
        scale = self._scale_for_epoch(epoch)
        for generator in self.generators:
            if generator is not None and hasattr(generator, "set_auxiliary_feedback_scale"):
                generator.set_auxiliary_feedback_scale(scale)
        print(f"\nAuxiliary feedback scale for epoch {epoch + 1}: {scale:.3f}")


class MacroAuxiliaryScheduler(Callback):
    def __init__(
            self,
            generators,
            warmup_epochs=3,
            ramp_epochs=5,
            start_scale=0.0,
            final_scale=1.0,
    ):
        super().__init__()
        self.generators = list(generators)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.start_scale = float(np.clip(start_scale, 0.0, 1.0))
        self.final_scale = float(np.clip(final_scale, 0.0, 1.0))

    def _scale_for_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            return self.start_scale
        progress = min((epoch - self.warmup_epochs + 1) / float(self.ramp_epochs), 1.0)
        return self.start_scale + progress * (self.final_scale - self.start_scale)

    def on_epoch_begin(self, epoch, logs=None):
        scale = self._scale_for_epoch(epoch)
        for generator in self.generators:
            if generator is not None and hasattr(generator, "set_macro_auxiliary_scale"):
                generator.set_macro_auxiliary_scale(scale)
        print(f"\nMacro auxiliary scale for epoch {epoch + 1}: {scale:.3f}")


MICRO_EPOCH_ENCODER_PREFIXES = (
    "td_raw_",
    "td_v3_raw_",
    "td_segment_",
    "td_v3_segment_",
)
MICRO_EPOCH_ENCODER_LAYER_NAMES = {
    "sequence_raw_domain_mixstyle",
    "sequence_v3_raw_domain_mixstyle",
    "sequence_segment_domain_mixstyle",
    "sequence_v3_segment_domain_mixstyle",
    "raw_feature_dropout",
    "v3_raw_feature_dropout",
    "segment_feature_dropout",
    "v3_segment_feature_dropout",
}


def set_micro_epoch_encoder_trainable(model, trainable):
    affected_layers = []
    for layer in model.layers:
        if layer.name in MICRO_EPOCH_ENCODER_LAYER_NAMES or layer.name.startswith(MICRO_EPOCH_ENCODER_PREFIXES):
            layer.trainable = bool(trainable)
            affected_layers.append(layer.name)
    return affected_layers


def fit_with_micro_macro_schedule(
        model,
        train_gen,
        val_gen,
        callbacks,
        total_epochs,
        base_learning_rate,
        recompile_fn,
):
    pretrain_epochs = max(0, min(int(MICRO_PRETRAIN_EPOCHS), int(total_epochs)))
    freeze_epochs = max(0, min(int(MACRO_STABILIZATION_EPOCHS), int(total_epochs) - pretrain_epochs))
    finetune_epochs = max(0, int(total_epochs) - pretrain_epochs - freeze_epochs)
    if pretrain_epochs + freeze_epochs + finetune_epochs <= 0:
        return

    print(
        "Micro-macro staged optimization: "
        f"micro_pretrain_epochs={pretrain_epochs}, "
        f"macro_stabilization_epochs={freeze_epochs}, "
        f"joint_finetune_epochs={finetune_epochs}, "
        f"macro_aux_ramp_epochs={MACRO_AUX_RAMP_EPOCHS}, "
        f"finetune_lr_scale={MICRO_FINETUNE_LR_SCALE:.3f}"
    )

    phases = []
    start_epoch = 0
    if pretrain_epochs > 0:
        phases.append(
            {
                "name": "micro morphology warm-start",
                "start": start_epoch,
                "end": start_epoch + pretrain_epochs,
                "micro_trainable": True,
                "learning_rate": float(base_learning_rate),
            }
        )
        start_epoch += pretrain_epochs
    if freeze_epochs > 0:
        phases.append(
            {
                "name": "macro concept-transition stabilization",
                "start": start_epoch,
                "end": start_epoch + freeze_epochs,
                "micro_trainable": False,
                "learning_rate": float(base_learning_rate),
            }
        )
        start_epoch += freeze_epochs
    if finetune_epochs > 0:
        phases.append(
            {
                "name": "joint micro-macro fine-tuning",
                "start": start_epoch,
                "end": start_epoch + finetune_epochs,
                "micro_trainable": True,
                "learning_rate": float(base_learning_rate) * float(MICRO_FINETUNE_LR_SCALE),
            }
        )

    for phase in phases:
        is_final_phase = phase is phases[-1]
        for callback in callbacks:
            if hasattr(callback, "restore_on_train_end"):
                callback.restore_on_train_end = bool(is_final_phase)
        affected_layers = set_micro_epoch_encoder_trainable(model, phase["micro_trainable"])
        recompile_fn(model, learning_rate=phase["learning_rate"])
        model.stop_training = False
        print(
            "\nMicro-macro phase: "
            f"{phase['name']} | "
            f"epochs={phase['start'] + 1}-{phase['end']} | "
            f"micro_encoder_trainable={phase['micro_trainable']} | "
            f"lr={phase['learning_rate']:.6f} | "
            f"affected_micro_layers={len(affected_layers)}"
        )
        model.fit(
            train_gen,
            initial_epoch=phase["start"],
            epochs=phase["end"],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=2,
        )
        if model.stop_training:
            print(
                "Micro-macro staged optimization stopped early during "
                f"{phase['name']}."
            )
            break


class SleepInputGenerator(Sequence):
    def __init__(self, X_raw, X_segment, batch_size=64):
        self.X_raw = X_raw
        self.X_segment = X_segment
        self.batch_size = int(batch_size)

    def __len__(self):
        return int(np.ceil(len(self.X_raw) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.X_raw))
        return {
            "raw_input": self.X_raw[start:end],
            "segment_input": self.X_segment[start:end],
        }


def _as_numpy_output(output):
    if hasattr(output, "numpy"):
        return output.numpy()
    return np.asarray(output)


def _concatenate_model_outputs(batch_outputs):
    first_output = batch_outputs[0]
    if isinstance(first_output, (list, tuple)):
        return [
            np.concatenate([_as_numpy_output(outputs[idx]) for outputs in batch_outputs], axis=0)
            for idx in range(len(first_output))
        ]
    if isinstance(first_output, dict):
        return {
            name: np.concatenate([_as_numpy_output(outputs[name]) for outputs in batch_outputs], axis=0)
            for name in first_output
        }
    return np.concatenate([_as_numpy_output(outputs) for outputs in batch_outputs], axis=0)


def _call_model_batch(model, inputs):
    if callable(model):
        return model(inputs, training=False)
    if hasattr(model, "predict"):
        return model.predict(
            [inputs["raw_input"], inputs["segment_input"]],
            batch_size=len(inputs["raw_input"]),
            verbose=0,
        )
    raise TypeError(f"Model object {type(model)!r} is neither callable nor predict-capable.")


def _slice_model_inputs(inputs, start, end):
    return {name: value[start:end] for name, value in inputs.items()}


def _is_recoverable_predict_oom(error):
    message = str(error).lower()
    return (
        isinstance(error, tf.errors.ResourceExhaustedError)
        or "resource_exhausted" in message
        or "no algorithm worked" in message
        or "cudnn" in message
        or "scratch space" in message
        or "failed to allocate" in message
        or "out of memory" in message
    )


def _call_model_batch_with_auto_split(model, inputs):
    try:
        return _call_model_batch(model, inputs)
    except (
            tf.errors.ResourceExhaustedError,
            tf.errors.NotFoundError,
            tf.errors.UnknownError,
            tf.errors.InternalError,
    ) as error:
        batch_len = len(inputs["raw_input"])
        if batch_len <= 1 or not _is_recoverable_predict_oom(error):
            raise
        mid = max(1, batch_len // 2)
        print(
            "Prediction batch exceeded GPU workspace; "
            f"splitting batch {batch_len} into {mid}+{batch_len - mid}."
        )
        return _concatenate_model_outputs(
            [
                _call_model_batch_with_auto_split(model, _slice_model_inputs(inputs, 0, mid)),
                _call_model_batch_with_auto_split(model, _slice_model_inputs(inputs, mid, batch_len)),
            ]
        )


def predict_sequence_model_outputs(
        model,
        X_raw,
        X_segment,
        file_sample_counts,
        batch_size=64,
        seq_len=25,
        sequence_stride=5,
        verbose=0,
):
    if len(X_raw) != len(X_segment):
        raise ValueError("X_raw and X_segment must contain the same number of samples.")
    if len(X_raw) == 0:
        raise ValueError("Cannot predict sequence model outputs for an empty input array.")

    generator = SleepSequenceInputGenerator(
        X_raw,
        X_segment,
        file_sample_counts=file_sample_counts,
        seq_len=seq_len,
        stride=sequence_stride,
        batch_size=batch_size,
    )
    output_sums = None
    output_counts = None
    window_offset = 0
    batch_range = range(len(generator))
    if verbose:
        batch_range = tqdm(batch_range, total=len(generator), desc="Predict sequences", leave=False)

    for batch_idx in batch_range:
        inputs = generator[batch_idx]
        outputs = _call_model_batch_with_auto_split(model, inputs)
        if isinstance(outputs, dict):
            main_probs = outputs.get("main_output")
        else:
            main_probs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if main_probs is None:
            raise ValueError("Sequence model outputs do not contain main_output probabilities.")
        main_probs = _as_numpy_output(main_probs).astype(np.float32)
        if main_probs.ndim != 3:
            raise ValueError(
                "Sequence model must return probabilities with shape "
                f"(batch, seq_len, classes), got {main_probs.shape}"
            )
        if output_sums is None:
            output_sums = np.zeros((len(X_raw), main_probs.shape[-1]), dtype=np.float32)
            output_counts = np.zeros((len(X_raw), 1), dtype=np.float32)

        batch_windows = generator.windows[window_offset: window_offset + len(main_probs)]
        for row_idx, (start, valid_len) in enumerate(batch_windows):
            epoch_indices = _window_epoch_indices(start, valid_len, generator.seq_len)[:valid_len]
            output_sums[epoch_indices] += main_probs[row_idx, :valid_len]
            output_counts[epoch_indices] += 1.0
        window_offset += len(main_probs)

    if output_sums is None:
        raise ValueError("Sequence prediction produced no outputs.")
    missing = output_counts[:, 0] <= 0
    if np.any(missing):
        raise ValueError(f"Sequence prediction missed {int(np.sum(missing))} epoch(s).")
    return output_sums / np.maximum(output_counts, 1.0)


def predict_model_outputs(
        model,
        X_raw,
        X_segment,
        batch_size=64,
        verbose=0,
        sequence_mode=False,
        file_sample_counts=None,
        seq_len=25,
        sequence_stride=5,
):
    if sequence_mode:
        if file_sample_counts is None:
            raise ValueError("file_sample_counts is required for sequence model prediction.")
        return predict_sequence_model_outputs(
            model,
            X_raw,
            X_segment,
            file_sample_counts=file_sample_counts,
            batch_size=batch_size,
            seq_len=seq_len,
            sequence_stride=sequence_stride,
            verbose=verbose,
        )

    if len(X_raw) != len(X_segment):
        raise ValueError("X_raw and X_segment must contain the same number of samples.")
    if len(X_raw) == 0:
        raise ValueError("Cannot predict model outputs for an empty input array.")

    batch_size = max(1, int(batch_size))
    batch_starts = range(0, len(X_raw), batch_size)
    if verbose:
        batch_starts = tqdm(
            batch_starts,
            total=int(np.ceil(len(X_raw) / batch_size)),
            desc="Predict",
            leave=False,
        )

    batch_outputs = []
    for start in batch_starts:
        end = min(start + batch_size, len(X_raw))
        outputs = _call_model_batch_with_auto_split(
            model,
            {
                "raw_input": X_raw[start:end],
                "segment_input": X_segment[start:end],
            },
        )
        batch_outputs.append(outputs)
    return _concatenate_model_outputs(batch_outputs)


class ModelWrapperForLIME:
    def __init__(self, base_model):
        self.base_model = base_model

    def predict(self, inputs, **kwargs):
        verbose = kwargs.pop("verbose", 0)
        batch_size = kwargs.pop("batch_size", 64)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            preds = predict_model_outputs(
                self.base_model,
                inputs[0],
                inputs[1],
                batch_size=batch_size,
                verbose=verbose,
            )
        else:
            preds = self.base_model(inputs, training=False)
            if isinstance(preds, (list, tuple)):
                preds = [_as_numpy_output(output) for output in preds]
            elif isinstance(preds, dict):
                preds = {name: _as_numpy_output(output) for name, output in preds.items()}
            else:
                preds = _as_numpy_output(preds)
        main_probs, _, _ = extract_prediction_heads(preds)
        return main_probs


def normalize_lime_explanation(explanation, num_channels, num_segments):
    if isinstance(explanation, dict):
        return explanation

    if isinstance(explanation, (list, tuple)) and len(explanation) >= 5:
        target_class, original_pred, segment_importance, raw_dccs, calibrated_dccs = explanation[:5]
        zeros_matrix = np.zeros((num_channels, num_segments), dtype=np.float32)
        zeros_vector = np.zeros((num_segments,), dtype=np.float32)
        return {
            "target_class": int(target_class),
            "original_pred": np.asarray(original_pred, dtype=np.float32),
            "segment_importance": np.asarray(segment_importance, dtype=np.float32),
            "raw_dccs": np.asarray(raw_dccs, dtype=np.float32),
            "calibrated_dccs": np.asarray(calibrated_dccs, dtype=np.float32),
            "segment_matrix": zeros_matrix,
            "prior_weighted_matrix": zeros_matrix,
            "temporal_net": zeros_vector,
            "cooperation_score": zeros_vector,
            "conflict_score": zeros_vector,
            "prior_alignment": 0.0,
            "channel_disagreement": np.zeros((num_channels,), dtype=np.float32),
            "clinical_concept_names": [],
            "clinical_concept_matrix": np.zeros((0, num_segments), dtype=np.float32),
            "clinical_concept_evidence_matrix": np.zeros((0, num_segments), dtype=np.float32),
            "raw_concept_dccs": np.zeros((0,), dtype=np.float32),
            "calibrated_concept_dccs": np.zeros((0,), dtype=np.float32),
            "concept_prior": np.zeros((0,), dtype=np.float32),
            "concept_alignment": 0.0,
            "perturbation_mode": "legacy",
        }

    raise TypeError(f"Unsupported LIME explanation type: {type(explanation)!r}")


def call_with_supported_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(*args, **kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **supported_kwargs)


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    if DETERMINISTIC_TF:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def scale_learning_rate_for_batch(
        base_learning_rate,
        global_batch_size,
        reference_batch_size=REFERENCE_BATCH_SIZE,
        max_learning_rate=MAX_MULTI_GPU_LEARNING_RATE,
):
    batch_ratio = max(float(global_batch_size) / float(reference_batch_size), 1.0)
    scaled_learning_rate = float(base_learning_rate) * np.sqrt(batch_ratio)
    return min(scaled_learning_rate, float(max_learning_rate))


def get_strategy_worker_devices(strategy):
    try:
        return tuple(strategy.extended.worker_devices)
    except (AttributeError, RuntimeError, TypeError):
        return ()


def configure_training_strategy():
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass

    if len(physical_gpus) > 1:
        mirrored_devices = [f"/GPU:{idx}" for idx in range(len(physical_gpus))]
        strategy = tf.distribute.MirroredStrategy(devices=mirrored_devices)
    else:
        strategy = tf.distribute.get_strategy()

    replicas = int(strategy.num_replicas_in_sync)
    worker_devices = get_strategy_worker_devices(strategy)
    train_batch_size = PER_REPLICA_BATCH_SIZE * replicas
    # Selector/evaluation inference calls model(inputs) directly instead of
    # distributing a Keras predict step, so keep this as a per-process batch.
    predict_batch_size = PREDICT_PER_REPLICA_BATCH_SIZE
    learning_rate = scale_learning_rate_for_batch(
        BASE_LEARNING_RATE,
        train_batch_size,
    )
    return strategy, replicas, train_batch_size, predict_batch_size, learning_rate, physical_gpus, worker_devices


def unpack_quality_load_result(result, include_channel_quality):
    if include_channel_quality:
        X_raw, y_raw, quality_scores, channel_quality = result
        return X_raw, y_raw, quality_scores, channel_quality
    X_raw, y_raw, quality_scores = result
    return X_raw, y_raw, quality_scores, None


def _load_npz_array_key(data, keys):
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32)
    return None


def _validate_lime_dccs_array(values, expected_len, split_name):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"{split_name} LIME-DCCS targets must have shape (samples, 3), got {values.shape}"
        )
    if values.shape[0] != expected_len:
        raise ValueError(
            f"{split_name} LIME-DCCS targets have {values.shape[0]} rows, expected {expected_len}"
        )
    return values


def _validate_optional_vector(values, expected_len, split_name, name, dtype=np.float32):
    if values is None:
        return None
    values = np.asarray(values, dtype=dtype).reshape(-1)
    if values.shape[0] != expected_len:
        raise ValueError(
            f"{split_name} {name} has {values.shape[0]} rows, expected {expected_len}"
        )
    return values


def _load_split_array(data, split_name, aliases):
    keys = []
    for alias in aliases:
        keys.extend(
            [
                f"{split_name}_{alias}",
                f"{split_name}_lime_{alias}",
                f"{split_name}_lime_dccs_{alias}",
                f"{split_name}_{alias}_targets",
            ]
        )
    return _load_npz_array_key(data, keys)


def _split_combined_array(values, train_len, val_len, name):
    if values is None:
        return None, None
    values = np.asarray(values)
    expected = train_len + val_len
    if values.shape[0] != expected:
        raise ValueError(f"Combined {name} must contain {expected} rows, got {values.shape[0]}")
    return values[:train_len], values[train_len:]


def load_lime_dccs_feedback_data(target_path, train_len, val_len):
    empty = {
        "train_targets": None,
        "val_targets": None,
        "train_confidence": None,
        "val_confidence": None,
        "train_predicted_labels": None,
        "val_predicted_labels": None,
        "train_correct": None,
        "val_correct": None,
        "train_mask": None,
        "val_mask": None,
    }
    if not target_path:
        return empty
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"LIME-DCCS target file not found: {target_path}")

    result = dict(empty)
    if target_path.endswith(".npy"):
        all_targets = np.asarray(np.load(target_path), dtype=np.float32)
        if all_targets.shape[0] != train_len + val_len:
            raise ValueError(
                "A single .npy LIME-DCCS target file must contain train+val rows "
                f"({train_len + val_len}), got {all_targets.shape[0]}"
            )
        result["train_targets"] = all_targets[:train_len]
        result["val_targets"] = all_targets[train_len:]
    else:
        with np.load(target_path) as data:
            train_targets = _load_npz_array_key(
                data,
                ["train", "train_lime_dccs", "train_lime_dccs_targets"],
            )
            val_targets = _load_npz_array_key(
                data,
                ["val", "validation", "val_lime_dccs", "val_lime_dccs_targets"],
            )
            if train_targets is None and val_targets is None:
                all_targets = _load_npz_array_key(data, ["all", "lime_dccs", "lime_dccs_targets"])
                if all_targets is None:
                    raise ValueError(
                        "LIME-DCCS .npz must contain train/val arrays or one all/lime_dccs array"
                    )
                if all_targets.shape[0] != train_len + val_len:
                    raise ValueError(
                        "Combined LIME-DCCS targets must contain train+val rows "
                        f"({train_len + val_len}), got {all_targets.shape[0]}"
                    )
                train_targets = all_targets[:train_len]
                val_targets = all_targets[train_len:]

            result["train_targets"] = train_targets
            result["val_targets"] = val_targets

            train_confidence = _load_split_array(data, "train", ["confidence", "conf", "max_prob"])
            val_confidence = _load_split_array(data, "val", ["confidence", "conf", "max_prob"])
            if train_confidence is None and val_confidence is None:
                train_confidence, val_confidence = _split_combined_array(
                    _load_npz_array_key(data, ["confidence", "conf", "max_prob", "lime_confidence"]),
                    train_len,
                    val_len,
                    "LIME-DCCS confidence",
                )
            result["train_confidence"] = train_confidence
            result["val_confidence"] = val_confidence

            train_predicted = _load_split_array(
                data,
                "train",
                ["pred", "prediction", "predicted", "predicted_label", "predicted_labels"],
            )
            val_predicted = _load_split_array(
                data,
                "val",
                ["pred", "prediction", "predicted", "predicted_label", "predicted_labels"],
            )
            if train_predicted is None and val_predicted is None:
                train_predicted, val_predicted = _split_combined_array(
                    _load_npz_array_key(
                        data,
                        ["pred", "prediction", "predicted", "predicted_label", "predicted_labels"],
                    ),
                    train_len,
                    val_len,
                    "LIME-DCCS predicted labels",
                )
            result["train_predicted_labels"] = train_predicted
            result["val_predicted_labels"] = val_predicted

            train_correct = _load_split_array(data, "train", ["correct", "is_correct"])
            val_correct = _load_split_array(data, "val", ["correct", "is_correct"])
            if train_correct is None and val_correct is None:
                train_correct, val_correct = _split_combined_array(
                    _load_npz_array_key(data, ["correct", "is_correct", "lime_correct"]),
                    train_len,
                    val_len,
                    "LIME-DCCS correctness mask",
                )
            result["train_correct"] = train_correct
            result["val_correct"] = val_correct

            train_mask = _load_split_array(data, "train", ["mask", "feedback_mask", "sample_mask"])
            val_mask = _load_split_array(data, "val", ["mask", "feedback_mask", "sample_mask"])
            if train_mask is None and val_mask is None:
                train_mask, val_mask = _split_combined_array(
                    _load_npz_array_key(data, ["mask", "feedback_mask", "sample_mask"]),
                    train_len,
                    val_len,
                    "LIME-DCCS feedback mask",
                )
            result["train_mask"] = train_mask
            result["val_mask"] = val_mask

    if result["train_targets"] is not None:
        result["train_targets"] = _validate_lime_dccs_array(result["train_targets"], train_len, "Train")
    if result["val_targets"] is not None:
        result["val_targets"] = _validate_lime_dccs_array(result["val_targets"], val_len, "Validation")

    result["train_confidence"] = _validate_optional_vector(
        result["train_confidence"], train_len, "Train", "confidence"
    )
    result["val_confidence"] = _validate_optional_vector(
        result["val_confidence"], val_len, "Validation", "confidence"
    )
    result["train_predicted_labels"] = _validate_optional_vector(
        result["train_predicted_labels"], train_len, "Train", "predicted labels", dtype=np.int32
    )
    result["val_predicted_labels"] = _validate_optional_vector(
        result["val_predicted_labels"], val_len, "Validation", "predicted labels", dtype=np.int32
    )
    result["train_correct"] = _validate_optional_vector(
        result["train_correct"], train_len, "Train", "correctness mask"
    )
    result["val_correct"] = _validate_optional_vector(
        result["val_correct"], val_len, "Validation", "correctness mask"
    )
    result["train_mask"] = _validate_optional_vector(
        result["train_mask"], train_len, "Train", "feedback mask"
    )
    result["val_mask"] = _validate_optional_vector(
        result["val_mask"], val_len, "Validation", "feedback mask"
    )
    return result


def load_lime_dccs_feedback_targets(target_path, train_len, val_len):
    feedback_data = load_lime_dccs_feedback_data(target_path, train_len, val_len)
    return feedback_data["train_targets"], feedback_data["val_targets"]


def _lime_filter_key(filter_mode):
    return str(filter_mode or "none").strip().lower().replace("-", "_")


def _lime_correct_mask(labels, predicted_labels=None, correct_mask=None, split_name="Train"):
    if correct_mask is not None:
        return np.asarray(correct_mask, dtype=np.float32).reshape(-1) > 0.5
    if predicted_labels is not None:
        return np.asarray(predicted_labels, dtype=np.int32).reshape(-1) == np.asarray(labels, dtype=np.int32).reshape(-1)
    raise ValueError(
        f"{split_name} LIME-DCCS feedback filter requires predicted labels or a correctness mask. "
        "Add train_predicted_labels/train_correct to the target .npz, or choose "
        "LFTS_LIME_DCCS_FEEDBACK_FILTER=high_confidence/train_only/none."
    )


def _lime_confidence_mask(confidence, min_confidence, split_name="Train"):
    if confidence is None:
        raise ValueError(
            f"{split_name} LIME-DCCS feedback filter requires confidence scores. "
            "Add train_confidence to the target .npz, or choose "
            "LFTS_LIME_DCCS_FEEDBACK_FILTER=correct/train_only/none."
        )
    return np.asarray(confidence, dtype=np.float32).reshape(-1) >= float(min_confidence)


def build_lime_feedback_sample_weights(
        labels,
        feedback_data,
        split_name,
        filter_mode="correct_and_high_confidence",
        min_confidence=0.70,
        use_val_feedback=False,
        soft_confidence_weighting=True,
):
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    split_key = "train" if split_name.lower().startswith("train") else "val"
    filter_key = _lime_filter_key(filter_mode)

    if split_key == "val" and not use_val_feedback:
        return np.zeros(len(labels), dtype=np.float32)

    explicit_mask = feedback_data.get(f"{split_key}_mask")
    mask = np.ones(len(labels), dtype=bool)
    if explicit_mask is not None:
        mask &= np.asarray(explicit_mask, dtype=np.float32).reshape(-1) > 0.5

    confidence = feedback_data.get(f"{split_key}_confidence")
    predicted_labels = feedback_data.get(f"{split_key}_predicted_labels")
    correct_mask = feedback_data.get(f"{split_key}_correct")

    if filter_key in {"none", "all", "all_train", "train", "train_only"}:
        pass
    elif filter_key == "high_confidence":
        mask &= _lime_confidence_mask(confidence, min_confidence, split_name=split_name)
    elif filter_key == "correct":
        mask &= _lime_correct_mask(labels, predicted_labels, correct_mask, split_name=split_name)
    elif filter_key in {"correct_and_high_confidence", "high_confidence_correct"}:
        mask &= _lime_correct_mask(labels, predicted_labels, correct_mask, split_name=split_name)
        mask &= _lime_confidence_mask(confidence, min_confidence, split_name=split_name)
    elif filter_key in {"correct_or_high_confidence", "high_confidence_or_correct"}:
        mask &= (
            _lime_correct_mask(labels, predicted_labels, correct_mask, split_name=split_name)
            | _lime_confidence_mask(confidence, min_confidence, split_name=split_name)
        )
    else:
        raise ValueError(
            "Unknown LIME-DCCS feedback filter "
            f"{filter_mode!r}. Use none, train_only, high_confidence, correct, "
            "correct_and_high_confidence, or correct_or_high_confidence."
        )

    weights = mask.astype(np.float32)
    if soft_confidence_weighting and confidence is not None:
        weights *= np.clip(np.asarray(confidence, dtype=np.float32).reshape(-1), 0.0, 1.0)
    return weights.astype(np.float32)


def summarize_feedback_sample_weights(name, weights):
    weights = np.asarray(weights, dtype=np.float32)
    active = int(np.sum(weights > 0.0))
    total = int(len(weights))
    ratio = active / max(total, 1)
    print(f"{name} LIME-DCCS feedback active samples: {active}/{total} ({ratio:.1%})")


def _concept_sample_weight_mode_key(mode):
    return str(mode or "uniform").strip().lower().replace("-", "_")


def build_concept_sample_weights(
        num_samples,
        quality_scores=None,
        channel_quality=None,
        mode="uniform",
        floor=0.10,
):
    mode_key = _concept_sample_weight_mode_key(mode)
    weights = np.ones(int(num_samples), dtype=np.float32)
    if mode_key in {"uniform", "none", "off", "disabled"}:
        return weights

    valid_modes = {
        "epoch_quality",
        "quality",
        "channel_quality",
        "quality_gated",
        "quality_gate",
        "epoch_channel_quality",
    }
    if mode_key not in valid_modes:
        raise ValueError(
            f"Unknown LFTS_CONCEPT_SAMPLE_WEIGHT_MODE={mode!r}. "
            "Use uniform, epoch_quality, channel_quality, or quality_gated."
        )

    used_component = False
    if mode_key in {"epoch_quality", "quality", "quality_gated", "quality_gate", "epoch_channel_quality"}:
        if quality_scores is None:
            print("Concept sample weighting requested epoch quality, but quality_scores are unavailable.")
        else:
            epoch_quality = np.asarray(quality_scores, dtype=np.float32).reshape(-1)
            if len(epoch_quality) != len(weights):
                raise ValueError(
                    "quality_scores must align with concept sample weights: "
                    f"got {len(epoch_quality)} scores for {len(weights)} epochs"
                )
            weights *= np.clip(np.nan_to_num(epoch_quality, nan=0.0), 0.0, 1.0)
            used_component = True

    if mode_key in {"channel_quality", "quality_gated", "quality_gate", "epoch_channel_quality"}:
        if channel_quality is None:
            print("Concept sample weighting requested channel quality, but channel_quality is unavailable.")
        else:
            channel_quality = np.asarray(channel_quality, dtype=np.float32)
            if len(channel_quality) != len(weights):
                raise ValueError(
                    "channel_quality must align with concept sample weights: "
                    f"got {len(channel_quality)} scores for {len(weights)} epochs"
                )
            if channel_quality.ndim <= 1:
                channel_reliability = channel_quality.reshape(-1)
            else:
                channel_reliability = np.nanmean(channel_quality.reshape(len(weights), -1), axis=1)
            weights *= np.clip(np.nan_to_num(channel_reliability, nan=0.0), 0.0, 1.0)
            used_component = True

    if not used_component:
        return weights

    floor = float(np.clip(floor, 0.0, 1.0))
    return np.clip(weights, floor, 1.0).astype(np.float32)


def summarize_concept_sample_weights(name, weights, mode):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    low_ratio = float(np.mean(weights < 0.5)) if len(weights) else 0.0
    print(
        f"{name} concept sample weights: "
        f"mode={_concept_sample_weight_mode_key(mode)}, "
        f"mean={np.mean(weights):.3f}, "
        f"p10={np.percentile(weights, 10):.3f}, "
        f"min={np.min(weights):.3f}, "
        f"low_lt_0.5={low_ratio:.1%}"
    )


def print_class_distribution(name, y_raw, class_names):
    counts = np.bincount(y_raw, minlength=len(class_names))
    total = counts.sum()
    ratio = counts / max(total, 1)
    summary = ", ".join(
        f"{label}:{count}({pct:.1%})"
        for label, count, pct in zip(class_names, counts, ratio)
    )
    print(f"{name} distribution: {summary}")


def build_segment_feature_matrix(X_raw, semantic_segmentation, extract_segment_features):
    X_segments = []
    for instance in tqdm(X_raw, desc="segment feature extraction"):
        segments = semantic_segmentation(instance, target_segments=30)
        X_segments.append(
            extract_segment_features(
                instance,
                segments,
                max_segments=30,
                include_spectral=INCLUDE_SPECTRAL_FEATURES,
            )
        )
    return np.asarray(X_segments, dtype=np.float32)


def compute_main_sample_weights(
        y_raw,
        num_classes,
        n1_boost=1.20,
        min_weight=0.85,
        max_weight=1.60,
):
    counts = np.bincount(y_raw, minlength=num_classes).astype(np.float32)
    nonzero_counts = counts[counts > 0]
    reference_count = float(np.median(nonzero_counts)) if len(nonzero_counts) else 1.0
    class_weights = np.sqrt(reference_count / np.maximum(counts, 1.0))
    class_weights = np.clip(class_weights, min_weight, max_weight)
    if num_classes > 1:
        class_weights[1] = min(max_weight, class_weights[1] * n1_boost)
    sample_weights = class_weights[y_raw].astype(np.float32)
    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)
    return sample_weights, class_weights.astype(np.float32)


def _record_slices_from_counts(total_len, file_sample_counts=None):
    if file_sample_counts is None:
        return [(0, int(total_len))]

    slices = []
    start = 0
    for count in file_sample_counts:
        end = min(start + int(count), int(total_len))
        if end > start:
            slices.append((start, end))
        start = end
    if start < int(total_len):
        slices.append((start, int(total_len)))
    return slices


def _is_target_n1_transition(left_stage, right_stage):
    pair = (int(left_stage), int(right_stage))
    target_pairs = {
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (4, 1), (1, 4),
    }
    return pair in target_pairs


FINE_TRANSITION_LABELS = [
    "Non-boundary",
    "W<->N1",
    "N1<->N2",
    "N2<->N3",
    "N2<->REM",
    "W<->REM",
    "N1<->REM",
    "Others",
]


def _fine_transition_type_index(left_stage, right_stage):
    pair = (int(left_stage), int(right_stage))
    if pair[0] == pair[1]:
        return 0
    unordered_pair = frozenset(pair)
    if unordered_pair == {0, 1}:
        return 1
    if unordered_pair == {1, 2}:
        return 2
    if unordered_pair == {2, 3}:
        return 3
    if unordered_pair == {2, 4}:
        return 4
    if unordered_pair == {0, 4}:
        return 5
    if unordered_pair == {1, 4}:
        return 6
    return 7


def compute_transition_boundary_sample_weights(
        y_raw,
        file_sample_counts=None,
        radius=2,
        base_boost=1.15,
        n1_boundary_boost=1.25,
        target_transition_boost=1.20,
        min_weight=1.0,
        max_weight=2.0,
):
    labels = np.asarray(y_raw, dtype=np.int32).ravel()
    weights = np.ones(len(labels), dtype=np.float32)
    radius = max(0, int(radius))

    for start, end in _record_slices_from_counts(len(labels), file_sample_counts):
        record_labels = labels[start:end]
        for offset, (left_stage, right_stage) in enumerate(zip(record_labels[:-1], record_labels[1:])):
            if left_stage == right_stage:
                continue
            boundary_left = start + offset
            boundary_right = boundary_left + 1
            affected_start = max(start, boundary_left - radius)
            affected_end = min(end, boundary_right + radius + 1)
            boost = float(base_boost)
            if left_stage == 1 or right_stage == 1:
                boost *= float(n1_boundary_boost)
            if _is_target_n1_transition(left_stage, right_stage):
                boost *= float(target_transition_boost)
            weights[affected_start:affected_end] = np.maximum(
                weights[affected_start:affected_end],
                boost,
            )

    return np.clip(weights, float(min_weight), float(max_weight)).astype(np.float32)


def build_transition_boundary_auxiliary_targets(
        y_raw,
        file_sample_counts=None,
        radius=1,
        positive_weight=2.0,
        n1_transition_weight=1.5,
        fine_grained=False,
        num_transition_types=5,
):
    labels = np.asarray(y_raw, dtype=np.int32).ravel()
    num_transition_types = max(1, int(num_transition_types))
    target_width = num_transition_types if fine_grained else 1
    targets = np.zeros((len(labels), target_width), dtype=np.float32)
    sample_weights = np.ones(len(labels), dtype=np.float32)
    boundary_mask = np.zeros(len(labels), dtype=bool)
    n1_boundary_mask = np.zeros(len(labels), dtype=bool)
    transition_type_ids = np.zeros(len(labels), dtype=np.int32)
    radius = max(0, int(radius))

    for start, end in _record_slices_from_counts(len(labels), file_sample_counts):
        record_labels = labels[start:end]
        for offset, (left_stage, right_stage) in enumerate(zip(record_labels[:-1], record_labels[1:])):
            if left_stage == right_stage:
                continue
            boundary_left = start + offset
            boundary_right = boundary_left + 1
            affected_start = max(start, boundary_left - radius)
            affected_end = min(end, boundary_right + radius + 1)
            boundary_mask[affected_start:affected_end] = True
            if fine_grained:
                transition_type_ids[affected_start:affected_end] = _fine_transition_type_index(
                    left_stage,
                    right_stage,
                )
            if left_stage == 1 or right_stage == 1 or _is_target_n1_transition(left_stage, right_stage):
                n1_boundary_mask[affected_start:affected_end] = True

    if fine_grained:
        transition_type_ids = np.clip(transition_type_ids, 0, target_width - 1)
        targets[np.arange(len(labels)), transition_type_ids] = 1.0
    else:
        targets[boundary_mask, 0] = 1.0
    sample_weights[boundary_mask] *= float(positive_weight)
    sample_weights[n1_boundary_mask] *= float(n1_transition_weight)
    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)
    transition_type_counts = np.bincount(
        transition_type_ids,
        minlength=target_width,
    ).astype(np.int64)
    return {
        "targets": targets.astype(np.float32),
        "sample_weights": sample_weights.astype(np.float32),
        "boundary_count": int(np.sum(boundary_mask)),
        "n1_boundary_count": int(np.sum(n1_boundary_mask)),
        "coverage": float(np.mean(boundary_mask)) if len(boundary_mask) else 0.0,
        "n1_coverage": float(np.mean(n1_boundary_mask)) if len(n1_boundary_mask) else 0.0,
        "mean_positive_weight": float(np.mean(sample_weights[boundary_mask])) if np.any(boundary_mask) else 1.0,
        "max_weight": float(np.max(sample_weights)) if len(sample_weights) else 1.0,
        "transition_type_counts": transition_type_counts,
    }


def _clinical_concept_derivative_template(left_stage, right_stage):
    pair = (int(left_stage), int(right_stage))
    templates = {
        (0, 1): [0.00, 0.70, -0.60, -0.10, -0.40, 0.65, -0.10, 0.25, -0.45],
        (1, 0): [0.00, -0.45, 0.55, -0.20, 0.45, -0.40, -0.05, -0.35, 0.45],
        (1, 2): [0.10, -0.30, -0.25, 0.80, -0.25, -0.50, -0.10, 0.10, -0.10],
        (2, 1): [-0.10, 0.65, 0.10, -0.60, 0.10, 0.45, 0.05, 0.35, -0.45],
        (2, 3): [0.90, -0.15, -0.10, -0.45, -0.15, -0.10, -0.10, 0.05, -0.10],
        (3, 2): [-0.75, 0.10, 0.05, 0.55, 0.10, 0.00, 0.00, -0.05, 0.10],
        (2, 4): [-0.20, 0.00, -0.10, -0.35, 0.00, -0.05, 0.75, 0.70, -0.60],
        (4, 2): [0.20, 0.00, 0.10, 0.45, 0.00, 0.05, -0.70, -0.55, 0.45],
        (1, 4): [-0.05, -0.10, -0.05, -0.10, 0.00, -0.15, 0.75, 0.65, -0.55],
        (4, 1): [0.00, 0.50, 0.00, 0.00, 0.00, 0.55, -0.60, -0.30, 0.35],
        (0, 4): [-0.10, -0.05, -0.25, -0.10, -0.10, -0.10, 0.80, 0.65, -0.55],
        (4, 0): [0.05, -0.10, 0.45, -0.10, 0.35, -0.20, -0.70, -0.55, 0.55],
    }
    if pair in templates:
        return np.asarray(templates[pair], dtype=np.float32)
    if pair[0] == pair[1]:
        return np.zeros(9, dtype=np.float32)
    if pair[1] == 1:
        return np.asarray([0.00, 0.50, 0.00, -0.25, 0.00, 0.35, -0.10, 0.20, -0.25], dtype=np.float32)
    if pair[1] == 3:
        return np.asarray([0.70, -0.10, -0.05, -0.35, -0.10, -0.05, -0.05, 0.05, -0.05], dtype=np.float32)
    if pair[1] == 4:
        return np.asarray([-0.15, -0.05, -0.10, -0.25, 0.00, -0.05, 0.65, 0.55, -0.45], dtype=np.float32)
    return np.zeros(9, dtype=np.float32)


def build_concept_derivative_auxiliary_targets(
        y_raw,
        file_sample_counts=None,
        positive_weight=2.0,
        n1_transition_weight=1.5,
        max_weight=8.0,
):
    labels = np.asarray(y_raw, dtype=np.int32).ravel()
    targets = np.zeros((len(labels), OT_TACB_CONCEPT_DERIVATIVE_DIM), dtype=np.float32)
    sample_weights = np.zeros(len(labels), dtype=np.float32)
    transition_mask = np.zeros(len(labels), dtype=bool)
    n1_transition_mask = np.zeros(len(labels), dtype=bool)
    transition_type_counts = np.zeros(OT_TACB_FINE_TRANSITION_TYPES, dtype=np.int64)

    for start, end in _record_slices_from_counts(len(labels), file_sample_counts):
        record_labels = labels[start:end]
        for offset, (left_stage, right_stage) in enumerate(zip(record_labels[:-1], record_labels[1:])):
            if left_stage == right_stage:
                continue
            right_index = start + offset + 1
            derivative = _clinical_concept_derivative_template(left_stage, right_stage)
            if not np.any(np.abs(derivative) > 1e-6):
                continue
            targets[right_index, :9] = derivative
            targets[right_index, 9] = 1.0
            transition_mask[right_index] = True
            sample_weights[right_index] = float(positive_weight)
            transition_type_counts[_fine_transition_type_index(left_stage, right_stage)] += 1
            if left_stage == 1 or right_stage == 1 or _is_target_n1_transition(left_stage, right_stage):
                n1_transition_mask[right_index] = True
                sample_weights[right_index] *= float(n1_transition_weight)

    positive_mask = sample_weights > 0.0
    if np.any(positive_mask):
        sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)
        sample_weights = np.clip(sample_weights, 0.0, float(max_weight))

    return {
        "targets": targets.astype(np.float32),
        "sample_weights": sample_weights.astype(np.float32),
        "transition_count": int(np.sum(transition_mask)),
        "n1_transition_count": int(np.sum(n1_transition_mask)),
        "coverage": float(np.mean(transition_mask)) if len(transition_mask) else 0.0,
        "n1_coverage": float(np.mean(n1_transition_mask)) if len(n1_transition_mask) else 0.0,
        "mean_positive_weight": float(np.mean(sample_weights[positive_mask])) if np.any(positive_mask) else 0.0,
        "max_weight": float(np.max(sample_weights)) if len(sample_weights) else 0.0,
        "transition_type_counts": transition_type_counts,
    }


def compute_transition_boundary_metrics(
        y_true,
        y_pred,
        file_sample_counts=None,
        radius=0,
):
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")

    radius = max(0, int(radius))
    boundary_mask = np.zeros(len(y_true), dtype=bool)
    target_boundary_mask = np.zeros(len(y_true), dtype=bool)

    for start, end in _record_slices_from_counts(len(y_true), file_sample_counts):
        record_labels = y_true[start:end]
        for offset, (left_stage, right_stage) in enumerate(zip(record_labels[:-1], record_labels[1:])):
            if left_stage == right_stage:
                continue
            boundary_left = start + offset
            boundary_right = boundary_left + 1
            affected_start = max(start, boundary_left - radius)
            affected_end = min(end, boundary_right + radius + 1)
            boundary_mask[affected_start:affected_end] = True
            if _is_target_n1_transition(left_stage, right_stage):
                target_boundary_mask[affected_start:affected_end] = True

    target_n1_mask = target_boundary_mask & (y_true == 1)
    boundary_correct = boundary_mask & (y_true == y_pred)
    target_correct = target_boundary_mask & (y_true == y_pred)
    target_n1_correct = target_n1_mask & (y_pred == 1)

    boundary_count = int(np.sum(boundary_mask))
    target_boundary_count = int(np.sum(target_boundary_mask))
    target_n1_count = int(np.sum(target_n1_mask))
    return {
        "boundary_count": boundary_count,
        "boundary_accuracy": float(np.sum(boundary_correct) / max(boundary_count, 1)),
        "target_boundary_count": target_boundary_count,
        "target_boundary_accuracy": float(np.sum(target_correct) / max(target_boundary_count, 1)),
        "target_n1_count": target_n1_count,
        "target_n1_recall": float(np.sum(target_n1_correct) / max(target_n1_count, 1)),
    }


def print_quality_summary(name, quality_scores):
    if quality_scores is None:
        return
    quality_scores = np.asarray(quality_scores, dtype=np.float32)
    low_quality = float(np.mean(quality_scores < 0.70))
    severe_quality = float(np.mean(quality_scores < 0.40))
    print(
        f"{name} quality confidence: mean_Q={np.mean(quality_scores):.3f}, "
        f"low_Q={low_quality * 100:.1f}%, "
        f"severe_Q={severe_quality * 100:.1f}%, "
        f"min_Q={np.min(quality_scores):.3f}"
    )


def temperature_scale_probabilities(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    temperature = max(float(temperature), 1e-3)
    log_probs = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
    scaled = np.exp(log_probs)
    return (scaled / np.maximum(np.sum(scaled, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def negative_log_likelihood(y_true, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-8, 1.0))))


def target_entropy_loss(main_probs, concept_probs=None, consistency_weight=0.0):
    main_probs = tf.cast(main_probs, tf.float32)
    main_probs = tf.clip_by_value(main_probs, 1e-8, 1.0)
    main_probs = main_probs / tf.maximum(tf.reduce_sum(main_probs, axis=-1, keepdims=True), 1e-8)
    entropy = -tf.reduce_sum(main_probs * tf.math.log(main_probs), axis=-1)
    loss = tf.reduce_mean(entropy)

    if concept_probs is not None and float(consistency_weight) > 0.0:
        concept_probs = tf.cast(concept_probs, tf.float32)
        concept_probs = tf.clip_by_value(concept_probs, 1e-8, 1.0)
        concept_probs = concept_probs / tf.maximum(
            tf.reduce_sum(concept_probs, axis=-1, keepdims=True),
            1e-8,
        )
        consistency = tf.reduce_mean(tf.reduce_sum(tf.square(main_probs - concept_probs), axis=-1))
        loss = loss + float(consistency_weight) * consistency

    return loss


def class_precision_recall(y_true, y_pred, class_idx):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    true_mask = y_true == class_idx
    pred_mask = y_pred == class_idx
    true_positive = float(np.sum(true_mask & pred_mask))
    precision = true_positive / max(float(np.sum(pred_mask)), 1.0)
    recall = true_positive / max(float(np.sum(true_mask)), 1.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def class_pair_confusion_stats(y_true, y_pred, left_idx, right_idx):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    left_to_right = int(np.sum((y_true == left_idx) & (y_pred == right_idx)))
    right_to_left = int(np.sum((y_true == right_idx) & (y_pred == left_idx)))
    pair_true_count = int(np.sum((y_true == left_idx) | (y_true == right_idx)))
    total = left_to_right + right_to_left
    return {
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "total": total,
        "pair_true_count": pair_true_count,
        "rate": float(total / max(pair_true_count, 1)),
    }


def build_transition_window_mask(y_true, file_sample_counts=None, radius=None):
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    mask = np.zeros(len(labels), dtype=bool)
    if len(labels) == 0:
        return mask
    radius = TRANSITION_BOUNDARY_RADIUS if radius is None else int(radius)
    radius = max(0, int(radius))
    if file_sample_counts is None:
        file_sample_counts = [len(labels)]
    offset = 0
    for raw_count in file_sample_counts:
        count = int(raw_count)
        if count <= 0:
            continue
        start = offset
        end = min(offset + count, len(labels))
        if end - start <= 1:
            offset += count
            continue
        sequence = labels[start:end]
        transition_positions = np.where(sequence[1:] != sequence[:-1])[0] + 1
        for pos in transition_positions:
            left = start + max(0, int(pos) - radius)
            right = start + min(len(sequence), int(pos) + radius + 1)
            mask[left:right] = True
        offset += count
    return mask


def transition_window_accuracy(y_true, y_pred, file_sample_counts=None, radius=None):
    mask = build_transition_window_mask(y_true, file_sample_counts=file_sample_counts, radius=radius)
    if not np.any(mask):
        return {"accuracy": float("nan"), "count": 0}
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    return {
        "accuracy": float(np.mean(y_true[mask] == y_pred[mask])),
        "count": int(np.sum(mask)),
    }


def score_predictions(y_true, predictions, mode="macro_f1"):
    mode = str(mode or "macro_f1").strip().lower()
    accuracy = float(np.mean(predictions == y_true))
    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, predictions))
    if not np.isfinite(kappa):
        kappa = 0.0
    n1_precision, n1_recall, n1_f1 = class_precision_recall(y_true, predictions, 1)
    _, n2_recall, n2_f1 = class_precision_recall(y_true, predictions, 2)
    rem_precision, rem_recall, rem_f1 = class_precision_recall(y_true, predictions, 4)
    w_n1_confusion = class_pair_confusion_stats(y_true, predictions, 0, 1)
    n1_n2_confusion = class_pair_confusion_stats(y_true, predictions, 1, 2)

    if mode == "macro_f1":
        score = macro_f1
    elif mode in {"stable_macro_n1_kappa", "cross_dataset_stable", "stable"}:
        score = 0.50 * macro_f1 + 0.30 * n1_f1 + 0.20 * kappa
    elif mode == "macro_n1_kappa":
        score = 0.45 * macro_f1 + 0.35 * n1_f1 + 0.20 * kappa
    elif mode == "n1_n2":
        score = 0.45 * n1_f1 + 0.35 * n2_recall + 0.20 * accuracy
    elif mode == "accuracy":
        score = accuracy
    elif mode == "accuracy_macro":
        score = accuracy + 0.20 * macro_f1
    else:
        score = 0.35 * macro_f1 + 0.25 * n1_precision + 0.20 * n1_recall + 0.20 * n2_recall

    return {
        "score": float(score),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "n1_precision": n1_precision,
        "n1_recall": n1_recall,
        "n1_f1": n1_f1,
        "n2_recall": n2_recall,
        "n2_f1": n2_f1,
        "rem_precision": rem_precision,
        "rem_recall": rem_recall,
        "rem_f1": rem_f1,
        "w_n1_confusion": w_n1_confusion,
        "n1_n2_confusion": n1_n2_confusion,
    }


def set_runtime_stage_channel_prior(stage_channel_prior):
    global RUNTIME_STAGE_CHANNEL_PRIOR
    if stage_channel_prior is None:
        RUNTIME_STAGE_CHANNEL_PRIOR = None
    else:
        prior = np.asarray(stage_channel_prior, dtype=np.float32)
        RUNTIME_STAGE_CHANNEL_PRIOR = prior / np.maximum(
            np.sum(prior, axis=1, keepdims=True),
            1e-8,
        )


def format_metric_summary(metrics):
    return (
        f"macro_F1={metrics['macro_f1']:.4f} "
        f"N1_F1={metrics['n1_f1']:.4f} "
        f"REM_F1={metrics['rem_f1']:.4f} "
        f"kappa={metrics['kappa']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"N1_P={metrics['n1_precision']:.4f} "
        f"N1_R={metrics['n1_recall']:.4f}"
    )


def format_pair_confusion_summary(metrics, class_names):
    w_n1 = metrics["w_n1_confusion"]
    n1_n2 = metrics["n1_n2_confusion"]
    return (
        f"{class_names[0]}->{class_names[1]}={w_n1['left_to_right']} "
        f"{class_names[1]}->{class_names[0]}={w_n1['right_to_left']} "
        f"{class_names[0]}/{class_names[1]}_rate={w_n1['rate']:.4f} | "
        f"{class_names[1]}->{class_names[2]}={n1_n2['left_to_right']} "
        f"{class_names[2]}->{class_names[1]}={n1_n2['right_to_left']} "
        f"{class_names[1]}/{class_names[2]}_rate={n1_n2['rate']:.4f}"
    )


def print_evaluation_report(
    name,
    y_true,
    predictions,
    class_names,
    selector_mode=SELECTOR_MODE,
    file_sample_counts=None,
    transition_radius=None,
):
    metrics = score_predictions(y_true, predictions, selector_mode)
    print(f"\n{name}")
    print(format_metric_summary(metrics))
    print("Pair confusion: " + format_pair_confusion_summary(metrics, class_names))
    if file_sample_counts is not None:
        transition_metrics = transition_window_accuracy(
            y_true,
            predictions,
            file_sample_counts=file_sample_counts,
            radius=transition_radius,
        )
        if transition_metrics["count"] > 0:
            print(
                "Transition-window accuracy: "
                f"{transition_metrics['accuracy']:.4f} "
                f"(n={transition_metrics['count']}, radius={transition_radius if transition_radius is not None else TRANSITION_BOUNDARY_RADIUS})"
            )
        else:
            print(
                "Transition-window accuracy: n/a "
                f"(no label transitions, radius={transition_radius if transition_radius is not None else TRANSITION_BOUNDARY_RADIUS})"
            )
    print(classification_report(y_true, predictions, target_names=class_names, digits=4, zero_division=0))
    return metrics


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return (probabilities / np.maximum(np.sum(probabilities, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def extract_prediction_heads(model_outputs):
    if isinstance(model_outputs, (list, tuple)):
        main_probs = model_outputs[0]
        channel_weights = model_outputs[1] if len(model_outputs) > 1 else None
        concept_probs = model_outputs[2] if len(model_outputs) > 2 else None
        return main_probs, channel_weights, concept_probs
    if isinstance(model_outputs, dict):
        return (
            model_outputs.get("main_output"),
            model_outputs.get("channel_attention_weights"),
            model_outputs.get("concept_output"),
        )
    return model_outputs, None, None


def blend_probabilities(primary, auxiliary, auxiliary_weight=0.20):
    primary = normalize_probabilities(primary)
    if auxiliary is None:
        return primary

    auxiliary = normalize_probabilities(auxiliary)
    if auxiliary.shape != primary.shape:
        return primary

    weight = float(np.clip(auxiliary_weight, 0.0, 1.0))
    log_probs = (1.0 - weight) * np.log(primary) + weight * np.log(auxiliary)
    log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
    fused = np.exp(log_probs)
    return normalize_probabilities(fused)


def build_stage_channel_prior(num_classes, num_channels):
    if RUNTIME_STAGE_CHANNEL_PRIOR is not None:
        stage_channel_prior = np.asarray(RUNTIME_STAGE_CHANNEL_PRIOR, dtype=np.float32)
    else:
        stage_channel_prior = np.array(
            [
                [0.30, 0.20, 0.50],  # W: alpha/beta with muscle tone.
                [0.50, 0.40, 0.10],  # N1: EEG theta and slow eye movement.
                [0.85, 0.05, 0.10],  # N2: EEG sigma/spindle evidence.
                [0.90, 0.05, 0.05],  # N3: EEG delta evidence.
                [0.20, 0.30, 0.50],  # REM: eye movement and low muscle tone.
            ],
            dtype=np.float32,
        )
    if num_classes > stage_channel_prior.shape[0]:
        extra = np.ones((num_classes - stage_channel_prior.shape[0], 3), dtype=np.float32)
        stage_channel_prior = np.vstack([stage_channel_prior, extra])

    prior = stage_channel_prior[:num_classes, :num_channels]
    return prior / np.maximum(np.sum(prior, axis=1, keepdims=True), 1e-8)


def align_probabilities_to_channel_prior(probabilities, channel_weights, alignment_strength=0.30):
    probabilities = normalize_probabilities(probabilities)
    if channel_weights is None:
        return probabilities

    channel_weights = np.asarray(channel_weights, dtype=np.float32)
    if channel_weights.ndim != 2 or len(channel_weights) != len(probabilities):
        return probabilities

    num_classes = probabilities.shape[1]
    num_channels = min(channel_weights.shape[1], 3)
    channel_weights = np.clip(channel_weights[:, :num_channels], 1e-8, None)
    channel_weights = channel_weights / np.maximum(np.sum(channel_weights, axis=1, keepdims=True), 1e-8)
    stage_prior = build_stage_channel_prior(num_classes, num_channels)

    numerator = channel_weights @ stage_prior.T
    denom = np.maximum(
        np.linalg.norm(channel_weights, axis=1, keepdims=True) *
        np.linalg.norm(stage_prior, axis=1, keepdims=True).T,
        1e-8,
    )
    alignment = numerator / denom
    alignment = alignment - np.mean(alignment, axis=1, keepdims=True)

    log_probs = np.log(probabilities) + float(alignment_strength) * alignment
    log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
    return normalize_probabilities(np.exp(log_probs))


def viterbi_predictions_by_recording_adaptive(
        probabilities,
        file_sample_counts,
        transition_log,
        initial_log,
        emission_weight=HMM_EMISSION_WEIGHT,
        quality_scores=None,
        self_loop_boost=0.85,
):
    decoded_parts = []
    start = 0
    for count in file_sample_counts:
        end = start + count
        quality_part = None if quality_scores is None else quality_scores[start:end]
        decoded_parts.append(
            Viterbi_decode_adaptive_v2(
                probabilities[start:end],
                transition_log,
                initial_log,
                emission_weight=emission_weight,
                quality_scores=quality_part,
                self_loop_boost=self_loop_boost,
            )
        )
        start = end
    return np.concatenate(decoded_parts) if decoded_parts else np.array([], dtype=np.int32)


def tune_decoded_candidate(
        y_true,
        probabilities,
        file_sample_counts,
        transition_log,
        initial_log,
        selector_mode=SELECTOR_MODE,
        emission_weights=None,
        quality_scores=None,
        self_loop_boosts=None,
):
    emission_weights = emission_weights or EXTENDED_HMM_EMISSION_WEIGHTS
    self_loop_boosts = self_loop_boosts or (None,)
    best = None

    for emission_weight in emission_weights:
        for self_loop_boost in self_loop_boosts:
            if self_loop_boost is None:
                predictions = viterbi_predictions_by_recording(
                    probabilities,
                    file_sample_counts,
                    transition_log,
                    initial_log,
                    emission_weight=emission_weight,
                )
            else:
                predictions = viterbi_predictions_by_recording_adaptive(
                    probabilities,
                    file_sample_counts,
                    transition_log,
                    initial_log,
                    emission_weight=emission_weight,
                    quality_scores=quality_scores,
                    self_loop_boost=self_loop_boost,
                )
            metrics = score_predictions(y_true, predictions, selector_mode)
            if best is None or metrics["score"] > best["metrics"]["score"]:
                best = {
                    "emission_weight": float(emission_weight),
                    "self_loop_boost": None if self_loop_boost is None else float(self_loop_boost),
                    "metrics": metrics,
                    "predictions": predictions,
                }

    return best


def build_extended_ablation_predictions(
        y_val_raw,
        val_main_probs,
        val_concept_probs,
        val_channel_weights,
        val_file_sample_counts,
        val_quality_scores,
        y_test_main_probs,
        test_concept_probs,
        test_channel_weights,
        test_file_sample_counts,
        test_quality_scores,
        transition_log,
        initial_log,
        selector_mode=SELECTOR_MODE,
):
    ablations = []

    if val_concept_probs is not None and test_concept_probs is not None:
        best = None
        for concept_weight in EXTENDED_CONCEPT_WEIGHTS:
            val_probs = blend_probabilities(val_main_probs, val_concept_probs, concept_weight)
            candidate = tune_decoded_candidate(
                y_val_raw,
                val_probs,
                val_file_sample_counts,
                transition_log,
                initial_log,
                selector_mode=selector_mode,
            )
            if best is None or candidate["metrics"]["score"] > best["metrics"]["score"]:
                best = {**candidate, "concept_weight": float(concept_weight)}
        test_probs = blend_probabilities(y_test_main_probs, test_concept_probs, best["concept_weight"])
        test_predictions = viterbi_predictions_by_recording(
            test_probs,
            test_file_sample_counts,
            transition_log,
            initial_log,
            emission_weight=best["emission_weight"],
        )
        ablations.append(
            {
                "name": "concept-prior fusion + tuned clinical HMM",
                "predictions": test_predictions,
                "params": (
                    f"concept_weight={best['concept_weight']:.2f}, "
                    f"emission_weight={best['emission_weight']:.2f}"
                ),
                "validation_metrics": best["metrics"],
            }
        )

    if val_channel_weights is not None and test_channel_weights is not None:
        best = None
        for channel_strength in EXTENDED_CHANNEL_STRENGTHS:
            val_probs = align_probabilities_to_channel_prior(
                val_main_probs,
                val_channel_weights,
                alignment_strength=channel_strength,
            )
            candidate = tune_decoded_candidate(
                y_val_raw,
                val_probs,
                val_file_sample_counts,
                transition_log,
                initial_log,
                selector_mode=selector_mode,
            )
            if best is None or candidate["metrics"]["score"] > best["metrics"]["score"]:
                best = {**candidate, "channel_strength": float(channel_strength)}
        test_probs = align_probabilities_to_channel_prior(
            y_test_main_probs,
            test_channel_weights,
            alignment_strength=best["channel_strength"],
        )
        test_predictions = viterbi_predictions_by_recording(
            test_probs,
            test_file_sample_counts,
            transition_log,
            initial_log,
            emission_weight=best["emission_weight"],
        )
        ablations.append(
            {
                "name": "channel-prior attention alignment + tuned clinical HMM",
                "predictions": test_predictions,
                "params": (
                    f"channel_strength={best['channel_strength']:.2f}, "
                    f"emission_weight={best['emission_weight']:.2f}"
                ),
                "validation_metrics": best["metrics"],
            }
        )

    if (
            val_concept_probs is not None and test_concept_probs is not None and
            val_channel_weights is not None and test_channel_weights is not None
    ):
        best = None
        for concept_weight in EXTENDED_CONCEPT_WEIGHTS:
            val_concept_aligned = blend_probabilities(val_main_probs, val_concept_probs, concept_weight)
            for channel_strength in EXTENDED_CHANNEL_STRENGTHS:
                val_probs = align_probabilities_to_channel_prior(
                    val_concept_aligned,
                    val_channel_weights,
                    alignment_strength=channel_strength,
                )
                candidate = tune_decoded_candidate(
                    y_val_raw,
                    val_probs,
                    val_file_sample_counts,
                    transition_log,
                    initial_log,
                    selector_mode=selector_mode,
                )
                if best is None or candidate["metrics"]["score"] > best["metrics"]["score"]:
                    best = {
                        **candidate,
                        "concept_weight": float(concept_weight),
                        "channel_strength": float(channel_strength),
                    }

        test_probs = blend_probabilities(y_test_main_probs, test_concept_probs, best["concept_weight"])
        test_probs = align_probabilities_to_channel_prior(
            test_probs,
            test_channel_weights,
            alignment_strength=best["channel_strength"],
        )
        test_predictions = viterbi_predictions_by_recording(
            test_probs,
            test_file_sample_counts,
            transition_log,
            initial_log,
            emission_weight=best["emission_weight"],
        )
        ablations.append(
            {
                "name": "concept + channel-prior constrained HMM",
                "predictions": test_predictions,
                "params": (
                    f"concept_weight={best['concept_weight']:.2f}, "
                    f"channel_strength={best['channel_strength']:.2f}, "
                    f"emission_weight={best['emission_weight']:.2f}"
                ),
                "validation_metrics": best["metrics"],
            }
        )

        best_quality = None
        for concept_weight in EXTENDED_CONCEPT_WEIGHTS:
            val_concept_aligned = blend_probabilities(val_main_probs, val_concept_probs, concept_weight)
            for channel_strength in EXTENDED_CHANNEL_STRENGTHS:
                val_probs = align_probabilities_to_channel_prior(
                    val_concept_aligned,
                    val_channel_weights,
                    alignment_strength=channel_strength,
                )
                candidate = tune_decoded_candidate(
                    y_val_raw,
                    val_probs,
                    val_file_sample_counts,
                    transition_log,
                    initial_log,
                    selector_mode=selector_mode,
                    quality_scores=val_quality_scores,
                    self_loop_boosts=EXTENDED_SELF_LOOP_BOOSTS,
                )
                if best_quality is None or candidate["metrics"]["score"] > best_quality["metrics"]["score"]:
                    best_quality = {
                        **candidate,
                        "concept_weight": float(concept_weight),
                        "channel_strength": float(channel_strength),
                    }

        test_probs = blend_probabilities(y_test_main_probs, test_concept_probs, best_quality["concept_weight"])
        test_probs = align_probabilities_to_channel_prior(
            test_probs,
            test_channel_weights,
            alignment_strength=best_quality["channel_strength"],
        )
        test_predictions = viterbi_predictions_by_recording_adaptive(
            test_probs,
            test_file_sample_counts,
            transition_log,
            initial_log,
            emission_weight=best_quality["emission_weight"],
            quality_scores=test_quality_scores,
            self_loop_boost=best_quality["self_loop_boost"],
        )
        ablations.append(
            {
                "name": "concept + channel-prior + quality-aware HMM",
                "predictions": test_predictions,
                "params": (
                    f"concept_weight={best_quality['concept_weight']:.2f}, "
                    f"channel_strength={best_quality['channel_strength']:.2f}, "
                    f"emission_weight={best_quality['emission_weight']:.2f}, "
                    f"self_loop_boost={best_quality['self_loop_boost']:.2f}"
                ),
                "validation_metrics": best_quality["metrics"],
            }
        )

    return ablations


def print_extended_ablation_reports(
        y_true,
        class_names,
        ablations,
        baseline_predictions=None,
        title="Test-side extended ablation analysis (not final score)",
        summary_title="Test-side extended ablation analysis summary",
):
    if not ablations:
        print("\nTest-side extended ablation analysis skipped: auxiliary heads are not available.")
        return

    print("\n==========================================")
    print(title)
    print("==========================================")
    if baseline_predictions is not None:
        baseline_accuracy = float(np.mean(np.asarray(baseline_predictions) == y_true))
    else:
        baseline_accuracy = None

    summary_rows = []
    for item in ablations:
        predictions = np.asarray(item["predictions"], dtype=np.int32)
        print(f"\nTest-side analysis report ({item['name']} | {item['params']})")
        print(classification_report(y_true, predictions, target_names=class_names, digits=4, zero_division=0))
        metrics = score_predictions(y_true, predictions, mode="macro_f1")
        delta = "" if baseline_accuracy is None else f" delta_vs_clinical_hmm={metrics['accuracy'] - baseline_accuracy:+.4f}"
        summary_rows.append(
            f"{item['name']}: macro_f1={metrics['macro_f1']:.4f} "
            f"N1_F1={metrics['n1_f1']:.4f} "
            f"kappa={metrics['kappa']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"N1_P={metrics['n1_precision']:.4f} "
            f"N1_R={metrics['n1_recall']:.4f}{delta} | {item['params']}"
        )

    print(f"\n{summary_title}")
    for row in summary_rows:
        print(row)


def tune_temperature(y_true, probabilities, temperatures=None):
    temperatures = np.asarray(temperatures if temperatures is not None else TEMPERATURE_GRID, dtype=np.float32)
    best_temperature = 1.0
    best_nll = np.inf
    best_probabilities = probabilities

    for temperature in temperatures:
        scaled_probabilities = temperature_scale_probabilities(probabilities, temperature)
        nll = negative_log_likelihood(y_true, scaled_probabilities)
        if nll < best_nll:
            best_temperature = float(temperature)
            best_nll = nll
            best_probabilities = scaled_probabilities

    return best_temperature, best_probabilities, best_nll


def predict_with_n1_threshold(
        probabilities,
        n1_threshold,
        n1_margin=0.03,
        transition_context_bonus=0.08,
):
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    if probabilities.shape[1] <= 1:
        return predictions

    non_n1_probs = np.delete(probabilities, 1, axis=1)
    strongest_non_n1 = np.max(non_n1_probs, axis=1)
    n1_mask = (
            (probabilities[:, 1] >= n1_threshold) &
            (probabilities[:, 1] >= strongest_non_n1 - n1_margin)
    )
    if len(predictions) >= 3 and transition_context_bonus > 0:
        previous_stage = np.roll(predictions, 1)
        next_stage = np.roll(predictions, -1)
        previous_stage[0] = predictions[0]
        next_stage[-1] = predictions[-1]
        plausible_n1_context = (
                ((previous_stage == 0) & np.isin(next_stage, [1, 2])) |
                ((previous_stage == 4) & np.isin(next_stage, [1, 0, 2])) |
                ((next_stage == 2) & np.isin(previous_stage, [0, 1, 4]))
        )
        relaxed_threshold = max(0.0, float(n1_threshold) - float(transition_context_bonus))
        relaxed_margin = float(n1_margin) + float(transition_context_bonus)
        contextual_n1_mask = (
                plausible_n1_context &
                (probabilities[:, 1] >= relaxed_threshold) &
                (probabilities[:, 1] >= strongest_non_n1 - relaxed_margin)
        )
        n1_mask = n1_mask | contextual_n1_mask
    predictions[n1_mask] = 1
    return predictions


def predict_with_n1_threshold_by_recording(
        probabilities,
        file_sample_counts,
        n1_threshold,
        n1_margin=0.03,
        transition_context_bonus=0.08,
):
    parts = []
    for start, end in _record_slices_from_counts(len(probabilities), file_sample_counts):
        parts.append(
            predict_with_n1_threshold(
                probabilities[start:end],
                n1_threshold,
                n1_margin=n1_margin,
                transition_context_bonus=transition_context_bonus,
            )
        )
    return np.concatenate(parts) if parts else np.array([], dtype=np.int32)


def tune_n1_threshold(
        y_true,
        probabilities,
        file_sample_counts=None,
        selector_mode=SELECTOR_MODE,
        thresholds=None,
        max_accuracy_drop=None,
):
    base_predictions = np.argmax(probabilities, axis=1)
    base_accuracy = float(np.mean(base_predictions == y_true))
    thresholds = np.asarray(thresholds if thresholds is not None else N1_THRESHOLD_GRID, dtype=np.float32)
    max_accuracy_drop = (
        N1_THRESHOLD_MAX_ACC_DROP
        if max_accuracy_drop is None
        else float(max_accuracy_drop)
    )

    best_threshold = 0.50
    best_predictions = base_predictions
    best_accuracy = base_accuracy
    best_metrics = score_predictions(y_true, base_predictions, selector_mode)
    best_macro_f1 = best_metrics["macro_f1"]
    best_objective = best_metrics["score"]

    for threshold in thresholds:
        predictions = predict_with_n1_threshold_by_recording(
            probabilities,
            file_sample_counts,
            threshold,
        )
        accuracy = float(np.mean(predictions == y_true))
        if accuracy < base_accuracy - max_accuracy_drop:
            continue
        metrics = score_predictions(y_true, predictions, selector_mode)
        objective = metrics["score"]
        if objective > best_objective:
            best_threshold = float(threshold)
            best_predictions = predictions
            best_accuracy = accuracy
            best_metrics = metrics
            best_macro_f1 = metrics["macro_f1"]
            best_objective = objective

    return best_threshold, {
        "base_accuracy": base_accuracy,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "best_score": float(best_objective),
        "best_kappa": best_metrics["kappa"],
        "n1_f1": best_metrics["n1_f1"],
        "max_accuracy_drop": float(max_accuracy_drop),
        "n1_recall": float(np.mean(best_predictions[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0,
    }


def build_oversampled_indices(y_raw, n1_factor=2):
    base_indices = np.arange(len(y_raw))
    if n1_factor <= 1:
        return base_indices
    n1_indices = np.where(y_raw == 1)[0]
    if len(n1_indices) == 0:
        return base_indices
    extra_indices = np.tile(n1_indices, n1_factor - 1)
    return np.concatenate([base_indices, extra_indices])


def get_file_sample_counts(folder_path, files):
    counts = []
    for filename in files:
        with np.load(os.path.join(folder_path, filename), mmap_mode="r") as data:
            counts.append(int(data["x"].shape[0]))
    return counts


def build_recording_domain_targets(file_sample_counts):
    targets = [
        np.full(int(count), domain_idx, dtype=np.float32)
        for domain_idx, count in enumerate(file_sample_counts)
        if int(count) > 0
    ]
    if not targets:
        return np.array([], dtype=np.float32)
    return np.concatenate(targets).astype(np.float32)


def smooth_sequence_predictions(
        probabilities,
        window_size=5,
        min_confidence=0.35,
        base_predictions=None,
        protected_classes=None,
        protection_thresholds=None,
        n1_enter_threshold=None,
        n1_hold_threshold=None,
):
    if len(probabilities) == 0:
        return np.array([], dtype=np.int32)

    half_window = window_size // 2
    smoothed = (
        np.asarray(base_predictions, dtype=np.int32).copy()
        if base_predictions is not None
        else np.argmax(probabilities, axis=1).astype(np.int32)
    )
    protected_classes = set(protected_classes or [])
    protection_thresholds = protection_thresholds or {}

    for idx in range(len(smoothed)):
        left = max(0, idx - half_window)
        right = min(len(smoothed), idx + half_window + 1)
        local_probs = probabilities[left:right].sum(axis=0)
        candidate = int(np.argmax(local_probs))
        current = smoothed[idx]
        protection_threshold = protection_thresholds.get(current, min_confidence)
        if current == 1 and n1_hold_threshold is not None:
            protection_threshold = min(protection_threshold, float(n1_hold_threshold))
        if current in protected_classes and probabilities[idx, current] >= protection_threshold:
            continue
        if candidate == 1 and n1_enter_threshold is not None and probabilities[idx, 1] < n1_enter_threshold:
            candidate = current
        if probabilities[idx, current] < min_confidence:
            smoothed[idx] = candidate
        elif candidate != current and local_probs[candidate] > local_probs[current] * 1.05:
            smoothed[idx] = candidate

    cleaned = smoothed.copy()
    for idx in range(1, len(cleaned) - 1):
        current = cleaned[idx]
        protection_threshold = protection_thresholds.get(current, min_confidence)
        if current == 1 and n1_hold_threshold is not None:
            protection_threshold = min(protection_threshold, float(n1_hold_threshold))
        if current in protected_classes and probabilities[idx, current] >= protection_threshold:
            continue
        if cleaned[idx - 1] == cleaned[idx + 1] and cleaned[idx] != cleaned[idx - 1]:
            cleaned[idx] = cleaned[idx - 1]

    return cleaned


def smooth_predictions_by_recording(
        probabilities,
        file_sample_counts,
        window_size=5,
        min_confidence=0.35,
        base_predictions=None,
        protected_classes=None,
        protection_thresholds=None,
        n1_enter_threshold=None,
        n1_hold_threshold=None,
):
    smoothed_parts = []
    start = 0
    for count in file_sample_counts:
        end = start + count
        base_part = None if base_predictions is None else base_predictions[start:end]
        smoothed_parts.append(
            smooth_sequence_predictions(
                probabilities[start:end],
                window_size=window_size,
                min_confidence=min_confidence,
                base_predictions=base_part,
                protected_classes=protected_classes,
                protection_thresholds=protection_thresholds,
                n1_enter_threshold=n1_enter_threshold,
                n1_hold_threshold=n1_hold_threshold,
            )
        )
        start = end
    return np.concatenate(smoothed_parts) if smoothed_parts else np.array([], dtype=np.int32)


def build_clinical_transition_mask():
    return np.array(
        [
            [1, 1, 1, 0, 0],  # W cannot jump directly to N3 or REM.
            [1, 1, 1, 0, 1],  # N1 is a gateway to N2, W, or rare REM onset.
            [1, 1, 1, 1, 1],  # N2 is the central transition hub.
            [1, 0, 1, 1, 0],  # N3 usually returns through N2; direct REM is locked out.
            [1, 1, 1, 0, 1],  # REM may persist or exit to W/N1/N2, not N3.
        ],
        dtype=bool,
    )


def build_transition_model(
        y_raw,
        file_sample_counts,
        num_classes,
        smoothing=0.05,
        self_bias=1.50,
        illegal_transition_penalty=HMM_ILLEGAL_TRANSITION_PENALTY,
):
    mask = build_clinical_transition_mask()
    transition_counts = np.where(mask, smoothing, 0.0).astype(np.float32)
    initial_counts = np.ones(num_classes, dtype=np.float32) * smoothing

    start = 0
    for count in file_sample_counts:
        end = start + count
        labels = np.asarray(y_raw[start:end], dtype=np.int32)
        if len(labels) > 0:
            initial_counts[labels[0]] += 1.0
        for previous_stage, next_stage in zip(labels[:-1], labels[1:]):
            if mask[previous_stage, next_stage]:
                transition_counts[previous_stage, next_stage] += 1.0
        start = end

    for stage_idx in range(num_classes):
        transition_counts[stage_idx, stage_idx] += self_bias

    transition_matrix = transition_counts / np.maximum(
        transition_counts.sum(axis=1, keepdims=True),
        1e-8,
    )
    initial_probs = initial_counts / np.maximum(np.sum(initial_counts), 1e-8)
    transition_log = np.where(
        mask,
        np.log(np.clip(transition_matrix, 1e-8, 1.0)),
        float(illegal_transition_penalty),
    )
    initial_log = np.log(np.clip(initial_probs, 1e-8, 1.0))
    return transition_log.astype(np.float32), initial_log.astype(np.float32), transition_matrix.astype(np.float32)


def Viterbi_decode_adaptive_v2(
        probabilities,
        transition_log,
        initial_log,
        emission_weight=1.0,
        quality_scores=None,
        min_quality=0.10,
        low_quality_threshold=0.60,
        self_loop_boost=0.85,
):
    """
    Quality-aware adaptive Viterbi decoding.

    Q_t controls the entropy of the emission term. Low-quality epochs are
    pushed toward a uniform emission distribution, and the transition model
    receives an adaptive self-loop boost to suppress artifact-driven jumps.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(probabilities) == 0:
        return np.array([], dtype=np.int32)

    transition_log = np.asarray(transition_log, dtype=np.float32)
    initial_log = np.asarray(initial_log, dtype=np.float32)
    log_emissions = np.log(np.clip(probabilities, 1e-8, 1.0))
    num_steps, num_classes = log_emissions.shape

    if quality_scores is None:
        quality = np.ones((num_steps, 1), dtype=np.float32)
    else:
        quality = np.asarray(quality_scores, dtype=np.float32).reshape(-1, 1)
        if len(quality) != num_steps:
            raise ValueError(
                "quality_scores length must match probabilities length: "
                f"{len(quality)} vs {num_steps}"
            )
        quality = np.clip(quality, min_quality, 1.0)

    uniform_log = np.log(1.0 / float(num_classes))
    adapted_log_emissions = quality * log_emissions + (1.0 - quality) * uniform_log
    emissions = emission_weight * adapted_log_emissions

    scores = np.full((num_steps, num_classes), -np.inf, dtype=np.float32)
    backpointers = np.zeros((num_steps, num_classes), dtype=np.int32)
    diagonal = np.arange(num_classes)

    scores[0] = initial_log + emissions[0]
    for step_idx in range(1, num_steps):
        dynamic_transition_log = transition_log
        current_quality = float(quality[step_idx, 0])
        if current_quality < low_quality_threshold:
            adaptive_strength = (low_quality_threshold - current_quality) / max(
                low_quality_threshold - min_quality,
                1e-6,
            )
            adaptive_strength = float(np.clip(adaptive_strength, 0.0, 1.0))
            dynamic_transition_log = transition_log.copy()
            dynamic_transition_log[diagonal, diagonal] += self_loop_boost * adaptive_strength

        candidate_scores = scores[step_idx - 1][:, np.newaxis] + dynamic_transition_log
        backpointers[step_idx] = np.argmax(candidate_scores, axis=0)
        scores[step_idx] = np.max(candidate_scores, axis=0) + emissions[step_idx]

    decoded = np.zeros(num_steps, dtype=np.int32)
    decoded[-1] = int(np.argmax(scores[-1]))
    for step_idx in range(num_steps - 2, -1, -1):
        decoded[step_idx] = backpointers[step_idx + 1, decoded[step_idx + 1]]
    return decoded


def viterbi_decode(
        probabilities,
        transition_log,
        initial_log,
        emission_weight=1.0,
        quality_scores=None,
        min_quality=0.10,
):
    return Viterbi_decode_adaptive_v2(
        probabilities,
        transition_log,
        initial_log,
        emission_weight=emission_weight,
        quality_scores=quality_scores,
        min_quality=min_quality,
    )


def viterbi_predictions_by_recording(
        probabilities,
        file_sample_counts,
        transition_log,
        initial_log,
        emission_weight=HMM_EMISSION_WEIGHT,
        quality_scores=None,
):
    decoded_parts = []
    start = 0
    for count in file_sample_counts:
        end = start + count
        quality_part = None if quality_scores is None else quality_scores[start:end]
        decoded_parts.append(
            viterbi_decode(
                probabilities[start:end],
                transition_log,
                initial_log,
                emission_weight=emission_weight,
                quality_scores=quality_part,
            )
        )
        start = end
    return np.concatenate(decoded_parts) if decoded_parts else np.array([], dtype=np.int32)


def build_postprocessor_candidates(
        probabilities,
        file_sample_counts,
        n1_threshold,
        transition_log=None,
        initial_log=None,
        emission_weight=HMM_EMISSION_WEIGHT,
        emission_weights=None,
        include_hmm=True,
        quality_scores=None,
        include_quality_aware_hmm=USE_QUALITY_AWARE_HMM,
):
    calibrated_predictions = predict_with_n1_threshold_by_recording(
        probabilities,
        file_sample_counts,
        n1_threshold,
    )
    candidates = {
        "temperature_argmax": np.argmax(probabilities, axis=1).astype(np.int32),
        "n1_calibrated": calibrated_predictions,
        "n1_protected_smoothed": smooth_predictions_by_recording(
            probabilities,
            file_sample_counts,
            window_size=5,
            min_confidence=0.35,
            base_predictions=calibrated_predictions,
            protected_classes={1},
            protection_thresholds={1: n1_threshold},
        ),
    }
    if include_hmm and transition_log is not None and initial_log is not None:
        candidate_emission_weights = tuple(emission_weights or (emission_weight,))
        for candidate_emission_weight in candidate_emission_weights:
            suffix = "" if np.isclose(candidate_emission_weight, emission_weight) else f"_e{candidate_emission_weight:.2f}"
            candidates[f"clinical_hmm{suffix}"] = viterbi_predictions_by_recording(
                probabilities,
                file_sample_counts,
                transition_log,
                initial_log,
                emission_weight=candidate_emission_weight,
            )
            if include_quality_aware_hmm and quality_scores is not None:
                candidates[f"quality_aware_hmm{suffix}"] = viterbi_predictions_by_recording(
                    probabilities,
                    file_sample_counts,
                    transition_log,
                    initial_log,
                    emission_weight=candidate_emission_weight,
                    quality_scores=quality_scores,
                )
    return candidates


def select_postprocessor_candidate(y_true, candidates, selector_mode=SELECTOR_MODE):
    scored_candidates = {
        name: score_predictions(y_true, predictions, selector_mode)
        for name, predictions in candidates.items()
    }
    best_name = max(scored_candidates, key=lambda name: scored_candidates[name]["score"])
    if POSTPROCESS_ACCURACY_GUARD and str(selector_mode or "").strip().lower() != "accuracy":
        best_accuracy_name = max(
            scored_candidates,
            key=lambda name: (
                scored_candidates[name]["accuracy"],
                scored_candidates[name]["score"],
            ),
        )
        best_metrics = scored_candidates[best_name]
        best_accuracy_metrics = scored_candidates[best_accuracy_name]
        score_gap = best_metrics["score"] - best_accuracy_metrics["score"]
        accuracy_gap = best_accuracy_metrics["accuracy"] - best_metrics["accuracy"]
        if score_gap <= POSTPROCESS_SCORE_TIE_EPS and accuracy_gap >= POSTPROCESS_ACCURACY_GUARD_DROP:
            best_name = best_accuracy_name
    return best_name, scored_candidates[best_name], scored_candidates


class BestValidationSelector(Callback):
    def __init__(
            self,
            X_val_raw,
            X_val_seg,
            y_val_raw,
            val_file_sample_counts,
            val_quality_scores=None,
            batch_size=64,
            patience=8,
            min_delta=1e-4,
            selector_mode=SELECTOR_MODE,
            temperature_scaling=TEMPERATURE_SCALING,
            transition_log=None,
            initial_log=None,
            emission_weight=HMM_EMISSION_WEIGHT,
            emission_weights=None,
            use_hmm=False,
            sequence_mode=False,
            seq_len=25,
            sequence_stride=5,
    ):
        super().__init__()
        self.X_val_raw = X_val_raw
        self.X_val_seg = X_val_seg
        self.y_val_raw = y_val_raw
        self.val_file_sample_counts = val_file_sample_counts
        self.val_quality_scores = val_quality_scores
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.selector_mode = selector_mode
        self.temperature_scaling = temperature_scaling
        self.transition_log = transition_log
        self.initial_log = initial_log
        self.emission_weight = emission_weight
        self.emission_weights = tuple(emission_weights or (emission_weight,))
        self.use_hmm = use_hmm
        self.sequence_mode = bool(sequence_mode)
        self.seq_len = max(1, int(seq_len))
        self.sequence_stride = max(1, int(sequence_stride))
        self.wait = 0
        self.best_epoch = 0
        self.best_score = -np.inf
        self.best_weights = None
        self.best_n1_threshold = 0.50
        self.best_temperature = 1.0
        self.best_decoder_mode = "n1_protected_smoothed"
        self.best_metrics = None
        self.restore_on_train_end = True

    def _predict_probabilities(self):
        outputs = predict_model_outputs(
            self.model,
            self.X_val_raw,
            self.X_val_seg,
            batch_size=self.batch_size,
            verbose=0,
            sequence_mode=self.sequence_mode,
            file_sample_counts=self.val_file_sample_counts,
            seq_len=self.seq_len,
            sequence_stride=self.sequence_stride,
        )
        main_probs, _, _ = extract_prediction_heads(outputs)
        return main_probs

    def _evaluate_current_model(self):
        probabilities = self._predict_probabilities()
        temperature = 1.0
        nll = negative_log_likelihood(self.y_val_raw, probabilities)
        if self.temperature_scaling:
            temperature, probabilities, nll = tune_temperature(self.y_val_raw, probabilities)

        n1_threshold, threshold_metrics = tune_n1_threshold(
            self.y_val_raw,
            probabilities,
            file_sample_counts=self.val_file_sample_counts,
            selector_mode=self.selector_mode,
        )
        candidates = build_postprocessor_candidates(
            probabilities,
            self.val_file_sample_counts,
            n1_threshold,
            transition_log=self.transition_log,
            initial_log=self.initial_log,
            emission_weight=self.emission_weight,
            emission_weights=self.emission_weights,
            include_hmm=self.use_hmm,
            quality_scores=self.val_quality_scores,
        )
        decoder_mode, selected_metrics, candidate_metrics = select_postprocessor_candidate(
            self.y_val_raw,
            candidates,
            self.selector_mode,
        )
        argmax_predictions = candidates["temperature_argmax"]
        smoothed_metrics = candidate_metrics["n1_protected_smoothed"]

        return {
            "score": selected_metrics["score"],
            "decoder_mode": decoder_mode,
            "temperature": temperature,
            "nll": nll,
            "n1_threshold": n1_threshold,
            "argmax_accuracy": float(np.mean(argmax_predictions == self.y_val_raw)),
            "calibrated_accuracy": threshold_metrics["best_accuracy"],
            "smoothed_accuracy": smoothed_metrics["accuracy"],
            "smoothed_macro_f1": smoothed_metrics["macro_f1"],
            "selected_accuracy": selected_metrics["accuracy"],
            "selected_macro_f1": selected_metrics["macro_f1"],
            "selected_kappa": selected_metrics["kappa"],
            "n1_precision": selected_metrics["n1_precision"],
            "n1_recall": selected_metrics["n1_recall"],
            "n1_f1": selected_metrics["n1_f1"],
            "n2_recall": selected_metrics["n2_recall"],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = self._evaluate_current_model()
        logs["val_selection_score"] = metrics["score"]
        logs["val_smoothed_accuracy"] = metrics["smoothed_accuracy"]
        logs["val_smoothed_macro_f1"] = metrics["smoothed_macro_f1"]

        improved = metrics["score"] > self.best_score + self.min_delta
        if improved:
            self.best_score = metrics["score"]
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.best_n1_threshold = metrics["n1_threshold"]
            self.best_temperature = metrics["temperature"]
            self.best_decoder_mode = metrics["decoder_mode"]
            self.best_metrics = metrics
            self.wait = 0
            status = "new best"
        else:
            self.wait += 1
            status = f"wait={self.wait}/{self.patience}"

        print(
            "\nValidation selector "
            f"({status}) | score={metrics['score']:.4f} "
            f"acc={metrics['selected_accuracy']:.4f} "
            f"macro_f1={metrics['selected_macro_f1']:.4f} "
            f"kappa={metrics['selected_kappa']:.4f} "
            f"N1_P={metrics['n1_precision']:.4f} "
            f"N1_R={metrics['n1_recall']:.4f} "
            f"N1_F1={metrics['n1_f1']:.4f} "
            f"N2_R={metrics['n2_recall']:.4f} "
            f"decoder={metrics['decoder_mode']} "
            f"T={metrics['temperature']:.2f} "
            f"thr={metrics['n1_threshold']:.3f}"
        )

        if self.wait >= self.patience:
            self.model.stop_training = True
            print(
                "Validation selector stopping: "
                f"no score improvement for {self.patience} epochs."
            )

    def on_train_end(self, logs=None):
        if not self.restore_on_train_end:
            print(
                "Validation selector retained best weights for final restore; "
                "skipping intermediate phase restore."
            )
            return
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                "Validation selector restored best weights from "
                f"epoch {self.best_epoch} | score={self.best_score:.4f} "
                f"decoder={self.best_decoder_mode} "
                f"T={self.best_temperature:.2f} "
                f"thr={self.best_n1_threshold:.3f}"
            )


def find_transition_windows(labels, window_radius=2):
    transitions = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            left = max(0, idx - window_radius)
            right = min(len(labels), idx + window_radius + 1)
            transitions.append((idx - 1, idx, left, right))
    return transitions


def save_academic_results(y_true, y_pred, class_names, output_dir):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report_dict).transpose()
    report_df = df.drop(index=["accuracy"], errors="ignore").drop(columns=["support"], errors="ignore")
    report_df.columns = ["Precision", "Recall", "F1-Score"]

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        report_df.astype(float),
        annot=True,
        cmap="RdBu_r",
        fmt=".4f",
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=1.5,
        annot_kws={"size": 13, "weight": "bold"},
    )
    metrics = score_predictions(y_true, y_pred, mode="macro_f1")
    plt.title(
        (
            "Classification Report | "
            f"macro-F1: {metrics['macro_f1']:.4f} | "
            f"N1 F1: {metrics['n1_f1']:.4f} | "
            f"kappa: {metrics['kappa']:.4f} | "
            f"acc: {metrics['accuracy']:.4f}"
        ),
        fontweight="bold",
        fontsize=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_report_visual.png"), dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def save_prediction_probabilities(
        output_dir,
        timestamp,
        log_suffix,
        seed,
        run_tag,
        class_names,
        y_true,
        test_probs_raw,
        test_probs_calibrated,
        y_pred_argmax,
        y_pred_temperature_argmax,
        y_pred_final,
        test_candidates,
        test_file_sample_counts,
        decoder_mode,
        temperature,
        n1_threshold,
):
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    filename = f"result_{timestamp}{log_suffix}_seed{int(seed)}_predictions.npz"
    save_path = os.path.join(prediction_dir, filename)
    candidate_names = list(test_candidates.keys())
    candidate_predictions = np.stack(
        [np.asarray(test_candidates[name], dtype=np.int16) for name in candidate_names],
        axis=0,
    )
    np.savez_compressed(
        save_path,
        y_true=np.asarray(y_true, dtype=np.int16),
        test_probs_raw=np.asarray(test_probs_raw, dtype=np.float32),
        test_probs_calibrated=np.asarray(test_probs_calibrated, dtype=np.float32),
        y_pred_argmax=np.asarray(y_pred_argmax, dtype=np.int16),
        y_pred_temperature_argmax=np.asarray(y_pred_temperature_argmax, dtype=np.int16),
        y_pred_final=np.asarray(y_pred_final, dtype=np.int16),
        candidate_names=np.asarray(candidate_names),
        candidate_predictions=candidate_predictions,
        test_file_sample_counts=np.asarray(test_file_sample_counts, dtype=np.int32),
        class_labels=np.asarray(class_names),
        seed=np.asarray(int(seed), dtype=np.int64),
        run_tag=np.asarray(str(run_tag or "")),
        decoder_mode=np.asarray(str(decoder_mode or "")),
        temperature=np.asarray(float(temperature), dtype=np.float32),
        n1_threshold=np.asarray(float(n1_threshold), dtype=np.float32),
    )
    print(f"Saved prediction probabilities: {save_path}")
    return save_path


def build_recording_epoch_axes(file_sample_counts):
    recording_ids = []
    epoch_indices = []
    for recording_idx, count in enumerate(file_sample_counts):
        count = int(count)
        if count <= 0:
            continue
        recording_ids.append(np.full(count, recording_idx, dtype=np.int32))
        epoch_indices.append(np.arange(count, dtype=np.int32))
    if not recording_ids:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(recording_ids), np.concatenate(epoch_indices)


def probability_logits(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = np.sum(probabilities, axis=1, keepdims=True)
    probabilities = probabilities / np.maximum(row_sums, 1e-12)
    return np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32)


def split_prediction_payload(
        split_name,
        labels,
        probabilities,
        file_sample_counts,
        quality_scores=None,
        final_predictions=None,
        calibrated_probabilities=None,
):
    labels = np.asarray(labels, dtype=np.int16).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError(
            f"{split_name} probabilities must be 2D, got shape {probabilities.shape}."
        )
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split_name} probabilities/labels length mismatch: "
            f"{probabilities.shape[0]} vs {labels.shape[0]}."
        )
    counts = np.asarray(file_sample_counts, dtype=np.int32).reshape(-1)
    if int(np.sum(counts)) != int(labels.shape[0]):
        raise ValueError(
            f"{split_name} file_sample_counts sum to {int(np.sum(counts))}, "
            f"expected {labels.shape[0]}."
        )
    recording_id, epoch_index = build_recording_epoch_axes(counts)
    argmax_predictions = np.argmax(probabilities, axis=1).astype(np.int16)
    if final_predictions is None:
        final_predictions = argmax_predictions
    final_predictions = np.asarray(final_predictions, dtype=np.int16).reshape(-1)
    if final_predictions.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split_name} final prediction length mismatch: "
            f"{final_predictions.shape[0]} vs {labels.shape[0]}."
        )
    payload = {
        f"{split_name}_y_true": labels,
        f"{split_name}_probs_raw": probabilities,
        f"{split_name}_final_logits": probability_logits(probabilities),
        f"{split_name}_pred_argmax": argmax_predictions,
        f"{split_name}_pred": final_predictions,
        f"{split_name}_y_pred_final": final_predictions,
        f"{split_name}_file_sample_counts": counts,
        f"{split_name}_recording_id": recording_id,
        f"{split_name}_epoch_index": epoch_index,
    }
    if calibrated_probabilities is not None:
        calibrated = np.asarray(calibrated_probabilities, dtype=np.float32)
        if calibrated.shape != probabilities.shape:
            raise ValueError(
                f"{split_name} calibrated probability shape mismatch: "
                f"{calibrated.shape} vs {probabilities.shape}."
            )
        payload[f"{split_name}_probs_calibrated"] = calibrated
    if quality_scores is not None:
        quality = np.asarray(quality_scores, dtype=np.float32).reshape(-1)
        if quality.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{split_name} quality length mismatch: {quality.shape[0]} vs {labels.shape[0]}."
            )
        payload[f"{split_name}_quality"] = quality
    return payload


def save_split_prediction_probabilities(
        output_dir,
        timestamp,
        log_suffix,
        seed,
        run_tag,
        class_names,
        train_labels,
        train_probs_raw,
        train_file_sample_counts,
        val_labels,
        val_probs_raw,
        val_file_sample_counts,
        test_labels,
        test_probs_raw,
        test_file_sample_counts,
        train_quality_scores=None,
        val_quality_scores=None,
        test_quality_scores=None,
        train_final_predictions=None,
        val_final_predictions=None,
        test_final_predictions=None,
        train_probs_calibrated=None,
        val_probs_calibrated=None,
        test_probs_calibrated=None,
        decoder_mode=None,
        temperature=1.0,
        n1_threshold=0.0,
):
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    filename = f"result_{timestamp}{log_suffix}_seed{int(seed)}_split_predictions.npz"
    save_path = os.path.join(prediction_dir, filename)
    payload = {
        **split_prediction_payload(
            "train",
            train_labels,
            train_probs_raw,
            train_file_sample_counts,
            quality_scores=train_quality_scores,
            final_predictions=train_final_predictions,
            calibrated_probabilities=train_probs_calibrated,
        ),
        **split_prediction_payload(
            "val",
            val_labels,
            val_probs_raw,
            val_file_sample_counts,
            quality_scores=val_quality_scores,
            final_predictions=val_final_predictions,
            calibrated_probabilities=val_probs_calibrated,
        ),
        **split_prediction_payload(
            "test",
            test_labels,
            test_probs_raw,
            test_file_sample_counts,
            quality_scores=test_quality_scores,
            final_predictions=test_final_predictions,
            calibrated_probabilities=test_probs_calibrated,
        ),
        "class_labels": np.asarray(class_names),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "run_tag": np.asarray(str(run_tag or "")),
        "decoder_mode": np.asarray(str(decoder_mode or "")),
        "temperature": np.asarray(float(temperature), dtype=np.float32),
        "n1_threshold": np.asarray(float(n1_threshold), dtype=np.float32),
    }
    np.savez_compressed(save_path, **payload)
    print(f"Saved train/val/test split predictions for C11: {save_path}")
    return save_path


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = sanitize_run_tag(RUN_TAG)
    log_suffix = f"_{run_tag}" if run_tag else ""
    log_path = os.path.join(LOG_DIR, f"result_{timestamp}{log_suffix}.txt")
    sys.stdout = Logger(log_path)

    (
        strategy,
        num_replicas,
        train_batch_size,
        predict_batch_size,
        learning_rate,
        physical_gpus,
        worker_devices,
    ) = configure_training_strategy()
    set_random_seed(SEED)

    from src.data_processing import get_npz_file_list, load_npz_data, split_npz_files
    from src.grad_cam import run_hybrid_grad_cam  # noqa: F401
    from src.lime_explanation import (
        extract_segment_features,
        run_multimodal_lime,
        semantic_segmentation,
        summarize_transition_explanation,
    )
    from src.model_training import (
        build_adaptive_clinical_priors,
        build_explanation_feedback_targets,
        create_hybrid_model_v2,
        create_sequence_sleep_model,
        recompile_sleep_model,
        reset_adaptive_priors,
        set_adaptive_priors,
    )
    from src.visualization import plot_xai_multimodal_dashboard
    try:
        from src.visualization import plot_transition_explanation_dashboard
    except ImportError:
        plot_transition_explanation_dashboard = None

    print("Core modules loaded.")
    print(
        "Run protocol: "
        f"version={RUN_MAIN_PROTOCOL_VERSION}, "
        f"profile={PROTOCOL_PROFILE}, "
        f"allow_env_overrides={ALLOW_PROTOCOL_ENV_OVERRIDES}, "
        f"run_tag={run_tag or 'none'}, "
        f"save_prediction_probs={SAVE_PREDICTION_PROBS}, "
        f"save_split_prediction_probs={SAVE_SPLIT_PREDICTION_PROBS}, "
        f"split_predict_batch={SPLIT_PREDICT_BATCH_SIZE or 'auto'}, "
        f"script={os.path.abspath(__file__)}"
    )
    if _IGNORED_PROTOCOL_ENV_OVERRIDES:
        ignored_summary = ", ".join(
            f"{name}={value!r}->{_FORCED_PROTOCOL_ENV[name]!r}"
            for name, value in sorted(_IGNORED_PROTOCOL_ENV_OVERRIDES.items())
        )
        print(
            "Protocol lock ignored stale environment overrides: "
            f"{ignored_summary}"
        )
    visible_gpu_names = ", ".join(gpu.name for gpu in physical_gpus) if physical_gpus else "CPU only"
    worker_device_names = ", ".join(worker_devices) if worker_devices else "default device"
    print(
        "Distributed training: "
        f"requested_server_gpus={TARGET_GPU_IDS}, "
        f"memory_profile={MEMORY_PROFILE}, "
        f"deterministic_ops={DETERMINISTIC_TF}, "
        f"cudnn_workspace_mb={os.environ.get('TF_CUDNN_WORKSPACE_LIMIT_IN_MB')}, "
        f"tf_visible_devices={visible_gpu_names}, "
        f"strategy_worker_devices={worker_device_names}, "
        f"replicas={num_replicas}, "
        f"per_replica_batch={PER_REPLICA_BATCH_SIZE}, "
        f"global_train_batch={train_batch_size}, "
        f"predict_batch={predict_batch_size}, "
        f"learning_rate={learning_rate:.6f}"
    )
    if REQUIRE_REQUESTED_GPUS and REQUESTED_GPU_COUNT > 1 and num_replicas < REQUESTED_GPU_COUNT:
        if len(physical_gpus) == 0:
            raise RuntimeError(
                "TensorFlow did not detect any GPU. The model is still wrapped in strategy.scope(), "
                "but this Python environment cannot use the server GPUs yet. "
                f"Requested CUDA_VISIBLE_DEVICES={TARGET_GPU_IDS}. "
                "The TensorFlow startup log says CUDA drivers/GPU libraries were not found, "
                "so check that this venv/container has GPU-enabled TensorFlow and the CUDA/cuDNN "
                "runtime libraries visible to it."
            )
        raise RuntimeError(
            "TensorFlow did not create the requested multi-GPU strategy: "
            f"requested {REQUESTED_GPU_COUNT} GPU(s) via CUDA_VISIBLE_DEVICES={TARGET_GPU_IDS}, "
            f"but strategy has {num_replicas} replica(s). "
            "Check the CUDA_VISIBLE_DEVICES mapping and restart Python so TensorFlow sees GPUs before import. "
            "Set LFTS_REQUIRE_REQUESTED_GPUS=0 only if you intentionally want CPU/single-GPU fallback."
        )
    print(
        "Extended ablation grid configured: "
        f"enabled={EXTENDED_ABLATION_STUDY}, "
        f"selector_mode={SELECTOR_MODE}, "
        f"concept_weights={EXTENDED_CONCEPT_WEIGHTS}, "
        f"channel_strengths={EXTENDED_CHANNEL_STRENGTHS}, "
        f"emission_weights={EXTENDED_HMM_EMISSION_WEIGHTS}, "
        f"validation_emission_weights={VALIDATION_HMM_EMISSION_WEIGHTS}, "
        f"self_loop_boosts={EXTENDED_SELF_LOOP_BOOSTS}, "
        f"accuracy_selector={EXTENDED_ACCURACY_SELECTOR_MODE}"
    )
    print(
        "Clean training protocol: "
        f"experiment_protocol={EXPERIMENT_PROTOCOL}, "
        f"stratified_subject_split={STRATIFIED_SUBJECT_SPLIT}, "
        f"model_input_mode={MODEL_INPUT_MODE}, "
        f"sequence_model={USE_SEQUENCE_MODEL}, "
        f"seq_len={SEQUENCE_LENGTH}, "
        f"sequence_train_stride={SEQUENCE_TRAIN_STRIDE}, "
        f"sequence_predict_stride={SEQUENCE_PREDICT_STRIDE}, "
        f"sequence_backbone={SEQUENCE_BACKBONE}, "
        f"epoch_encoder_variant={EPOCH_ENCODER_VARIANT}, "
        f"sequence_transformer={SEQUENCE_USE_TRANSFORMER}, "
        f"micro_macro_training={MICRO_MACRO_TRAINING}, "
        f"micro_pretrain_epochs={MICRO_PRETRAIN_EPOCHS}, "
        f"macro_stabilization_epochs={MACRO_STABILIZATION_EPOCHS}, "
        f"macro_aux_ramp_epochs={MACRO_AUX_RAMP_EPOCHS}, "
        f"micro_finetune_lr_scale={MICRO_FINETUNE_LR_SCALE:.3f}, "
        f"loss={CLASSIFICATION_LOSS_NAME}, "
        f"generalization_preset={GENERALIZATION_PRESET}, "
        f"domain_mixstyle={USE_DOMAIN_MIXSTYLE}, "
        f"mixstyle_p={DOMAIN_MIXSTYLE_P:.2f}, "
        f"mixstyle_alpha={DOMAIN_MIXSTYLE_ALPHA:.2f}, "
        f"domain_alignment={USE_DOMAIN_ALIGNMENT}, "
        f"domain_alignment_mode={DOMAIN_ALIGNMENT_MODE}, "
        f"domain_alignment_stage_conditioned={DOMAIN_ALIGNMENT_STAGE_CONDITIONED}, "
        f"domain_alignment_loss_weight={DOMAIN_ALIGNMENT_LOSS_WEIGHT:.4f}, "
        f"n1_supcon={USE_N1_SUPCON}, "
        f"n1_supcon_loss_weight={N1_SUPCON_LOSS_WEIGHT:.3f}, "
        f"n1_supcon_projection_dim={N1_SUPCON_PROJECTION_DIM}, "
        f"balanced_sequence_sampling={USE_BALANCED_SEQUENCE_SAMPLING}, "
        f"balance_n1_window_boost={BALANCE_N1_WINDOW_BOOST:.2f}, "
        f"balance_n3_window_boost={BALANCE_N3_WINDOW_BOOST:.2f}, "
        f"balance_transition_window_boost={BALANCE_TRANSITION_WINDOW_BOOST:.2f}, "
        f"transition_boundary_weights={USE_TRANSITION_BOUNDARY_WEIGHTS}, "
        f"transition_boundary_radius={TRANSITION_BOUNDARY_RADIUS}, "
        f"transition_boundary_base_boost={TRANSITION_BOUNDARY_BASE_BOOST:.2f}, "
        f"transition_boundary_n1_boost={TRANSITION_BOUNDARY_N1_BOOST:.2f}, "
        f"transition_boundary_target_boost={TRANSITION_BOUNDARY_TARGET_BOOST:.2f}, "
        f"transition_boundary_max_weight={TRANSITION_BOUNDARY_MAX_WEIGHT:.2f}, "
        f"transition_boundary_aux={USE_TRANSITION_BOUNDARY_AUX}, "
        f"fine_transition_aux={USE_FINE_TRANSITION_AUX}, "
        f"transition_boundary_aux_radius={TRANSITION_BOUNDARY_AUX_RADIUS}, "
        f"transition_boundary_aux_loss_weight={TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT:.3f}, "
        f"transition_boundary_aux_pos_weight={TRANSITION_BOUNDARY_AUX_POS_WEIGHT:.2f}, "
        f"transition_boundary_aux_n1_weight={TRANSITION_BOUNDARY_AUX_N1_WEIGHT:.2f}, "
        f"n1_threshold_grid=({float(np.min(N1_THRESHOLD_GRID)):.2f},"
        f"{float(np.max(N1_THRESHOLD_GRID)):.2f},n={len(N1_THRESHOLD_GRID)}), "
        f"n1_threshold_max_acc_drop={N1_THRESHOLD_MAX_ACC_DROP:.3f}, "
        f"postprocess_accuracy_guard={POSTPROCESS_ACCURACY_GUARD}, "
        f"postprocess_score_tie_eps={POSTPROCESS_SCORE_TIE_EPS:.4f}, "
        f"postprocess_accuracy_guard_drop={POSTPROCESS_ACCURACY_GUARD_DROP:.4f}, "
        f"hmm_enabled={USE_HMM_VITERBI}, "
        f"quality_hmm_enabled={USE_QUALITY_AWARE_HMM}, "
        f"adaptive_clinical_priors={ADAPTIVE_CLINICAL_PRIORS}, "
        f"adaptive_prior_alpha={ADAPTIVE_PRIOR_ALPHA:.2f}, "
        f"adaptive_prior_min_class_samples={ADAPTIVE_PRIOR_MIN_CLASS_SAMPLES}, "
        f"adaptive_prior_contrast_power={ADAPTIVE_PRIOR_CONTRAST_POWER:.2f}, "
        f"adaptive_prior_reliability_power={ADAPTIVE_PRIOR_RELIABILITY_POWER:.2f}, "
        f"sequence_clinical_concepts={USE_SEQUENCE_CLINICAL_CONCEPTS}, "
        f"clinical_concept_version={CLINICAL_CONCEPT_VERSION}, "
        f"concept_language={USE_CONCEPT_LANGUAGE}, "
        f"concept_language_stage_loss_weight={CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT:.3f}, "
        f"concept_language_bottleneck_loss_weight={CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT:.3f}, "
        f"late_clinical_concept_residual={USE_LATE_CLINICAL_CONCEPT_RESIDUAL}, "
        f"late_concept_residual_scale={LATE_CONCEPT_RESIDUAL_SCALE:.3f}, "
        f"late_concept_residual_trainable={LATE_CONCEPT_RESIDUAL_TRAINABLE}, "
        f"late_concept_stop_gradient={LATE_CONCEPT_STOP_GRADIENT}, "
        f"late_concept_confidence_gate={USE_LATE_CONCEPT_CONFIDENCE_GATE}, "
        f"late_concept_gate_low_confidence={LATE_CONCEPT_GATE_LOW_CONFIDENCE:.3f}, "
        f"late_concept_gate_high_confidence={LATE_CONCEPT_GATE_HIGH_CONFIDENCE:.3f}, "
        f"late_concept_gate_mid_value={LATE_CONCEPT_GATE_MID_VALUE:.3f}, "
        f"concept_feature_modulation={USE_CONCEPT_FEATURE_MODULATION}, "
        f"concept_modulation_strength={CONCEPT_MODULATION_STRENGTH:.3f}, "
        f"concept_modulation_stop_gradient={CONCEPT_MODULATION_STOP_GRADIENT}, "
        f"concept_modulation_mode={CONCEPT_MODULATION_MODE}, "
        f"ot_tacb={USE_OT_TACB}, "
        f"ot_tacb_sinkhorn_loss_weight={OT_TACB_SINKHORN_LOSS_WEIGHT:.3f}, "
        f"ot_tacb_fine_boundary_loss_weight={OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT:.3f}, "
        f"ot_tacb_concept_derivative_loss_weight={OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT:.3f}, "
        f"concept_derivative_aux={USE_CONCEPT_DERIVATIVE_AUX}, "
        f"concept_derivative_loss_weight={CONCEPT_DERIVATIVE_LOSS_WEIGHT:.3f}, "
        f"ot_tacb_sinkhorn_epsilon={OT_TACB_SINKHORN_EPSILON:.3f}, "
        f"ot_tacb_sinkhorn_iterations={OT_TACB_SINKHORN_ITERATIONS}, "
        f"ot_tacb_cosine_weight={OT_TACB_COSINE_WEIGHT:.2f}, "
        f"dpcm={USE_MPDG}, "
        f"dpcm_concept_dim={MPDG_CONCEPT_DIM}, "
        f"dpcm_mask_amplitude={DPCM_MASK_AMPLITUDE:.2f}, "
        f"dpcm_prior_init_scale={DPCM_PRIOR_INIT_SCALE:.3f}, "
        f"dpcm_boundary_loss_weight={DPCM_BOUNDARY_LOSS_WEIGHT:.3f}, "
        f"dpcm_boundary_contrast_loss_weight={DPCM_BOUNDARY_CONTRAST_LOSS_WEIGHT:.3f}, "
        f"dpcm_boundary_contrast_margin={DPCM_BOUNDARY_CONTRAST_MARGIN:.2f}, "
        f"clinical_prior_trainable={CLINICAL_PRIOR_TRAINABLE}, "
        f"xai_dashboards={RUN_XAI_DASHBOARDS}"
    )
    print(f"Clinical concept version: {CLINICAL_CONCEPT_VERSION}")
    if OT_TACB_DISABLED_LIME_FEEDBACK_MODE is not None:
        print(
            "OT-TACB disabled online LIME-DCCS distillation: "
            f"requested_mode={OT_TACB_DISABLED_LIME_FEEDBACK_MODE!r}, active_mode={EXPLANATION_FEEDBACK_MODE!r}"
        )
    print(
        "Explanation feedback training: "
        f"enabled={EXPLANATION_FEEDBACK_TRAINING}, "
        f"mode={EXPLANATION_FEEDBACK_MODE}, "
        f"concept_prior_weight={CONCEPT_TARGET_PRIOR_WEIGHT}, "
        f"dccs_channel_prior_weight={DCCS_CHANNEL_PRIOR_WEIGHT}, "
        f"lime_dccs_weights=({LIME_DCCS_FEEDBACK_ALPHA:.2f}, "
        f"{LIME_DCCS_FEEDBACK_BETA:.2f}, {LIME_DCCS_FEEDBACK_GAMMA:.2f}), "
        f"lime_dccs_target_path={LIME_DCCS_TARGET_PATH or 'none'}, "
        f"lime_dccs_filter={LIME_DCCS_FEEDBACK_FILTER}, "
        f"lime_dccs_min_confidence={LIME_DCCS_MIN_CONFIDENCE:.2f}, "
        f"lime_dccs_use_val_feedback={LIME_DCCS_USE_VAL_FEEDBACK}, "
        f"lime_dccs_soft_sample_weights={LIME_DCCS_SOFT_SAMPLE_WEIGHTS}, "
        f"channel_quality_feedback={USE_CHANNEL_QUALITY_FEEDBACK}, "
        f"quality_routing={USE_CHANNEL_QUALITY_ROUTING}, "
        f"concept_sample_weight_mode={_concept_sample_weight_mode_key(CONCEPT_SAMPLE_WEIGHT_MODE)}, "
        f"concept_sample_weight_floor={CONCEPT_SAMPLE_WEIGHT_FLOOR:.2f}, "
        f"concept_output_loss_weight={CONCEPT_OUTPUT_LOSS_WEIGHT}, "
        f"attention_loss_weight={ATTENTION_FEEDBACK_LOSS_WEIGHT}, "
        f"concept_bottleneck_loss_weight={CONCEPT_BOTTLENECK_LOSS_WEIGHT}, "
        f"aux_warmup_epochs={AUX_FEEDBACK_WARMUP_EPOCHS}, "
        f"aux_ramp_epochs={AUX_FEEDBACK_RAMP_EPOCHS}"
    )

    all_files = get_npz_file_list(MULTI_CHANNEL_PATH)
    train_files, val_files, test_files = split_npz_files(
        all_files,
        test_size=0.15,
        val_size=0.15,
        seed=SEED,
        folder_path=MULTI_CHANNEL_PATH,
        class_names=CLASS_LABELS,
        log_summary=True,
        stratify_by_labels=STRATIFIED_SUBJECT_SPLIT,
    )

    print(f"Found {len(all_files)} recording files.")
    print(f"Train files ({len(train_files)}): {train_files}")
    print(f"Val files   ({len(val_files)}): {val_files}")
    print(f"Test files  ({len(test_files)}): {test_files}")

    X_train_raw, y_train_raw, train_quality_scores, train_channel_quality = unpack_quality_load_result(
        load_npz_data(
            MULTI_CHANNEL_PATH,
            file_subset=train_files,
            preprocess=PREPROCESS_SIGNALS,
            filter_profile=PREPROCESS_FILTER_PROFILE,
            normalization_mode=PREPROCESS_NORMALIZATION_MODE,
            clip_sigma=QUALITY_CLIP_SIGMA,
            enable_amplitude_clipping=ENABLE_AMPLITUDE_CLIPPING,
            return_quality=True,
            return_channel_quality=USE_CHANNEL_QUALITY_FEEDBACK,
        ),
        include_channel_quality=USE_CHANNEL_QUALITY_FEEDBACK,
    )
    X_val_raw, y_val_raw, val_quality_scores, val_channel_quality = unpack_quality_load_result(
        load_npz_data(
            MULTI_CHANNEL_PATH,
            file_subset=val_files,
            preprocess=PREPROCESS_SIGNALS,
            filter_profile=PREPROCESS_FILTER_PROFILE,
            normalization_mode=PREPROCESS_NORMALIZATION_MODE,
            clip_sigma=QUALITY_CLIP_SIGMA,
            enable_amplitude_clipping=ENABLE_AMPLITUDE_CLIPPING,
            return_quality=True,
            return_channel_quality=USE_CHANNEL_QUALITY_FEEDBACK,
        ),
        include_channel_quality=USE_CHANNEL_QUALITY_FEEDBACK,
    )
    X_test_raw, y_test_raw, test_quality_scores, _ = unpack_quality_load_result(
        load_npz_data(
            MULTI_CHANNEL_PATH,
            file_subset=test_files,
            preprocess=PREPROCESS_SIGNALS,
            filter_profile=PREPROCESS_FILTER_PROFILE,
            normalization_mode=PREPROCESS_NORMALIZATION_MODE,
            clip_sigma=QUALITY_CLIP_SIGMA,
            enable_amplitude_clipping=ENABLE_AMPLITUDE_CLIPPING,
            return_quality=True,
            return_channel_quality=False,
        ),
        include_channel_quality=False,
    )
    train_file_sample_counts = get_file_sample_counts(MULTI_CHANNEL_PATH, train_files)
    val_file_sample_counts = get_file_sample_counts(MULTI_CHANNEL_PATH, val_files)
    test_file_sample_counts = get_file_sample_counts(MULTI_CHANNEL_PATH, test_files)
    train_domain_targets = build_recording_domain_targets(train_file_sample_counts)
    val_domain_targets = build_recording_domain_targets(val_file_sample_counts)

    print_class_distribution("Train", y_train_raw, CLASS_LABELS)
    print_class_distribution("Val", y_val_raw, CLASS_LABELS)
    print_class_distribution("Test", y_test_raw, CLASS_LABELS)
    print_quality_summary("Train", train_quality_scores)
    print_quality_summary("Val", val_quality_scores)
    print_quality_summary("Test", test_quality_scores)
    print(
        "Clinical harmonization protocol: "
        f"filter_profile={PREPROCESS_FILTER_PROFILE}, "
        f"normalization={PREPROCESS_NORMALIZATION_MODE}; "
        "no train/val/test population scaler fit."
    )

    X_train_seg = build_segment_feature_matrix(X_train_raw, semantic_segmentation, extract_segment_features)
    X_val_seg = build_segment_feature_matrix(X_val_raw, semantic_segmentation, extract_segment_features)
    X_test_seg = build_segment_feature_matrix(X_test_raw, semantic_segmentation, extract_segment_features)

    reset_adaptive_priors()
    set_runtime_stage_channel_prior(None)
    if ADAPTIVE_CLINICAL_PRIORS:
        adaptive_prior_info = build_adaptive_clinical_priors(
            X_train_seg,
            y_train_raw,
            num_classes=NUM_CLASSES,
            alpha=ADAPTIVE_PRIOR_ALPHA,
            min_class_samples=ADAPTIVE_PRIOR_MIN_CLASS_SAMPLES,
            contrast_power=ADAPTIVE_PRIOR_CONTRAST_POWER,
            reliability_power=ADAPTIVE_PRIOR_RELIABILITY_POWER,
            concept_version=CLINICAL_CONCEPT_VERSION,
        )
        set_adaptive_priors(
            adaptive_prior_info["stage_concept_prior"],
            adaptive_prior_info["stage_channel_prior"],
        )
        set_runtime_stage_channel_prior(adaptive_prior_info["stage_channel_prior"])
        print(
            "Train-adaptive clinical priors fitted: "
            f"alpha={adaptive_prior_info['alpha']:.2f}, "
            f"contrast_power={adaptive_prior_info['contrast_power']:.2f}, "
            f"reliability_power={adaptive_prior_info['reliability_power']:.2f}, "
            f"concept_version={adaptive_prior_info['concept_version']}, "
            f"min_class_samples={adaptive_prior_info['min_class_samples']}, "
            f"class_counts={dict(zip(CLASS_LABELS, adaptive_prior_info['class_counts'].tolist()))}"
        )
        print("Adaptive prior calibration summary:")
        print(
            pd.DataFrame(
                {
                    "count": adaptive_prior_info["class_counts"],
                    "alpha": adaptive_prior_info["class_adaptive_alpha"],
                    "concept_shift_l1": adaptive_prior_info["concept_prior_shift_l1"],
                    "channel_shift_l1": adaptive_prior_info["channel_prior_shift_l1"],
                },
                index=CLASS_LABELS,
            ).round(3)
        )
        print("Adaptive stage-channel prior:")
        print(
            pd.DataFrame(
                adaptive_prior_info["stage_channel_prior"],
                index=CLASS_LABELS,
                columns=["EEG", "EOG", "EMG"],
            ).round(3)
        )
        print("Adaptive stage-concept prior:")
        print(
            pd.DataFrame(
                adaptive_prior_info["stage_concept_prior"],
                index=[
                    "EEG Delta",
                    "EEG Theta",
                    "EEG Alpha",
                    "EEG Sigma",
                    "EEG Beta",
                    "EOG Slow",
                    "EOG Rapid",
                    "EMG Low",
                    "EMG Active",
                ],
                columns=CLASS_LABELS,
            ).round(3)
        )
    else:
        print("Train-adaptive clinical priors disabled; using base clinical priors.")

    y_train = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val_raw, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

    lime_feedback_active = uses_lime_dccs_feedback(EXPLANATION_FEEDBACK_MODE)
    if lime_feedback_active:
        lime_feedback_data = load_lime_dccs_feedback_data(
            LIME_DCCS_TARGET_PATH,
            train_len=len(y_train_raw),
            val_len=len(y_val_raw),
        )
    else:
        lime_feedback_data = {"train_targets": None, "val_targets": None}
    train_lime_dccs_targets = lime_feedback_data["train_targets"]
    val_lime_dccs_targets = lime_feedback_data["val_targets"]
    if lime_feedback_active and train_lime_dccs_targets is None:
        raise ValueError(
            f"feedback mode {EXPLANATION_FEEDBACK_MODE!r} requires train LIME-DCCS targets. "
            "Set LFTS_LIME_DCCS_TARGET_PATH to a .npz/.npy file containing train targets."
        )
    if lime_feedback_active and val_lime_dccs_targets is None:
        val_lime_dccs_targets = np.ones((len(y_val_raw), 3), dtype=np.float32) / 3.0

    if train_lime_dccs_targets is not None:
        print(
            "Loaded LIME-DCCS feedback targets: "
            f"train={train_lime_dccs_targets.shape}, "
            f"val={None if lime_feedback_data['val_targets'] is None else lime_feedback_data['val_targets'].shape}"
        )

    train_attention_sample_weights = np.ones(len(y_train_raw), dtype=np.float32)
    val_attention_sample_weights = np.ones(len(y_val_raw), dtype=np.float32)
    if lime_feedback_active:
        train_attention_sample_weights = build_lime_feedback_sample_weights(
            y_train_raw,
            lime_feedback_data,
            split_name="Train",
            filter_mode=LIME_DCCS_FEEDBACK_FILTER,
            min_confidence=LIME_DCCS_MIN_CONFIDENCE,
            use_val_feedback=True,
            soft_confidence_weighting=LIME_DCCS_SOFT_SAMPLE_WEIGHTS,
        )
        val_attention_sample_weights = build_lime_feedback_sample_weights(
            y_val_raw,
            lime_feedback_data,
            split_name="Validation",
            filter_mode=LIME_DCCS_FEEDBACK_FILTER,
            min_confidence=LIME_DCCS_MIN_CONFIDENCE,
            use_val_feedback=LIME_DCCS_USE_VAL_FEEDBACK,
            soft_confidence_weighting=LIME_DCCS_SOFT_SAMPLE_WEIGHTS,
        )
        summarize_feedback_sample_weights("Train", train_attention_sample_weights)
        summarize_feedback_sample_weights("Validation", val_attention_sample_weights)
        if np.sum(train_attention_sample_weights > 0.0) == 0:
            raise ValueError(
                "No training samples passed the LIME-DCCS feedback filter. "
                "Lower LFTS_LIME_DCCS_MIN_CONFIDENCE or choose a less strict filter."
            )

    train_feedback_targets = build_explanation_feedback_targets(
        X_train_seg,
        y_train_raw,
        num_classes=NUM_CLASSES,
        concept_prior_weight=CONCEPT_TARGET_PRIOR_WEIGHT,
        channel_prior_weight=DCCS_CHANNEL_PRIOR_WEIGHT,
        feedback_mode=EXPLANATION_FEEDBACK_MODE,
        lime_dccs_targets=train_lime_dccs_targets,
        feedback_weights=(
            LIME_DCCS_FEEDBACK_ALPHA,
            LIME_DCCS_FEEDBACK_BETA,
            LIME_DCCS_FEEDBACK_GAMMA,
        ),
        channel_quality=train_channel_quality if USE_CHANNEL_QUALITY_FEEDBACK else None,
        concept_version=CLINICAL_CONCEPT_VERSION,
    )
    val_feedback_targets = build_explanation_feedback_targets(
        X_val_seg,
        y_val_raw,
        num_classes=NUM_CLASSES,
        concept_prior_weight=CONCEPT_TARGET_PRIOR_WEIGHT,
        channel_prior_weight=DCCS_CHANNEL_PRIOR_WEIGHT,
        feedback_mode=EXPLANATION_FEEDBACK_MODE,
        lime_dccs_targets=val_lime_dccs_targets,
        feedback_weights=(
            LIME_DCCS_FEEDBACK_ALPHA,
            LIME_DCCS_FEEDBACK_BETA,
            LIME_DCCS_FEEDBACK_GAMMA,
        ),
        channel_quality=val_channel_quality if USE_CHANNEL_QUALITY_FEEDBACK else None,
        concept_version=CLINICAL_CONCEPT_VERSION,
    )
    print(
        "Explanation feedback targets: "
        f"mode={train_feedback_targets['feedback_mode']}, "
        f"component_weights={train_feedback_targets['feedback_component_weights']}, "
        f"concept={train_feedback_targets['concept_targets'].shape}, "
        f"attention={train_feedback_targets['attention_feedback_targets'].shape}"
    )
    sequence_concept_loss_weight = (
        0.0
        if USE_SEQUENCE_MODEL and USE_MPDG
        else (
            OT_TACB_SINKHORN_LOSS_WEIGHT
            if USE_SEQUENCE_MODEL and USE_OT_TACB
            else CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT
        )
    )
    sequence_bottleneck_loss_weight = (
        CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT
        if USE_SEQUENCE_MODEL and USE_MPDG
        else (
            0.0
            if USE_SEQUENCE_MODEL and USE_OT_TACB
            else CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT
        )
    )
    include_concept_derivative_aux = bool(
        USE_SEQUENCE_MODEL
        and not USE_MPDG
        and USE_SEQUENCE_CLINICAL_CONCEPTS
        and USE_CONCEPT_LANGUAGE
        and USE_CONCEPT_DERIVATIVE_AUX
        and CONCEPT_DERIVATIVE_LOSS_WEIGHT > 0.0
    )
    concept_language_active = bool(
        USE_SEQUENCE_MODEL
        and USE_SEQUENCE_CLINICAL_CONCEPTS
        and USE_CONCEPT_LANGUAGE
        and (
            sequence_concept_loss_weight > 0.0
            or sequence_bottleneck_loss_weight > 0.0
            or include_concept_derivative_aux
        )
    )
    print(
        "Clinical concept language: "
        f"{'enabled' if concept_language_active else 'disabled'} "
        f"(stage_or_sinkhorn_loss={sequence_concept_loss_weight:.3f}, "
        f"bottleneck_loss={sequence_bottleneck_loss_weight:.3f}, "
        f"derivative_aux={include_concept_derivative_aux})"
    )
    train_concept_sample_weights = None
    val_concept_sample_weights = None
    if concept_language_active:
        train_concept_sample_weights = build_concept_sample_weights(
            len(y_train_raw),
            quality_scores=train_quality_scores,
            channel_quality=train_channel_quality if USE_CHANNEL_QUALITY_FEEDBACK else None,
            mode=CONCEPT_SAMPLE_WEIGHT_MODE,
            floor=CONCEPT_SAMPLE_WEIGHT_FLOOR,
        )
        val_concept_sample_weights = build_concept_sample_weights(
            len(y_val_raw),
            quality_scores=val_quality_scores,
            channel_quality=val_channel_quality if USE_CHANNEL_QUALITY_FEEDBACK else None,
            mode=CONCEPT_SAMPLE_WEIGHT_MODE,
            floor=CONCEPT_SAMPLE_WEIGHT_FLOOR,
        )
        summarize_concept_sample_weights("Train", train_concept_sample_weights, CONCEPT_SAMPLE_WEIGHT_MODE)
        summarize_concept_sample_weights("Validation", val_concept_sample_weights, CONCEPT_SAMPLE_WEIGHT_MODE)
    elif _concept_sample_weight_mode_key(CONCEPT_SAMPLE_WEIGHT_MODE) not in {"uniform", "none", "off", "disabled"}:
        print(
            "Concept sample weighting requested, but clinical concept language is disabled; "
            "ignoring concept sample weights."
        )

    train_sample_weights, class_weights = compute_main_sample_weights(
        y_train_raw,
        NUM_CLASSES,
        n1_boost=N1_WEIGHT_BOOST,
    )
    if QUALITY_WEIGHTING:
        print("QUALITY_WEIGHTING is ignored: quality scores are used only by adaptive Viterbi.")
    if USE_TRANSITION_BOUNDARY_WEIGHTS:
        boundary_weights = compute_transition_boundary_sample_weights(
            y_train_raw,
            file_sample_counts=train_file_sample_counts,
            radius=TRANSITION_BOUNDARY_RADIUS,
            base_boost=TRANSITION_BOUNDARY_BASE_BOOST,
            n1_boundary_boost=TRANSITION_BOUNDARY_N1_BOOST,
            target_transition_boost=TRANSITION_BOUNDARY_TARGET_BOOST,
            min_weight=1.0,
            max_weight=TRANSITION_BOUNDARY_MAX_WEIGHT,
        )
        train_sample_weights = train_sample_weights * boundary_weights
        train_sample_weights = train_sample_weights / (np.mean(train_sample_weights) + 1e-8)
        boundary_mask = boundary_weights > 1.0 + 1e-6
        boundary_count = int(np.sum(boundary_mask))
        print(
            "Transition boundary weighting: enabled, "
            f"radius={TRANSITION_BOUNDARY_RADIUS}, "
            f"boundary_epochs={boundary_count}/{len(boundary_weights)} "
            f"({boundary_count / max(1, len(boundary_weights)):.1%}), "
            f"mean_boundary_weight={np.mean(boundary_weights[boundary_mask]) if boundary_count else 1.0:.3f}, "
            f"max_boundary_weight={np.max(boundary_weights):.3f}, "
            f"combined_mean={np.mean(train_sample_weights):.3f}"
        )
    else:
        print("Transition boundary weighting: disabled")
    transition_boundary_loss_weight = (
        DPCM_BOUNDARY_LOSS_WEIGHT
        if USE_SEQUENCE_MODEL and USE_MPDG
        else OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT
        if USE_SEQUENCE_MODEL and USE_OT_TACB
        else TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT
    )
    transition_boundary_fine_grained = bool(
        USE_SEQUENCE_MODEL
        and (USE_OT_TACB or USE_FINE_TRANSITION_AUX)
    )
    transition_boundary_num_classes = (
        OT_TACB_FINE_TRANSITION_TYPES
        if transition_boundary_fine_grained
        else 1
    )
    include_transition_boundary_aux = (
        USE_SEQUENCE_MODEL
        and USE_TRANSITION_BOUNDARY_AUX
        and transition_boundary_loss_weight > 0.0
    )
    train_boundary_aux = None
    val_boundary_aux = None
    if include_transition_boundary_aux:
        train_boundary_aux = build_transition_boundary_auxiliary_targets(
            y_train_raw,
            file_sample_counts=train_file_sample_counts,
            radius=TRANSITION_BOUNDARY_AUX_RADIUS,
            positive_weight=TRANSITION_BOUNDARY_AUX_POS_WEIGHT,
            n1_transition_weight=TRANSITION_BOUNDARY_AUX_N1_WEIGHT,
            fine_grained=transition_boundary_fine_grained,
            num_transition_types=transition_boundary_num_classes,
        )
        val_boundary_aux = build_transition_boundary_auxiliary_targets(
            y_val_raw,
            file_sample_counts=val_file_sample_counts,
            radius=TRANSITION_BOUNDARY_AUX_RADIUS,
            positive_weight=TRANSITION_BOUNDARY_AUX_POS_WEIGHT,
            n1_transition_weight=TRANSITION_BOUNDARY_AUX_N1_WEIGHT,
            fine_grained=transition_boundary_fine_grained,
            num_transition_types=transition_boundary_num_classes,
        )
        transition_type_summary = ", ".join(
            f"{label}:{int(count)}"
            for label, count in zip(FINE_TRANSITION_LABELS, train_boundary_aux["transition_type_counts"])
        )
        print(
            "Transition boundary auxiliary: enabled, "
            f"radius={TRANSITION_BOUNDARY_AUX_RADIUS}, "
            f"loss_weight={transition_boundary_loss_weight:.3f}, "
            f"num_types={transition_boundary_num_classes}, "
            f"train_boundary_epochs={train_boundary_aux['boundary_count']}/{len(y_train_raw)} "
            f"({train_boundary_aux['coverage']:.1%}), "
            f"train_n1_boundary_epochs={train_boundary_aux['n1_boundary_count']}/{len(y_train_raw)} "
            f"({train_boundary_aux['n1_coverage']:.1%}), "
            f"mean_positive_weight={train_boundary_aux['mean_positive_weight']:.3f}, "
            f"max_weight={train_boundary_aux['max_weight']:.3f}, "
            f"transition_types={transition_type_summary}"
        )
    else:
        print("Transition boundary auxiliary: disabled")
    train_derivative_aux = None
    val_derivative_aux = None
    if include_concept_derivative_aux:
        train_derivative_aux = build_concept_derivative_auxiliary_targets(
            y_train_raw,
            file_sample_counts=train_file_sample_counts,
            positive_weight=TRANSITION_BOUNDARY_AUX_POS_WEIGHT,
            n1_transition_weight=TRANSITION_BOUNDARY_AUX_N1_WEIGHT,
        )
        val_derivative_aux = build_concept_derivative_auxiliary_targets(
            y_val_raw,
            file_sample_counts=val_file_sample_counts,
            positive_weight=TRANSITION_BOUNDARY_AUX_POS_WEIGHT,
            n1_transition_weight=TRANSITION_BOUNDARY_AUX_N1_WEIGHT,
        )
        derivative_type_summary = ", ".join(
            f"{label}:{int(count)}"
            for label, count in zip(FINE_TRANSITION_LABELS, train_derivative_aux["transition_type_counts"])
        )
        print(
            "Concept derivative energy auxiliary: enabled, "
            f"loss_weight={CONCEPT_DERIVATIVE_LOSS_WEIGHT:.3f}, "
            f"train_transition_epochs={train_derivative_aux['transition_count']}/{len(y_train_raw)} "
            f"({train_derivative_aux['coverage']:.1%}), "
            f"train_n1_transition_epochs={train_derivative_aux['n1_transition_count']}/{len(y_train_raw)} "
            f"({train_derivative_aux['n1_coverage']:.1%}), "
            f"mean_positive_weight={train_derivative_aux['mean_positive_weight']:.3f}, "
            f"max_weight={train_derivative_aux['max_weight']:.3f}, "
            f"transition_types={derivative_type_summary}"
        )
    else:
        print("Concept derivative energy auxiliary: disabled")
    train_indices = build_oversampled_indices(y_train_raw, n1_factor=N1_OVERSAMPLE_FACTOR)
    class_weight_summary = ", ".join(
        f"{label}:{weight:.3f}" for label, weight in zip(CLASS_LABELS, class_weights)
    )
    print(f"Moderate class weights: {class_weight_summary}")
    if USE_SEQUENCE_MODEL:
        print(
            "N1 balancing: class-balanced temporal sample weights inside sequence windows; "
            f"raw_epochs={len(y_train_raw)}"
        )
    else:
        print(
            f"N1 balancing: factor={N1_OVERSAMPLE_FACTOR}, "
            f"training samples per epoch={len(train_indices)}, raw={len(y_train_raw)}"
        )
    transition_log, initial_log, transition_matrix = build_transition_model(
        y_train_raw,
        train_file_sample_counts,
        NUM_CLASSES,
    )
    print("Clinical HMM transition matrix:")
    print(pd.DataFrame(transition_matrix, index=CLASS_LABELS, columns=CLASS_LABELS).round(3))

    with strategy.scope():
        if USE_SEQUENCE_MODEL:
            model = create_sequence_sleep_model(
                raw_shape=X_train_raw.shape[1:],
                segment_shape=X_train_seg.shape[1:],
                seq_len=SEQUENCE_LENGTH,
                num_classes=NUM_CLASSES,
                learning_rate=learning_rate,
                sequence_backbone=SEQUENCE_BACKBONE,
                epoch_encoder_variant=EPOCH_ENCODER_VARIANT,
                use_transformer=SEQUENCE_USE_TRANSFORMER,
                loss_name=CLASSIFICATION_LOSS_NAME,
                use_domain_mixstyle=USE_DOMAIN_MIXSTYLE,
                mixstyle_p=DOMAIN_MIXSTYLE_P,
                mixstyle_alpha=DOMAIN_MIXSTYLE_ALPHA,
                use_domain_alignment=USE_DOMAIN_ALIGNMENT,
                domain_alignment_loss_weight=DOMAIN_ALIGNMENT_LOSS_WEIGHT,
                domain_alignment_mode=DOMAIN_ALIGNMENT_MODE,
                use_clinical_concept_features=USE_SEQUENCE_CLINICAL_CONCEPTS,
                clinical_concept_version=CLINICAL_CONCEPT_VERSION,
                clinical_prior_trainable=CLINICAL_PRIOR_TRAINABLE,
                use_late_clinical_concept_residual=USE_LATE_CLINICAL_CONCEPT_RESIDUAL,
                late_concept_residual_scale=LATE_CONCEPT_RESIDUAL_SCALE,
                late_concept_residual_trainable=LATE_CONCEPT_RESIDUAL_TRAINABLE,
                late_concept_stop_gradient=LATE_CONCEPT_STOP_GRADIENT,
                use_late_concept_confidence_gate=USE_LATE_CONCEPT_CONFIDENCE_GATE,
                late_concept_gate_low_confidence=LATE_CONCEPT_GATE_LOW_CONFIDENCE,
                late_concept_gate_high_confidence=LATE_CONCEPT_GATE_HIGH_CONFIDENCE,
                late_concept_gate_mid_value=LATE_CONCEPT_GATE_MID_VALUE,
                use_concept_feature_modulation=USE_CONCEPT_FEATURE_MODULATION,
                concept_modulation_strength=CONCEPT_MODULATION_STRENGTH,
                concept_modulation_stop_gradient=CONCEPT_MODULATION_STOP_GRADIENT,
                concept_modulation_mode=CONCEPT_MODULATION_MODE,
                use_concept_language=concept_language_active,
                concept_loss_weight=sequence_concept_loss_weight,
                concept_bottleneck_loss_weight=sequence_bottleneck_loss_weight,
                use_ot_tacb=USE_OT_TACB,
                ot_sinkhorn_epsilon=OT_TACB_SINKHORN_EPSILON,
                ot_sinkhorn_iterations=OT_TACB_SINKHORN_ITERATIONS,
                ot_cosine_weight=OT_TACB_COSINE_WEIGHT,
                use_mpdg=USE_MPDG,
                mpdg_concept_dim=MPDG_CONCEPT_DIM,
                dpcm_mask_amplitude=DPCM_MASK_AMPLITUDE,
                dpcm_prior_init_scale=DPCM_PRIOR_INIT_SCALE,
                dpcm_boundary_contrast_loss_weight=DPCM_BOUNDARY_CONTRAST_LOSS_WEIGHT,
                dpcm_boundary_contrast_margin=DPCM_BOUNDARY_CONTRAST_MARGIN,
                use_transition_boundary_auxiliary=include_transition_boundary_aux,
                transition_boundary_loss_weight=transition_boundary_loss_weight,
                transition_boundary_num_classes=transition_boundary_num_classes,
                use_concept_derivative_auxiliary=include_concept_derivative_aux,
                concept_derivative_loss_weight=CONCEPT_DERIVATIVE_LOSS_WEIGHT,
                use_n1_supervised_contrastive=USE_N1_SUPCON,
                n1_contrastive_loss_weight=N1_SUPCON_LOSS_WEIGHT,
                n1_contrastive_projection_dim=N1_SUPCON_PROJECTION_DIM,
            )
        else:
            model = create_hybrid_model_v2(
                raw_shape=X_train_raw.shape[1:],
                segment_shape=X_train_seg.shape[1:],
                num_classes=NUM_CLASSES,
                learning_rate=learning_rate,
                attention_loss_weight=ATTENTION_FEEDBACK_LOSS_WEIGHT if EXPLANATION_FEEDBACK_TRAINING else 0.0,
                concept_loss_weight=CONCEPT_OUTPUT_LOSS_WEIGHT,
                concept_bottleneck_loss_weight=(
                    CONCEPT_BOTTLENECK_LOSS_WEIGHT
                    if EXPLANATION_FEEDBACK_TRAINING and uses_concept_bottleneck_feedback(EXPLANATION_FEEDBACK_MODE)
                    else 0.0
                ),
                use_channel_quality_routing=USE_CHANNEL_QUALITY_ROUTING,
                channel_quality_bias_strength=CHANNEL_QUALITY_BIAS_STRENGTH,
                loss_name=CLASSIFICATION_LOSS_NAME,
                use_domain_mixstyle=USE_DOMAIN_MIXSTYLE,
                mixstyle_p=DOMAIN_MIXSTYLE_P,
                mixstyle_alpha=DOMAIN_MIXSTYLE_ALPHA,
                use_domain_alignment=USE_DOMAIN_ALIGNMENT,
                domain_alignment_loss_weight=DOMAIN_ALIGNMENT_LOSS_WEIGHT,
                domain_alignment_mode=DOMAIN_ALIGNMENT_MODE,
                clinical_concept_version=CLINICAL_CONCEPT_VERSION,
                clinical_prior_trainable=CLINICAL_PRIOR_TRAINABLE,
            )

    monitor_metric = (
        "val_main_output_loss"
        if USE_SEQUENCE_MODEL and (
            concept_language_active
            or include_transition_boundary_aux
            or include_concept_derivative_aux
            or USE_N1_SUPCON
            or USE_DOMAIN_ALIGNMENT
        )
        else ("val_loss" if USE_SEQUENCE_MODEL else "val_main_output_loss")
    )
    callbacks = [
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    validation_selector = None
    if USE_METRIC_SELECTOR:
        validation_selector = BestValidationSelector(
            X_val_raw,
            X_val_seg,
            y_val_raw,
            val_file_sample_counts,
            val_quality_scores=val_quality_scores,
            batch_size=predict_batch_size,
            patience=8,
            selector_mode=SELECTOR_MODE,
            temperature_scaling=TEMPERATURE_SCALING,
            transition_log=transition_log,
            initial_log=initial_log,
            emission_weight=HMM_EMISSION_WEIGHT,
            emission_weights=VALIDATION_HMM_EMISSION_WEIGHTS,
            use_hmm=USE_HMM_VITERBI,
            sequence_mode=USE_SEQUENCE_MODEL,
            seq_len=SEQUENCE_LENGTH,
            sequence_stride=SEQUENCE_PREDICT_STRIDE,
        )
        callbacks.insert(0, validation_selector)
    else:
        callbacks.insert(
            0,
            EarlyStopping(
                monitor=monitor_metric,
                patience=8,
                restore_best_weights=True,
                mode="min",
            ),
        )

    if USE_SEQUENCE_MODEL:
        train_gen = SleepSequenceDataGenerator(
            X_train_raw,
            X_train_seg,
            y_train,
            file_sample_counts=train_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            stride=SEQUENCE_TRAIN_STRIDE,
            batch_size=train_batch_size,
            shuffle=True,
            sample_weights=train_sample_weights,
            domain_targets=train_domain_targets,
            include_domain_alignment=USE_DOMAIN_ALIGNMENT,
            domain_alignment_stage_conditioned=DOMAIN_ALIGNMENT_STAGE_CONDITIONED,
            concept_targets=train_feedback_targets["concept_targets"],
            concept_sample_weights=train_concept_sample_weights,
            include_concept_language=concept_language_active,
            transition_boundary_targets=(
                train_boundary_aux["targets"] if include_transition_boundary_aux else None
            ),
            transition_boundary_sample_weights=(
                train_boundary_aux["sample_weights"] if include_transition_boundary_aux else None
            ),
            include_transition_boundary_auxiliary=include_transition_boundary_aux,
            concept_derivative_targets=(
                train_derivative_aux["targets"] if include_concept_derivative_aux else None
            ),
            concept_derivative_sample_weights=(
                train_derivative_aux["sample_weights"] if include_concept_derivative_aux else None
            ),
            include_concept_derivative_auxiliary=include_concept_derivative_aux,
            include_n1_contrastive=USE_N1_SUPCON,
            balanced_sequence_sampling=USE_BALANCED_SEQUENCE_SAMPLING,
            balance_n1_window_boost=BALANCE_N1_WINDOW_BOOST,
            balance_n3_window_boost=BALANCE_N3_WINDOW_BOOST,
            balance_transition_window_boost=BALANCE_TRANSITION_WINDOW_BOOST,
        )
        val_gen = SleepSequenceDataGenerator(
            X_val_raw,
            X_val_seg,
            y_val,
            file_sample_counts=val_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            stride=SEQUENCE_PREDICT_STRIDE,
            batch_size=train_batch_size,
            shuffle=False,
            domain_targets=val_domain_targets,
            include_domain_alignment=USE_DOMAIN_ALIGNMENT,
            domain_alignment_stage_conditioned=DOMAIN_ALIGNMENT_STAGE_CONDITIONED,
            concept_targets=val_feedback_targets["concept_targets"],
            concept_sample_weights=val_concept_sample_weights,
            include_concept_language=concept_language_active,
            transition_boundary_targets=(
                val_boundary_aux["targets"] if include_transition_boundary_aux else None
            ),
            transition_boundary_sample_weights=(
                val_boundary_aux["sample_weights"] if include_transition_boundary_aux else None
            ),
            include_transition_boundary_auxiliary=include_transition_boundary_aux,
            concept_derivative_targets=(
                val_derivative_aux["targets"] if include_concept_derivative_aux else None
            ),
            concept_derivative_sample_weights=(
                val_derivative_aux["sample_weights"] if include_concept_derivative_aux else None
            ),
            include_concept_derivative_auxiliary=include_concept_derivative_aux,
            include_n1_contrastive=USE_N1_SUPCON,
            balanced_sequence_sampling=False,
        )
        print(
            "Sequence windows: "
            f"train={len(train_gen.windows)} "
            f"val={len(val_gen.windows)} "
            f"seq_len={SEQUENCE_LENGTH} "
            f"train_stride={SEQUENCE_TRAIN_STRIDE} "
            f"predict_stride={SEQUENCE_PREDICT_STRIDE} "
            f"balanced_sampling={USE_BALANCED_SEQUENCE_SAMPLING}"
        )
    else:
        train_gen = SleepDataGenerator(
            X_train_raw,
            X_train_seg,
            y_train,
            batch_size=train_batch_size,
            shuffle=True,
            sample_weights=train_sample_weights,
            indices=train_indices,
            attention_targets=train_feedback_targets["attention_feedback_targets"],
            concept_targets=train_feedback_targets["concept_targets"],
            attention_sample_weights=train_attention_sample_weights,
            domain_targets=train_domain_targets,
            include_domain_alignment=USE_DOMAIN_ALIGNMENT,
            include_auxiliary_sample_weights=(
                EXPLANATION_FEEDBACK_TRAINING
                and (
                    ATTENTION_FEEDBACK_LOSS_WEIGHT > 0.0
                    or CONCEPT_BOTTLENECK_LOSS_WEIGHT > 0.0
                )
            ),
        )
        val_gen = SleepDataGenerator(
            X_val_raw,
            X_val_seg,
            y_val,
            batch_size=train_batch_size,
            shuffle=False,
            attention_targets=val_feedback_targets["attention_feedback_targets"],
            concept_targets=val_feedback_targets["concept_targets"],
            attention_sample_weights=val_attention_sample_weights,
            domain_targets=val_domain_targets,
            include_domain_alignment=USE_DOMAIN_ALIGNMENT,
            include_auxiliary_sample_weights=(
                EXPLANATION_FEEDBACK_TRAINING
                and (
                    ATTENTION_FEEDBACK_LOSS_WEIGHT > 0.0
                    or CONCEPT_BOTTLENECK_LOSS_WEIGHT > 0.0
                )
            ),
        )

    if (
            not USE_SEQUENCE_MODEL
            and EXPLANATION_FEEDBACK_TRAINING
            and (ATTENTION_FEEDBACK_LOSS_WEIGHT > 0.0 or CONCEPT_BOTTLENECK_LOSS_WEIGHT > 0.0)
    ):
        callbacks.insert(
            0,
            AuxiliaryFeedbackScheduler(
                [train_gen, val_gen],
                warmup_epochs=AUX_FEEDBACK_WARMUP_EPOCHS,
                ramp_epochs=AUX_FEEDBACK_RAMP_EPOCHS,
                start_scale=AUX_FEEDBACK_START_SCALE,
            ),
        )
    if USE_SEQUENCE_MODEL and MICRO_MACRO_TRAINING:
        callbacks.insert(
            0,
            MacroAuxiliaryScheduler(
                [train_gen, val_gen],
                warmup_epochs=MICRO_PRETRAIN_EPOCHS,
                ramp_epochs=MACRO_AUX_RAMP_EPOCHS,
                start_scale=MACRO_AUX_START_SCALE,
                final_scale=MACRO_AUX_FINAL_SCALE,
            ),
        )

    print(
        "\nTraining with subject-level split, per-recording robust normalization, "
        "spectral segment features, moderate class-balanced weights, "
        f"{'sequence context' if USE_SEQUENCE_MODEL else 'single-epoch baseline'}, "
        f"{'micro-macro staged optimization, ' if USE_SEQUENCE_MODEL and MICRO_MACRO_TRAINING else ''}"
        f"and {num_replicas} synchronous replica(s)."
    )
    if USE_SEQUENCE_MODEL and MICRO_MACRO_TRAINING:
        fit_with_micro_macro_schedule(
            model,
            train_gen,
            val_gen,
            callbacks,
            EPOCHS,
            learning_rate,
            recompile_sleep_model,
        )
    else:
        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=2,
        )

    print("\n==========================================")
    print("4. Global test inference")
    print("==========================================")

    decoder_mode = "n1_protected_smoothed"
    if validation_selector is not None and validation_selector.best_metrics is not None:
        n1_threshold = validation_selector.best_n1_threshold
        temperature = validation_selector.best_temperature
        decoder_mode = validation_selector.best_decoder_mode
        print(
            "Selected validation model: "
            f"epoch={validation_selector.best_epoch} "
            f"score={validation_selector.best_score:.4f} "
            f"val_acc={validation_selector.best_metrics['selected_accuracy']:.4f} "
            f"val_macro_f1={validation_selector.best_metrics['selected_macro_f1']:.4f} "
            f"val_kappa={validation_selector.best_metrics['selected_kappa']:.4f} "
            f"val_N1_P={validation_selector.best_metrics['n1_precision']:.4f} "
            f"val_N1_R={validation_selector.best_metrics['n1_recall']:.4f} "
            f"val_N1_F1={validation_selector.best_metrics['n1_f1']:.4f} "
            f"val_N2_R={validation_selector.best_metrics['n2_recall']:.4f} "
            f"decoder={decoder_mode} "
            f"T={temperature:.2f} "
            f"thr={n1_threshold:.3f}"
        )
    else:
        val_outputs = predict_model_outputs(
            model,
            X_val_raw,
            X_val_seg,
            batch_size=predict_batch_size,
            verbose=0,
            sequence_mode=USE_SEQUENCE_MODEL,
            file_sample_counts=val_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            sequence_stride=SEQUENCE_PREDICT_STRIDE,
        )
        y_val_probs_all, _, _ = extract_prediction_heads(val_outputs)
        temperature = 1.0
        if TEMPERATURE_SCALING:
            temperature, y_val_probs_all, val_nll = tune_temperature(y_val_raw, y_val_probs_all)
            print(f"Validation temperature scaling: T={temperature:.2f} val_nll={val_nll:.4f}")
        n1_threshold, threshold_metrics = tune_n1_threshold(
            y_val_raw,
            y_val_probs_all,
            file_sample_counts=val_file_sample_counts,
            selector_mode=SELECTOR_MODE,
        )
        print(
            "Validation-calibrated N1 threshold: "
            f"{n1_threshold:.3f} | "
            f"val_acc={threshold_metrics['best_accuracy']:.4f} "
            f"val_macro_f1={threshold_metrics['best_macro_f1']:.4f} "
            f"val_N1_F1={threshold_metrics['n1_f1']:.4f} "
            f"val_n1_recall={threshold_metrics['n1_recall']:.4f}"
        )
        val_candidates = build_postprocessor_candidates(
            y_val_probs_all,
            val_file_sample_counts,
            n1_threshold,
            transition_log=transition_log,
            initial_log=initial_log,
            emission_weight=HMM_EMISSION_WEIGHT,
            emission_weights=VALIDATION_HMM_EMISSION_WEIGHTS,
            include_hmm=USE_HMM_VITERBI,
            quality_scores=val_quality_scores,
        )
        decoder_mode, decoder_metrics, _ = select_postprocessor_candidate(
            y_val_raw,
            val_candidates,
            SELECTOR_MODE,
        )
        print(
            "Validation-selected decoder: "
            f"{decoder_mode} | score={decoder_metrics['score']:.4f} "
            f"acc={decoder_metrics['accuracy']:.4f} "
            f"macro_f1={decoder_metrics['macro_f1']:.4f} "
            f"kappa={decoder_metrics['kappa']:.4f} "
            f"N1_P={decoder_metrics['n1_precision']:.4f} "
            f"N1_R={decoder_metrics['n1_recall']:.4f} "
            f"N1_F1={decoder_metrics['n1_f1']:.4f} "
            f"N2_R={decoder_metrics['n2_recall']:.4f}"
        )

    pred_outputs = predict_model_outputs(
        model,
        X_test_raw,
        X_test_seg,
        batch_size=predict_batch_size,
        verbose=1,
        sequence_mode=USE_SEQUENCE_MODEL,
        file_sample_counts=test_file_sample_counts,
        seq_len=SEQUENCE_LENGTH,
        sequence_stride=SEQUENCE_PREDICT_STRIDE,
    )
    y_pred_probs_all, test_channel_weights, test_concept_probs = extract_prediction_heads(pred_outputs)
    y_pred_probs_calibrated = temperature_scale_probabilities(y_pred_probs_all, temperature)
    y_true = np.argmax(y_test, axis=1)
    y_pred_argmax = np.argmax(y_pred_probs_all, axis=1)
    test_candidates = build_postprocessor_candidates(
        y_pred_probs_calibrated,
        test_file_sample_counts,
        n1_threshold,
        transition_log,
        initial_log,
        emission_weight=HMM_EMISSION_WEIGHT,
        emission_weights=VALIDATION_HMM_EMISSION_WEIGHTS,
        include_hmm=USE_HMM_VITERBI,
        quality_scores=test_quality_scores,
    )
    y_pred_temperature_argmax = test_candidates["temperature_argmax"]
    y_pred_smoothed = test_candidates["n1_protected_smoothed"]
    y_pred_viterbi = test_candidates.get("clinical_hmm", y_pred_smoothed)
    y_pred_quality_viterbi = test_candidates.get("quality_aware_hmm", y_pred_viterbi)
    if decoder_mode not in test_candidates:
        print(
            "Validation-selected decoder is unavailable on test candidates; "
            f"falling back to temperature_argmax. decoder={decoder_mode}"
        )
    y_pred_final = test_candidates.get(decoder_mode, y_pred_temperature_argmax)
    y_pred_report = y_pred_final
    best_test_name, best_test_metrics, test_candidate_metrics = select_postprocessor_candidate(
        y_true,
        test_candidates,
        SELECTOR_MODE,
    )
    y_pred_best_test_analysis = test_candidates[best_test_name]

    argmax_metrics = print_evaluation_report(
        "Evaluation: argmax raw",
        y_true,
        y_pred_argmax,
        CLASS_LABELS,
        file_sample_counts=test_file_sample_counts,
    )
    selected_metrics = print_evaluation_report(
        f"Evaluation: val-selected postprocess | decoder={decoder_mode} | T={temperature:.2f}",
        y_true,
        y_pred_report,
        CLASS_LABELS,
        file_sample_counts=test_file_sample_counts,
    )
    print_evaluation_report(
        (
            "Evaluation: best test-side analysis "
            f"(not final score) | decoder={best_test_name}"
        ),
        y_true,
        y_pred_best_test_analysis,
        CLASS_LABELS,
        file_sample_counts=test_file_sample_counts,
    )

    print("\nPostprocess candidate summary")
    for candidate_name, metrics in sorted(
            test_candidate_metrics.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
    ):
        print(f"{candidate_name}: {format_metric_summary(metrics)} score={metrics['score']:.4f}")
    print(
        "\nPrimary metric summary | "
        f"argmax_raw: {format_metric_summary(argmax_metrics)} | "
        f"val_selected: {format_metric_summary(selected_metrics)}"
    )

    if SAVE_PREDICTION_PROBS:
        save_prediction_probabilities(
            OUTPUT_DIR,
            timestamp,
            log_suffix,
            SEED,
            run_tag,
            CLASS_LABELS,
            y_true,
            y_pred_probs_all,
            y_pred_probs_calibrated,
            y_pred_argmax,
            y_pred_temperature_argmax,
            y_pred_report,
            test_candidates,
            test_file_sample_counts,
            decoder_mode,
            temperature,
            n1_threshold,
        )

    if SAVE_SPLIT_PREDICTION_PROBS:
        print("\nSaving train/val/test split predictions for C11 calibration...")
        split_predict_batch_size = (
            min(predict_batch_size, SPLIT_PREDICT_BATCH_SIZE)
            if SPLIT_PREDICT_BATCH_SIZE > 0
            else predict_batch_size
        )
        print(
            "Split prediction batch size: "
            f"{split_predict_batch_size} (base predict_batch={predict_batch_size})"
        )
        train_split_outputs = predict_model_outputs(
            model,
            X_train_raw,
            X_train_seg,
            batch_size=split_predict_batch_size,
            verbose=1,
            sequence_mode=USE_SEQUENCE_MODEL,
            file_sample_counts=train_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            sequence_stride=SEQUENCE_PREDICT_STRIDE,
        )
        train_split_probs, _, _ = extract_prediction_heads(train_split_outputs)
        val_split_outputs = predict_model_outputs(
            model,
            X_val_raw,
            X_val_seg,
            batch_size=split_predict_batch_size,
            verbose=1,
            sequence_mode=USE_SEQUENCE_MODEL,
            file_sample_counts=val_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            sequence_stride=SEQUENCE_PREDICT_STRIDE,
        )
        val_split_probs, _, _ = extract_prediction_heads(val_split_outputs)
        train_split_probs_calibrated = temperature_scale_probabilities(train_split_probs, temperature)
        val_split_probs_calibrated = temperature_scale_probabilities(val_split_probs, temperature)
        train_split_candidates = build_postprocessor_candidates(
            train_split_probs_calibrated,
            train_file_sample_counts,
            n1_threshold,
            transition_log,
            initial_log,
            emission_weight=HMM_EMISSION_WEIGHT,
            emission_weights=VALIDATION_HMM_EMISSION_WEIGHTS,
            include_hmm=USE_HMM_VITERBI,
            quality_scores=train_quality_scores,
        )
        val_split_candidates = build_postprocessor_candidates(
            val_split_probs_calibrated,
            val_file_sample_counts,
            n1_threshold,
            transition_log,
            initial_log,
            emission_weight=HMM_EMISSION_WEIGHT,
            emission_weights=VALIDATION_HMM_EMISSION_WEIGHTS,
            include_hmm=USE_HMM_VITERBI,
            quality_scores=val_quality_scores,
        )
        train_split_pred_final = train_split_candidates.get(
            decoder_mode,
            train_split_candidates["temperature_argmax"],
        )
        val_split_pred_final = val_split_candidates.get(
            decoder_mode,
            val_split_candidates["temperature_argmax"],
        )
        save_split_prediction_probabilities(
            OUTPUT_DIR,
            timestamp,
            log_suffix,
            SEED,
            run_tag,
            CLASS_LABELS,
            y_train_raw,
            train_split_probs,
            train_file_sample_counts,
            y_val_raw,
            val_split_probs,
            val_file_sample_counts,
            y_true,
            y_pred_probs_all,
            test_file_sample_counts,
            train_quality_scores=train_quality_scores,
            val_quality_scores=val_quality_scores,
            test_quality_scores=test_quality_scores,
            train_final_predictions=train_split_pred_final,
            val_final_predictions=val_split_pred_final,
            test_final_predictions=y_pred_report,
            train_probs_calibrated=train_split_probs_calibrated,
            val_probs_calibrated=val_split_probs_calibrated,
            test_probs_calibrated=y_pred_probs_calibrated,
            decoder_mode=decoder_mode,
            temperature=temperature,
            n1_threshold=n1_threshold,
        )

    if EXTENDED_ABLATION_STUDY:
        print("\nPreparing test-side extended ablation analysis with validation-tuned constraints...")
        print(
            "Test-side ablation grid | "
            f"concept_weights={EXTENDED_CONCEPT_WEIGHTS} "
            f"channel_strengths={EXTENDED_CHANNEL_STRENGTHS} "
            f"emission_weights={EXTENDED_HMM_EMISSION_WEIGHTS} "
            f"self_loop_boosts={EXTENDED_SELF_LOOP_BOOSTS} "
            f"accuracy_selector={EXTENDED_ACCURACY_SELECTOR_MODE}"
        )
        val_ablation_outputs = predict_model_outputs(
            model,
            X_val_raw,
            X_val_seg,
            batch_size=predict_batch_size,
            verbose=0,
            sequence_mode=USE_SEQUENCE_MODEL,
            file_sample_counts=val_file_sample_counts,
            seq_len=SEQUENCE_LENGTH,
            sequence_stride=SEQUENCE_PREDICT_STRIDE,
        )
        y_val_probs_for_ablation, val_channel_weights, val_concept_probs = extract_prediction_heads(
            val_ablation_outputs)
        y_val_probs_for_ablation = temperature_scale_probabilities(y_val_probs_for_ablation, temperature)
        extended_ablations = build_extended_ablation_predictions(
            y_val_raw=y_val_raw,
            val_main_probs=y_val_probs_for_ablation,
            val_concept_probs=val_concept_probs,
            val_channel_weights=val_channel_weights,
            val_file_sample_counts=val_file_sample_counts,
            val_quality_scores=val_quality_scores,
            y_test_main_probs=y_pred_probs_calibrated,
            test_concept_probs=test_concept_probs,
            test_channel_weights=test_channel_weights,
            test_file_sample_counts=test_file_sample_counts,
            test_quality_scores=test_quality_scores,
            transition_log=transition_log,
            initial_log=initial_log,
            selector_mode=SELECTOR_MODE,
        )
        print_extended_ablation_reports(
            y_true,
            CLASS_LABELS,
            extended_ablations,
            baseline_predictions=y_pred_viterbi,
        )
        accuracy_ablations = build_extended_ablation_predictions(
            y_val_raw=y_val_raw,
            val_main_probs=y_val_probs_for_ablation,
            val_concept_probs=val_concept_probs,
            val_channel_weights=val_channel_weights,
            val_file_sample_counts=val_file_sample_counts,
            val_quality_scores=val_quality_scores,
            y_test_main_probs=y_pred_probs_calibrated,
            test_concept_probs=test_concept_probs,
            test_channel_weights=test_channel_weights,
            test_file_sample_counts=test_file_sample_counts,
            test_quality_scores=test_quality_scores,
            transition_log=transition_log,
            initial_log=initial_log,
            selector_mode=EXTENDED_ACCURACY_SELECTOR_MODE,
        )
        print_extended_ablation_reports(
            y_true,
            CLASS_LABELS,
            accuracy_ablations,
            baseline_predictions=y_pred_viterbi,
            title=(
                "Accuracy-oriented test-side ablation analysis "
                f"(validation selector={EXTENDED_ACCURACY_SELECTOR_MODE})"
            ),
            summary_title="Accuracy-oriented test-side ablation analysis summary",
        )

    save_academic_results(y_true, y_pred_report, CLASS_LABELS, OUTPUT_DIR)
    save_confusion_matrix(y_true, y_pred_report, CLASS_LABELS, OUTPUT_DIR)

    if not RUN_XAI_DASHBOARDS:
        print("\n5. XAI dashboards skipped by default for the clean training/evaluation protocol.")
        print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")
        return
    if USE_SEQUENCE_MODEL:
        print("\n5. XAI dashboards skipped: the current LIME wrapper is for the single-epoch baseline.")
        print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")
        return

    print("\n5. Running representative LIME dashboards")
    lime_wrapper = ModelWrapperForLIME(model)

    for stage_idx, stage_name in enumerate(CLASS_LABELS):
        correct_indices = np.where((y_true == stage_idx) & (y_pred_report == stage_idx))[0]
        if len(correct_indices) == 0:
            print(f"Stage {stage_name}: no correctly predicted sample; skipping.")
            continue

        target_idx = correct_indices[0]
        instance_raw = X_test_raw[target_idx]
        print(f"Explaining stage {stage_name}, test index {target_idx}")

        segments = semantic_segmentation(instance_raw, target_segments=30)
        explanation = run_multimodal_lime(
            model=lime_wrapper,
            instance_multimodal=instance_raw,
            segments=segments,
            num_perturbations=300,
        )
        explanation = normalize_lime_explanation(
            explanation,
            num_channels=instance_raw.shape[1],
            num_segments=len(segments),
        )
        save_path = os.path.join(OUTPUT_DIR, f"XAI_Diagnosis_{stage_name}.png")
        call_with_supported_kwargs(
            plot_xai_multimodal_dashboard,
            signal=instance_raw,
            segments=segments,
            segment_importance=explanation["segment_importance"],
            raw_dccs=explanation["raw_dccs"],
            calibrated_dccs=explanation["calibrated_dccs"],
            segment_matrix=explanation["segment_matrix"],
            prior_weighted_matrix=explanation["prior_weighted_matrix"],
            temporal_net=explanation["temporal_net"],
            cooperation_score=explanation["cooperation_score"],
            conflict_score=explanation["conflict_score"],
            prior_alignment=explanation["prior_alignment"],
            channel_disagreement=explanation["channel_disagreement"],
            clinical_concept_names=explanation.get("clinical_concept_names"),
            clinical_concept_matrix=explanation.get("clinical_concept_matrix"),
            clinical_concept_evidence_matrix=explanation.get("clinical_concept_evidence_matrix"),
            raw_concept_dccs=explanation.get("raw_concept_dccs"),
            calibrated_concept_dccs=explanation.get("calibrated_concept_dccs"),
            concept_prior=explanation.get("concept_prior"),
            concept_alignment=explanation.get("concept_alignment"),
            perturbation_mode=explanation.get("perturbation_mode"),
            channel_names=["EEG", "EOG", "EMG"],
            target_class_name=CLASS_LABELS[explanation["target_class"]],
            save_path=save_path,
        )

    print("\n6. Running transition-aware XAI dashboards")
    if plot_transition_explanation_dashboard is None:
        print("Transition dashboard function is not available; skipping transition XAI figures.")
        print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")
        return

    transition_generated = 0
    offset = 0
    for file_idx, sample_count in enumerate(test_file_sample_counts):
        file_true = y_true[offset: offset + sample_count]
        file_pred = y_pred_report[offset: offset + sample_count]
        transitions = find_transition_windows(file_true, window_radius=2)
        for _, transition_idx, left, right in transitions:
            if file_pred[transition_idx - 1] != file_true[transition_idx - 1]:
                continue
            if file_pred[transition_idx] != file_true[transition_idx]:
                continue

            epoch_explanations = []
            for local_idx in range(left, right):
                global_idx = offset + local_idx
                instance_raw = X_test_raw[global_idx]
                segments = semantic_segmentation(instance_raw, target_segments=30)
                explanation = run_multimodal_lime(
                    model=lime_wrapper,
                    instance_multimodal=instance_raw,
                    segments=segments,
                    num_perturbations=200,
                )
                explanation = normalize_lime_explanation(
                    explanation,
                    num_channels=instance_raw.shape[1],
                    num_segments=len(segments),
                )
                epoch_explanations.append(explanation)

            from_stage = CLASS_LABELS[file_true[transition_idx - 1]]
            to_stage = CLASS_LABELS[file_true[transition_idx]]
            transition_summary = summarize_transition_explanation(
                epoch_explanations,
                ["EEG", "EOG", "EMG"],
                from_stage=from_stage,
                to_stage=to_stage,
            )
            if transition_summary is None:
                continue

            transition_path = os.path.join(
                OUTPUT_DIR,
                f"XAI_Transition_{test_files[file_idx].replace('.npz', '')}_{from_stage}_to_{to_stage}.png",
            )
            call_with_supported_kwargs(
                plot_transition_explanation_dashboard,
                transition_summary,
                from_stage=from_stage,
                to_stage=to_stage,
                save_path=transition_path,
            )
            transition_generated += 1
            break
        offset += sample_count

    print(f"Transition dashboards generated: {transition_generated}")
    print(f"\nDone. Logs and figures saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
