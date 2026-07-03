#!/usr/bin/env python
"""Run run_main.py with a fixed split produced by build_sleep_dataset_protocol.py.

This wrapper keeps dataset-specific protocol handling outside the shared
training files. It sets LFTS_MULTI_CHANNEL_PATH, monkey-patches the split
function before run_main.main() imports it, and then delegates to run_main.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_CLEAN_SEQUENCE_BASE: Dict[str, str] = {
    "LFTS_ALLOW_PROTOCOL_OVERRIDES": "1",
    "LFTS_SELECTOR_MODE": "accuracy_macro",
    "LFTS_LOSS_NAME": "categorical_crossentropy",
    "LFTS_GENERALIZATION_PRESET": "mixstyle_stage_coral",
    "LFTS_N1_WEIGHT_BOOST": "1.10",
    "LFTS_USE_N1_SUPCON": "0",
    "LFTS_BALANCE_N1_WINDOW_BOOST": "1.25",
    "LFTS_BALANCE_N3_WINDOW_BOOST": "1.30",
    "LFTS_BALANCE_TRANSITION_WINDOW_BOOST": "1.05",
    "LFTS_TRANSITION_BOUNDARY_BASE_BOOST": "1.05",
    "LFTS_TRANSITION_BOUNDARY_N1_BOOST": "1.05",
    "LFTS_TRANSITION_BOUNDARY_TARGET_BOOST": "1.10",
    "LFTS_TRANSITION_BOUNDARY_MAX_WEIGHT": "1.60",
    "LFTS_TRANSITION_BOUNDARY_AUX_LOSS_WEIGHT": "0.020",
    "LFTS_TRANSITION_BOUNDARY_AUX_POS_WEIGHT": "1.30",
    "LFTS_TRANSITION_BOUNDARY_AUX_N1_WEIGHT": "1.15",
    "LFTS_USE_OT_TACB": "0",
    "LFTS_OT_TACB_SINKHORN_LOSS_WEIGHT": "0.0",
    "LFTS_OT_TACB_FINE_BOUNDARY_LOSS_WEIGHT": "0.0",
    "LFTS_OT_TACB_CONCEPT_DERIVATIVE_LOSS_WEIGHT": "0.0",
    "LFTS_USE_CONCEPT_DERIVATIVE_AUX": "0",
    "LFTS_CONCEPT_DERIVATIVE_LOSS_WEIGHT": "0.0",
    "LFTS_MACRO_AUX_FINAL_SCALE": "0.40",
    "LFTS_MICRO_FINETUNE_LR_SCALE": "0.10",
    "LFTS_EXPLANATION_FEEDBACK_MODE": "clinical",
    "LFTS_EXPLANATION_FEEDBACK_TRAINING": "0",
    "LFTS_ATTENTION_FEEDBACK_LOSS_WEIGHT": "0.0",
    "LFTS_CONCEPT_OUTPUT_LOSS_WEIGHT": "0.0",
    "LFTS_CONCEPT_BOTTLENECK_LOSS_WEIGHT": "0.0",
    "LFTS_CLINICAL_PRIOR_TRAINABLE": "0",
    "LFTS_EXTENDED_ABLATION_STUDY": "1",
}

_NO_CONCEPT_BRANCH: Dict[str, str] = {
    "LFTS_USE_SEQUENCE_CLINICAL_CONCEPTS": "0",
    "LFTS_USE_CONCEPT_FEATURE_MODULATION": "0",
    "LFTS_USE_CONCEPT_LANGUAGE": "0",
    "LFTS_CONCEPT_LANGUAGE_STAGE_LOSS_WEIGHT": "0.0",
    "LFTS_CONCEPT_LANGUAGE_BOTTLENECK_LOSS_WEIGHT": "0.0",
    "LFTS_CONCEPT_TARGET_PRIOR_WEIGHT": "0.0",
    "LFTS_DCCS_CHANNEL_PRIOR_WEIGHT": "0.0",
    "LFTS_ADAPTIVE_CLINICAL_PRIORS": "0",
}

_LATE_CONCEPT_RESIDUAL_BRANCH: Dict[str, str] = {
    **_NO_CONCEPT_BRANCH,
    "LFTS_USE_LATE_CLINICAL_CONCEPT_RESIDUAL": "1",
    "LFTS_LATE_CONCEPT_RESIDUAL_SCALE": "0.05",
    "LFTS_LATE_CONCEPT_RESIDUAL_TRAINABLE": "1",
    "LFTS_LATE_CONCEPT_STOP_GRADIENT": "1",
    "LFTS_CLINICAL_PRIOR_TRAINABLE": "0",
}

_CONCEPT_FILM_MODULATION_BRANCH: Dict[str, str] = {
    **_NO_CONCEPT_BRANCH,
    "LFTS_USE_CONCEPT_FEATURE_MODULATION": "1",
    "LFTS_CONCEPT_MODULATION_STRENGTH": "0.10",
    "LFTS_CONCEPT_MODULATION_STOP_GRADIENT": "1",
    "LFTS_CONCEPT_MODULATION_MODE": "real",
    "LFTS_USE_LATE_CLINICAL_CONCEPT_RESIDUAL": "0",
    "LFTS_CLINICAL_PRIOR_TRAINABLE": "0",
}


_ACCURACY_SELECTOR_ONLY: Dict[str, str] = {
    "LFTS_SELECTOR_MODE": "accuracy",
    "LFTS_N1_THRESHOLD_MAX_ACC_DROP": "0.000",
    "LFTS_POSTPROCESS_ACCURACY_GUARD_DROP": "0.0000",
}


DATASET_PRESETS: Dict[str, Dict[str, str]] = {
    "none": {},
    "A1_concept_film": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_RUN_TAG": "A1_concept_film",
    },
    "A1_concept_film_s005": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.05",
        "LFTS_RUN_TAG": "A1_concept_film_s005",
    },
    "A1_concept_film_s015": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.15",
        "LFTS_RUN_TAG": "A1_concept_film_s015",
    },
    "A1_concept_film_zero": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "zero",
        "LFTS_RUN_TAG": "A1_concept_film_zero",
    },
    "A1_concept_film_shuffled": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "shuffled",
        "LFTS_RUN_TAG": "A1_concept_film_shuffled",
    },
    "edf_c0_no_concept": {
        **_CLEAN_SEQUENCE_BASE,
        **_NO_CONCEPT_BRANCH,
        "LFTS_RUN_TAG": "edf_c0_no_concept",
    },
    "edf_c0_accuracy_selector_only": {
        **_CLEAN_SEQUENCE_BASE,
        **_NO_CONCEPT_BRANCH,
        **_ACCURACY_SELECTOR_ONLY,
        "LFTS_RUN_TAG": "edf_c0_accuracy_selector_only",
    },
    "edf_c10_late_concept_residual": {
        **_CLEAN_SEQUENCE_BASE,
        **_LATE_CONCEPT_RESIDUAL_BRANCH,
        "LFTS_RUN_TAG": "edf_c10_late_concept_residual",
    },
    "edf_A1_concept_film": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_RUN_TAG": "edf_A1_concept_film",
    },
    "edf_A1_concept_film_s005": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.05",
        "LFTS_RUN_TAG": "edf_A1_concept_film_s005",
    },
    "edf_A1_concept_film_s015": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.15",
        "LFTS_RUN_TAG": "edf_A1_concept_film_s015",
    },
    "edf_A1_concept_film_zero": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "zero",
        "LFTS_RUN_TAG": "edf_A1_concept_film_zero",
    },
    "edf_A1_concept_film_shuffled": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "shuffled",
        "LFTS_RUN_TAG": "edf_A1_concept_film_shuffled",
    },
    "isruc_c0_no_concept": {
        **_CLEAN_SEQUENCE_BASE,
        **_NO_CONCEPT_BRANCH,
        "LFTS_RUN_TAG": "isruc_c0_no_concept",
    },
    "isruc_c0_accuracy_selector_only": {
        **_CLEAN_SEQUENCE_BASE,
        **_NO_CONCEPT_BRANCH,
        **_ACCURACY_SELECTOR_ONLY,
        "LFTS_RUN_TAG": "isruc_c0_accuracy_selector_only",
    },
    "isruc_c10_late_concept_residual": {
        **_CLEAN_SEQUENCE_BASE,
        **_LATE_CONCEPT_RESIDUAL_BRANCH,
        "LFTS_RUN_TAG": "isruc_c10_late_concept_residual",
    },
    "isruc_A1_concept_film": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_RUN_TAG": "isruc_A1_concept_film",
    },
    "isruc_A1_concept_film_s005": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.05",
        "LFTS_RUN_TAG": "isruc_A1_concept_film_s005",
    },
    "isruc_A1_concept_film_s015": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_STRENGTH": "0.15",
        "LFTS_RUN_TAG": "isruc_A1_concept_film_s015",
    },
    "isruc_A1_concept_film_zero": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "zero",
        "LFTS_RUN_TAG": "isruc_A1_concept_film_zero",
    },
    "isruc_A1_concept_film_shuffled": {
        **_CLEAN_SEQUENCE_BASE,
        **_CONCEPT_FILM_MODULATION_BRANCH,
        "LFTS_CONCEPT_MODULATION_MODE": "shuffled",
        "LFTS_RUN_TAG": "isruc_A1_concept_film_shuffled",
    },
}

DATASET_PRESET_BATCH_LIMITS: Dict[str, Dict[str, int]] = {
    # SleepEDF has long recordings and the sequence frontend flattens
    # per_replica_batch * seq_len epochs through the raw encoder. Keep this
    # conservative for 10-12GB cards such as RTX 3080.
    "A1_concept_film": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "A1_concept_film_s005": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "A1_concept_film_s015": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "A1_concept_film_zero": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "A1_concept_film_shuffled": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_c0_no_concept": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_c0_accuracy_selector_only": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_c10_late_concept_residual": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_A1_concept_film": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_A1_concept_film_s005": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_A1_concept_film_s015": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_A1_concept_film_zero": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "edf_A1_concept_film_shuffled": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_c0_no_concept": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_c0_accuracy_selector_only": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_c10_late_concept_residual": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_A1_concept_film": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_A1_concept_film_s005": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_A1_concept_film_s015": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_A1_concept_film_zero": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
    "isruc_A1_concept_film_shuffled": {
        "per_replica_batch": 8,
        "predict_per_replica_batch": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch run_main.py with fixed train/val/test files from a protocol split.json."
    )
    parser.add_argument("--split-json", required=True, help="Path to dataset_protocols/<dataset>/split.json.")
    parser.add_argument(
        "--npz-dir",
        default=None,
        help="Override NPZ directory. Defaults to source_npz_dir/npz_dir stored in split.json.",
    )
    parser.add_argument("--gpus", default=None, help="Optional CUDA device list, e.g. 4,5,6,7.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training RNG seed. This does not change the fixed train/val/test split.",
    )
    parser.add_argument("--per-replica-batch", type=int, default=None)
    parser.add_argument("--predict-per-replica-batch", type=int, default=None)
    parser.add_argument("--memory-profile", choices=("safe", "balanced", "max"), default=None)
    parser.add_argument(
        "--dataset-preset",
        choices=tuple(DATASET_PRESETS.keys()),
        default="none",
        help="Optional dataset-specific launch preset kept outside run_main.py.",
    )
    parser.add_argument(
        "--allow-protocol-overrides",
        action="store_true",
        help="Set LFTS_ALLOW_PROTOCOL_OVERRIDES=1 before importing run_main.py.",
    )
    parser.add_argument(
        "--no-preset-batch-clamp",
        action="store_true",
        help="Do not clamp batch sizes for dataset presets.",
    )
    return parser.parse_args()


def load_split_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def normalize_split_files(raw_split_files: object) -> Dict[str, List[str]]:
    if not isinstance(raw_split_files, dict):
        raise ValueError("split.json must contain a split_files object.")
    normalized: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        values = raw_split_files.get(split)
        if values is None:
            values = raw_split_files.get(split.capitalize())
        if not isinstance(values, list):
            raise ValueError(f"split_files.{split} must be a list.")
        normalized[split] = [str(item) for item in values]
    return normalized


def source_npz_dir(split_data: Mapping[str, object], override: Optional[str]) -> str:
    if override:
        return str(Path(override).expanduser().resolve())
    raw_value = split_data.get("source_npz_dir") or split_data.get("npz_dir")
    if not raw_value:
        raise ValueError("NPZ directory not provided. Pass --npz-dir or include source_npz_dir in split.json.")
    return str(Path(str(raw_value)).expanduser().resolve())


def apply_launch_environment(args: argparse.Namespace, npz_dir: str) -> None:
    os.environ["LFTS_MULTI_CHANNEL_PATH"] = npz_dir
    preset_env = DATASET_PRESETS.get(str(args.dataset_preset), {})
    for name, value in preset_env.items():
        os.environ[name] = value
    batch_limits = (
        {}
        if bool(args.no_preset_batch_clamp)
        else DATASET_PRESET_BATCH_LIMITS.get(str(args.dataset_preset), {})
    )
    if args.gpus is not None:
        os.environ["LFTS_CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.seed is not None:
        os.environ["LFTS_SEED"] = str(int(args.seed))

    per_replica_batch = args.per_replica_batch
    per_replica_limit = batch_limits.get("per_replica_batch")
    if per_replica_batch is not None and per_replica_limit is not None:
        per_replica_batch = min(int(per_replica_batch), int(per_replica_limit))
    elif per_replica_batch is None and per_replica_limit is not None:
        per_replica_batch = int(per_replica_limit)
    if per_replica_batch is not None:
        os.environ["LFTS_PER_REPLICA_BATCH_SIZE"] = str(per_replica_batch)

    predict_batch = args.predict_per_replica_batch
    predict_limit = batch_limits.get("predict_per_replica_batch")
    if predict_batch is not None and predict_limit is not None:
        predict_batch = min(int(predict_batch), int(predict_limit))
    elif predict_batch is None and predict_limit is not None:
        predict_batch = int(predict_limit)
    if predict_batch is not None:
        os.environ["LFTS_PREDICT_PER_REPLICA_BATCH_SIZE"] = str(predict_batch)

    if args.memory_profile is not None:
        os.environ["LFTS_MEMORY_PROFILE"] = str(args.memory_profile)
    if args.allow_protocol_overrides:
        os.environ["LFTS_ALLOW_PROTOCOL_OVERRIDES"] = "1"


def build_fixed_split_function(split_files: Mapping[str, List[str]]):
    train_files = list(split_files["train"])
    val_files = list(split_files["val"])
    test_files = list(split_files["test"])

    def fixed_split_npz_files(
        files,
        test_size=0.15,
        val_size=0.15,
        seed=42,
        folder_path=None,
        class_names=None,
        logger=None,
        log_summary=False,
        stratify_by_labels=False,
    ):
        del test_size, val_size, seed, class_names, stratify_by_labels
        available = set(str(item) for item in files)
        requested = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }
        missing = {
            split: [filename for filename in split_items if filename not in available]
            for split, split_items in requested.items()
        }
        missing = {split: split_items for split, split_items in missing.items() if split_items}
        if missing:
            raise FileNotFoundError(
                "Protocol split references files that are not present in the active NPZ folder: "
                f"{missing}"
            )
        if log_summary:
            print("Split mode: fixed protocol split from split.json")
            print(f"Protocol train files ({len(train_files)}): {train_files}")
            print(f"Protocol val files ({len(val_files)}): {val_files}")
            print(f"Protocol test files ({len(test_files)}): {test_files}")
            if folder_path is not None:
                print(f"Protocol NPZ folder: {folder_path}")
        return train_files, val_files, test_files

    return fixed_split_npz_files


def main() -> None:
    args = parse_args()
    split_path = Path(args.split_json).expanduser().resolve()
    split_data = load_split_json(split_path)
    split_files = normalize_split_files(split_data.get("split_files"))
    npz_dir = source_npz_dir(split_data, args.npz_dir)
    apply_launch_environment(args, npz_dir)

    print("Fixed dataset protocol launcher")
    print(f"split_json={split_path}")
    print(f"npz_dir={npz_dir}")
    print(
        "split_sizes="
        f"train:{len(split_files['train'])}, "
        f"val:{len(split_files['val'])}, "
        f"test:{len(split_files['test'])}"
    )
    if args.seed is not None:
        print(f"seed={int(args.seed)} (fixed split files unchanged)")
    if args.dataset_preset != "none":
        preset_env = DATASET_PRESETS[str(args.dataset_preset)]
        print(f"dataset_preset={args.dataset_preset}")
        print(
            "preset_env="
            + ", ".join(f"{name}={value}" for name, value in sorted(preset_env.items()))
        )
        if not args.no_preset_batch_clamp and str(args.dataset_preset) in DATASET_PRESET_BATCH_LIMITS:
            print(
                "preset_batch_limits="
                + ", ".join(
                    f"{name}={value}"
                    for name, value in sorted(DATASET_PRESET_BATCH_LIMITS[str(args.dataset_preset)].items())
                )
            )
            print(
                "effective_batch_env="
                f"LFTS_PER_REPLICA_BATCH_SIZE={os.environ.get('LFTS_PER_REPLICA_BATCH_SIZE')}, "
                f"LFTS_PREDICT_PER_REPLICA_BATCH_SIZE={os.environ.get('LFTS_PREDICT_PER_REPLICA_BATCH_SIZE')}"
            )

    import run_main  # noqa: WPS433 - import after launch env is intentional.
    import src.data_processing as data_processing  # noqa: WPS433

    data_processing.split_npz_files = build_fixed_split_function(split_files)
    run_main.main()


if __name__ == "__main__":
    main()
