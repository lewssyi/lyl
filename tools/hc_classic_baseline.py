#!/usr/bin/env python
"""Equal-budget Classic Sequence-LIME validation baseline for HC feasibility."""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.xai.hierarchical_context_attribution import (  # noqa: E402
    FAITHFULNESS_QUERIES,
    TOTAL_QUERY_BUDGET,
    cache_identity,
    deterministic_masks,
)
from src.xai.model_faithful_sequence_lime import (  # noqa: E402
    resolve_target_protocols,
    transform_surrogate_targets,
)
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402
from tools.hierarchical_context_xai_feasibility import (  # noqa: E402
    ALPHAS,
    KERNELS,
    LEVELS,
    PROTOCOLS,
    ROOT,
    SEEDS,
    authority_guard,
    epoch_mask_modifiers,
    fit_y2,
    load_sample,
    predict_specs,
    write_json,
)


BASELINE_FEATURES = 90
SURROGATE_TRAIN_QUERIES = 1200
SURROGATE_VALIDATION_QUERIES = 306
SURROGATE_QUERIES = SURROGATE_TRAIN_QUERIES + SURROGATE_VALIDATION_QUERIES
PROTOCOL_VERSION = "HC_FEASIBILITY_CLASSIC_B0_V1"


def run_one(args: argparse.Namespace) -> None:
    authority_guard()
    row, artifact, x_raw, x_seg, target_id, _artifact_start = load_sample(
        args.data_split, args.sample_id
    )
    model = evaluator.randomized_model(
        x_raw.shape[1:], x_seg.shape[1:], args.level, seed=args.seed
    )
    current_vector = evaluator.sequence_target_vector(
        model,
        x_raw,
        x_seg,
        np.asarray([len(x_raw)]),
        target_id,
        x_raw[target_id],
        x_seg[target_id],
        batch_windows=16,
    )
    original_vector = np.asarray(
        artifact["probs"][int(args.sample_id)], dtype=np.float32
    )
    protocols = resolve_target_protocols(original_vector, current_vector)
    model_identity = evaluator.evaluated_model_identity(
        model,
        f"{PROTOCOL_VERSION}_{args.level}_{args.seed}",
        randomization_level=args.level,
        randomization_seed=args.seed,
        method_id="CLASSIC_SEQUENCE_LIME_B0",
        protocol_id="T0_T1_Y2",
    )
    mask_seed = args.seed * 1000003 + args.sample_id * 101 + BASELINE_FEATURES
    masks = deterministic_masks(BASELINE_FEATURES, SURROGATE_QUERIES, mask_seed)
    modifiers = epoch_mask_modifiers(
        x_raw,
        x_seg,
        target_id,
        masks,
        int(protocols["T0"]),
        BASELINE_FEATURES,
    )
    vectors, surrogate_accounting = predict_specs(
        model, x_raw, x_seg, target_id, modifiers
    )

    sample_rows, attribution_rows, runtime_rows = [], [], []
    fractions = np.linspace(0.0, 1.0, FAITHFULNESS_QUERIES // 2)
    for protocol in PROTOCOLS:
        target_class = int(protocols[protocol])
        score, fidelity, response = fit_y2(
            masks, vectors, target_class, SURROGATE_TRAIN_QUERIES
        )
        order = np.argsort(-np.abs(score))
        faith_modifiers = []
        for insertion in (False, True):
            for fraction in fractions:
                count = int(round(float(fraction) * len(score)))
                mask = (
                    np.zeros(len(score), dtype=np.int32)
                    if insertion
                    else np.ones(len(score), dtype=np.int32)
                )
                mask[order[:count]] = 1 if insertion else 0
                faith_modifiers.append(
                    epoch_mask_modifiers(
                        x_raw,
                        x_seg,
                        target_id,
                        mask[None, :],
                        target_class,
                        BASELINE_FEATURES,
                    )[0]
                )
        faith_vectors, faith_accounting = predict_specs(
            model, x_raw, x_seg, target_id, faith_modifiers
        )
        faith_response = transform_surrogate_targets(
            faith_vectors, target_class
        )["Y2"]
        deletion_auc = float(
            np.trapz(faith_response[: len(fractions)], fractions)
        )
        insertion_auc = float(
            np.trapz(faith_response[len(fractions) :], fractions)
        )
        sample_rows.append(
            {
                "data_split": args.data_split,
                "source_recording": row["source_recording"],
                "sample_id": args.sample_id,
                "true_stage": row["true_stage"],
                "predicted_stage": row["predicted_stage"],
                "correct": row["correct"],
                "transition_boundary": row["transition_boundary"],
                "confidence": row["confidence"],
                "candidate": "B0_CLASSIC_90",
                "group_scheme": "NONE",
                "feature_count": BASELINE_FEATURES,
                "replacement": "zero",
                "randomization_level": args.level,
                "randomization_seed": args.seed,
                "target_protocol": protocol,
                "target_class": target_class,
                "context_surrogate_R2": np.nan,
                "local_surrogate_R2": float(
                    fidelity["validation_weighted_r2"]
                ),
                "hierarchical_surrogate_R2": float(
                    fidelity["validation_weighted_r2"]
                ),
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "faithfulness": insertion_auc - deletion_auc,
                "response_variance": float(np.var(response)),
                "effective_sample_size": float(
                    fidelity["effective_sample_size"]
                ),
                "within_epoch_importance": json.dumps(
                    score.astype(float).tolist()
                ),
                "query_budget": TOTAL_QUERY_BUDGET,
                "surrogate_train_queries": SURROGATE_TRAIN_QUERIES,
                "surrogate_validation_queries": SURROGATE_VALIDATION_QUERIES,
                "faithfulness_queries": FAITHFULNESS_QUERIES,
                "protocol_version": PROTOCOL_VERSION,
                "model_weight_fingerprint": model_identity[
                    "model_weight_fingerprint"
                ],
                "cache_key": cache_identity(
                    model_identity["model_weight_fingerprint"],
                    args.sample_id,
                    "B0_CLASSIC_90",
                    args.level,
                    args.seed,
                    protocol,
                ),
                "true_label_used_for_attribution": False,
                "stage_prior_used": False,
            }
        )
        attribution_rows.append(
            {
                "data_split": args.data_split,
                "sample_id": args.sample_id,
                "candidate": "B0_CLASSIC_90",
                "feature_count": BASELINE_FEATURES,
                "randomization_level": args.level,
                "randomization_seed": args.seed,
                "target_protocol": protocol,
                "target_class": target_class,
                "importance": json.dumps(score.astype(float).tolist()),
                "local_surrogate_R2": float(
                    fidelity["validation_weighted_r2"]
                ),
                "response_variance": float(np.var(response)),
                "effective_sample_size": float(
                    fidelity["effective_sample_size"]
                ),
            }
        )
        runtime_rows.append(
            {
                "sample_id": args.sample_id,
                "candidate": "B0_CLASSIC_90",
                "target_protocol": protocol,
                "logical_queries": TOTAL_QUERY_BUDGET,
                "surrogate_window_instances": surrogate_accounting[
                    "window_instances"
                ],
                "faithfulness_window_instances": faith_accounting[
                    "window_instances"
                ],
            }
        )

    run_id = (
        f"{args.stage}_classic_sample{args.sample_id}_{args.level}_"
        f"seed{args.seed}_zero"
    )
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_csv(
        out_dir / "classic_sample_level.csv", index=False
    )
    pd.DataFrame(attribution_rows).to_csv(
        out_dir / "classic_attribution.csv", index=False
    )
    pd.DataFrame(runtime_rows).to_csv(
        out_dir / "runtime_query_summary.csv", index=False
    )
    required = (
        out_dir / "classic_sample_level.csv",
        out_dir / "classic_attribution.csv",
        out_dir / "runtime_query_summary.csv",
    )
    write_json(
        out_dir / "run_manifest.json",
        {
            "status": "COMPLETE",
            "run_id": run_id,
            "stage": args.stage,
            "data_split": args.data_split,
            "sample_id": args.sample_id,
            "candidate": "B0_CLASSIC_90",
            "feature_count": BASELINE_FEATURES,
            "replacement": "zero",
            "randomization_level": args.level,
            "randomization_seed": args.seed,
            "target_protocols": list(PROTOCOLS),
            "logical_query_budget_per_candidate": TOTAL_QUERY_BUDGET,
            "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
            "model_weight_fingerprint": model_identity[
                "model_weight_fingerprint"
            ],
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "all_outputs_nonempty": all(
                path.exists() and path.stat().st_size > 1 for path in required
            ),
        },
    )
    print(json.dumps({"status": "COMPLETE", "run_id": run_id}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--data-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--level", choices=LEVELS, default="R0")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3101)
    args = parser.parse_args()
    if args.stage == "test":
        frozen = ROOT / "frozen_validation_selection.json"
        if not frozen.exists() or not json.loads(
            frozen.read_text(encoding="utf-8")
        ).get("fixed_test_allowed"):
            parser.error("fixed test is blocked until validation hard gates pass")
    return args


if __name__ == "__main__":
    run_one(parse_args())
