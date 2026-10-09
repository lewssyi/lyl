#!/usr/bin/env python
"""Validation-only MF-SeqLIME selection and frozen fixed-36 test evaluation."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_lime_abc_diagnosis as abc  # noqa: E402
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ROOT = PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42"
ABC_ROOT = PROJECT / "outputs/sequence_lime_abc_diagnosis/isruc_seed42"
VALIDATION_MANIFEST = ROOT / "validation_selection_samples.json"
FROZEN_SELECTION = ROOT / "frozen_validation_selection.json"
CANDIDATES = ("V_M0", "V_ZERO", "V_PHYS", "V_EQUAL", "V_ZERO25", "V_PHYS25")
COMPARATORS = ("M0", "M3", "M4", "M5", "M7", "M8")


def load_split_inputs(split: str):
    z = np.load(evaluator.PRED_PATH)
    artifact = evaluator.load_artifact_split(z, "val" if split == "validation" else split)
    split_files = evaluator.normalize_split_files(evaluator.load_split_spec())
    if split == "test":
        x_raw, x_seg, _ = evaluator.load_or_build_test_inputs(
            evaluator.OUT_ROOT, split_files, artifact
        )
        current = evaluator.OUT_ROOT / "cache/current_sequence_probs_float32.npy"
        if current.exists():
            artifact = dict(artifact)
            artifact["probs"] = np.load(current).astype(np.float32)
            artifact["pred"] = np.argmax(artifact["probs"], axis=1).astype(np.int16)
        return artifact, x_raw, x_seg

    evaluator.ensure_lime_imports()
    cache = ROOT / "cache/validation_inputs_segments.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cached = np.load(cache)
        return artifact, cached["x_raw"].astype(np.float32), cached["x_seg"].astype(np.float32)
    from src.data_processing import load_npz_data

    spec = evaluator.load_split_spec()
    loaded = load_npz_data(
        str(Path(str(spec["source_npz_dir"])).expanduser()),
        file_subset=list(split_files["val"]),
        preprocess=True,
        filter_profile="clinical_canonical",
        filter_backend="scipy",
        normalization_mode="recording_robust_zscore",
        clip_sigma=None,
        enable_amplitude_clipping=False,
        return_quality=True,
    )
    x_raw, labels, _quality = loaded
    if not np.array_equal(np.asarray(labels, dtype=np.int16), artifact["y_true"]):
        raise RuntimeError("validation labels differ from the frozen prediction artifact")
    features = []
    for index, epoch in enumerate(x_raw):
        if index and index % 1000 == 0:
            print(f"validation segment features {index}/{len(x_raw)}", flush=True)
        segments = evaluator.semantic_segmentation(epoch, target_segments=30)
        features.append(
            evaluator.extract_segment_features(
                epoch, segments, max_segments=30, include_spectral=True
            )
        )
    x_seg = np.asarray(features, dtype=np.float32)
    np.savez_compressed(cache, x_raw=np.asarray(x_raw, dtype=np.float32), x_seg=x_seg)
    return artifact, np.asarray(x_raw, dtype=np.float32), x_seg


def prepare_validation_manifest() -> None:
    artifact, _x_raw, _x_seg = load_split_inputs("validation")
    labels = np.asarray(artifact["y_true"], dtype=np.int16)
    predicted = np.asarray(artifact["pred"], dtype=np.int16)
    recording = np.asarray(artifact["recording_id"], dtype=np.int32)
    transition = np.zeros(len(labels), dtype=bool)
    for index in range(len(labels)):
        left = index > 0 and recording[index - 1] == recording[index] and labels[index - 1] != labels[index]
        right = index + 1 < len(labels) and recording[index + 1] == recording[index] and labels[index + 1] != labels[index]
        transition[index] = bool(left or right)
    frame = pd.DataFrame({
        "global_epoch_index": np.arange(len(labels), dtype=np.int32),
        "recording_id": recording,
        "subject_id": recording,
        "epoch_index_in_recording": artifact["epoch_index"],
        "true_stage": [evaluator.CLASS_LABELS[int(value)] for value in labels],
        "predicted_stage": [evaluator.CLASS_LABELS[int(value)] for value in predicted],
        "confidence": np.max(artifact["probs"], axis=1),
        "correct": labels == predicted,
        "transition_boundary": transition,
        "selection_split": "validation",
    })
    rng = np.random.default_rng(42017)
    selected = []
    for stage in evaluator.CLASS_LABELS:
        for correct in (True, False):
            group = frame.loc[(frame.true_stage == stage) & (frame.correct == correct)]
            if len(group):
                selected.append(int(rng.choice(group.global_epoch_index.to_numpy())))
    remaining = frame.loc[~frame.global_epoch_index.isin(selected)].copy()
    for transition_value in (True, False):
        if len(selected) >= 12:
            break
        group = remaining.loc[remaining.transition_boundary == transition_value]
        needed = min(12 - len(selected), len(group))
        if needed:
            selected.extend(int(value) for value in rng.choice(group.global_epoch_index.to_numpy(), size=needed, replace=False))
    selected_frame = frame.set_index("global_epoch_index").loc[selected[:12]].reset_index()
    payload = {
        "schema_version": "mf_validation_selection_v1",
        "selection_split": "validation",
        "test_data_used": False,
        "selection_rule": "seeded stage-by-correctness stratification, then transition/stable fill; no attribution or faithfulness endpoint inspected",
        "selection_seed": 42017,
        "sample_count": int(len(selected_frame)),
        "prediction_artifact_sha256": evaluator.file_sha256(evaluator.PRED_PATH),
        "split_sha256": evaluator.file_sha256(evaluator.SPLIT_PATH),
        "samples": json.loads(selected_frame.to_json(orient="records")),
    }
    abc.write_json(VALIDATION_MANIFEST, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))


def evaluation_frame(split: str) -> pd.DataFrame:
    if split == "validation":
        payload = json.loads(VALIDATION_MANIFEST.read_text())
        return pd.DataFrame(payload["samples"])
    payload = json.loads((ABC_ROOT / "fixed_36_samples.json").read_text())
    return pd.DataFrame(payload["samples"])


def run_shard(args) -> None:
    frame = evaluation_frame(args.data_split)
    artifact, x_raw, x_seg = load_split_inputs(args.data_split)
    prefix = "validation" if args.data_split == "validation" else "test"
    run_id = f"{prefix}_{args.randomization_level}_seed{args.randomization_seed}"
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sample_level.csv"
    include_candidates = args.data_split == "validation" or bool(args.include_validation_candidates)
    method_filter = None
    if args.data_split == "test":
        frozen = json.loads(FROZEN_SELECTION.read_text())
        method_filter = list(COMPARATORS)
        selected = str(frozen["selected_candidate"])
        if selected not in method_filter:
            method_filter.append(selected)
    expected_methods = len(CANDIDATES) + 9 if include_candidates else 9
    if method_filter:
        expected_methods = len(method_filter)
    expected_rows = expected_methods * 3 * 2
    existing = pd.read_csv(csv_path) if args.resume and csv_path.exists() else pd.DataFrame()
    completed = set()
    if not existing.empty:
        counts = existing.groupby("global_epoch_index").size()
        completed = {int(index) for index, count in counts.items() if int(count) == expected_rows}
    model = evaluator.randomized_model(
        x_raw.shape[1:], x_seg.shape[1:], args.randomization_level, seed=args.randomization_seed
    )
    frames = [existing] if not existing.empty else []
    for _, row in frame.iterrows():
        sample_id = int(row.global_epoch_index)
        if sample_id in completed:
            continue
        sample = row.to_dict()
        sample["_include_validation_candidates"] = include_candidates
        if method_filter:
            sample["_method_filter"] = method_filter
        print(f"start {run_id} sample={sample_id}", flush=True)
        result = abc.run_sample(
            sample,
            model,
            args.randomization_level,
            args.randomization_seed,
            args.train_masks,
            args.validation_masks,
            artifact,
            x_raw,
            x_seg,
            out_dir,
        )
        if len(result) != expected_rows:
            raise RuntimeError(f"expected {expected_rows} rows, observed {len(result)}")
        frames.append(result)
        pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        print(f"done {run_id} sample={sample_id}", flush=True)
    final = pd.concat(frames, ignore_index=True)
    manifest = {
        "schema_version": "two_version_shard_v1",
        "status": "COMPLETE",
        "data_split": args.data_split,
        "test_used_for_selection": False,
        "randomization_level": args.randomization_level,
        "randomization_seed": args.randomization_seed,
        "sample_count": int(final.global_epoch_index.nunique()),
        "row_count": int(len(final)),
        "expected_rows_per_sample": expected_rows,
        "checkpoint_sha256": evaluator.file_sha256(evaluator.CKPT_PATH),
        "csv_sha256": evaluator.file_sha256(csv_path),
    }
    abc.write_json(out_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def attribution_similarity(first: str, second: str) -> float:
    return float(spearmanr(np.asarray(json.loads(first)), np.asarray(json.loads(second))).statistic)


def aggregate_validation() -> None:
    frames = []
    for level in ("R0", "R7"):
        for seed in (3101, 3102, 3103):
            path = ROOT / "runs" / f"validation_{level}_seed{seed}/sample_level.csv"
            data = pd.read_csv(path)
            if len(data) != 1080 or data.global_epoch_index.nunique() != 12:
                raise RuntimeError(f"validation shard incomplete: {path}")
            frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    keys = ["global_epoch_index", "randomization_seed", "target_class_protocol", "surrogate_target", "method"]
    r0 = data.loc[data.randomization_level == "R0"]
    r7 = data.loc[data.randomization_level == "R7"]
    reference = r0[keys + ["attribution_score"]].rename(columns={"attribution_score": "r0_score"})
    compared = r7.merge(reference, on=keys, validate="one_to_one")
    compared["r7_similarity"] = [
        attribution_similarity(first, second)
        for first, second in zip(compared.r0_score, compared.attribution_score)
    ]
    baseline = data.loc[data.method == "M8", [
        "global_epoch_index", "randomization_level", "randomization_seed", "target_class_protocol", "surrogate_target",
        "deletion_auc", "insertion_auc",
    ]].rename(columns={"deletion_auc": "random_deletion", "insertion_auc": "random_insertion"})
    enriched = data.merge(
        baseline,
        on=["global_epoch_index", "randomization_level", "randomization_seed", "target_class_protocol", "surrogate_target"],
        validate="many_to_one",
    )
    enriched["faithfulness_advantage"] = 0.5 * (
        enriched.random_deletion - enriched.deletion_auc
        + enriched.insertion_auc - enriched.random_insertion
    )
    candidates = []
    for candidate, target in itertools.product(CANDIDATES, ("Y0", "Y1", "Y2")):
        original = enriched.loc[
            (enriched.randomization_level == "R0")
            & (enriched.method == candidate)
            & (enriched.surrogate_target == target)
        ]
        randomized = compared.loc[(compared.method == candidate) & (compared.surrogate_target == target)]
        stability_values = []
        for (_sample, protocol), group in original.groupby(["global_epoch_index", "target_class_protocol"]):
            scores = group.sort_values("randomization_seed").attribution_score.tolist()
            stability_values.extend(
                attribution_similarity(scores[a], scores[b]) for a, b in ((0, 1), (0, 2), (1, 2))
            )
        seed_similarity = randomized.groupby("randomization_seed").r7_similarity.mean()
        candidates.append({
            "candidate": candidate,
            "surrogate_target": target,
            "selection_split": "validation",
            "test_data_used": False,
            "validation_weighted_r2": float(original.validation_weighted_r2.mean()),
            "validation_mae": float(original.validation_mae.mean()),
            "validation_rmse": float(original.validation_rmse.mean()),
            "faithfulness_advantage_vs_random": float(original.faithfulness_advantage.mean()),
            "mask_seed_spearman": float(np.mean(stability_values)),
            "r7_similarity": float(randomized.r7_similarity.mean()),
            "randomization_seed_pass_count": int((seed_similarity <= 0.21).sum()),
            "equal_query_budget": bool(original.query_count.nunique() == 1),
            "query_count": int(original.query_count.iloc[0]),
        })
    result = pd.DataFrame(candidates)
    baseline_rows = result.loc[result.candidate == "V_M0"].set_index("surrogate_target")
    result["fidelity_noninferior"] = [
        row.validation_weighted_r2 >= baseline_rows.loc[row.surrogate_target].validation_weighted_r2 - 1e-6
        for row in result.itertuples()
    ]
    result["faithfulness_noninferior"] = [
        row.faithfulness_advantage_vs_random >= baseline_rows.loc[row.surrogate_target].faithfulness_advantage_vs_random - 0.01
        for row in result.itertuples()
    ]
    result["stability_noninferior"] = [
        row.mask_seed_spearman >= baseline_rows.loc[row.surrogate_target].mask_seed_spearman - 0.05
        for row in result.itertuples()
    ]
    result["randomization_pass"] = result.randomization_seed_pass_count >= 2
    result["eligible"] = result[[
        "fidelity_noninferior", "faithfulness_noninferior", "stability_noninferior", "randomization_pass", "equal_query_budget"
    ]].all(axis=1)
    result["validation_score"] = (
        result.validation_weighted_r2
        + result.faithfulness_advantage_vs_random
        + result.mask_seed_spearman
        - result.r7_similarity.abs()
    )
    eligible = result.loc[result.eligible]
    if eligible.empty:
        selected = result.loc[(result.candidate == "V_M0") & (result.surrogate_target == "Y2")].iloc[0]
        status = "FALLBACK_PURE_M0"
    else:
        selected = eligible.sort_values(["validation_score", "candidate"], ascending=[False, True]).iloc[0]
        status = "VALIDATION_SELECTED"
    result["selected"] = False
    result.loc[selected.name, "selected"] = True
    result.to_csv(ROOT / "validation_method_selection.csv", index=False)
    frozen = {
        "schema_version": "mf_frozen_validation_selection_v1",
        "status": status,
        "selection_split": "validation",
        "test_data_used": False,
        "selected_candidate": str(selected.candidate),
        "selected_surrogate_target": str(selected.surrogate_target),
        "selected_validation_score": float(selected.validation_score),
        "selected_query_count": int(selected.query_count),
        "predicted_stage_prior_used": False,
        "true_stage_prior_used": False,
        "fixed_stage_channel_prior_used": False,
        "test_evaluation_allowed": True,
    }
    abc.write_json(FROZEN_SELECTION, frozen)
    print(json.dumps(frozen, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-validation", action="store_true")
    parser.add_argument("--aggregate-validation", action="store_true")
    parser.add_argument("--data-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--randomization-level", choices=("R0", "R7"), default="R0")
    parser.add_argument("--randomization-seed", type=int, choices=(3101, 3102, 3103), default=3101)
    parser.add_argument("--train-masks", type=int, default=128)
    parser.add_argument("--validation-masks", type=int, default=32)
    parser.add_argument("--include-validation-candidates", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare_validation:
        prepare_validation_manifest()
    elif args.aggregate_validation:
        aggregate_validation()
    else:
        run_shard(args)
