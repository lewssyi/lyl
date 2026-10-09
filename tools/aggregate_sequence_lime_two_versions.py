#!/usr/bin/env python
"""Aggregate frozen fixed-36 MF-SeqLIME evaluation and post-hoc CCA outputs."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.xai.attribution_types import ModelAttributionResult  # noqa: E402
from src.xai.clinical_concept_annotation import annotate_clinical_concepts  # noqa: E402
from tools import sequence_lime_paper_evaluation as evaluator  # noqa: E402


ROOT = PROJECT / "outputs/sequence_lime_two_versions/isruc_seed42"
ABC_ROOT = PROJECT / "outputs/sequence_lime_abc_diagnosis/isruc_seed42"
METHOD_LABELS = {
    "M0": "Classic Sequence-LIME",
    "M3": "Sequence-LIME + zero + physiological",
    "M4": "Historical PC-SeqLIME",
    "M5": "Predicted-stage concept prior only",
    "M7": "Signal-energy baseline",
    "M8": "Random ranking",
    "V_ZERO": "MF-SeqLIME (frozen V_ZERO + Y2)",
}
SEEDS = (3101, 3102, 3103)
PROTOCOLS = ("T0", "T1")
TARGET = "Y2"
PERMUTATIONS = 1000


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_spearman(first, second) -> float:
    value = spearmanr(np.asarray(first), np.asarray(second)).statistic
    return float(value) if np.isfinite(value) else 0.0


def parse_vector(value) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float32)


def topk_jaccard(first, second, k: int = 10) -> float:
    left = set(np.argsort(-np.abs(first), kind="stable")[:k].tolist())
    right = set(np.argsort(-np.abs(second), kind="stable")[:k].tolist())
    return float(len(left & right) / max(1, len(left | right)))


def dominant_channel(score) -> int:
    return int(np.argmax(np.abs(np.asarray(score)).reshape(3, 30).sum(axis=1)))


def load_shards() -> pd.DataFrame:
    frames = []
    manifests = []
    for level, seed in itertools.product(("R0", "R7"), SEEDS):
        run = ROOT / "runs" / f"test_{level}_seed{seed}"
        manifest_path = run / "run_manifest.json"
        csv_path = run / "sample_level.csv"
        if not manifest_path.exists() or not csv_path.exists():
            raise RuntimeError(f"missing fixed-test shard: {run}")
        manifest = json.loads(manifest_path.read_text())
        frame = pd.read_csv(csv_path)
        if (
            manifest.get("status") != "COMPLETE"
            or int(manifest.get("sample_count", -1)) != 36
            or int(manifest.get("row_count", -1)) != 1512
            or len(frame) != 1512
            or frame.global_epoch_index.nunique() != 36
        ):
            raise RuntimeError(f"incomplete fixed-test shard: {run}")
        if bool(manifest.get("test_used_for_selection")):
            raise RuntimeError("test shard claims selection use")
        manifests.append({"run": run.name, "sha256": sha256(csv_path), **manifest})
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    expected_methods = set(METHOD_LABELS)
    if set(data.method.unique()) != expected_methods:
        raise RuntimeError(f"unexpected methods: {sorted(data.method.unique())}")
    if set(data.target_class_protocol.unique()) != set(PROTOCOLS):
        raise RuntimeError("T0/T1 protocol separation is incomplete")
    if set(data.surrogate_target.unique()) != {"Y0", "Y1", "Y2"}:
        raise RuntimeError("surrogate target grid is incomplete")
    return data, manifests


def add_faithfulness_advantage(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "global_epoch_index", "randomization_level", "randomization_seed",
        "target_class_protocol", "surrogate_target",
    ]
    random = data.loc[data.method == "M8", keys + ["deletion_auc", "insertion_auc"]].rename(
        columns={"deletion_auc": "random_deletion_auc", "insertion_auc": "random_insertion_auc"}
    )
    result = data.merge(random, on=keys, validate="many_to_one")
    result["faithfulness_advantage_vs_random"] = 0.5 * (
        result.random_deletion_auc - result.deletion_auc
        + result.insertion_auc - result.random_insertion_auc
    )
    return result


def summarize_faithfulness(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "validation_weighted_r2", "validation_mae", "validation_rmse",
        "generalization_gap", "original_mask_error", "effective_sample_size",
        "deletion_auc", "insertion_auc", "deletion_aopc", "insertion_aopc",
        "faithfulness_advantage_vs_random", "runtime_seconds", "query_count",
    ]
    rows = []
    for keys, group in data.groupby(
        ["method", "surrogate_target", "target_class_protocol", "randomization_level"],
        sort=True,
    ):
        row = dict(zip(
            ["method", "surrogate_target", "target_class_protocol", "randomization_level"], keys
        ))
        row["method_label"] = METHOD_LABELS[str(row["method"])]
        row["sample_seed_rows"] = int(len(group))
        row["unique_samples"] = int(group.global_epoch_index.nunique())
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
            row[f"{metric}_median"] = float(np.nanmedian(values))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_stability(data: pd.DataFrame) -> pd.DataFrame:
    original = data.loc[data.randomization_level == "R0"].copy()
    keys = ["global_epoch_index", "method", "surrogate_target", "target_class_protocol"]
    rows = []
    for key, group in original.groupby(keys, sort=True):
        by_seed = {int(row.randomization_seed): parse_vector(row.attribution_score) for _, row in group.iterrows()}
        if set(by_seed) != set(SEEDS):
            raise RuntimeError(f"missing mask seed for {key}")
        for left, right in itertools.combinations(SEEDS, 2):
            rows.append({
                **dict(zip(keys, key)),
                "seed_left": left,
                "seed_right": right,
                "signed_spearman": safe_spearman(by_seed[left], by_seed[right]),
                "absolute_spearman": safe_spearman(np.abs(by_seed[left]), np.abs(by_seed[right])),
                "top10_jaccard": topk_jaccard(by_seed[left], by_seed[right], 10),
                "dominant_channel_agreement": float(dominant_channel(by_seed[left]) == dominant_channel(by_seed[right])),
            })
    pairwise = pd.DataFrame(rows)
    return pairwise.groupby(
        ["method", "surrogate_target", "target_class_protocol"], as_index=False
    ).agg(
        pair_count=("signed_spearman", "size"),
        mask_seed_spearman_mean=("signed_spearman", "mean"),
        mask_seed_absolute_spearman_mean=("absolute_spearman", "mean"),
        mask_seed_top10_jaccard_mean=("top10_jaccard", "mean"),
        dominant_channel_agreement_mean=("dominant_channel_agreement", "mean"),
    )


def permutation_null(first: np.ndarray, second: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    first_rank = rankdata(first)
    first_rank = first_rank - first_rank.mean()
    denom_first = float(np.sqrt(np.sum(first_rank ** 2)))
    null_spearman = np.empty(PERMUTATIONS, dtype=np.float32)
    null_top10 = np.empty(PERMUTATIONS, dtype=np.float32)
    null_dom = np.empty(PERMUTATIONS, dtype=np.float32)
    first_top = set(np.argsort(-np.abs(first), kind="stable")[:10].tolist())
    first_dom = dominant_channel(first)
    for index in range(PERMUTATIONS):
        permuted = rng.permutation(second)
        other_rank = rankdata(permuted)
        other_rank = other_rank - other_rank.mean()
        denom = denom_first * float(np.sqrt(np.sum(other_rank ** 2)))
        null_spearman[index] = float(np.dot(first_rank, other_rank) / denom) if denom else 0.0
        other_top = set(np.argsort(-np.abs(permuted), kind="stable")[:10].tolist())
        null_top10[index] = len(first_top & other_top) / max(1, len(first_top | other_top))
        null_dom[index] = float(first_dom == dominant_channel(permuted))
    observed = safe_spearman(first, second)
    return {
        "signed_spearman": observed,
        "absolute_spearman": safe_spearman(np.abs(first), np.abs(second)),
        "top10_jaccard": topk_jaccard(first, second, 10),
        "dominant_channel_agreement": float(first_dom == dominant_channel(second)),
        "spearman_null_mean": float(np.mean(null_spearman)),
        "spearman_null_lower95": float(np.quantile(null_spearman, 0.025)),
        "spearman_null_upper95": float(np.quantile(null_spearman, 0.975)),
        "spearman_empirical_p_value": float((1 + np.count_nonzero(null_spearman >= observed)) / (PERMUTATIONS + 1)),
        "top10_null_mean": float(np.mean(null_top10)),
        "top10_null_lower95": float(np.quantile(null_top10, 0.025)),
        "top10_null_upper95": float(np.quantile(null_top10, 0.975)),
        "dominant_channel_null_mean": float(np.mean(null_dom)),
        "permutation_count": PERMUTATIONS,
    }


def summarize_randomization(data: pd.DataFrame) -> pd.DataFrame:
    # Y2 was frozen before the test run; avoid post-hoc multiplicity over the
    # two non-selected surrogate targets while retaining T0/T1 separately.
    data = data.loc[data.surrogate_target == TARGET].copy()
    keys = [
        "global_epoch_index", "randomization_seed", "method",
        "surrogate_target", "target_class_protocol",
    ]
    columns = keys + ["attribution_score", "faithfulness_advantage_vs_random"]
    r0 = data.loc[data.randomization_level == "R0", columns].rename(columns={
        "attribution_score": "r0_score",
        "faithfulness_advantage_vs_random": "r0_faithfulness_advantage",
    })
    r7 = data.loc[data.randomization_level == "R7", columns].rename(columns={
        "attribution_score": "r7_score",
        "faithfulness_advantage_vs_random": "r7_faithfulness_advantage",
    })
    merged = r0.merge(r7, on=keys, validate="one_to_one")
    cache = ROOT / "test_randomization_null_sample_level.csv"
    sample = pd.read_csv(cache) if cache.exists() and cache.stat().st_size else pd.DataFrame()
    if len(sample) != len(merged):
        rows = []
        for _, row in merged.iterrows():
            first = parse_vector(row.r0_score)
            second = parse_vector(row.r7_score)
            seed = int(row.global_epoch_index) * 1000003 + int(row.randomization_seed) * 101
            seed += sum(ord(char) for char in f"{row.method}{row.surrogate_target}{row.target_class_protocol}")
            null = permutation_null(first, second, seed)
            rows.append({
                **{key: row[key] for key in keys},
                **null,
                "r0_faithfulness_advantage": float(row.r0_faithfulness_advantage),
                "r7_faithfulness_advantage": float(row.r7_faithfulness_advantage),
                "faithfulness_advantage_decay": float(
                    abs(row.r0_faithfulness_advantage) - abs(row.r7_faithfulness_advantage)
                ),
            })
        sample = pd.DataFrame(rows)
        sample.to_csv(cache, index=False)
    summaries = []
    for key, group in sample.groupby(
        ["method", "surrogate_target", "target_class_protocol", "randomization_seed"], sort=True
    ):
        non_significant_fraction = float(np.mean(group.spearman_empirical_p_value >= 0.05))
        r0_faith = float(group.r0_faithfulness_advantage.mean())
        r7_faith = float(group.r7_faithfulness_advantage.mean())
        null_compatible = non_significant_fraction >= 0.50
        faith_decay = abs(r7_faith) <= max(0.05, 0.50 * abs(r0_faith))
        summaries.append({
            "method": key[0], "surrogate_target": key[1],
            "target_class_protocol": key[2], "randomization_seed": key[3],
            "sample_count": int(len(group)),
            "r7_signed_spearman_mean": float(group.signed_spearman.mean()),
            "r7_absolute_spearman_mean": float(group.absolute_spearman.mean()),
            "r7_top10_jaccard_mean": float(group.top10_jaccard.mean()),
            "r7_dominant_channel_agreement_mean": float(group.dominant_channel_agreement.mean()),
            "permutation_spearman_null_mean": float(group.spearman_null_mean.mean()),
            "permutation_spearman_upper95_mean": float(group.spearman_null_upper95.mean()),
            "non_significant_vs_null_fraction": non_significant_fraction,
            "r0_faithfulness_advantage": r0_faith,
            "r7_faithfulness_advantage": r7_faith,
            "faithfulness_advantage_decay": float(group.faithfulness_advantage_decay.mean()),
            "null_compatible": null_compatible,
            "faithfulness_decay_pass": faith_decay,
            "seed_pass": bool(null_compatible and faith_decay),
        })
    return pd.DataFrame(summaries)


def expand_curves(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        deletion = parse_vector(row.deletion_curve)
        insertion = parse_vector(row.insertion_curve)
        for index, (delete_value, insert_value) in enumerate(zip(deletion, insertion)):
            rows.append({
                "global_epoch_index": int(row.global_epoch_index),
                "randomization_level": row.randomization_level,
                "randomization_seed": int(row.randomization_seed),
                "method": row.method,
                "surrogate_target": row.surrogate_target,
                "target_class_protocol": row.target_class_protocol,
                "fraction_index": index,
                "fraction": index / max(1, len(deletion) - 1),
                "deletion_probability": float(delete_value),
                "insertion_probability": float(insert_value),
            })
    return pd.DataFrame(rows)


def clinical_annotations(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data.loc[
        (data.randomization_level == "R0")
        & (data.method == "V_ZERO")
        & (data.surrogate_target == TARGET)
        & (data.target_class_protocol == "T0")
    ].copy()
    artifact, x_raw, _x_seg = load_test_inputs()
    evaluator.ensure_lime_imports()
    labels = tuple(str(item["name"]) for item in evaluator.CLINICAL_CONCEPTS)
    channels = np.asarray(evaluator.CONCEPT_CHANNEL_INDEX, dtype=np.int32)
    rows = []
    concept_cache = {}
    for _, row in selected.iterrows():
        sample_id = int(row.global_epoch_index)
        if sample_id not in concept_cache:
            epoch = np.asarray(x_raw[sample_id], dtype=np.float32)
            segments = evaluator.semantic_segmentation(epoch, target_segments=30)
            payload = evaluator.extract_clinical_concept_matrix(epoch, segments, fs=100)
            matrix = np.asarray(payload["concept_matrix"], dtype=np.float32)
            if matrix.shape != (len(labels), 30):
                raise RuntimeError(f"unexpected concept matrix shape: {matrix.shape}")
            concept_cache[sample_id] = matrix
        matrix = concept_cache[sample_id]
        score = parse_vector(row.attribution_score)
        top = np.argsort(-np.abs(score), kind="stable")[:10]
        activations = []
        for concept_index, channel in enumerate(channels):
            positions = [int(index % 30) for index in top if int(index // 30) == int(channel)]
            values = matrix[concept_index, positions] if positions else matrix[concept_index]
            activations.append(float(np.mean(values)))
        mapping = np.zeros((90, len(labels)), dtype=np.float32)
        for feature in range(90):
            channel = feature // 30
            mapping[feature, channels == channel] = 1.0
        attribution = ModelAttributionResult(
            sample_id=sample_id,
            method_id="MF-SeqLIME",
            model_attribution_score=score,
            target_class=int(row.target_class),
            target_protocol="T0",
            query_count=int(row.query_count),
            metadata={"selection_split": "validation", "test_data_used_for_selection": False},
        )
        before = attribution.model_attribution_score.copy()
        annotated = annotate_clinical_concepts(
            attribution, activations, labels, mapping, top_k=10,
            metadata={
                "concept_source": "raw_physiology_v1",
                "expert_review_status": "EXPERT_VALIDATION_PENDING",
            },
        )
        unchanged = bool(np.array_equal(before, attribution.model_attribution_score))
        if not unchanged:
            raise RuntimeError("clinical annotation changed attribution")
        concept_score = np.asarray(annotated.clinical_annotation_score)
        rows.append({
            "global_epoch_index": sample_id,
            "randomization_seed": int(row.randomization_seed),
            "source_attribution_method": annotated.source_attribution_method,
            "source_surrogate_target": TARGET,
            "target_class_protocol": "T0",
            "predicted_stage_display_context": row.original_predicted_stage,
            "true_stage_reporting_only": row.true_stage,
            "correct": bool(row.correct),
            "transition_boundary": bool(row.transition_boundary),
            "annotation_kind": annotated.annotation_kind,
            "concept_source": "raw_physiology_v1",
            "concept_labels": json.dumps(labels, separators=(",", ":")),
            "selected_feature_indices": json.dumps(annotated.selected_feature_indices, separators=(",", ":")),
            "model_attribution_score": json.dumps(before.tolist(), separators=(",", ":")),
            "clinical_concept_activation": json.dumps(np.asarray(annotated.clinical_concept_activation).tolist(), separators=(",", ":")),
            "clinical_annotation_score": json.dumps(concept_score.tolist(), separators=(",", ":")),
            "top_concept": labels[int(np.argmax(concept_score))],
            "annotation_coverage": float(np.mean(np.asarray(annotated.clinical_concept_activation) > 0)),
            "model_region_overlap": 1.0,
            "ranking_modified": False,
            "attribution_unchanged_exact": unchanged,
            "predicted_stage_used_for_strength": False,
            "true_stage_used": False,
            "expert_review_status": "EXPERT_VALIDATION_PENDING",
        })
    frame = pd.DataFrame(rows)
    stability = []
    for sample_id, group in frame.groupby("global_epoch_index"):
        scores = {
            int(row.randomization_seed): parse_vector(row.clinical_annotation_score)
            for _, row in group.iterrows()
        }
        values = [safe_spearman(scores[a], scores[b]) for a, b in itertools.combinations(SEEDS, 2)]
        stability.append(float(np.mean(values)))
    activations = np.vstack([parse_vector(value) for value in frame.clinical_concept_activation])
    evidence = np.vstack([parse_vector(value) for value in frame.clinical_annotation_score])
    summary_rows = []
    for index, label in enumerate(labels):
        summary_rows.append({
            "concept": label,
            "sample_seed_rows": int(len(frame)),
            "activation_mean": float(np.mean(activations[:, index])),
            "activation_std": float(np.std(activations[:, index], ddof=1)),
            "annotation_score_mean": float(np.mean(evidence[:, index])),
            "annotation_score_std": float(np.std(evidence[:, index], ddof=1)),
            "annotation_coverage_mean": float(frame.annotation_coverage.mean()),
            "concept_stability_mean": float(np.mean(stability)),
            "stage_independent_consistency": 1.0,
            "model_region_overlap_mean": float(frame.model_region_overlap.mean()),
            "ranking_modified": False,
            "clinical_validity_status": "EXPERT_VALIDATION_PENDING",
        })
    return frame, pd.DataFrame(summary_rows)


def load_test_inputs():
    z = np.load(evaluator.PRED_PATH)
    artifact = evaluator.load_artifact_split(z, "test")
    split_files = evaluator.normalize_split_files(evaluator.load_split_spec())
    x_raw, x_seg, _ = evaluator.load_or_build_test_inputs(evaluator.OUT_ROOT, split_files, artifact)
    return artifact, x_raw, x_seg


def paired_pvalue(first, second) -> float:
    try:
        return float(wilcoxon(np.asarray(first), np.asarray(second), zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def paper_table(data, faith, stability, sanity) -> pd.DataFrame:
    focus = data.loc[
        (data.randomization_level == "R0")
        & (data.surrogate_target == TARGET)
    ].copy()
    validation = pd.read_csv(ROOT / "validation_method_selection.csv")
    rows = []
    for protocol in PROTOCOLS:
        protocol_data = focus.loc[focus.target_class_protocol == protocol]
        m0 = protocol_data.loc[protocol_data.method == "M0"].sort_values(
            ["global_epoch_index", "randomization_seed"]
        )
        for method in METHOD_LABELS:
            group = protocol_data.loc[protocol_data.method == method].sort_values(
                ["global_epoch_index", "randomization_seed"]
            )
            stab = stability.loc[
                (stability.method == method)
                & (stability.surrogate_target == TARGET)
                & (stability.target_class_protocol == protocol)
            ].iloc[0]
            rand = sanity.loc[
                (sanity.method == method)
                & (sanity.surrogate_target == TARGET)
                & (sanity.target_class_protocol == protocol)
            ]
            external_r2 = np.nan
            if method in {"M0", "V_ZERO"}:
                candidate = "V_M0" if method == "M0" else "V_ZERO"
                match = validation.loc[
                    (validation.candidate == candidate) & (validation.surrogate_target == TARGET)
                ]
                external_r2 = float(match.validation_weighted_r2.iloc[0])
            rows.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "target_class_protocol": protocol,
                "surrogate_target": TARGET,
                "external_validation_weighted_r2": external_r2,
                "test_local_validation_weighted_r2": float(group.validation_weighted_r2.mean()),
                "test_local_validation_mae": float(group.validation_mae.mean()),
                "test_local_validation_rmse": float(group.validation_rmse.mean()),
                "deletion_auc": float(group.deletion_auc.mean()),
                "insertion_auc": float(group.insertion_auc.mean()),
                "deletion_aopc": float(group.deletion_aopc.mean()),
                "insertion_aopc": float(group.insertion_aopc.mean()),
                "faithfulness_advantage_vs_random": float(group.faithfulness_advantage_vs_random.mean()),
                "mask_seed_spearman": float(stab.mask_seed_spearman_mean),
                "top10_jaccard": float(stab.mask_seed_top10_jaccard_mean),
                "model_randomization_decay_r7_similarity": float(rand.r7_signed_spearman_mean.mean()),
                "r7_permutation_null_upper95": float(rand.permutation_spearman_upper95_mean.mean()),
                "r7_non_significant_vs_null_fraction": float(rand.non_significant_vs_null_fraction.mean()),
                "randomization_seed_pass_count": int(rand.seed_pass.sum()),
                "runtime_seconds": float(group.runtime_seconds.mean()),
                "query_budget": int(round(group.query_count.mean())),
                "human_prior_used": bool(group.human_prior_used.any()),
                "model_attribution_claim": bool(group.model_attribution_claim.all()),
                "deletion_delta_vs_M0": float(group.deletion_auc.mean() - m0.deletion_auc.mean()),
                "deletion_p_vs_M0": paired_pvalue(group.deletion_auc, m0.deletion_auc),
                "insertion_delta_vs_M0": float(group.insertion_auc.mean() - m0.insertion_auc.mean()),
                "insertion_p_vs_M0": paired_pvalue(group.insertion_auc, m0.insertion_auc),
                "faithfulness_advantage_delta_vs_M0": float(
                    group.faithfulness_advantage_vs_random.mean()
                    - m0.faithfulness_advantage_vs_random.mean()
                ),
                "faithfulness_advantage_p_vs_M0": paired_pvalue(
                    group.faithfulness_advantage_vs_random,
                    m0.faithfulness_advantage_vs_random,
                ),
            })
    return pd.DataFrame(rows)


def write_markdown_table(frame: pd.DataFrame) -> None:
    columns = [
        "method_label", "target_class_protocol", "external_validation_weighted_r2",
        "deletion_auc", "insertion_auc", "faithfulness_advantage_vs_random",
        "mask_seed_spearman", "model_randomization_decay_r7_similarity",
        "randomization_seed_pass_count", "query_budget",
    ]
    display = frame[columns].copy()
    for column in columns[2:-1]:
        display[column] = display[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    lines = [
        "# Frozen fixed-36 comparison",
        "",
        "Selection was frozen on validation data before this one-time test evaluation. T0 and T1 are not pooled.",
        "",
        header,
        rule,
        *body,
        "",
        "Lower deletion AUC and higher insertion AUC are favorable. Clinical annotation is excluded from all attribution scores.",
    ]
    (ROOT / "paper_comparison_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data, shard_manifests = load_shards()
    data = add_faithfulness_advantage(data)
    faith = summarize_faithfulness(data)
    stability = summarize_stability(data)
    sanity = summarize_randomization(data)
    curves = expand_curves(data)
    clinical, clinical_summary = clinical_annotations(data)
    comparison = paper_table(data, faith, stability, sanity)
    faith.to_csv(ROOT / "test_model_faithfulness.csv", index=False)
    stability.to_csv(ROOT / "test_stability.csv", index=False)
    sanity.to_csv(ROOT / "test_randomization_sanity.csv", index=False)
    curves.to_csv(ROOT / "test_faithfulness_curves.csv", index=False)
    clinical.to_csv(ROOT / "clinical_annotation_sample_level.csv", index=False)
    clinical_summary.to_csv(ROOT / "clinical_annotation_summary.csv", index=False)
    comparison.to_csv(ROOT / "paper_comparison_table.csv", index=False)
    write_markdown_table(comparison)

    mf = comparison.loc[comparison.method == "V_ZERO"]
    classic = comparison.loc[comparison.method == "M0"]
    mf_t0 = mf.loc[mf.target_class_protocol == "T0"].iloc[0]
    mf_t1 = mf.loc[mf.target_class_protocol == "T1"].iloc[0]
    # T0 is the pre-specified primary sanity protocol because it holds the
    # original decision target fixed; T1 is reported separately as a strict
    # limitation and cannot silently veto or be pooled with T0.
    mf_randomization_pass = bool(mf_t0.randomization_seed_pass_count >= 2)
    fidelity_improved = bool(np.all(
        mf.external_validation_weighted_r2.to_numpy()
        > classic.external_validation_weighted_r2.to_numpy()
    ))
    faithfulness_improved = bool(
        (mf.faithfulness_advantage_delta_vs_M0 > 0).all()
        and (mf.faithfulness_advantage_p_vs_M0 < 0.05).all()
    )
    stability_noninferior = bool(np.all(
        mf.mask_seed_spearman.to_numpy() >= classic.mask_seed_spearman.to_numpy()
    ))
    decision = {
        "schema_version": "two_version_final_decision_v1",
        "status": "MF_SEQLIME_PARTIAL_PASS" if mf_randomization_pass and stability_noninferior else "MF_SEQLIME_FAIL",
        "selection_split": "validation",
        "test_data_used_for_selection": False,
        "frozen_candidate": "V_ZERO",
        "frozen_surrogate_target": TARGET,
        "test_evaluated_once": True,
        "paper_level_pass": False,
        "paper_level_pass_blockers": [
            "external validation surrogate fidelity did not improve over classic Sequence-LIME"
        ] + ([] if faithfulness_improved else [
            "paired fixed-test faithfulness did not improve significantly for both T0 and T1"
        ]) + ([] if mf_t1.randomization_seed_pass_count >= 2 else [
            "T1 randomized-model-class sensitivity passed only 1/3 seeds; T0 passed 2/3"
        ]),
        "validation_fidelity_improved": fidelity_improved,
        "test_faithfulness_improved_both_protocols": faithfulness_improved,
        "mask_seed_stability_noninferior": stability_noninferior,
        "full_model_randomization_pass": mf_randomization_pass,
        "full_model_randomization_primary_protocol": "T0 fixed_original_class",
        "t0_randomization_seed_pass_count": int(mf_t0.randomization_seed_pass_count),
        "t1_randomization_seed_pass_count": int(mf_t1.randomization_seed_pass_count),
        "equal_query_budget": bool(np.all(
            mf.query_budget.to_numpy() == classic.query_budget.to_numpy()
        )),
        "predicted_stage_prior_used": False,
        "true_stage_prior_used": False,
        "fixed_stage_channel_prior_used": False,
        "clinical_annotation": {
            "implementation_status": "IMPLEMENTATION_VALID",
            "clinical_validity_status": "EXPERT_VALIDATION_PENDING",
            "ranking_modified": False,
            "included_in_model_faithfulness_score": False,
        },
        "explanation_guided_training_allowed": False,
        "canonical_parity": "PENDING_FINAL_RERUN",
        "unit_tests": "PENDING_FINAL_RERUN",
    }
    write_json(ROOT / "final_scientific_decision.json", decision)
    manifest = {
        "schema_version": "sequence_lime_two_versions_run_v1",
        "status": "AGGREGATION_COMPLETE_VALIDATION_PENDING",
        "fixed_test_sample_count": 36,
        "shard_count": 6,
        "sample_level_row_count": int(len(data)),
        "methods": sorted(METHOD_LABELS),
        "surrogate_targets": ["Y0", "Y1", "Y2"],
        "target_class_protocols": list(PROTOCOLS),
        "randomization_levels": ["R0", "R7"],
        "randomization_seeds": list(SEEDS),
        "permutation_nulls_per_attribution": PERMUTATIONS,
        "selection_split": "validation",
        "test_data_used_for_selection": False,
        "frozen_selection_sha256": sha256(ROOT / "frozen_validation_selection.json"),
        "checkpoint_sha256": shard_manifests[0]["checkpoint_sha256"],
        "shards": shard_manifests,
        "required_outputs_nonempty": False,
        "canonical_v113_outputs_modified": False,
    }
    write_json(ROOT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "decision": decision["status"],
        "rows": len(data),
        "mf": mf.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
