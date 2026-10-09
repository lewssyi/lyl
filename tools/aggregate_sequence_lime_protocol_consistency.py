#!/usr/bin/env python
"""Validate and aggregate the unified Classic Sequence-LIME consistency audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_lime_protocol_consistency_audit as audit  # noqa: E402


ROOT = audit.ROOT
PERMUTATIONS = 1000


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


def normalized_ranks(value: str) -> np.ndarray:
    ranks = rankdata(parse_vector(value), method="average").astype(np.float64)
    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    return ranks / max(norm, 1e-12)


def top_jaccard(first: str, second: str, count: int) -> float:
    a = set(np.argsort(-np.abs(parse_vector(first)), kind="stable")[:count].tolist())
    b = set(np.argsort(-np.abs(parse_vector(second)), kind="stable")[:count].tolist())
    return float(len(a.intersection(b)) / max(len(a.union(b)), 1))


def dominant_channel(value: str) -> int:
    vector = parse_vector(value).reshape(3, 30)
    return int(np.argmax(np.sum(np.abs(vector), axis=1)))


def deterministic_seed(*values: object) -> int:
    text = "|".join(str(value) for value in values)
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def sample_null(first: str, second: str, *identity: object) -> tuple[float, float, float, np.ndarray]:
    a = normalized_ranks(first)
    b = normalized_ranks(second)
    observed = float(np.dot(a, b))
    rng = np.random.default_rng(deterministic_seed(*identity))
    null = np.empty(PERMUTATIONS, dtype=np.float32)
    for index in range(PERMUTATIONS):
        null[index] = abs(float(np.dot(a, rng.permutation(b))))
    q95 = float(np.quantile(null, 0.95))
    p_value = float((1 + np.sum(null >= abs(observed))) / (PERMUTATIONS + 1))
    return observed, abs(observed), q95, null


def expected_run(group: str, level: str, seed: int, shard_index: int, shard_count: int = 12) -> Path:
    run_id = f"group{group}_{level}_seed{seed}_shard{shard_index + 1}of{shard_count}"
    return ROOT / "runs" / run_id


def load_groups(groups: Iterable[str]) -> pd.DataFrame:
    frames = []
    failures = []
    for group in groups:
        expected_samples = len(audit.group_frame(group))
        for level in audit.LEVELS:
            for seed in audit.SEEDS:
                group_frames = []
                for shard_index in range(12):
                    run = expected_run(group, level, seed, shard_index)
                    manifest_path = run / "run_manifest.json"
                    csv_path = run / "sample_level_raw.csv"
                    if not manifest_path.exists() or not csv_path.exists() or csv_path.stat().st_size <= 1:
                        failures.append(f"missing:{run.relative_to(ROOT)}")
                        continue
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("status") != "COMPLETE" or manifest.get("protocol_version") != audit.PROTOCOL_VERSION:
                        failures.append(f"invalid_manifest:{run.relative_to(ROOT)}")
                        continue
                    frame = pd.read_csv(csv_path)
                    if len(frame) != 2 or frame.global_epoch_index.nunique() != 1:
                        failures.append(f"invalid_csv:{run.relative_to(ROOT)}")
                        continue
                    if set(frame.query_budget) != {audit.QUERY_BUDGET}:
                        failures.append(f"query_budget:{run.relative_to(ROOT)}")
                        continue
                    group_frames.append(frame)
                if len(group_frames) == 12:
                    combined = pd.concat(group_frames, ignore_index=True)
                    if combined.global_epoch_index.nunique() != expected_samples or len(combined) != expected_samples * 2:
                        failures.append(f"coverage:group{group}:{level}:seed{seed}")
                    frames.append(combined)
    validation = {
        "schema_version": "sequence_lime_protocol_consistency_output_validation_v1",
        "status": "PASS" if not failures else "FAIL",
        "groups": list(groups),
        "expected_shards": int(len(list(groups)) * len(audit.LEVELS) * len(audit.SEEDS) * 12),
        "validated_frames": int(len(frames)),
        "failures": failures,
    }
    audit.write_json(ROOT / "output_validation.json", validation)
    if failures:
        raise RuntimeError(f"output validation failed: {failures[:8]}")
    return pd.concat(frames, ignore_index=True)


def enrich_sample_level(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    keys = ["audit_group", "global_epoch_index", "randomization_seed", "target_class_protocol"]
    r0 = data.loc[data.randomization_level == "R0", keys + ["attribution_score", "faithfulness_advantage"]].rename(
        columns={"attribution_score": "r0_attribution", "faithfulness_advantage": "r0_faithfulness_advantage"}
    )
    merged = data.merge(r0, on=keys, validate="many_to_one")
    signed, absolute, top5, top10, dominant_agreement = [], [], [], [], []
    q95_values, p_values, decay, pass_values = [], [], [], []
    null_arrays: dict[str, np.ndarray] = {}
    for row in merged.itertuples():
        observed, abs_observed, q95, null = sample_null(
            row.r0_attribution,
            row.attribution_score,
            row.audit_group,
            row.global_epoch_index,
            row.randomization_level,
            row.randomization_seed,
            row.target_class_protocol,
        )
        key = (
            f"group{row.audit_group}__sample{int(row.global_epoch_index)}__{row.randomization_level}__"
            f"seed{int(row.randomization_seed)}__{row.target_class_protocol}"
        )
        null_arrays[key] = null
        signed.append(observed)
        absolute.append(abs_observed)
        top5.append(top_jaccard(row.r0_attribution, row.attribution_score, 5))
        top10.append(top_jaccard(row.r0_attribution, row.attribution_score, 10))
        dominant_agreement.append(dominant_channel(row.r0_attribution) == dominant_channel(row.attribution_score))
        q95_values.append(q95)
        p_values.append(float((1 + np.sum(null >= abs_observed)) / (PERMUTATIONS + 1)))
        faith_decay = float(abs(row.r0_faithfulness_advantage) - abs(row.faithfulness_advantage))
        decay.append(faith_decay)
        pass_values.append(bool(
            row.randomization_level == "R7"
            and abs_observed <= q95
            and abs(row.faithfulness_advantage) <= max(0.05, 0.5 * abs(row.r0_faithfulness_advantage))
        ))
    merged["signed_spearman"] = signed
    merged["absolute_spearman"] = absolute
    merged["top_5_jaccard"] = top5
    merged["top_10_jaccard"] = top10
    merged["dominant_channel_agreement"] = dominant_agreement
    merged["permutation_null_q95"] = q95_values
    merged["empirical_p_value"] = p_values
    merged["r7_faithfulness_decay"] = decay
    merged["sample_randomization_pass"] = pass_values
    return merged, null_arrays


def group_null(group: pd.DataFrame, *identity: object) -> tuple[float, float, np.ndarray]:
    pairs = [(normalized_ranks(row.r0_attribution), normalized_ranks(row.attribution_score)) for row in group.itertuples()]
    observed = float(np.mean([abs(float(np.dot(first, second))) for first, second in pairs]))
    rng = np.random.default_rng(deterministic_seed(*identity))
    null = np.empty(PERMUTATIONS, dtype=np.float32)
    for index in range(PERMUTATIONS):
        null[index] = np.mean([abs(float(np.dot(first, rng.permutation(second)))) for first, second in pairs])
    return observed, float(np.quantile(null, 0.95)), null


def summarize_randomization(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    r7 = sample.loc[sample.randomization_level == "R7"].copy()
    rows, arrays = [], {}
    for (group_name, protocol, seed), frame in r7.groupby(
        ["audit_group", "target_class_protocol", "randomization_seed"], sort=True
    ):
        observed, q95, null = group_null(frame, group_name, protocol, seed)
        arrays[f"group{group_name}__{protocol}__seed{int(seed)}"] = null
        r0_faith = float(frame.r0_faithfulness_advantage.mean())
        r7_faith = float(frame.faithfulness_advantage.mean())
        faith_decay_pass = bool(abs(r7_faith) <= max(0.05, 0.5 * abs(r0_faith)))
        rows.append({
            "audit_group": group_name,
            "target_class_protocol": protocol,
            "randomization_seed": int(seed),
            "sample_count": int(len(frame)),
            "signed_spearman_mean": float(frame.signed_spearman.mean()),
            "absolute_spearman_mean": observed,
            "top_5_jaccard_mean": float(frame.top_5_jaccard.mean()),
            "top_10_jaccard_mean": float(frame.top_10_jaccard.mean()),
            "dominant_channel_agreement": float(frame.dominant_channel_agreement.mean()),
            "permutation_count": PERMUTATIONS,
            "permutation_null_mean": float(np.mean(null)),
            "permutation_null_q95": q95,
            "empirical_p_value": float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)),
            "null_compatible": bool(observed <= q95),
            "r0_faithfulness_advantage": r0_faith,
            "r7_faithfulness_advantage": r7_faith,
            "faithfulness_advantage_decay": float(abs(r0_faith) - abs(r7_faith)),
            "faithfulness_decay_pass": faith_decay_pass,
            "seed_pass": bool(observed <= q95 and faith_decay_pass),
        })
    summary = pd.DataFrame(rows)
    summary["protocol_seed_pass_count"] = summary.groupby(
        ["audit_group", "target_class_protocol"]
    ).seed_pass.transform("sum").astype(int)
    summary["protocol_pass"] = summary.protocol_seed_pass_count >= 2
    return summary, arrays


def bootstrap_difference(first: np.ndarray, second: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(2000, dtype=np.float64)
    for index in range(len(values)):
        values[index] = float(np.mean(rng.choice(first, len(first), replace=True)) - np.mean(rng.choice(second, len(second), replace=True)))
    return float(np.mean(first) - np.mean(second)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def compare_a_c(summary: pd.DataFrame, sample: pd.DataFrame) -> Mapping[str, object]:
    rows = []
    materially_different = False
    r7 = sample.loc[sample.randomization_level == "R7"]
    for protocol in audit.PROTOCOLS:
        first = r7.loc[(r7.audit_group == "A") & (r7.target_class_protocol == protocol), "absolute_spearman"].to_numpy(float)
        second = r7.loc[(r7.audit_group == "C") & (r7.target_class_protocol == protocol), "absolute_spearman"].to_numpy(float)
        difference, low, high = bootstrap_difference(first, second, deterministic_seed("A-C", protocol))
        a_pass = int(summary.loc[(summary.audit_group == "A") & (summary.target_class_protocol == protocol), "protocol_seed_pass_count"].iloc[0])
        c_pass = int(summary.loc[(summary.audit_group == "C") & (summary.target_class_protocol == protocol), "protocol_seed_pass_count"].iloc[0])
        distinct = bool(low > 0 or high < 0 or a_pass != c_pass)
        materially_different = materially_different or distinct
        rows.append({
            "target_class_protocol": protocol,
            "a_minus_c_absolute_spearman": difference,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "a_seed_pass_count": a_pass,
            "c_seed_pass_count": c_pass,
            "material_difference": distinct,
        })
    pd.DataFrame(rows).to_csv(ROOT / "group_a_vs_c_bootstrap.csv", index=False)
    return {
        "material_a_c_difference": materially_different,
        "group_b_required_by_protocol": materially_different,
        "criterion": "bootstrap CI excludes zero or protocol seed-pass count differs",
    }


def group_endpoint_summaries(sample: pd.DataFrame) -> None:
    sample.groupby(
        ["audit_group", "randomization_level", "target_class_protocol"], as_index=False
    ).agg(
        sample_count=("global_epoch_index", "nunique"),
        weighted_r2=("surrogate_weighted_r2", "mean"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        effective_sample_size=("effective_sample_size", "mean"),
        local_response_standard_deviation=("local_response_standard_deviation", "mean"),
        local_response_range=("local_response_range", "mean"),
    ).to_csv(ROOT / "group_level_fidelity_summary.csv", index=False)
    sample.groupby(
        ["audit_group", "randomization_level", "target_class_protocol"], as_index=False
    ).agg(
        sample_count=("global_epoch_index", "nunique"),
        deletion_auc=("deletion_auc", "mean"),
        insertion_auc=("insertion_auc", "mean"),
        faithfulness_advantage=("faithfulness_advantage", "mean"),
        r7_faithfulness_decay=("r7_faithfulness_decay", "mean"),
    ).to_csv(ROOT / "group_level_faithfulness_summary.csv", index=False)


def aggregate(groups: list[str]) -> None:
    data = load_groups(groups)
    sample, sample_nulls = enrich_sample_level(data)
    randomization, group_nulls = summarize_randomization(sample)
    sample.to_csv(ROOT / "sample_level_protocol_consistency.csv", index=False)
    randomization.to_csv(ROOT / "group_level_randomization_summary.csv", index=False)
    np.savez_compressed(ROOT / "permutation_nulls_1000.npz", **sample_nulls, **group_nulls)
    group_endpoint_summaries(sample)
    decision = {
        "schema_version": "sequence_lime_protocol_consistency_interim_v1",
        "status": "UNIFIED_RERUN_AGGREGATED",
        "groups_completed": groups,
        "protocol_version": audit.PROTOCOL_VERSION,
        "query_budget": audit.QUERY_BUDGET,
        "all_formal_csv_nonempty": True,
    }
    if set(groups) >= {"A", "C"}:
        decision.update(compare_a_c(randomization, sample))
    audit.write_json(ROOT / "interim_decision.json", decision)
    for name in (
        "sample_level_protocol_consistency.csv",
        "group_level_randomization_summary.csv",
        "group_level_fidelity_summary.csv",
        "group_level_faithfulness_summary.csv",
    ):
        if (ROOT / name).stat().st_size <= 1:
            raise RuntimeError(f"empty formal CSV: {name}")
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=("A", "B", "C"), required=True)
    args = parser.parse_args()
    aggregate(list(args.groups))


if __name__ == "__main__":
    main()
