#!/usr/bin/env python
"""Validate and aggregate the 12-sample MF surrogate screening stage."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.xai.surrogate_selection import score_and_gate_candidates  # noqa: E402


ROOT = PROJECT / "outputs/mf_seqlime_surrogate_improvement/isruc_seed42"
LEVELS = ("R0", "R3", "R5", "R7")
SEEDS = (3101, 3102, 3103)
CANDIDATES = (
    "C0_CLASSIC_Y2", "C1_STRUCTURED_M30", "C1_STRUCTURED_M50",
    "C1_STRUCTURED_M70", "C2_BOUNDARY_M50", "C3_WEAK_ZERO_A00",
    "C3_WEAK_ZERO_A10", "C3_WEAK_ZERO_A20", "C3_WEAK_ZERO_A30",
)
PROTOCOLS = ("T0", "T1")
PERMUTATIONS = 1000


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def vector(value) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


def signed_spearman(first, second) -> float:
    value = float(spearmanr(vector(first), vector(second)).statistic)
    return 0.0 if not np.isfinite(value) else value


def top10_jaccard(first, second) -> float:
    a, b = vector(first), vector(second)
    ia = set(np.argsort(-a, kind="stable")[:10])
    ib = set(np.argsort(-b, kind="stable")[:10])
    return float(len(ia & ib) / max(len(ia | ib), 1))


def dominant_channel(value) -> int:
    score = vector(value).reshape(3, 30)
    return int(np.argmax(np.sum(np.abs(score), axis=1)))


def null_for_pairs(pairs, seed: int) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    aggregate = np.zeros(PERMUTATIONS, dtype=np.float64)
    observed = []
    for first_value, second_value in pairs:
        first = rankdata(vector(first_value), method="average")
        second = rankdata(vector(second_value), method="average")
        first = (first - first.mean()) / max(first.std(), 1e-12)
        second = (second - second.mean()) / max(second.std(), 1e-12)
        observed.append(abs(float(np.mean(first * second))))
        permutations = np.asarray(
            [rng.permutation(len(second)) for _ in range(PERMUTATIONS)], dtype=np.int16
        )
        aggregate += np.abs(np.mean(first[None, :] * second[permutations], axis=1))
    aggregate /= max(len(pairs), 1)
    return float(np.mean(observed)), float(np.quantile(aggregate, 0.95)), aggregate


def load_and_validate():
    sample_frames, mask_frames, accounting_frames = [], [], []
    failures = []
    baseline = json.loads((ROOT / "baseline_manifest.json").read_text(encoding="utf-8"))
    checkpoint = str(baseline["checkpoint_sha256"])
    for level, seed, shard in itertools.product(LEVELS, SEEDS, range(1, 13)):
        run_id = f"validation_small_{level}_seed{seed}_shard{shard}of12_disjoint_v2"
        run = ROOT / "runs" / run_id
        paths = {
            "manifest": run / "run_manifest.json",
            "sample": run / "sample_level_metrics.csv",
            "mask": run / "mask_distribution_summary.csv",
            "accounting": run / "query_accounting.csv",
        }
        for name, path in paths.items():
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"{run_id}: missing/empty {name}")
        if any(item.startswith(run_id) for item in failures):
            continue
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        sample = pd.read_csv(paths["sample"])
        masks = pd.read_csv(paths["mask"])
        accounting = pd.read_csv(paths["accounting"])
        if manifest.get("status") != "COMPLETE" or manifest.get("checkpoint_sha256") != checkpoint:
            failures.append(f"{run_id}: status/checkpoint mismatch")
        if len(sample) != 18 or set(sample.candidate) != set(CANDIDATES):
            failures.append(f"{run_id}: sample row/candidate mismatch")
        if set(sample.target_class_protocol) != set(PROTOCOLS):
            failures.append(f"{run_id}: protocol mismatch")
        if set(sample.query_count.astype(int)) != {1530}:
            failures.append(f"{run_id}: query budget mismatch")
        if len(masks) != 9 or set(masks.mask_count.astype(int)) != {1506}:
            failures.append(f"{run_id}: mask row/budget mismatch")
        if (masks.unique_mask_count.astype(int) != 1506).any():
            failures.append(f"{run_id}: duplicate candidate masks")
        if accounting.empty or set(accounting.status) != {"WINDOW_INSTANCE_ACCOUNTING_PASS"}:
            failures.append(f"{run_id}: window accounting failure")
        if sample.cache_identity_digest.isna().any() or sample.model_weight_fingerprint.isna().any():
            failures.append(f"{run_id}: missing model/cache identity")
        if sample.test_data_used_for_selection.astype(bool).any():
            failures.append(f"{run_id}: test selection contamination")
        if sample.human_annotation_used.astype(bool).any() or sample.true_label_used_by_method.astype(bool).any():
            failures.append(f"{run_id}: forbidden method input")
        sample["run_id"] = run_id
        masks["run_id"] = run_id
        accounting["run_id"] = run_id
        sample_frames.append(sample)
        mask_frames.append(masks)
        accounting_frames.append(accounting)
    payload = {
        "schema_version": "mf_surrogate_small_output_validation_v1",
        "status": "PASS" if not failures else "FAIL",
        "expected_shards": 144,
        "validated_shards": len(sample_frames),
        "failures": failures,
    }
    write_json(ROOT / "small_output_validation.json", payload)
    if failures:
        raise RuntimeError(f"small output validation failed: {failures[:5]}")
    return (
        pd.concat(sample_frames, ignore_index=True),
        pd.concat(mask_frames, ignore_index=True),
        pd.concat(accounting_frames, ignore_index=True),
    )


def main() -> None:
    data, masks, accounting = load_and_validate()
    if data.global_epoch_index.nunique() != 12 or len(data) != 2592:
        raise RuntimeError("small validation coverage is incomplete")
    key = ["global_epoch_index", "randomization_seed", "target_class_protocol", "candidate"]
    r0 = data.loc[data.randomization_level == "R0"].copy()
    r7 = data.loc[data.randomization_level == "R7"].copy()
    paired = r0[key + ["attribution_score"]].rename(columns={"attribution_score": "r0_score"}).merge(
        r7[key + ["attribution_score"]].rename(columns={"attribution_score": "r7_score"}),
        on=key, validate="one_to_one",
    )
    paired["signed_spearman"] = [signed_spearman(a, b) for a, b in zip(paired.r0_score, paired.r7_score)]
    paired["absolute_spearman"] = paired.signed_spearman.abs()
    paired["top10_jaccard"] = [top10_jaccard(a, b) for a, b in zip(paired.r0_score, paired.r7_score)]
    paired["dominant_channel_agreement"] = [
        dominant_channel(a) == dominant_channel(b) for a, b in zip(paired.r0_score, paired.r7_score)
    ]

    randomization_rows = []
    null_arrays = {}
    for (candidate, protocol, seed), group in paired.groupby(
        ["candidate", "target_class_protocol", "randomization_seed"], sort=True
    ):
        deterministic = int(hashlib.sha256(f"{candidate}|{protocol}|{seed}".encode()).hexdigest()[:12], 16)
        observed, q95, null = null_for_pairs(list(zip(group.r0_score, group.r7_score)), deterministic)
        null_arrays[(candidate, protocol, int(seed))] = null
        randomization_rows.append({
            "candidate": candidate,
            "target_class_protocol": protocol,
            "randomization_seed": int(seed),
            "sample_count": int(len(group)),
            "signed_spearman_mean": float(group.signed_spearman.mean()),
            "absolute_spearman_mean": observed,
            "top10_jaccard_mean": float(group.top10_jaccard.mean()),
            "dominant_channel_agreement": float(group.dominant_channel_agreement.mean()),
            "permutation_count": PERMUTATIONS,
            "permutation_null_mean": float(np.mean(null)),
            "permutation_null_q95": q95,
            "seed_pass": bool(observed <= q95),
        })
    randomization = pd.DataFrame(randomization_rows)
    randomization.to_csv(ROOT / "validation_randomization_summary.csv", index=False)
    np.savez_compressed(
        ROOT / "validation_randomization_nulls_1000.npz",
        **{f"{candidate}__{protocol}__seed{seed}": values for (candidate, protocol, seed), values in null_arrays.items()},
    )

    stability_rows = []
    for (candidate, sample_id, protocol), group in r0.groupby(
        ["candidate", "global_epoch_index", "target_class_protocol"], sort=True
    ):
        by_seed = {int(row.randomization_seed): row for row in group.itertuples()}
        if set(by_seed) != set(SEEDS):
            raise RuntimeError("R0 mask-seed coverage failure")
        for component, column in (
            ("final", "attribution_score"),
            ("lime", "classic_component_score"),
            ("zero", "zero_component_score"),
        ):
            values = []
            for first_seed, second_seed in ((3101, 3102), (3101, 3103), (3102, 3103)):
                values.append(signed_spearman(
                    getattr(by_seed[first_seed], column), getattr(by_seed[second_seed], column)
                ))
            stability_rows.append({
                "candidate": candidate, "global_epoch_index": int(sample_id),
                "target_class_protocol": protocol, "component": component,
                "mask_seed_spearman": float(np.mean(values)),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(ROOT / "validation_stability_summary.csv", index=False)

    level_endpoints = data.groupby(["randomization_level", "candidate"], as_index=False).agg(
        deletion_auc=("deletion_auc", "mean"), insertion_auc=("insertion_auc", "mean")
    )
    endpoint_lookup = level_endpoints.set_index(["randomization_level", "candidate"])
    summary_rows = []
    for candidate in CANDIDATES:
        original = r0.loc[r0.candidate == candidate]
        rand = randomization.loc[randomization.candidate == candidate]
        base_r0 = endpoint_lookup.loc[("R0", "C0_CLASSIC_Y2")]
        base_r7 = endpoint_lookup.loc[("R7", "C0_CLASSIC_Y2")]
        cand_r0 = endpoint_lookup.loc[("R0", candidate)]
        cand_r7 = endpoint_lookup.loc[("R7", candidate)]
        advantage_r0 = 0.5 * (
            float(base_r0.deletion_auc - cand_r0.deletion_auc)
            + float(cand_r0.insertion_auc - base_r0.insertion_auc)
        )
        advantage_r7 = 0.5 * (
            float(base_r7.deletion_auc - cand_r7.deletion_auc)
            + float(cand_r7.insertion_auc - base_r7.insertion_auc)
        )
        final_stability = stability.loc[
            (stability.candidate == candidate) & (stability.component == "final"),
            "mask_seed_spearman",
        ]
        summary_rows.append({
            "candidate": candidate,
            "validation_weighted_r2": float(original.validation_weighted_r2.mean()),
            "validation_mae": float(original.validation_mae.mean()),
            "validation_rmse": float(original.validation_rmse.mean()),
            "generalization_gap": float(original.generalization_gap.mean()),
            "deletion_auc": float(cand_r0.deletion_auc),
            "insertion_auc": float(cand_r0.insertion_auc),
            "faithfulness_advantage_r0": advantage_r0,
            "faithfulness_advantage_r7": advantage_r7,
            "stability_spearman": float(final_stability.mean()),
            "r7_similarity": float(rand.absolute_spearman_mean.mean()),
            "r7_null_q95": float(rand.permutation_null_q95.mean()),
            "r7_t0_seed_pass_count": int(rand.loc[rand.target_class_protocol == "T0", "seed_pass"].sum()),
            "r7_t1_seed_pass_count": int(rand.loc[rand.target_class_protocol == "T1", "seed_pass"].sum()),
            "query_count": int(original.query_count.iloc[0]),
            "runtime_seconds": float(original.runtime_seconds_amortized.sum()),
        })
    summary = pd.DataFrame(summary_rows)
    scored, decision = score_and_gate_candidates(summary)
    eligible = scored.loc[scored.hard_gate_pass & (scored.candidate != "C0_CLASSIC_Y2")].sort_values(
        ["validation_multiobjective_score", "candidate"], ascending=[False, True]
    )
    shortlist = eligible.head(3).candidate.tolist()
    decision = dict(decision)
    decision.update({
        "schema_version": "mf_surrogate_small_validation_decision_v1",
        "stage": "small_validation_12",
        "sample_count": 12,
        "permutation_null_draws": PERMUTATIONS,
        "qualified_candidate_count": int(len(eligible)),
        "shortlist_maximum": 3,
        "shortlisted_candidates": shortlist,
        "final_validation_allowed": bool(shortlist),
        "test_evaluation_allowed": False,
        "test_data_used": False,
    })
    if not shortlist:
        decision["status"] = "VALIDATION_NO_QUALIFIED_CANDIDATE"
        decision["selected_candidate"] = None
    else:
        decision["status"] = "SMALL_VALIDATION_SHORTLIST_FROZEN"
        decision["selected_candidate"] = None
    scored.to_csv(ROOT / "validation_candidate_metrics.csv", index=False)
    scored.to_csv(ROOT / "surrogate_candidate_summary.csv", index=False)
    masks.to_csv(ROOT / "mask_distribution_summary.csv", index=False)
    runtime = data.groupby(["randomization_level", "candidate"], as_index=False).agg(
        requested_query_budget=("query_count", "first"),
        runtime_seconds=("runtime_seconds_amortized", "sum"),
        peak_rss_kib=("peak_rss_kib", "max"),
    )
    runtime.to_csv(ROOT / "runtime_query_summary.csv", index=False)
    write_json(ROOT / "small_validation_decision.json", decision)
    if not shortlist:
        write_json(ROOT / "frozen_validation_selection.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
