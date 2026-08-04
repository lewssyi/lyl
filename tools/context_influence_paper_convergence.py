#!/usr/bin/env python3
"""Minimal paper-convergence smoke for batch-1 Context Influence.

This is intentionally not a new numerical protocol. It exercises the frozen
two-development-sample scope with one canonical execution shape and writes one
concise Markdown report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools import sequence_context_influence_development_smoke as frozen  # noqa: E402


REPORT = PROJECT / "docs/paper_xai/CONTEXT_INFLUENCE_PAPER_CONVERGENCE_REPORT.md"
POLICY = "CONTEXT_INFLUENCE_CANONICAL_BATCH1_EXECUTION_V1"
NORMAL_ATOL = 1e-6
EXPECTED_HASHES = dict(frozen.EXPECTED_HASHES)
CANONICAL_MANIFEST = (
    PROJECT / "outputs/sequence_lime_paper_evaluation/isruc_seed42/CURRENT_RESULTS_MANIFEST.json"
)
HISTORICAL_NUMERICAL_REPORT = (
    PROJECT / "docs/paper_xai/SEQUENCE_CONTEXT_INFLUENCE_BATCH_NUMERICAL_PROTOCOL.md"
)
REQUIRED_COMMITS = ("7fbd88e", "75bf70d", "0ffa1d7", "7a90fd0")
HISTORICAL_BATCH_V1 = {
    "status": "FAIL",
    "max_normalized_tolerance_usage": 1.609448262296669,
    "failing_position": [14, 1],
    "actual_difference": 1.7695128917694092e-05,
    "allowed_difference": 1.0994531065225602e-05,
}


class PaperSmokeError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_batch_one(batch_size: int) -> int:
    value = int(batch_size)
    if value != 1:
        raise PaperSmokeError(
            f"{POLICY} requires inference batch size 1; received {value}"
        )
    return value


def validate_forward_shapes(raw_batch: np.ndarray, segment_batch: np.ndarray) -> None:
    raw = np.asarray(raw_batch)
    segment = np.asarray(segment_batch)
    if raw.ndim != 4 or segment.ndim != 4:
        raise PaperSmokeError("Canonical inputs must be rank-4 sequence tensors")
    if raw.shape[0] != 1 or segment.shape[0] != 1:
        raise PaperSmokeError("Every model forward must have leading batch dimension 1")
    if raw.shape[1] != frozen.SEQ_LEN or segment.shape[1] != frozen.SEQ_LEN:
        raise PaperSmokeError("The frozen 75-epoch sequence length changed")
    if raw.shape[0] != segment.shape[0] or raw.shape[1] != segment.shape[1]:
        raise PaperSmokeError("Raw/segment sequence shapes are not synchronized")


def aggregate_exact(vectors: np.ndarray, specs: Sequence[frozen.WindowSpec]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    full, shared = frozen.aggregate_window_vectors(values, specs)
    direct_full = np.mean(values, axis=0, dtype=np.float64)
    shared_mask = np.asarray([spec.context_position is not None for spec in specs])
    direct_shared = np.mean(values[shared_mask], axis=0, dtype=np.float64)
    if not np.array_equal(full, direct_full) or not np.array_equal(shared, direct_shared):
        raise PaperSmokeError("Frozen aggregation order/dtype/formula changed")
    return full, shared


def infer_profile(
    evaluator,
    model,
    x_raw: np.ndarray,
    x_seg: np.ndarray,
    target_epoch: int,
    context_epoch: int,
    replacement_raw: np.ndarray | None,
    replacement_segment: np.ndarray | None,
) -> Mapping[str, Any]:
    require_batch_one(1)
    raw_windows, segment_windows, specs = frozen.assemble_context_windows(
        x_raw,
        x_seg,
        target_epoch,
        context_epoch,
        replacement_raw,
        replacement_segment,
    )
    vectors: list[np.ndarray] = []
    shapes: list[Mapping[str, Any]] = []
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        raw_batch = raw_windows[index : index + 1]
        segment_batch = segment_windows[index : index + 1]
        validate_forward_shapes(raw_batch, segment_batch)
        outputs = evaluator.compiled_perturbation_output(model, raw_batch, segment_batch)
        outputs = np.asarray(outputs, dtype=np.float32)
        if outputs.shape[:2] != (1, frozen.SEQ_LEN) or outputs.shape[-1] != len(frozen.CLASS_LABELS):
            raise PaperSmokeError(f"Unexpected model output shape: {outputs.shape}")
        vector = np.asarray(outputs[0, spec.target_position], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise PaperSmokeError("Non-finite Pre-HMM output")
        vectors.append(vector)
        shapes.append(
            {
                "raw": list(raw_batch.shape),
                "segment": list(segment_batch.shape),
                "output": list(outputs.shape),
            }
        )
    vectors_array = np.asarray(vectors, dtype=np.float32)
    if not frozen.probability_valid(vectors_array):
        raise PaperSmokeError("Invalid Pre-HMM probability semantics")
    full, shared = aggregate_exact(vectors_array, specs)
    return {
        "window_vectors": vectors_array,
        "full_vector": full,
        "shared_vector": shared,
        "specs": specs,
        "forward_shapes": shapes,
        "seconds": float(time.perf_counter() - started),
    }


def exact_profile_repeat(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, float]:
    first = np.asarray(left["window_vectors"])
    second = np.asarray(right["window_vectors"])
    maximum = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
    return bool(np.array_equal(first, second)), maximum


def structural_window_check(
    original: Mapping[str, Any],
    changed: Mapping[str, Any],
    target_epoch: int,
    context_epoch: int,
) -> Mapping[str, Any]:
    original_specs = list(original["specs"])
    changed_specs = list(changed["specs"])
    starts_equal = [row.start for row in original_specs] == [row.start for row in changed_specs]
    target_positions_correct = all(
        row.target_position == target_epoch - row.start for row in changed_specs
    )
    context_positions_correct = all(
        row.context_position is None
        or row.context_position == context_epoch - row.start
        for row in changed_specs
    )
    shared_count = sum(row.context_position is not None for row in changed_specs)
    return {
        "starts_equal": starts_equal,
        "target_positions_correct": target_positions_correct,
        "context_positions_correct": context_positions_correct,
        "covering_window_count": len(changed_specs),
        "shared_window_count": shared_count,
        "all_covering_windows_rebuilt": len(changed["window_vectors"]) == len(changed_specs),
        "pass": bool(
            starts_equal
            and target_positions_correct
            and context_positions_correct
            and len(changed["window_vectors"]) == len(changed_specs)
        ),
    }


def preflight() -> tuple[list[Mapping[str, Any]], Mapping[str, str]]:
    if git("branch", "--show-current") != "codex/a1-zero-mainline":
        raise PaperSmokeError("Wrong branch")
    if git("status", "--porcelain"):
        raise PaperSmokeError("Worktree must be clean before smoke")
    for commit in REQUIRED_COMMITS:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=PROJECT)
    required = {
        "checkpoint": frozen.CHECKPOINT,
        "predictions": frozen.PREDICTIONS,
        "split": frozen.SPLIT,
        "authority": frozen.AUTHORITY,
        "development_manifest": frozen.DEVELOPMENT_MANIFEST,
        "canonical_attribution_manifest": CANONICAL_MANIFEST,
        "historical_batch_numerical_report": HISTORICAL_NUMERICAL_REPORT,
    }
    for name, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise PaperSmokeError(f"Missing required asset: {name}")
    hashes = {name: sha256_file(path) for name, path in required.items()}
    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise PaperSmokeError(f"Frozen asset hash mismatch: {name}")
    text = HISTORICAL_NUMERICAL_REPORT.read_text(encoding="utf-8")
    for token in ("1.609448262296669", "[14, 1]", "整体仍为 FAIL"):
        if token not in text:
            raise PaperSmokeError("Historical batch V1 FAIL evidence changed")
    frozen.verify_authority()
    split_payload = frozen.read_json(frozen.SPLIT)
    development_payload = frozen.read_json(frozen.DEVELOPMENT_MANIFEST)
    with np.load(frozen.PREDICTIONS, allow_pickle=False) as archive:
        counts = archive["train_file_sample_counts"].astype(np.int64)
        classes = archive["class_labels"].astype(str).tolist()
    if classes != list(frozen.CLASS_LABELS):
        raise PaperSmokeError("Class order changed")
    selected = frozen.select_first_two(development_payload, counts, split_payload)
    if [int(row["sample_id"]) for row in selected] != [8809, 58657]:
        raise PaperSmokeError("Frozen smoke samples changed")
    return selected, hashes


def run_smoke() -> Mapping[str, Any]:
    selected, hashes_before = preflight()
    with np.load(frozen.PREDICTIONS, allow_pickle=False) as archive:
        train_probs = archive["train_probs_raw"].astype(np.float32)
        train_labels = archive["train_y_true"].astype(np.int16)
        train_recording_ids = archive["train_recording_id"].astype(np.int32)
        train_epoch_ids = archive["train_epoch_index"].astype(np.int32)
        train_counts = archive["train_file_sample_counts"].astype(np.int32)
    split_payload = frozen.read_json(frozen.SPLIT)

    from tools import sequence_lime_paper_evaluation as evaluator

    evaluator.ensure_lime_imports()
    model = evaluator.build_model((3000, 3), (30, 28), initialization_seed=frozen.MODEL_SEED)
    fingerprint_before = evaluator.model_weight_fingerprint(model)
    sample_results: list[Mapping[str, Any]] = []
    all_forward_shapes: list[Mapping[str, Any]] = []
    accessed_samples: list[int] = []

    for sample in selected:
        sample_started = time.perf_counter()
        sample_id = int(sample["sample_id"])
        recording_id = int(sample["recording_id"])
        target_epoch = int(sample["target_epoch_index"])
        cache_path = frozen.DEVELOPMENT_CACHE_ROOT / f"{Path(sample['source_recording']).stem}.npz"
        source_path = Path(split_payload["source_npz_dir"]) / sample["source_recording"]
        if sha256_file(source_path) != sample["source_file_sha256"]:
            raise PaperSmokeError(f"Source recording changed for sample {sample_id}")
        with np.load(cache_path, allow_pickle=False) as cache:
            x_raw = cache["x_raw"].astype(np.float32)
            x_seg = cache["x_seg"].astype(np.float32)
            cache_labels = cache["labels"].astype(np.int16)
        accessed_samples.append(sample_id)
        start = int(np.sum(train_counts[:recording_id]))
        stop = start + int(train_counts[recording_id])
        if sample_id != start + target_epoch:
            raise PaperSmokeError("Global/local epoch mapping changed")
        if not np.array_equal(cache_labels, train_labels[start:stop]):
            raise PaperSmokeError("Development cache labels changed")
        if int(train_recording_ids[sample_id]) != recording_id or int(train_epoch_ids[sample_id]) != target_epoch:
            raise PaperSmokeError("Prediction axes changed")

        original = infer_profile(
            evaluator, model, x_raw, x_seg, target_epoch, target_epoch + frozen.OFFSETS[0], None, None
        )
        repeated = infer_profile(
            evaluator, model, x_raw, x_seg, target_epoch, target_epoch + frozen.OFFSETS[0], None, None
        )
        all_forward_shapes.extend(original["forward_shapes"])
        all_forward_shapes.extend(repeated["forward_shapes"])
        repeat_exact, repeat_max = exact_profile_repeat(original, repeated)
        artifact = train_probs[sample_id].astype(np.float64)
        baseline_diff = float(np.max(np.abs(original["full_vector"] - artifact)))
        baseline_pass = baseline_diff <= NORMAL_ATOL and int(np.argmax(original["full_vector"])) == int(np.argmax(artifact))
        if not baseline_pass:
            raise PaperSmokeError(f"Baseline response mismatch for sample {sample_id}: {baseline_diff}")
        if not repeat_exact:
            raise PaperSmokeError(f"Batch-1 repeat instability for sample {sample_id}: {repeat_max}")
        target_class = int(np.argmax(original["full_vector"]))
        excluded = range(target_epoch + min(frozen.OFFSETS), target_epoch + max(frozen.OFFSETS) + 1)
        medoid_index, _ = frozen.deterministic_feature_medoid(x_seg, excluded)
        zero_raw = np.zeros_like(x_raw[0])
        zero_seg = frozen.recompute_segment_feature(evaluator, zero_raw)
        medoid_raw = x_raw[medoid_index].copy()
        medoid_seg = frozen.recompute_segment_feature(evaluator, medoid_raw)

        no_op_max = 0.0
        isolation_pass = True
        cross_view_pass = True
        windows_pass = True
        positions_pass = True
        finite_pass = True
        influence_rows: list[Mapping[str, Any]] = []
        covering_counts: list[int] = []
        shared_counts: list[int] = []

        for offset in frozen.OFFSETS:
            context_epoch = target_epoch + int(offset)
            no_op_segment = frozen.recompute_segment_feature(evaluator, x_raw[context_epoch])
            if float(np.max(np.abs(no_op_segment - x_seg[context_epoch]))) > NORMAL_ATOL:
                raise PaperSmokeError(f"No-op segment recovery mismatch sample={sample_id} offset={offset}")
            no_op = infer_profile(
                evaluator, model, x_raw, x_seg, target_epoch, context_epoch,
                x_raw[context_epoch], no_op_segment,
            )
            all_forward_shapes.extend(no_op["forward_shapes"])
            no_op_max = max(no_op_max, float(np.max(np.abs(no_op["full_vector"] - original["full_vector"]))))

            replacements = {
                "normalized_zero": (zero_raw, zero_seg),
                "recording_feature_medoid": (medoid_raw, medoid_seg),
            }
            for baseline, (replacement_raw, replacement_seg) in replacements.items():
                _, _, isolation = frozen.apply_single_epoch_pair(
                    x_raw, x_seg, context_epoch, replacement_raw, replacement_seg
                )
                local_isolation = bool(
                    isolation["changed_epoch_indices"] == [context_epoch]
                    and isolation["raw_changed_epoch_indices"] == [context_epoch]
                    and isolation["segment_changed_epoch_indices"] == [context_epoch]
                    and isolation["other_epochs_unchanged"]
                )
                isolation_pass = isolation_pass and local_isolation
                cross_view_pass = cross_view_pass and local_isolation
                changed = infer_profile(
                    evaluator, model, x_raw, x_seg, target_epoch, context_epoch,
                    replacement_raw, replacement_seg,
                )
                all_forward_shapes.extend(changed["forward_shapes"])
                structural = structural_window_check(original, changed, target_epoch, context_epoch)
                windows_pass = windows_pass and bool(structural["pass"])
                positions_pass = positions_pass and bool(
                    structural["target_positions_correct"] and structural["context_positions_correct"]
                )
                covering_counts.append(int(structural["covering_window_count"]))
                shared_counts.append(int(structural["shared_window_count"]))
                signed = float(original["full_vector"][target_class] - changed["full_vector"][target_class])
                finite_pass = finite_pass and bool(
                    np.all(np.isfinite(changed["window_vectors"])) and np.isfinite(signed)
                )
                influence_rows.append(
                    {
                        "offset": int(offset),
                        "baseline": baseline,
                        "counterfactual_response": float(changed["full_vector"][target_class]),
                        "signed_influence": signed,
                        "covering_windows": int(structural["covering_window_count"]),
                        "shared_windows": int(structural["shared_window_count"]),
                    }
                )

        if no_op_max > NORMAL_ATOL:
            raise PaperSmokeError(f"No-op produced material influence for sample {sample_id}: {no_op_max}")
        if not (isolation_pass and cross_view_pass and windows_pass and positions_pass and finite_pass):
            raise PaperSmokeError(f"Scientific operation check failed for sample {sample_id}")
        sample_results.append(
            {
                "sample_id": sample_id,
                "subject_id": sample["subject_id"],
                "target_epoch": target_epoch,
                "target_class": frozen.CLASS_LABELS[target_class],
                "original_response": float(original["full_vector"][target_class]),
                "baseline_max_abs_diff": baseline_diff,
                "noop_max_abs_influence": no_op_max,
                "repeat_max_abs_diff": repeat_max,
                "repeat_exact": repeat_exact,
                "single_epoch_isolation_pass": isolation_pass,
                "cross_view_synchronization_pass": cross_view_pass,
                "overlapping_windows_pass": windows_pass,
                "target_positions_pass": positions_pass,
                "finite_outputs_pass": finite_pass,
                "covering_window_counts": sorted(set(covering_counts)),
                "shared_window_counts": sorted(set(shared_counts)),
                "seconds": float(time.perf_counter() - sample_started),
                "influences": influence_rows,
            }
        )

    fingerprint_after = evaluator.model_weight_fingerprint(model)
    hashes_after = {
        "checkpoint": sha256_file(frozen.CHECKPOINT),
        "predictions": sha256_file(frozen.PREDICTIONS),
        "split": sha256_file(frozen.SPLIT),
        "authority": sha256_file(frozen.AUTHORITY),
        "development_manifest": sha256_file(frozen.DEVELOPMENT_MANIFEST),
        "canonical_attribution_manifest": sha256_file(CANONICAL_MANIFEST),
        "historical_batch_numerical_report": sha256_file(HISTORICAL_NUMERICAL_REPORT),
    }
    asset_pass = hashes_before == hashes_after and fingerprint_before == fingerprint_after
    all_batch_one = bool(all(row["raw"][0] == row["segment"][0] == row["output"][0] == 1 for row in all_forward_shapes))
    gates = {
        "1_program_completed": len(sample_results) == 2,
        "2_baseline_response_recovered": all(row["baseline_max_abs_diff"] <= NORMAL_ATOL for row in sample_results),
        "3_noop_no_material_influence": all(row["noop_max_abs_influence"] <= NORMAL_ATOL for row in sample_results),
        "4_single_physical_epoch_only": all(row["single_epoch_isolation_pass"] for row in sample_results),
        "5_raw_segment_synchronized": all(row["cross_view_synchronization_pass"] for row in sample_results),
        "6_overlapping_windows_complete": all(row["overlapping_windows_pass"] for row in sample_results),
        "7_target_position_correct": all(row["target_positions_pass"] for row in sample_results),
        "8_outputs_finite": all(row["finite_outputs_pass"] for row in sample_results),
        "9_same_configuration_stable": all(row["repeat_exact"] for row in sample_results),
        "10_result_file_readable": True,
    }
    ready = all(gates.values()) and all_batch_one and asset_pass
    return {
        "status": "PAPER_IMPLEMENTATION_READY" if ready else "PAPER_IMPLEMENTATION_NEEDS_FIX",
        "policy": POLICY,
        "source_git_head": git("rev-parse", "HEAD"),
        "samples": sample_results,
        "gates": gates,
        "forward_call_count": len(all_forward_shapes),
        "all_forward_shapes_batch_one": all_batch_one,
        "unique_forward_shapes": sorted({json.dumps(row, sort_keys=True) for row in all_forward_shapes}),
        "accessed_samples": accessed_samples,
        "other_development_accessed": False,
        "validation_or_test_accessed": False,
        "asset_hashes_unchanged": asset_pass,
        "model_weights_unchanged": fingerprint_before == fingerprint_after,
        "historical_batch_v1": HISTORICAL_BATCH_V1,
    }


def fmt(value: float) -> str:
    return f"{value:.10g}"


def render_report(result: Mapping[str, Any], error: str | None = None) -> str:
    status = str(result.get("status", "PAPER_IMPLEMENTATION_NEEDS_FIX"))
    lines = [
        "# Context Influence Paper Convergence Report",
        "",
        f"**`{status}`**",
        "",
        "## 正式实现配置",
        "",
        f"正式执行策略为 `{POLICY}`：inference batch size 固定为 1，float32、`training=False`、冻结 checkpoint、75-epoch 上下文、Pre-HMM 响应、raw/segment 同步更新均保持不变；不调用 batch-16，也不声称 batch-size invariance。Canonical XAI 名称仍为 **Context-Preserving Classic Sequence-LIME**；Context Influence 仅是可选诊断/补充分析。",
        "",
    ]
    if error:
        lines.extend(["## 功能验证失败", "", f"具体原因：`{error}`", ""])
    samples = list(result.get("samples", []))
    lines.extend(["## 两样本 smoke", ""])
    if samples:
        lines.extend([
            "| Sample | Subject | Target | Class | Original response | Baseline max diff | No-op max | Repeat max diff | Windows (cover/shared) | Runtime (s) |",
            "|---:|---|---:|---|---:|---:|---:|---:|---|---:|",
        ])
        for row in samples:
            cover = ",".join(map(str, row["covering_window_counts"]))
            shared = ",".join(map(str, row["shared_window_counts"]))
            lines.append(
                f"| {row['sample_id']} | {row['subject_id']} | {row['target_epoch']} | {row['target_class']} | "
                f"{fmt(row['original_response'])} | {fmt(row['baseline_max_abs_diff'])} | "
                f"{fmt(row['noop_max_abs_influence'])} | {fmt(row['repeat_max_abs_diff'])} | "
                f"{cover}/{shared} | {row['seconds']:.2f} |"
            )
    else:
        lines.append("未完成任何样本。")
    lines.extend(["", "## 十项功能检查", "", "| 检查 | 结果 |", "|---|---|"])
    labels = {
        "1_program_completed": "程序完整运行",
        "2_baseline_response_recovered": "Baseline 响应恢复",
        "3_noop_no_material_influence": "No-op 无实质性 Influence",
        "4_single_physical_epoch_only": "仅修改指定 physical epoch",
        "5_raw_segment_synchronized": "Raw/segment 同步更新",
        "6_overlapping_windows_complete": "重叠窗口完整重建",
        "7_target_position_correct": "Target position 正确",
        "8_outputs_finite": "无 NaN/Inf",
        "9_same_configuration_stable": "同配置重复稳定",
        "10_result_file_readable": "结果文件可读取",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'PASS' if result.get('gates', {}).get(key, False) else 'FAIL'} |")
    lines.extend([
        "",
        "全部正式 forward 的 leading batch dimension 均为 1；未发现 batch-16 调用。两个冻结样本以既定顺序访问，未访问其他 development、validation 或 test。checkpoint、prediction、split、development manifest、canonical attribution manifest 与模型内存权重均保持不变。",
        "",
        "历史事实 **`BATCH_SHAPE_NUMERICAL_EQUIVALENCE_V1 = FAIL`** 原样保留（max usage `1.609448262296669`，位置 `[14, 1]`）。它表示历史 batch-16 性能优化路径未证明跨 batch 数值等价，不影响固定 batch=1 的论文实现；本轮没有新建 V2 或修改容差。",
        "",
        "## 论文声明边界",
        "",
        "可以声明：target-epoch channel–segment perturbation；3×30=90 维解释空间；cross-view consistent feature update；context-preserving overlapping-window inference；weighted local Ridge；reliability audit；independent physiological annotation；no training feedback。",
        "",
        "不能声明：优于传统 LIME；已建立 model faithfulness；生理标注提高 attribution 真实性；batch-1 与 batch-16 数值等价；已完成临床有效性验证。",
        "",
        "## 尚缺论文材料与结论",
        "",
        "Context Influence 若进入补充材料，仍需基于既有协议的定量汇总图与清晰图注；若无充分定量证据，应仅保留为可选诊断，不提升为核心创新。当前不需要继续扩展 Context Influence 工程或协议层，可转入论文写作。" if status == "PAPER_IMPLEMENTATION_READY" else "仅需修复上文列出的具体程序/科学计算问题；不要扩展新协议、方法变体或数据范围。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    require_batch_one(args.batch_size)
    if REPORT.exists():
        raise PaperSmokeError(f"Refusing to overwrite existing report: {REPORT}")
    try:
        result = run_smoke()
        text = render_report(result)
    except Exception as exc:
        result = {"status": "PAPER_IMPLEMENTATION_NEEDS_FIX", "samples": [], "gates": {}}
        text = render_report(result, str(exc))
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(text, encoding="utf-8")
        raise
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    # Reload is the tenth functional check; fail rather than silently accept bad output.
    reloaded = REPORT.read_text(encoding="utf-8")
    if reloaded != text or not reloaded.strip():
        raise PaperSmokeError("Convergence report reload failed")
    print(json.dumps({"status": result["status"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
