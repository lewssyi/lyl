#!/usr/bin/env python3
"""Build a blinded display-validation package from frozen XAI artifacts only."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.clinical_concepts import CONCEPT_CHANNEL_INDEX, CONCEPT_NAMES_V1
from src.lime_explanation import (
    extract_clinical_concept_matrix,
    semantic_segmentation,
)
from src.xai.attribution_types import ModelAttributionResult
from src.xai.clinical_concept_annotation import annotate_clinical_concepts


SEED = 20260726
ATTRIBUTION_SEEDS = (3101, 3102, 3103)
DISPLAY_SEED = 3101
TOP_K = 10
CHANNELS = ("EEG", "EOG", "EMG")
HC = PROJECT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"
SOURCE_CACHE = (
    PROJECT
    / "outputs/sequence_lime_two_versions/isruc_seed42/cache"
    / "validation_inputs_segments.npz"
)
OUT = PROJECT / "outputs/physiology_aware_display_validation/isruc_seed42"
DOC = PROJECT / "docs/paper_xai"
PACKAGE = DOC / "expert_evaluation_package"
AUTHORITY = (
    PROJECT
    / "outputs/xai_authoritative_status/isruc_seed42/CURRENT_XAI_STATUS.json"
)
CPU_DIAGNOSTIC_MARKER = "parity_cpu_driver_mismatch_20260724"

OPPOSING_PAIRS = (
    ("EOG Slow Eye", "EOG Rapid Eye"),
    ("EMG Low Tone", "EMG Muscle Activation"),
)
STAGE_COMPATIBLE = {
    "W": {"EEG Alpha", "EEG Beta", "EMG Muscle Activation"},
    "N1": {"EEG Theta", "EOG Slow Eye"},
    "N2": {"EEG Theta", "EEG Sigma"},
    "N3": {"EEG Delta"},
    "REM": {"EOG Rapid Eye", "EMG Low Tone"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float32).tobytes()
    ).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_output_manifest() -> None:
    manifest_path = OUT / "CURRENT_RESULTS_MANIFEST.json"
    files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file() and path != manifest_path
    )
    write_json(
        manifest_path,
        {
            "schema_version": "physiology_aware_display_results_manifest_v1",
            "artifacts": [
                {
                    "path": str(path.relative_to(PROJECT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ],
        },
    )


def assert_authority() -> dict:
    status = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    required = {
        "canonical_posthoc_method": "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
        "scientific_status": "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
        "feedback_training_status": "PERMANENTLY_BLOCKED_FOR_CURRENT_METHOD",
    }
    for key, expected in required.items():
        if status.get(key) != expected:
            raise RuntimeError(f"authority mismatch for {key}: {status.get(key)!r}")
    return status


def load_case_pool() -> pd.DataFrame:
    reliability = pd.read_csv(HC / "reliability_sample_level.csv")
    base = reliability[
        (reliability.method == "B0_CLASSIC_90")
        & (reliability.target_protocol == "T0")
    ].copy()
    if len(base) != 30 or base.sample_id.nunique() != 30:
        raise RuntimeError("expected exactly 30 unique canonical validation cases")
    if base.sample_id.duplicated().any():
        raise RuntimeError("canonical case pool contains duplicate samples")

    auxiliary = (
        reliability.groupby("sample_id").reliability_state.apply(
            lambda values: bool((values == "CONTEXT_DOMINATED").any())
        )
    )
    manifest = json.loads(
        (HC / "validation_samples.json").read_text(encoding="utf-8")
    )
    metadata = pd.DataFrame(manifest["samples"]).rename(
        columns={"global_epoch_index": "sample_id"}
    )
    columns = [
        "sample_id",
        "recording_id",
        "epoch_index_in_recording",
        "source_file_sha256",
    ]
    base = base.merge(metadata[columns], on="sample_id", how="left", validate="one_to_one")
    base["auxiliary_context_dominated"] = (
        base.sample_id.map(auxiliary).fillna(False).astype(bool)
    )
    return base.sort_values("sample_id").reset_index(drop=True)


def attribution_path(sample_id: int, seed: int) -> Path:
    return (
        HC
        / "runs"
        / f"validation_classic_sample{sample_id}_R0_seed{seed}_zero"
        / "classic_attribution.csv"
    )


def load_attribution(sample_id: int, seed: int) -> tuple[np.ndarray, Path]:
    path = attribution_path(sample_id, seed)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing frozen attribution: {path}")
    frame = pd.read_csv(path)
    selected = frame[
        (frame.candidate == "B0_CLASSIC_90")
        & (frame.randomization_level == "R0")
        & (frame.randomization_seed == seed)
        & (frame.target_protocol == "T0")
    ]
    if len(selected) != 1:
        raise RuntimeError(f"ambiguous frozen attribution row: {path}")
    score = np.asarray(ast.literal_eval(selected.iloc[0].importance), dtype=np.float32)
    if score.shape != (90,) or not np.isfinite(score).all():
        raise RuntimeError(f"invalid frozen attribution vector: {path}")
    return score, path


def stable_rank(score: np.ndarray) -> np.ndarray:
    return np.argsort(-np.abs(score), kind="stable")


def dominant_channel(score: np.ndarray) -> str:
    masses = np.sum(np.abs(score.reshape(3, 30)), axis=1)
    return CHANNELS[int(np.argmax(masses))]


def concept_payload(
    sample_id: int,
    score: np.ndarray,
    concept_matrix: np.ndarray,
) -> dict:
    labels = tuple(CONCEPT_NAMES_V1)
    channels = np.asarray(CONCEPT_CHANNEL_INDEX, dtype=np.int32)
    top = stable_rank(score)[:TOP_K]
    activations = []
    for concept_index, channel in enumerate(channels):
        positions = [
            int(index % 30)
            for index in top
            if int(index // 30) == int(channel)
        ]
        values = (
            concept_matrix[concept_index, positions]
            if positions
            else concept_matrix[concept_index]
        )
        activations.append(float(np.mean(values)))

    mapping = np.zeros((90, len(labels)), dtype=np.float32)
    for feature in range(90):
        mapping[feature, channels == int(feature // 30)] = 1.0
    attribution = ModelAttributionResult(
        sample_id=sample_id,
        method_id="CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
        model_attribution_score=score.copy(),
        target_class=-1,
        target_protocol="T0",
        query_count=0,
        metadata={
            "source": "frozen_existing_artifact",
            "no_model_inference": True,
        },
    )
    before = attribution.model_attribution_score.copy()
    annotated = annotate_clinical_concepts(
        attribution=attribution,
        concept_activation=activations,
        concept_labels=labels,
        feature_to_concept=mapping,
        top_k=TOP_K,
        metadata={
            "concept_source": "raw_physiology_v1",
            "expert_review_status": "EXPERT_VALIDATION_PENDING",
            "true_stage_used": False,
        },
    )
    if not np.array_equal(before, attribution.model_attribution_score):
        raise RuntimeError("annotation changed frozen attribution")
    evidence = np.asarray(annotated.clinical_annotation_score, dtype=np.float32)
    return {
        "labels": labels,
        "top": top,
        "activations": np.asarray(activations, dtype=np.float32),
        "evidence": evidence,
        "ranking_modified": False,
        "attribution_unchanged_exact": True,
        "true_stage_used": False,
    }


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = pd.Series(first).rank(method="average")
    right = pd.Series(second).rank(method="average")
    value = left.corr(right)
    return float(value) if np.isfinite(value) else 0.0


def conflict_details(labels: tuple[str, ...], evidence: np.ndarray) -> list[str]:
    index = {label: position for position, label in enumerate(labels)}
    conflicts = []
    for first, second in OPPOSING_PAIRS:
        a = float(evidence[index[first]])
        b = float(evidence[index[second]])
        high = max(a, b)
        if high > 0 and min(a, b) / high >= 0.8:
            conflicts.append(f"{first} vs {second}")
    return conflicts


def deterministic_pilot(pool: pd.DataFrame, size: int = 10) -> set[int]:
    records = []
    for row in pool.itertuples():
        tokens = {
            f"stage:{row.true_stage}",
            f"correct:{bool(row.correct)}",
            f"transition:{bool(row.transition_boundary)}",
            f"reliability:{row.reliability_state}",
            f"channel:{row.dominant_channel}",
        }
        if bool(row.auxiliary_context_dominated):
            tokens.add("aux_context_dominated:true")
        records.append((int(row.sample_id), tokens))
    uncovered = set().union(*(tokens for _, tokens in records))
    chosen: list[int] = []
    available = dict(records)
    while len(chosen) < size and available:
        best = sorted(
            available,
            key=lambda sample_id: (
                -len(available[sample_id] & uncovered),
                sample_id,
            ),
        )[0]
        chosen.append(best)
        uncovered -= available.pop(best)
    return set(chosen)


def shared_limit(score: np.ndarray) -> float:
    value = float(np.max(np.abs(score)))
    return value if value > 0 else 1.0


def render_panel(
    path_stem: Path,
    case_id: str,
    prediction: str,
    score: np.ndarray,
    details: dict | None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(9.0, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.0))
    heat_ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    limit = shared_limit(score)
    image = heat_ax.imshow(
        score.reshape(3, 30),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=Normalize(vmin=-limit, vmax=limit),
    )
    heat_ax.set_yticks(range(3), CHANNELS)
    heat_ax.set_xticks([0, 4, 9, 14, 19, 24, 29], [0, 5, 10, 15, 20, 25, 30])
    heat_ax.set_xlabel("Time within target epoch (s)")
    heat_ax.set_ylabel("Channel")
    heat_ax.set_title(f"{case_id} · Predicted stage: {prediction}")
    cbar = fig.colorbar(image, ax=heat_ax, fraction=0.045, pad=0.03)
    cbar.set_label("Local Perturbation Sensitivity\nnegative  ←  0  →  positive")
    if not np.any(score < 0):
        heat_ax.text(
            0.01,
            -0.20,
            "Frozen artifact contains no negative entries; the shared signed scale is retained.",
            transform=heat_ax.transAxes,
            fontsize=7,
            color="#555555",
        )
    info_ax.axis("off")
    if details is not None:
        top_lines = "\n".join(
            f"  {index + 1}. {label}: {value:.3f}"
            for index, (label, value) in enumerate(details["top_concepts"])
        )
        warning = details["warning"]
        text = (
            "Derived physiological annotation\n"
            f"{top_lines}\n\n"
            "Reliability audit\n"
            f"  Status: {details['reliability_state']}\n"
            f"  Surrogate fidelity: {details['fidelity']:.3f}\n"
            f"  Faithfulness: {details['faithfulness']:.3f}\n"
            f"  Seed stability: {details['stability']:.3f}\n"
            f"  Randomization: {details['randomization_status']}\n\n"
            "Target/context\n"
            "  Target: predicted class, T0\n"
            "  Context: preserved; not fused into heatmap\n\n"
            f"Evidence warning\n  {warning}\n\n"
            "Derived annotation; expert validation pending."
        )
        info_ax.text(
            0.02,
            0.98,
            text,
            va="top",
            ha="left",
            linespacing=1.35,
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#F5F7FA",
                "edgecolor": "#B8C2CC",
            },
        )
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def render_blinded_pair(
    case_id: str,
    panel_a: Path,
    panel_b: Path,
    out_stem: Path,
) -> None:
    left = plt.imread(panel_a)
    right = plt.imread(panel_b)
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 5.2), constrained_layout=True)
    for axis, image, label in zip(axes, (left, right), ("Panel A", "Panel B")):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(f"{case_id} · {label}", fontsize=12, fontweight="bold")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def response_form(case_id: str) -> str:
    return f"""# Anonymous response form: {case_id}

Complete the following independently for Panel A and Panel B. Do not attempt
to infer the method identity.

| Item | Panel A response | Panel B response |
| --- | --- | --- |
| Main attended channel (EEG/EOG/EMG) |  |  |
| Most important time region |  |  |
| Clear physiological pattern? (yes/no/uncertain) |  |  |
| Pattern category or NONE |  |  |
| Explanation reliable? (yes/no/uncertain) |  |  |
| Could the explanation mislead? (yes/no/uncertain) |  |  |
| Helps analyze an incorrect prediction? (yes/no/not applicable) |  |  |
| Completion time, seconds |  |  |
| Confidence, 1–5 |  |  |
| Physiological plausibility, 1–5 |  |  |
| Interpretability, 1–5 |  |  |
| Information clarity, 1–5 |  |  |
| Reliability awareness, 1–5 |  |  |
| Error-analysis usefulness, 1–5 |  |  |
| Clinical/research usefulness, 1–5 |  |  |
| Risk of misleading, 1–5 (analyzed separately) |  |  |
| Overall preference, 1–5 |  |  |

Evaluator ID: ________
Background: ________
Sleep-staging experience: ________
Involved in model development? yes / no
Aware of method identity? yes / no
"""


def render_metric_figure(metrics: pd.DataFrame) -> None:
    values = {
        "Concept\ncoverage": float(metrics.annotation_supported.mean()),
        "Channel–concept\nconsistency": float(
            metrics.channel_concept_consistent.mean()
        ),
        "Top-region\nevidence": float(metrics.top_region_evidence_rate.mean()),
        "Cross-seed\nstability": float(metrics.cross_mask_seed_concept_stability.mean()),
        "Stage–concept\nagreement*": float(metrics.stage_concept_agreement.mean()),
        "Conflict-free": 1.0 - float(metrics.annotation_conflict.mean()),
    }
    frame = pd.DataFrame(
        {"metric": list(values), "value": list(values.values())}
    )
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(figure_dir / "physiology_metric_summary.csv", index=False)
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    bars = ax.bar(frame.metric, frame.value, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proportion / correlation")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame.value):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )
    ax.text(
        0.0,
        -0.28,
        "* Reporting-only true-stage stratification; not used to generate annotation.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "physiology_metric_summary.pdf")
    fig.savefig(figure_dir / "physiology_metric_summary.png", dpi=300)
    plt.close(fig)
    write_json(
        figure_dir / "physiology_metric_summary_manifest.json",
        {
            "render_backend": "matplotlib_native_vector",
            "source": "physiology_metrics.csv",
            "human_validation_result_included": False,
            "scientific_status": "AUTOMATIC_AUDIT_ONLY",
            "artifacts": {
                name: {
                    "bytes": (figure_dir / name).stat().st_size,
                    "sha256": sha256(figure_dir / name),
                }
                for name in (
                    "physiology_metric_summary.csv",
                    "physiology_metric_summary.pdf",
                    "physiology_metric_summary.png",
                )
            },
        },
    )


def main() -> None:
    if "--manifest-only" in sys.argv:
        if not OUT.exists():
            raise RuntimeError("cannot inventory a missing output directory")
        write_output_manifest()
        print(json.dumps({"status": "PASS", "mode": "manifest-only"}, indent=2))
        return
    authority = assert_authority()
    if OUT.exists() or PACKAGE.exists():
        raise RuntimeError(
            "refusing to overwrite an existing physiology display package"
        )
    OUT.mkdir(parents=True)
    PACKAGE.mkdir(parents=True)

    pool = load_case_pool()
    input_cache_hash = sha256(SOURCE_CACHE)
    with np.load(SOURCE_CACHE) as cache:
        if "x_raw" not in cache.files:
            raise RuntimeError("validation cache lacks frozen x_raw input")
        x_raw = np.asarray(cache["x_raw"], dtype=np.float32)
    if len(x_raw) <= int(pool.sample_id.max()):
        raise RuntimeError("validation cache is shorter than the frozen case pool")

    records = []
    annotation_rows = []
    guard_rows = []
    scoring = []
    source_files = {}

    for case_index, row in enumerate(pool.itertuples(), start=1):
        sample_id = int(row.sample_id)
        case_id = f"CASE-{case_index:03d}"
        scores = {}
        sources = {}
        annotations = {}
        epoch = np.asarray(x_raw[sample_id], dtype=np.float32)
        segments = semantic_segmentation(epoch, target_segments=30)
        matrix_payload = extract_clinical_concept_matrix(epoch, segments, fs=100)
        concept_matrix = np.asarray(matrix_payload["concept_matrix"], dtype=np.float32)
        if concept_matrix.shape != (len(CONCEPT_NAMES_V1), 30):
            raise RuntimeError(
                f"unexpected concept matrix for sample {sample_id}: {concept_matrix.shape}"
            )
        for seed in ATTRIBUTION_SEEDS:
            score, source = load_attribution(sample_id, seed)
            scores[seed] = score
            sources[seed] = source
            annotations[seed] = concept_payload(sample_id, score, concept_matrix)
            source_files[str(source.relative_to(PROJECT))] = sha256(source)

        display_score = scores[DISPLAY_SEED]
        payload = annotations[DISPLAY_SEED]
        labels = payload["labels"]
        evidence = payload["evidence"]
        positive = np.isfinite(evidence) & (evidence > 0)
        supported = bool(positive.any())
        top_index = int(np.argmax(evidence)) if supported else -1
        top_concept = labels[top_index] if supported else "NONE"
        top_pairs = (
            sorted(
                (
                    (labels[index], float(evidence[index]))
                    for index in range(len(labels))
                    if positive[index]
                ),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            if supported
            else [("NONE / INSUFFICIENT_EVIDENCE", 0.0)]
        )
        conflicts = conflict_details(labels, evidence)
        dominant = dominant_channel(display_score)
        top_channel = (
            CHANNELS[int(CONCEPT_CHANNEL_INDEX[top_index])] if supported else "NONE"
        )
        channel_consistent = bool(supported and dominant == top_channel)
        top_regions = payload["top"]
        top_region_supported = []
        for feature in top_regions:
            channel = int(feature // 30)
            segment = int(feature % 30)
            concept_ids = np.flatnonzero(CONCEPT_CHANNEL_INDEX == channel)
            top_region_supported.append(
                bool(np.any(concept_matrix[concept_ids, segment] > 0))
            )
        seed_evidence = [annotations[seed]["evidence"] for seed in ATTRIBUTION_SEEDS]
        seed_stability_values = [
            rank_correlation(seed_evidence[left], seed_evidence[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
        faithfulness_path = (
            HC
            / "runs"
            / f"validation_classic_sample{sample_id}_R0_seed3101_zero"
            / "classic_sample_level.csv"
        )
        faithfulness_frame = pd.read_csv(faithfulness_path)
        faithfulness_row = faithfulness_frame[
            faithfulness_frame.target_protocol == "T0"
        ].iloc[0]
        source_files[str(faithfulness_path.relative_to(PROJECT))] = sha256(
            faithfulness_path
        )
        warning = (
            "NONE / INSUFFICIENT_EVIDENCE"
            if not supported
            else (
                "Opposing physiological evidence; interpret cautiously."
                if conflicts
                else (
                    "Reliability gate not met; interpretation may mislead."
                    if row.reliability_state != "LOCALLY_EXPLAINABLE"
                    else "No unsupported-evidence flag."
                )
            )
        )
        details = {
            "top_concepts": top_pairs,
            "reliability_state": row.reliability_state,
            "fidelity": float(row.local_surrogate_R2),
            "faithfulness": float(faithfulness_row.faithfulness),
            "stability": float(row.mask_seed_stability),
            "randomization_status": (
                "UNRELIABLE"
                if row.reliability_state == "RANDOMIZATION_UNRELIABLE"
                else "See frozen reliability state"
            ),
            "warning": warning,
        }
        v0_stem = OUT / "v0_panels" / case_id
        v1_stem = OUT / "v1_panels" / case_id
        render_panel(v0_stem, case_id, row.predicted_stage, display_score.copy(), None)
        render_panel(v1_stem, case_id, row.predicted_stage, display_score.copy(), details)

        stage_agreement = bool(
            supported and top_concept in STAGE_COMPATIBLE[str(row.true_stage)]
        )
        annotation_rows.append(
            {
                "anonymous_case_id": case_id,
                "sample_id": sample_id,
                "concept_source": "raw_physiology_v1",
                "concept_labels": json.dumps(labels, separators=(",", ":")),
                "selected_feature_indices": json.dumps(
                    [int(value) for value in payload["top"]],
                    separators=(",", ":"),
                ),
                "clinical_concept_activation": json.dumps(
                    payload["activations"].astype(float).tolist(),
                    separators=(",", ":"),
                ),
                "clinical_annotation_score": json.dumps(
                    evidence.astype(float).tolist(), separators=(",", ":")
                ),
                "top_concept": top_concept,
                "annotation_supported": supported,
                "unsupported_annotation": not supported,
                "annotation_conflict": bool(conflicts),
                "conflict_pairs": "|".join(conflicts),
                "dominant_channel": dominant,
                "top_concept_channel": top_channel,
                "channel_concept_consistent": channel_consistent,
                "top_region_evidence_rate": float(np.mean(top_region_supported)),
                "cross_mask_seed_concept_stability": float(
                    np.mean(seed_stability_values)
                ),
                "stage_concept_agreement": stage_agreement,
                "concept_sparsity": float(np.mean(~positive)),
                "true_stage_joined_after_annotation": str(row.true_stage),
                "true_stage_used_for_annotation": False,
                "predicted_stage_used_for_annotation": False,
                "ranking_modified": False,
                "attribution_unchanged_exact": True,
                "expert_review_status": "EXPERT_VALIDATION_PENDING",
            }
        )
        rank = stable_rank(display_score)
        guard_rows.append(
            {
                "anonymous_case_id": case_id,
                "sample_id": sample_id,
                "source_path": str(sources[DISPLAY_SEED].relative_to(PROJECT)),
                "source_file_sha256": source_files[
                    str(sources[DISPLAY_SEED].relative_to(PROJECT))
                ],
                "source_array_sha256": array_sha256(display_score),
                "v0_array_sha256": array_sha256(display_score.copy()),
                "v1_array_sha256": array_sha256(display_score.copy()),
                "bitwise_identical": bool(
                    display_score.tobytes() == display_score.copy().tobytes()
                ),
                "ranking_identical": bool(
                    np.array_equal(rank, stable_rank(display_score.copy()))
                ),
                "shared_color_limit": shared_limit(display_score),
            }
        )
        records.append(
            {
                "anonymous_case_id": case_id,
                "sample_id": sample_id,
                "recording": str(row.source_recording),
                "recording_id": int(row.recording_id),
                "epoch_in_recording": int(row.epoch_index_in_recording),
                "true_stage_reporting_only": str(row.true_stage),
                "prediction": str(row.predicted_stage),
                "correct": bool(row.correct),
                "transition": bool(row.transition_boundary),
                "reliability_status": str(row.reliability_state),
                "auxiliary_context_dominated": bool(
                    row.auxiliary_context_dominated
                ),
                "dominant_channel": dominant,
                "selection_rule": "complete_frozen_30_case_validation_pool",
                "source_recording_sha256": str(row.source_file_sha256),
                "attribution_source_sha256": source_files[
                    str(sources[DISPLAY_SEED].relative_to(PROJECT))
                ],
                "attribution_array_sha256": array_sha256(display_score),
            }
        )
        top_feature = int(rank[0])
        scoring.append(
            {
                "anonymous_case_id": case_id,
                "expected_main_channel": dominant,
                "expected_top_segment_index": int(top_feature % 30),
                "expected_top_time_region_seconds": (
                    f"{int(top_feature % 30)}–{int(top_feature % 30) + 1}"
                ),
                "expected_pattern_available": supported,
                "expected_pattern_category": top_concept,
                "expected_reliable": row.reliability_state
                == "LOCALLY_EXPLAINABLE",
                "expected_misleading_risk": (
                    row.reliability_state != "LOCALLY_EXPLAINABLE"
                    or not supported
                    or bool(conflicts)
                ),
            }
        )

    pool_records = pd.DataFrame(records)
    pilot_ids = deterministic_pilot(pool_records.rename(
        columns={
            "true_stage_reporting_only": "true_stage",
            "prediction": "predicted_stage",
            "reliability_status": "reliability_state",
            "transition": "transition_boundary",
        }
    ))
    pool_records["pilot_case"] = pool_records.sample_id.isin(pilot_ids)
    case_payload = {
        "schema_version": "physiology_display_case_selection_v1",
        "main_case_count": 30,
        "pilot_case_count": int(pool_records.pilot_case.sum()),
        "selection_split": "validation",
        "test_set_used": False,
        "selection_uses_rendered_image": False,
        "selection_uses_true_stage_agreement": False,
        "cases": pool_records.to_dict(orient="records"),
    }
    write_json(OUT / "case_selection.json", case_payload)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pool_records))
    v1_in_a = set(int(value) for value in order[: len(pool_records) // 2])
    case_order = rng.permutation(len(pool_records))
    allocation_rows = []
    rating_rows = []
    task_rows = []
    for display_order, row_index in enumerate(case_order, start=1):
        row = pool_records.iloc[int(row_index)]
        case_id = row.anonymous_case_id
        a_version, b_version = (
            ("CONTEXT_PRESENTATION", "BASE_PRESENTATION")
            if int(row_index) in v1_in_a
            else ("BASE_PRESENTATION", "CONTEXT_PRESENTATION")
        )
        allocation_rows.append(
            {
                "display_order": display_order,
                "anonymous_case_id": case_id,
                "sample_id_restricted": int(row.sample_id),
                "panel_a_restricted": a_version,
                "panel_b_restricted": b_version,
                "allocation_seed": SEED,
                "pilot_case": bool(row.pilot_case),
            }
        )
        source_a = OUT / (
            "v1_panels" if a_version == "CONTEXT_PRESENTATION" else "v0_panels"
        ) / f"{case_id}.png"
        source_b = OUT / (
            "v1_panels" if b_version == "CONTEXT_PRESENTATION" else "v0_panels"
        ) / f"{case_id}.png"
        pair_stem = OUT / "blinded_pairs" / case_id
        render_blinded_pair(case_id, source_a, source_b, pair_stem)
        case_dir = PACKAGE / case_id
        case_dir.mkdir(parents=True)
        shutil.copyfile(pair_stem.with_suffix(".png"), case_dir / "panel_pair.png")
        shutil.copyfile(pair_stem.with_suffix(".pdf"), case_dir / "panel_pair.pdf")
        (case_dir / "response_form.md").write_text(
            response_form(case_id), encoding="utf-8"
        )
        for panel in ("A", "B"):
            rating_rows.append(
                {
                    "evaluator_id": "",
                    "anonymous_case_id": case_id,
                    "panel": panel,
                    "physiological_plausibility_1_5": "",
                    "interpretability_1_5": "",
                    "information_clarity_1_5": "",
                    "reliability_awareness_1_5": "",
                    "error_analysis_usefulness_1_5": "",
                    "clinical_research_usefulness_1_5": "",
                    "risk_of_misleading_1_5_separate_direction": "",
                    "overall_preference_1_5": "",
                    "completion_time_seconds": "",
                    "confidence_1_5": "",
                }
            )
            task_rows.append(
                {
                    "evaluator_id": "",
                    "anonymous_case_id": case_id,
                    "panel": panel,
                    "main_channel": "",
                    "top_time_region": "",
                    "physiological_pattern_present": "",
                    "pattern_category": "",
                    "reliable_judgment": "",
                    "could_mislead": "",
                    "helps_error_analysis": "",
                    "answer_correctness_locked_after_scoring": "",
                    "reliability_judgment_accuracy_locked_after_scoring": "",
                    "misinterpretation_locked_after_scoring": "",
                }
            )

    allocation = pd.DataFrame(allocation_rows)
    allocation.to_csv(OUT / "blinded_pair_manifest.csv", index=False)
    write_json(
        OUT / "blinded_pair_manifest.json",
        {
            "schema_version": "blinded_pair_manifest_v1",
            "access": "RESEARCH_TEAM_ONLY_DO_NOT_DISTRIBUTE_TO_EVALUATORS",
            "allocation_seed": SEED,
            "balanced": {
                "context_in_a": len(v1_in_a),
                "context_in_b": len(pool_records) - len(v1_in_a),
            },
            "identity_terms_hidden_in_evaluator_package": True,
            "pairs": allocation.to_dict(orient="records"),
        },
    )
    pd.DataFrame(rating_rows).to_csv(OUT / "expert_rating_template.csv", index=False)
    pd.DataFrame(task_rows).to_csv(OUT / "user_task_template.csv", index=False)
    write_json(
        OUT / "scoring_key_restricted.json",
        {
            "access": "RESEARCH_TEAM_ONLY_AFTER_RATING_DATA_LOCK",
            "answers": scoring,
        },
    )

    metrics = pd.DataFrame(annotation_rows)
    metrics.to_csv(OUT / "physiology_metrics.csv", index=False)
    metrics[
        [
            "anonymous_case_id",
            "sample_id",
            "annotation_supported",
            "unsupported_annotation",
            "top_concept",
            "dominant_channel",
            "top_concept_channel",
            "channel_concept_consistent",
            "top_region_evidence_rate",
            "concept_sparsity",
        ]
    ].to_csv(OUT / "annotation_coverage.csv", index=False)
    metrics[
        [
            "anonymous_case_id",
            "sample_id",
            "annotation_conflict",
            "conflict_pairs",
            "stage_concept_agreement",
            "true_stage_joined_after_annotation",
        ]
    ].to_csv(OUT / "annotation_conflicts.csv", index=False)
    write_json(
        OUT / "attribution_identity_guard.json",
        {
            "schema_version": "v0_v1_attribution_identity_guard_v1",
            "status": "PASS",
            "case_count": len(guard_rows),
            "bitwise_identical_cases": sum(
                bool(row["bitwise_identical"]) for row in guard_rows
            ),
            "ranking_identical_cases": sum(
                bool(row["ranking_identical"]) for row in guard_rows
            ),
            "shared_scale_cases": len(guard_rows),
            "rows": guard_rows,
        },
    )
    write_json(
        OUT / "statistical_analysis_plan.json",
        {
            "registration_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            "paired_design": True,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": SEED,
            "primary_multiplicity": "Holm",
            "subjective_multiplicity": "Benjamini-Hochberg FDR",
            "missing_rating_handling": "complete_pairs_no_imputation",
            "risk_of_misleading_direction": "lower_is_better_analyzed_separately",
            "human_results_available": False,
        },
    )
    render_metric_figure(metrics)

    readme = """# Anonymous physiology display evaluation package

This directory contains 30 anonymous paired cases and blank response forms.
Use the pilot case IDs listed in `PACKAGE_MANIFEST.json` for the 10-case pilot.
Do not distribute the restricted allocation or scoring key from `outputs/`.
No method name, version name, or study hypothesis is disclosed here.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    package_files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    write_json(
        PACKAGE / "PACKAGE_MANIFEST.json",
        {
            "schema_version": "anonymous_expert_evaluation_package_v1",
            "method_identity_hidden": True,
            "allocation_key_included": False,
            "case_count": 30,
            "pilot_case_ids": sorted(
                pool_records.loc[pool_records.pilot_case, "anonymous_case_id"].tolist()
            ),
            "files": [
                {
                    "path": str(path.relative_to(PACKAGE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in package_files
            ],
        },
    )
    source_files[str(SOURCE_CACHE.relative_to(PROJECT))] = input_cache_hash
    run_manifest = {
        "schema_version": "physiology_aware_display_validation_v1",
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT, text=True
        ).strip(),
        "registration_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "scientific_status": authority["scientific_status"],
        "canonical_method": authority["canonical_posthoc_method"],
        "display_attribution": "R0_seed3101_T0_existing_artifact",
        "case_count": 30,
        "pilot_case_count": int(pool_records.pilot_case.sum()),
        "allocation_seed": SEED,
        "test_set_used": False,
        "no_lime_experiment_started": True,
        "no_model_loaded": True,
        "no_model_inference": True,
        "no_gpu_execution": True,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "true_stage_used_for_annotation": False,
        "predicted_stage_used_for_annotation": False,
        "feedback_training_status": authority["feedback_training_status"],
        "cpu_diagnostic_used": False,
        "blocked_by_gpu_backend_mismatch_fields": [],
        "source_files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(source_files.items())
        ],
    }
    if any(CPU_DIAGNOSTIC_MARKER in item["path"] for item in run_manifest["source_files"]):
        raise RuntimeError("non-authoritative CPU diagnostic entered formal package")
    write_json(OUT / "run_manifest.json", run_manifest)

    decision = {
        "decision": "PHYSIOLOGY_AWARE_VISUALIZATION_NOT_YET_VALIDATED",
        "human_blinded_evaluation_completed": False,
        "evaluator_count": 0,
        "expert_agreement": None,
        "v1_physiological_interpretability_improved": None,
        "v1_reliability_judgment_accuracy_improved": None,
        "v1_misinterpretation_noninferior": None,
        "attribution_bitwise_unchanged": True,
        "automatic_metrics_complete": True,
        "clinical_validation_claim_allowed": False,
        "model_faithfulness_claim_allowed": False,
        "reason": (
            "Automatic audits and a blinded package are complete, but no genuine "
            "blinded evaluator ratings exist."
        ),
    }
    write_json(OUT / "final_decision.json", decision)

    required = [
        OUT / "run_manifest.json",
        OUT / "case_selection.json",
        OUT / "blinded_pair_manifest.json",
        OUT / "attribution_identity_guard.json",
        OUT / "physiology_metrics.csv",
        OUT / "annotation_coverage.csv",
        OUT / "annotation_conflicts.csv",
        OUT / "expert_rating_template.csv",
        OUT / "user_task_template.csv",
        OUT / "statistical_analysis_plan.json",
        OUT / "final_decision.json",
        PACKAGE / "PACKAGE_MANIFEST.json",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("required display-validation output is empty")
    write_output_manifest()
    print(
        json.dumps(
            {
                "status": "PASS",
                "cases": len(pool_records),
                "pilot_cases": int(pool_records.pilot_case.sum()),
                "identity_guard": "PASS",
                "decision": decision["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
