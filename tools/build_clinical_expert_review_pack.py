#!/usr/bin/env python
"""Build a blinded expert-review pack from precomputed offline XAI targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clinical_concepts import concepts_from_raw_waveform  # noqa: E402


INPUT_CACHE = PROJECT_ROOT / "outputs/sequence_lime_paper_evaluation/isruc_seed42/cache/test_inputs_segments.npz"
PREDICTIONS = PROJECT_ROOT / "outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz"
DEFAULT_TARGETS = PROJECT_ROOT / "outputs/xai_training_targets/isruc_seed42/targets.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_targets(run_dirs, perturbation_mode="physiological"):
    target_map = {}
    provenance = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        paths = sorted((run_dir / "cache").glob("sample_*.json"))
        if not paths:
            raise FileNotFoundError(f"No completed sample cache in {run_dir}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mode_payload = payload.get("perturbation_results", {}).get(
                perturbation_mode
            )
            if not mode_payload:
                raise KeyError(f"{path} lacks perturbation mode {perturbation_mode}")
            segment_count = int(payload["segments_count"])
            coefficient = np.asarray(mode_payload["coef"], dtype=np.float32)
            if coefficient.size != 3 * segment_count:
                raise ValueError(
                    f"{path} coefficient size {coefficient.size} is not "
                    f"3 x {segment_count}"
                )
            global_index = int(payload["global_epoch_index"])
            if global_index in target_map:
                raise ValueError(f"Duplicate evaluation sample {global_index}")
            target_map[global_index] = coefficient.reshape(3, segment_count).T
            provenance.append({"path": str(path), "sha256": sha256(path)})
    return target_map, provenance


def transition_flags(labels: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(labels), dtype=bool)
    if len(labels) > 1:
        flags[1:] |= labels[1:] != labels[:-1]
        flags[:-1] |= labels[:-1] != labels[1:]
    return flags


def stratified_indices(labels, predictions, available, per_cell, seed):
    rng = np.random.default_rng(seed)
    transition = transition_flags(labels)
    rows = []
    for stage in range(5):
        for correct in (False, True):
            for is_transition in (False, True):
                candidates = np.flatnonzero(
                    available
                    & (labels == stage)
                    & ((predictions == labels) == correct)
                    & (transition == is_transition)
                )
                if len(candidates):
                    chosen = rng.choice(candidates, size=min(per_cell, len(candidates)), replace=False)
                    rows.extend(chosen.tolist())
    rng.shuffle(rows)
    return rows, transition


def render_item(path, waveform, importance, concept_matrix):
    fig, axes = plt.subplots(5, 1, figsize=(11, 8), constrained_layout=True)
    names = ("EEG", "EOG", "EMG")
    for channel in range(3):
        axes[channel].plot(waveform[:, channel], linewidth=0.5, color=("#3b4cc0", "#2a9d8f", "#e76f51")[channel])
        axes[channel].set_ylabel(names[channel])
        axes[channel].set_xticks([])
    axes[3].imshow(np.asarray(importance).T, aspect="auto", cmap="coolwarm")
    axes[3].set_ylabel("XAI channel")
    axes[3].set_title("Offline local attribution (segment x channel; signed)")
    axes[4].imshow(concept_matrix, aspect="auto", cmap="viridis")
    axes[4].set_ylabel("raw concept")
    axes[4].set_xlabel("segment")
    axes[4].set_title("raw_physiology_v1 evidence")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--evaluation-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed evaluator run directory; repeat for disjoint shards.",
    )
    parser.add_argument(
        "--perturbation-mode",
        default="physiological",
        choices=("zero", "physiological", "gaussian_noise", "local_mean"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    required_paths = [INPUT_CACHE, PREDICTIONS]
    if not args.evaluation_run_dir:
        required_paths.append(args.targets)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    cache, pred = np.load(INPUT_CACHE), np.load(PREDICTIONS)
    labels = pred["test_y_true"].astype(int)
    predictions = np.argmax(pred["test_probs_raw"], axis=1)
    if args.evaluation_run_dir:
        target_map, source_provenance = load_evaluation_targets(
            args.evaluation_run_dir,
            perturbation_mode=args.perturbation_mode,
        )
        target_source = "completed_sequence_lime_evaluation"
    else:
        targets = np.load(args.targets)
        target_indices = targets["global_epoch_index"].astype(int)
        importance = targets["importance"].astype(np.float32)
        target_map = {
            int(index): row for index, row in zip(target_indices, importance)
        }
        source_provenance = [
            {"path": str(args.targets.resolve()), "sha256": sha256(args.targets)}
        ]
        target_source = "precomputed_target_artifact"
    target_indices = np.asarray(sorted(target_map), dtype=int)
    if np.any(target_indices < 0) or np.any(target_indices >= len(labels)):
        raise ValueError("Review target indices fall outside the frozen test artifact")
    available = np.zeros(len(labels), dtype=bool)
    available[target_indices] = True
    selected, transition = stratified_indices(
        labels, predictions, available, max(1, args.per_cell), args.seed
    )
    reviewer_dir = args.output_dir / "reviewer_pack"
    item_dir = reviewer_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    private_rows, rating_rows = [], []
    for order, index in enumerate(selected, 1):
        item_id = f"XAI-{order:04d}"
        waveform = cache["x_raw"][index].astype(np.float32)
        item_importance = target_map[index]
        boundaries = np.linspace(
            0, waveform.shape[0], int(item_importance.shape[0]) + 1, dtype=int
        )
        segments = list(zip(boundaries[:-1], boundaries[1:]))
        concepts = concepts_from_raw_waveform(waveform, segments)["concept_matrix"]
        render_item(item_dir / f"{item_id}.png", waveform, item_importance, concepts)
        private_rows.append(
            {
                "item_id": item_id,
                "global_epoch_index": index,
                "true_stage": int(labels[index]),
                "model_stage": int(predictions[index]),
                "correct": bool(labels[index] == predictions[index]),
                "transition": bool(transition[index]),
            }
        )
        rating_rows.append(
            {
                "reviewer_id": "",
                "item_id": item_id,
                "physiological_plausibility": "",
                "temporal_localization": "",
                "concept_evidence_coherence": "",
                "clinical_usefulness": "",
                "reviewer_stage": "",
                "artifact_flag": "",
                "comments": "",
            }
        )
    pd.DataFrame(rating_rows).to_csv(reviewer_dir / "rating_form.csv", index=False)
    pd.DataFrame(private_rows).to_csv(args.output_dir / "private_key.csv", index=False)
    manifest = {
        "status": "clinical expert validation pending",
        "blinded": True,
        "seed": args.seed,
        "item_count": len(selected),
        "target_source": target_source,
        "target_source_files": source_provenance,
        "perturbation_mode": args.perturbation_mode,
        "prediction_artifact_sha256": sha256(PREDICTIONS),
        "private_key_separate": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
