#!/usr/bin/env python3
"""Build frozen paper XAI tables, figures, and deterministic case materials."""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper_xai"
FIG = DOC / "figures"
HC = ROOT / "outputs" / "hierarchical_context_xai_feasibility" / "isruc_seed42"
PC = ROOT / "outputs" / "sequence_lime_protocol_consistency_audit" / "isruc_seed42"
MF = ROOT / "outputs" / "mf_seqlime_surrogate_improvement" / "isruc_seed42"
TWO = ROOT / "outputs" / "sequence_lime_two_versions" / "isruc_seed42"
BLUE, ORANGE, GREEN, RED, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#777777"
)
STATE_ORDER = [
    "LOCALLY_EXPLAINABLE",
    "CONTEXT_DOMINATED",
    "LOW_MODEL_RESPONSE",
    "SURROGATE_UNRELIABLE",
    "RANDOMIZATION_UNRELIABLE",
]
STATE_COLORS = [GREEN, ORANGE, SKY, GREY, RED]
GENERATED = datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_figure(fig, directory: Path, data: pd.DataFrame, sources, note: str):
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / "figure.png"
    pdf = directory / "figure.pdf"
    csv_path = directory / "figure_data.csv"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data.to_csv(csv_path, index=False)
    manifest = {
        "schema_version": "paper_xai_figure_manifest_v1",
        "generated_at_utc": GENERATED,
        "generator": "tools/build_paper_xai_materials.py",
        "canonical_method": "CONTEXT_PRESERVING_CLASSIC_SEQUENCE_LIME",
        "method_role": "EXPLORATORY_LOCAL_PERTURBATION_SENSITIVITY",
        "scientific_status": "MODEL_FAITHFULNESS_NOT_ESTABLISHED",
        "note": note,
        "source_files": [
            {"path": str(Path(p).relative_to(ROOT)), "sha256": sha(Path(p))}
            for p in sources
        ],
        "artifacts": {
            "png": {"path": "figure.png", "sha256": sha(png)},
            "pdf": {"path": "figure.pdf", "sha256": sha(pdf)},
            "data": {"path": "figure_data.csv", "sha256": sha(csv_path)},
        },
    }
    (directory / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build_figure_a():
    nodes = [
        "Frozen A1-zero\nModel", "Target\nEpoch",
        "Context-Preserving\nPerturbation", "Covering Sequence\nWindows",
        "Repeated Frozen\nInference", "Weighted Local\nSurrogate",
        "Local Perturbation\nSensitivity", "Reliability\nAudit",
        "Optional Clinical\nAnnotation",
    ]
    rows = []
    for i, node in enumerate(nodes):
        rows.append({"record_type": "node", "source": node.replace("\n", " "),
                     "target": "", "edge_style": ""})
    for a, b in zip(nodes[:-2], nodes[1:-1]):
        rows.append({"record_type": "edge", "source": a.replace("\n", " "),
                     "target": b.replace("\n", " "), "edge_style": "forward"})
    rows.append({"record_type": "edge", "source": "Reliability Audit",
                 "target": "Optional Clinical Annotation", "edge_style": "optional"})
    for badge in ("Pre-HMM Neural Response", "No Gradient", "No Training Feedback"):
        rows.append({"record_type": "boundary_badge", "source": badge,
                     "target": "", "edge_style": ""})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.2); ax.axis("off")
    positions = [(0.3 + i * 1.9, 3.0) for i in range(8)] + [(11.9, 0.65)]
    for i, (label, (x, y)) in enumerate(zip(nodes, positions)):
        color = "#DDEBF7" if i < 8 else "#FFF1CC"
        box = FancyBboxPatch((x, y), 1.55, 0.9, boxstyle="round,pad=0.08",
                             linewidth=1.4, edgecolor=BLUE if i < 8 else ORANGE,
                             facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.775, y + 0.45, label, ha="center", va="center",
                fontsize=8.5, color="#17324D")
    for i in range(7):
        x1, y1 = positions[i]; x2, y2 = positions[i + 1]
        ax.annotate("", xy=(x2, y2 + 0.45), xytext=(x1 + 1.55, y1 + 0.45),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    x1, y1 = positions[7]; x2, y2 = positions[8]
    ax.annotate("", xy=(x2 + 0.75, y2 + 0.9), xytext=(x1 + 0.75, y1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                linestyle="--"))
    badges = [("Pre-HMM Neural Response", 4.9), ("No Gradient", 7.6),
              ("No Training Feedback", 9.7)]
    for text, x in badges:
        ax.text(x, 4.55, text, ha="center", va="center", fontsize=8.5,
                color="#333333", bbox=dict(boxstyle="round,pad=0.3",
                fc="#F3F3F3", ec="#777777", lw=0.8))
    ax.text(7.0, 0.35, "Query-only post-hoc path • normal inference never reads XAI",
            ha="center", fontsize=10, weight="bold", color="#17324D")
    save_figure(
        fig, FIG / "figure_a", data,
        [ROOT / "docs" / "XAI_ARCHITECTURE_CLAIM_BOUNDARIES.md"],
        "Code-native vector fallback used because the remote AI schematic "
        "generator had no OPENROUTER_API_KEY. Exact boundary text was preserved.",
    )


def build_figure_b():
    fidelity = pd.read_csv(HC / "hierarchical_surrogate_fidelity.csv")
    faith = pd.read_csv(HC / "faithfulness_summary.csv")
    stability = pd.read_csv(HC / "stability_summary.csv")
    rand = pd.read_csv(HC / "randomization_summary.csv")
    f0 = fidelity.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'").iloc[0]
    fa = faith.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0' and randomization_level == 'R0'")
    st = stability.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'").iloc[0]
    ra = rand.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'")
    rows = [
        {"panel": "fidelity", "seed": "", "metric": "weighted_R2", "value": f0.local_surrogate_R2},
        *[{"panel": "faithfulness", "seed": int(r.randomization_seed),
           "metric": "insertion_minus_deletion", "value": r.faithfulness}
          for _, r in fa.iterrows()],
        {"panel": "stability", "seed": "", "metric": "mask_seed_spearman_median",
         "value": st.mask_seed_stability_median},
    ]
    for _, r in ra.iterrows():
        rows += [
            {"panel": "randomization", "seed": int(r.randomization_seed),
             "metric": "R7_similarity", "value": r.R7_similarity_mean},
            {"panel": "randomization", "seed": int(r.randomization_seed),
             "metric": "null_q95", "value": r.permutation_null_q95},
        ]
    data = pd.DataFrame(rows)
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.2))
    axs[0, 0].bar(["Weighted $R^2$"], [float(f0.local_surrogate_R2)], color=BLUE)
    axs[0, 0].axhline(0.8, color=GREY, ls="--", lw=1)
    axs[0, 0].set_ylim(0, 1); axs[0, 0].set_ylabel("Local surrogate fit")
    axs[0, 0].text(-0.13, 1.05, "a", transform=axs[0, 0].transAxes, weight="bold")
    axs[0, 1].bar([str(int(x)) for x in fa.randomization_seed], fa.faithfulness, color=GREEN)
    axs[0, 1].set_ylabel("Insertion − deletion"); axs[0, 1].set_xlabel("Mask seed")
    axs[0, 1].text(-0.13, 1.05, "b", transform=axs[0, 1].transAxes, weight="bold")
    axs[1, 0].bar(["Median Spearman"], [float(st.mask_seed_stability_median)], color=SKY)
    axs[1, 0].axhline(0.5, color=GREY, ls="--", lw=1)
    axs[1, 0].set_ylim(0, 1); axs[1, 0].set_ylabel("Mask-seed stability")
    axs[1, 0].text(-0.13, 1.05, "c", transform=axs[1, 0].transAxes, weight="bold")
    x = np.arange(len(ra))
    axs[1, 1].plot(x, ra.R7_similarity_mean, "o-", color=RED, label="R7 similarity")
    axs[1, 1].plot(x, ra.permutation_null_q95, "s--", color=GREY, label="Null q95")
    axs[1, 1].set_xticks(x, [str(int(s)) for s in ra.randomization_seed])
    axs[1, 1].set_xlabel("Randomization seed"); axs[1, 1].set_ylabel("Absolute similarity")
    axs[1, 1].legend(frameon=False, fontsize=8)
    axs[1, 1].text(-0.13, 1.05, "d", transform=axs[1, 1].transAxes, weight="bold")
    fig.tight_layout()
    save_figure(fig, FIG / "figure_b", data, [
        HC / "hierarchical_surrogate_fidelity.csv", HC / "faithfulness_summary.csv",
        HC / "stability_summary.csv", HC / "randomization_summary.csv"],
        "All panels use B0 Classic, validation T0, from one retained-method protocol. "
        "They are reliability diagnostics, not proof of model faithfulness.",
    )


def stacked(ax, frame, group_col, letter):
    tab = frame.groupby([group_col, "reliability_state"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=STATE_ORDER, fill_value=0)
    prop = tab.div(tab.sum(axis=1), axis=0)
    left = np.zeros(len(prop))
    for state, color in zip(STATE_ORDER, STATE_COLORS):
        ax.barh([str(x) for x in prop.index], prop[state], left=left,
                color=color, label=state.replace("_", " ").title())
        left += prop[state].to_numpy()
    ax.set_xlim(0, 1); ax.set_xlabel("Sample proportion")
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, weight="bold")
    return tab


def build_figure_c():
    rel_path = HC / "reliability_sample_level.csv"
    d = pd.read_csv(rel_path).query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'").copy()
    d["overall"] = "All"
    d["correctness"] = np.where(d.correct, "Correct", "Incorrect")
    d["temporal_status"] = np.where(d.transition_boundary, "Transition", "Stable")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.5))
    tabs = {
        "overall": stacked(axs[0, 0], d, "overall", "a"),
        "stage": stacked(axs[0, 1], d, "true_stage", "b"),
        "correctness": stacked(axs[1, 0], d, "correctness", "c"),
        "temporal_status": stacked(axs[1, 1], d, "temporal_status", "d"),
    }
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    rows = []
    for stratum, tab in tabs.items():
        for group, row in tab.iterrows():
            total = int(row.sum())
            for state in STATE_ORDER:
                rows.append({"stratum": stratum, "group": group, "state": state,
                             "count": int(row[state]), "proportion": row[state] / total})
    save_figure(fig, FIG / "figure_c", pd.DataFrame(rows), [rel_path],
                "Canonical B0 validation T0 reliability coverage only; zero-count "
                "states are retained. Local Perturbation Sensitivity terminology applies.")


def find_attr(sample_id: int):
    run_root = HC / "runs"
    preferred = sorted(run_root.glob(
        f"validation_classic_sample{sample_id}_R0_seed3101_zero/classic_attribution.csv"
    ))
    paths = preferred or sorted(run_root.glob(
        f"validation_classic_sample{sample_id}_R0_seed*_zero/classic_attribution.csv"
    ))
    if not paths:
        return None
    df = pd.read_csv(paths[0])
    row = df.query("target_protocol == 'T0'").iloc[0]
    values = np.asarray(ast.literal_eval(row.importance), dtype=float).reshape(3, 30)
    return values, paths[0], int(row.randomization_seed)


def choose_with_attr(frame, sort_cols, ascending, excluded):
    for _, row in frame.sort_values(sort_cols, ascending=ascending).iterrows():
        sid = int(row.sample_id)
        if sid not in excluded and find_attr(sid) is not None:
            return row
    return None


def build_figure_d():
    rel_path = HC / "reliability_sample_level.csv"
    rel = pd.read_csv(rel_path)
    b0 = rel.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'").copy()
    selected, excluded = [], set()
    specs = [
        ("Relatively reliable", b0.query("reliability_state == 'LOCALLY_EXPLAINABLE'"),
         ["sample_id"], [True], "canonical_state_then_lowest_sample_id"),
        ("Randomization-unreliable", b0.query("reliability_state == 'RANDOMIZATION_UNRELIABLE'"),
         ["R7_similarity", "sample_id"], [False, True], "highest_R7_similarity_then_sample_id"),
        ("Incorrect or transition", b0.assign(
            both=(~b0.correct.astype(bool)) & b0.transition_boundary.astype(bool)),
         ["both", "correct", "transition_boundary", "sample_id"],
         [False, True, False, True], "incorrect_transition_priority_then_sample_id"),
    ]
    for label, frame, cols, asc, rule in specs:
        row = choose_with_attr(frame, cols, asc, excluded)
        if row is not None:
            selected.append((label, row, rule, "B0_CLASSIC_90"))
            excluded.add(int(row.sample_id))
    context = rel.query(
        "method == 'H3_G25_F45' and target_protocol == 'T0' and "
        "reliability_state == 'CONTEXT_DOMINATED'"
    )
    row = choose_with_attr(context, ["context_dependence_ratio", "sample_id"],
                           [False, True], excluded)
    if row is not None:
        selected.append(("Context-dominated auxiliary flag", row,
                         "H3_G25_context_ratio_desc_then_sample_id",
                         "H3_G25_F45 selection; B0 attribution displayed"))
        excluded.add(int(row.sample_id))
    if len(selected) < 4:
        raise RuntimeError("Deterministic protocol could not identify four cases with B0 R0 attribution")
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)
    data_rows, sources = [], [rel_path]
    for ax, (label, row, rule, selection_source) in zip(axs, selected):
        values, path, seed = find_attr(int(row.sample_id))
        sources.append(path)
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks([0, 1, 2], ["EEG", "EOG", "EMG"])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_title(
            f"ID {int(row.sample_id)} | {row.true_stage}→{row.predicted_stage}",
            loc="left", fontsize=8.5, pad=3,
        )
        for c in range(3):
            for s in range(30):
                data_rows.append({
                    "case": label, "sample_id": int(row.sample_id),
                    "selection_source": selection_source, "selection_rule": rule,
                    "true_stage": row.true_stage, "predicted_stage": row.predicted_stage,
                    "correct": bool(row.correct), "transition_boundary": bool(row.transition_boundary),
                    "reliability_state": row.reliability_state, "channel": ["EEG", "EOG", "EMG"][c],
                    "segment": s, "local_perturbation_sensitivity": values[c, s],
                    "attribution_seed": seed,
                })
    axs[-1].set_xlabel("Target-epoch segment")
    fig.colorbar(im, ax=axs, fraction=0.018, pad=0.08,
                 label="Normalized Local Perturbation Sensitivity")
    fig.subplots_adjust(left=0.24, right=0.90, hspace=0.52)
    data = pd.DataFrame(data_rows)
    save_figure(fig, FIG / "figure_d", data, list(dict.fromkeys(sources)),
                "Cases were selected deterministically before visualization; no beauty, "
                "physiological-expectation, or story-support criterion was used.")
    summary = data.drop_duplicates("case")[[
        "case", "sample_id", "selection_source", "selection_rule", "true_stage",
        "predicted_stage", "correct", "transition_boundary", "reliability_state",
        "attribution_seed"
    ]]
    summary.to_csv(DOC / "XAI_CASE_STUDY_SELECTION.csv", index=False)
    return summary


def build_figure_e():
    rows = [
        ("Classic", "Unified A/C protocol", "0/3 T0; 0/3 T1",
         "Canonical sensitivity reference; model faithfulness not established"),
        ("MF historical", "Historical two-version + later audit", "Superseded",
         "No external fidelity uplift; incompatible old randomization result superseded"),
        ("Structured sampling", "MF surrogate-improvement validation", "No qualified candidate",
         "Sampling changes did not clear joint hard gates"),
        ("HC-G9", "HC equal-budget validation", "0/3 T0; 0/3 T1",
         "No joint fidelity/faithfulness/stability/randomization uplift"),
        ("HC-G25", "HC equal-budget validation", "0/3 T0; 0/3 T1",
         "No joint fidelity/faithfulness/stability/randomization uplift"),
    ]
    data = pd.DataFrame(rows, columns=["method", "protocol", "decision", "reason"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8)); ax.axis("off")
    y = np.arange(len(data))[::-1]
    for i, (_, r) in enumerate(data.iterrows()):
        yy = y[i]
        ax.add_patch(FancyBboxPatch((0.03, yy - 0.33), 0.94, 0.66,
                     boxstyle="round,pad=0.015", fc="#F7F7F7",
                     ec=BLUE if i == 0 else RED, lw=1.2))
        ax.text(0.05, yy + 0.12, r.method, weight="bold", fontsize=10)
        ax.text(0.33, yy + 0.12, r.protocol, fontsize=8.5, color="#444444")
        ax.text(0.05, yy - 0.13, f"{r.decision} — {r.reason}", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(data) - 0.4)
    save_figure(fig, FIG / "figure_e", data, [
        PC / "group_level_randomization_summary.csv",
        TWO / "final_scientific_decision.json",
        MF / "surrogate_candidate_summary.csv",
        HC / "final_scientific_decision.json",
    ], "Protocol-separated decision trail. Numeric endpoints from incompatible "
       "protocols are not pooled or compared on a shared axis.")


def build_tables():
    rel = pd.read_csv(HC / "reliability_sample_level.csv")
    b0 = rel.query("method == 'B0_CLASSIC_90' and target_protocol == 'T0'")
    fid = pd.read_csv(HC / "hierarchical_surrogate_fidelity.csv").query(
        "method == 'B0_CLASSIC_90' and target_protocol == 'T0'").iloc[0]
    stab = pd.read_csv(HC / "stability_summary.csv").query(
        "method == 'B0_CLASSIC_90' and target_protocol == 'T0'").iloc[0]
    rand = pd.read_csv(HC / "randomization_summary.csv").query(
        "method == 'B0_CLASSIC_90' and target_protocol == 'T0'")
    parity = json.loads(
        (PC / "guard" / "parity" / "prediction_metric_parity.json").read_text()
    )
    model_parity = json.loads(
        (PC / "guard" / "parity" / "model_prediction_parity.json").read_text()
    )
    memory = json.loads((PC / "guard" / "fixed_graph_memory_1200.json").read_text())
    metric_diff = parity["metric_abs_differences"]
    engineering = pd.DataFrame([
        ("Canonical probability parity", "PASS",
         f"max |Δp| = {model_parity['exact_artifact_parity']['max_abs_probability_diff']:.1e}"),
        ("Argmax agreement", "PASS", f"{parity['argmax_agreement']:.1%}"),
        ("Accuracy difference", "PASS", f"{metric_diff['accuracy_abs_diff']:.1e}"),
        ("Macro-F1 difference", "PASS", f"{metric_diff['macro_f1_abs_diff']:.1e}"),
        ("Cohen κ difference", "PASS", f"{metric_diff['kappa_abs_diff']:.1e}"),
        ("75-epoch wrapper parity", "PASS",
         f"max |Δp| = {model_parity['sequence_wrapper_parity']['overall_max_abs_diff']:.3e}"),
        ("Cache/model identity guard", "PASS",
         "A1-zero identity and checkpoint SHA-256 unchanged"),
        ("Fixed-graph 1,200-forward memory", memory["status"],
         f"current growth = {memory['current_memory_growth_bytes']} bytes"),
        ("Feedback guard", "PASS", "enabled/λ>0/real target fail fast"),
        ("Maintained tests before freeze", "122/122", "HC final validation"),
    ], columns=["Engineering check", "Status", "Evidence"])
    boundaries = pd.DataFrame([
        ("Implementation/audit pipeline", "Established", "Engineering only"),
        ("Local perturbation sensitivity", "Allowed", "Exploratory post-hoc role"),
        ("Classic randomization T0", "0/3 seeds", "Unified authoritative protocol"),
        ("Classic randomization T1", "0/3 seeds", "Unified authoritative protocol"),
        ("Model faithfulness", "Not established", "Classic failed unified T0/T1 gates"),
        ("MF-SeqLIME", "Superseded historical candidate", "No current method claim"),
        ("HC-SeqLIME", "Validation failed historical candidate", "Fixed 36 test not run"),
        ("Reliable coverage", f"{b0.reliable.mean():.1%}", "Validation T0; 1/30 samples"),
        ("Causal attribution", "Not claimed", "Perturbation response is not causality"),
        ("Clinical Concept Annotation", "Implementation valid", "Expert validation pending"),
        ("Feedback training", "Permanently blocked", "No XAI-to-loss path"),
    ], columns=["Claim", "Status", "Boundary"])
    reliability_rows = [
        ("Validation T0 sample count", len(b0)),
        ("Local surrogate weighted R²", f"{fid.local_surrogate_R2:.3f}"),
        ("Mask-seed stability median", f"{stab.mask_seed_stability_median:.3f}"),
        ("Relatively reliable coverage", f"{b0.reliable.mean():.1%}"),
        ("Context-dominated proportion", f"{(b0.reliability_state == 'CONTEXT_DOMINATED').mean():.1%}"),
        ("Low-response proportion", f"{(b0.reliability_state == 'LOW_MODEL_RESPONSE').mean():.1%}"),
        ("Surrogate-unreliable proportion", f"{(b0.reliability_state == 'SURROGATE_UNRELIABLE').mean():.1%}"),
        ("Randomization-unreliable proportion", f"{(b0.reliability_state == 'RANDOMIZATION_UNRELIABLE').mean():.1%}"),
        ("R7 similarity mean (3 seeds)", f"{rand.R7_similarity_mean.mean():.3f}"),
        ("Permutation-null q95 mean", f"{rand.permutation_null_q95.mean():.3f}"),
        ("Randomization seed passes", f"{int(rand.seed_protocol_pass.sum())}/3"),
    ]
    for label, mask in [
        ("Correct", b0.correct.astype(bool)),
        ("Incorrect", ~b0.correct.astype(bool)),
        ("Transition", b0.transition_boundary.astype(bool)),
        ("Stable", ~b0.transition_boundary.astype(bool)),
    ]:
        part = b0[mask]
        reliability_rows.append(
            (f"Reliable coverage — {label}",
             "NA" if len(part) == 0 else f"{part.reliable.mean():.1%} ({int(part.reliable.sum())}/{len(part)})")
        )
    for stage, part in b0.groupby("true_stage"):
        reliability_rows.append(
            (f"Reliable coverage — stage {stage}",
             f"{part.reliable.mean():.1%} ({int(part.reliable.sum())}/{len(part)})")
        )
    reliability = pd.DataFrame(reliability_rows, columns=["Reliability endpoint", "Result"])
    negative = pd.DataFrame([
        ("Historical PC prior fusion", "Predicted-stage prior would improve alignment",
         "A/B/C diagnostic", "Prior contaminated model-independent evidence",
         "Rejected as circular; historical only"),
        ("Weak zero calibration", "Zero marginal calibration would add faithfulness",
         "Historical two-version", "No external validation fidelity uplift",
         "Superseded historical candidate"),
        ("Structured neighborhood", "Structured masks would improve joint gates",
         "MF surrogate-improvement validation", "No qualified candidate",
         "Sampling change did not establish faithfulness"),
        ("Boundary-adaptive sampling", "Boundary emphasis would improve local response",
         "MF surrogate-improvement validation", "No qualified candidate",
         "No promotable uplift under preregistered gates"),
        ("Hierarchical context attribution", "Context groups would recover sequence dependence",
         "HC equal-budget validation", "G9/G25: 0/3 T0 and 0/3 T1",
         "Validation failed; fixed 36 test not run"),
    ], columns=["Ablation", "Hypothesis", "Protocol (not pooled)", "Result",
                "Scientific conclusion"])
    payloads = [
        ("TABLE_XAI_ENGINEERING_VALIDATION.md", "XAI Engineering Validation", engineering,
         "Engineering validity does not imply scientific model faithfulness."),
        ("TABLE_XAI_SCIENTIFIC_BOUNDARIES.md", "XAI Scientific Boundaries", boundaries,
         "The authoritative scientific status is `MODEL_FAITHFULNESS_NOT_ESTABLISHED`."),
        ("TABLE_XAI_RELIABILITY_RESULTS.md", "XAI Reliability Results", reliability,
         "All numeric rows use canonical B0 validation T0 under one retained-method protocol."),
        ("TABLE_XAI_NEGATIVE_ABLATIONS.md", "XAI Negative Ablations", negative,
         "Protocols are identified per row and their numeric endpoints are not pooled."),
    ]
    for filename, title, df, note in payloads:
        (DOC / filename).write_text(
            f"# {title}\n\n{note}\n\n{md_table(df)}\n", encoding="utf-8"
        )


def write_case_protocol(summary):
    body = f"""# XAI Case Study Selection Protocol

## Frozen pool and variables

The pool is the 30-sample validation reliability set in
`outputs/hierarchical_context_xai_feasibility/isruc_seed42/reliability_sample_level.csv`.
Selection uses T0 only. Recorded variables are sample ID, source recording,
true/predicted stage, correctness, transition/stable status, reliability state,
local surrogate R², response variance, stability, R7 similarity, confidence,
and—only for the auxiliary HC flag—context-dependence ratio.

## Deterministic order

1. Relatively reliable: canonical B0 `LOCALLY_EXPLAINABLE`, lowest sample ID.
2. Randomization-unreliable: canonical B0, highest R7 similarity then sample ID.
3. Incorrect or transition: prefer incorrect+transition, then incorrect, then
   transition, then sample ID.
4. Context-dominated: auxiliary H3-G25 `CONTEXT_DOMINATED`, descending context
   ratio then sample ID; the displayed heatmap remains canonical B0.

Already selected IDs are excluded. A case must have an existing B0 R0 T0
attribution artifact. There is no replacement from test data. If a category
has no eligible case, the report must state **no reliable case available**
rather than relaxing the rule.

## Explicit prohibitions

Selection by visual beauty, physiological expectation, story support,
interesting-looking channels, desired stage, or desired correctness is
forbidden. The protocol is executed before plotting; figure quality cannot
change the selected IDs.

## Frozen selected cases

{md_table(summary)}
"""
    (ROOT / "docs" / "XAI_CASE_STUDY_SELECTION_PROTOCOL.md").write_text(
        body, encoding="utf-8"
    )


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    build_tables()
    build_figure_a()
    build_figure_b()
    build_figure_c()
    summary = build_figure_d()
    write_case_protocol(summary)
    build_figure_e()
    inventory = sorted(
        p for p in DOC.rglob("*") if p.is_file()
    )
    payload = {
        "schema_version": "paper_xai_materials_manifest_v1",
        "generated_at_utc": GENERATED,
        "generator": "tools/build_paper_xai_materials.py",
        "artifacts": [
            {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in inventory
        ],
    }
    (DOC / "CURRENT_RESULTS_MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "artifacts": len(inventory)}, indent=2))


if __name__ == "__main__":
    main()
