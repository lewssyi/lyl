import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure_output_dir(save_path):
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _append_suffix(save_path, suffix):
    base_path, ext = os.path.splitext(save_path)
    return f"{base_path}_{suffix}{ext or '.png'}"


def _save_or_show(fig, save_path, message):
    if save_path:
        _ensure_output_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{message}: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _as_percentage(values):
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(np.abs(values)))
    if total <= 1e-12:
        return np.zeros_like(values)
    return np.abs(values) / total * 100.0


def _annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _optional_array(values):
    if values is None:
        return None
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def plot_xai_multimodal_dashboard(
    signal,
    segments,
    segment_importance,
    raw_dccs,
    calibrated_dccs,
    segment_matrix,
    prior_weighted_matrix,
    temporal_net,
    cooperation_score,
    conflict_score,
    prior_alignment,
    channel_disagreement,
    clinical_concept_names=None,
    clinical_concept_matrix=None,
    clinical_concept_evidence_matrix=None,
    raw_concept_dccs=None,
    calibrated_concept_dccs=None,
    concept_prior=None,
    concept_alignment=None,
    perturbation_mode=None,
    channel_names=None,
    target_class_name="REM",
    save_path=None,
):
    channel_names = channel_names or ["EEG", "EOG", "EMG"]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    num_channels = signal.shape[1]
    num_segments = len(segments)
    visible_channel_names = channel_names[:num_channels]
    cmap = plt.get_cmap("coolwarm")
    max_abs_weight = max(
        float(np.max(np.abs(segment_importance))),
        float(np.max(np.abs(segment_matrix))),
        1e-6,
    )

    if save_path:
        signal_path = _append_suffix(save_path, "signals")
        dccs_path = _append_suffix(save_path, "dccs_percent")
        evidence_path = save_path
    else:
        signal_path = dccs_path = evidence_path = None

    fig_signal, axes = plt.subplots(
        num_channels,
        1,
        figsize=(15, 2.8 * num_channels),
        sharex=True,
        constrained_layout=True,
    )
    if num_channels == 1:
        axes = [axes]
    time_axis = np.arange(signal.shape[0])
    for channel_idx in range(num_channels):
        ax = axes[channel_idx]
        ax.plot(time_axis, signal[:, channel_idx], color="black", linewidth=0.8, alpha=0.85)
        channel_weights = segment_matrix[channel_idx]
        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            norm_weight = 0.5 + (channel_weights[seg_idx] / (2 * max_abs_weight))
            norm_weight = float(np.clip(norm_weight, 0.0, 1.0))
            ax.axvspan(start_idx, end_idx, facecolor=cmap(norm_weight), alpha=0.45, edgecolor="none")
            ax.axvline(x=end_idx, color="gray", linestyle="--", linewidth=0.4, alpha=0.4)
        ax.set_ylabel(f"{channel_names[channel_idx]}\nAmplitude", fontweight="bold")
        ax.set_xlim(0, signal.shape[0])
    axes[0].set_title(
        f"Dynamic Physiological Segments | Predicted Stage: {target_class_name}",
        fontweight="bold",
        fontsize=15,
        loc="left",
    )
    axes[-1].set_xlabel("Time (Samples)", fontweight="bold")
    _save_or_show(fig_signal, signal_path, "XAI signal segments saved to")

    raw_pct = _as_percentage(raw_dccs)
    calibrated_pct = _as_percentage(calibrated_dccs)
    disagreement = np.nan_to_num(
        np.asarray(channel_disagreement, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    concept_names = list(clinical_concept_names or [])
    concept_evidence = _optional_array(clinical_concept_evidence_matrix)
    concept_activation = _optional_array(clinical_concept_matrix)
    raw_concept_pct = _as_percentage(raw_concept_dccs) if raw_concept_dccs is not None else np.array([])
    calibrated_concept_pct = (
        _as_percentage(calibrated_concept_dccs) if calibrated_concept_dccs is not None else np.array([])
    )
    concept_prior_pct = _as_percentage(concept_prior) if concept_prior is not None else np.array([])
    has_concepts = (
        len(concept_names) > 0
        and len(calibrated_concept_pct) == len(concept_names)
        and len(raw_concept_pct) == len(concept_names)
    )

    fig_dccs, axes_dccs = plt.subplots(
        2,
        2,
        figsize=(15, 8.2),
        gridspec_kw={"width_ratios": [1.45, 1.0], "height_ratios": [1.0, 1.2]},
        constrained_layout=True,
    )
    x = np.arange(num_channels)
    width = 0.34
    raw_bars = axes_dccs[0, 0].bar(x - width / 2, raw_pct, width, label="Raw DCCS", color="#4C72B0")
    calibrated_bars = axes_dccs[0, 0].bar(
        x + width / 2,
        calibrated_pct,
        width,
        label="Calibrated DCCS",
        color="#DD8452",
    )
    axes_dccs[0, 0].set_xticks(x)
    axes_dccs[0, 0].set_xticklabels(visible_channel_names)
    axes_dccs[0, 0].set_ylim(0, max(100.0, float(np.max([raw_pct.max(), calibrated_pct.max()])) * 1.2))
    axes_dccs[0, 0].set_ylabel("Channel Contribution (%)", fontweight="bold")
    axes_dccs[0, 0].set_title("DCCS Percentage by Channel", fontweight="bold")
    axes_dccs[0, 0].legend(fontsize=9)
    _annotate_bars(axes_dccs[0, 0], raw_bars)
    _annotate_bars(axes_dccs[0, 0], calibrated_bars)

    axes_dccs[0, 1].bar(
        visible_channel_names,
        disagreement,
        color=["#4C72B0", "#DD8452", "#55A868"][:num_channels],
    )
    axes_dccs[0, 1].set_ylim(0, max(1.0, float(np.max(disagreement)) * 1.2))
    axes_dccs[0, 1].set_ylabel("Disagreement", fontweight="bold")
    mode_suffix = f" | {perturbation_mode}" if perturbation_mode else ""
    axes_dccs[0, 1].set_title(f"Prior Alignment = {prior_alignment:.3f}{mode_suffix}", fontweight="bold")

    if has_concepts:
        concept_strength = raw_concept_pct + calibrated_concept_pct + concept_prior_pct
        top_count = min(8, len(concept_names))
        top_idx = np.argsort(concept_strength)[-top_count:][::-1]
        y_pos = np.arange(top_count)
        bar_height = 0.24
        axes_dccs[1, 0].barh(
            y_pos - bar_height,
            raw_concept_pct[top_idx],
            bar_height,
            label="Raw Concept DCCS",
            color="#4C72B0",
        )
        axes_dccs[1, 0].barh(
            y_pos,
            calibrated_concept_pct[top_idx],
            bar_height,
            label="Calibrated Concept DCCS",
            color="#DD8452",
        )
        if len(concept_prior_pct) == len(concept_names):
            axes_dccs[1, 0].barh(
                y_pos + bar_height,
                concept_prior_pct[top_idx],
                bar_height,
                label="Clinical Prior",
                color="#55A868",
            )
        axes_dccs[1, 0].set_yticks(y_pos)
        axes_dccs[1, 0].set_yticklabels([concept_names[idx] for idx in top_idx])
        axes_dccs[1, 0].invert_yaxis()
        axes_dccs[1, 0].set_xlabel("Contribution (%)", fontweight="bold")
        axes_dccs[1, 0].set_title("Clinical Concept DCCS", fontweight="bold")
        axes_dccs[1, 0].legend(fontsize=8)

        axes_dccs[1, 1].bar(
            ["Concept\nAlignment"],
            [float(concept_alignment or 0.0)],
            color="#8172B2",
        )
        axes_dccs[1, 1].set_ylim(0, 1.05)
        axes_dccs[1, 1].set_ylabel("Alignment", fontweight="bold")
        axes_dccs[1, 1].set_title("AASM Concept Prior Alignment", fontweight="bold")
    else:
        axes_dccs[1, 0].axis("off")
        axes_dccs[1, 1].axis("off")
    _save_or_show(fig_dccs, dccs_path, "XAI DCCS percentage saved to")

    has_concept_heatmap = concept_evidence is not None and concept_evidence.size > 0 and len(concept_names) > 0
    fig_evidence = plt.figure(figsize=(17, 12 if has_concept_heatmap else 10), constrained_layout=True)
    if has_concept_heatmap:
        gs = fig_evidence.add_gridspec(3, 2, hspace=0.25, wspace=0.2, height_ratios=[1.0, 0.9, 0.9])
    else:
        gs = fig_evidence.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    ax_joint = fig_evidence.add_subplot(gs[0, 0])
    sns.heatmap(
        segment_matrix,
        cmap="coolwarm",
        center=0.0,
        ax=ax_joint,
        cbar=True,
        xticklabels=[f"S{idx + 1}" for idx in range(num_segments)],
        yticklabels=visible_channel_names,
    )
    ax_joint.set_title("Joint Channel-Segment Evidence", fontweight="bold")
    ax_joint.set_xlabel("Dynamic Segment")
    ax_joint.set_ylabel("Physiological Channel")

    ax_calibrated = fig_evidence.add_subplot(gs[0, 1])
    sns.heatmap(
        prior_weighted_matrix,
        cmap="coolwarm",
        center=0.0,
        ax=ax_calibrated,
        cbar=True,
        xticklabels=[f"S{idx + 1}" for idx in range(num_segments)],
        yticklabels=visible_channel_names,
    )
    ax_calibrated.set_title("Clinically Calibrated Evidence", fontweight="bold")
    ax_calibrated.set_xlabel("Dynamic Segment")
    ax_calibrated.set_ylabel("Physiological Channel")

    if has_concept_heatmap:
        ax_concept = fig_evidence.add_subplot(gs[1, :])
        concept_heatmap = concept_evidence
        if concept_activation is not None and concept_activation.shape == concept_evidence.shape:
            concept_heatmap = concept_evidence * np.sign(np.sum(segment_matrix, axis=0, keepdims=True))
        sns.heatmap(
            concept_heatmap,
            cmap="coolwarm",
            center=0.0,
            ax=ax_concept,
            cbar=True,
            xticklabels=[f"S{idx + 1}" for idx in range(num_segments)],
            yticklabels=concept_names,
        )
        ax_concept.set_title("Clinical Concept-Segment Evidence", fontweight="bold")
        ax_concept.set_xlabel("Dynamic Segment")
        ax_concept.set_ylabel("Clinical Concept")

    ax_temporal = fig_evidence.add_subplot(gs[2 if has_concept_heatmap else 1, :])
    segment_axis = np.arange(num_segments)
    ax_temporal.plot(segment_axis, temporal_net, color="#3B5BA9", linewidth=2.0, label="Net evidence")
    ax_temporal.fill_between(segment_axis, 0, cooperation_score, color="#55A868", alpha=0.25, label="Cooperation")
    ax_temporal.fill_between(segment_axis, 0, conflict_score, color="#C44E52", alpha=0.2, label="Conflict")
    ax_temporal.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_temporal.set_title("Temporal Cooperation / Conflict", fontweight="bold")
    ax_temporal.set_xlabel("Dynamic Segment")
    ax_temporal.set_ylabel("Evidence")
    ax_temporal.legend(loc="upper right", fontsize=9)

    fig_evidence.suptitle(
        f"Evidence Dashboard | Predicted Stage: {target_class_name}",
        fontweight="bold",
        fontsize=16,
    )
    _save_or_show(fig_evidence, evidence_path, "XAI evidence dashboard saved to")


def plot_transition_explanation_dashboard(
    transition_summary,
    from_stage,
    to_stage,
    save_path=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    channel_names = transition_summary["channel_names"]
    mean_segment_matrix = transition_summary["mean_segment_matrix"]
    mean_temporal_net = transition_summary["mean_temporal_net"]
    transition_shift = transition_summary["transition_shift"]
    mean_cooperation = transition_summary["mean_cooperation"]
    mean_conflict = transition_summary["mean_conflict"]
    prior_alignment_trend = transition_summary["prior_alignment_trend"]
    channel_shift = transition_summary["channel_shift"]
    concept_names = transition_summary.get("clinical_concept_names", [])
    concept_shift = _optional_array(transition_summary.get("transition_concept_shift"))
    concept_prior = _optional_array(transition_summary.get("transition_concept_prior"))
    transition_prior_alignment = float(transition_summary.get("transition_prior_alignment", 0.0))
    transition_plausibility = float(transition_summary.get("transition_plausibility", 0.0))

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35, width_ratios=[1.25, 1.0, 1.0, 1.1])

    ax_heat = fig.add_subplot(gs[:, 0])
    sns.heatmap(
        mean_segment_matrix,
        cmap="coolwarm",
        center=0.0,
        ax=ax_heat,
        xticklabels=[f"S{idx + 1}" for idx in range(mean_segment_matrix.shape[1])],
        yticklabels=channel_names,
    )
    ax_heat.set_title(f"Transition Evidence: {from_stage} -> {to_stage}", fontweight="bold")
    ax_heat.set_xlabel("Dynamic Segment")
    ax_heat.set_ylabel("Physiological Channel")

    ax_shift = fig.add_subplot(gs[0, 1])
    ax_shift.plot(transition_shift, color="#C44E52", linewidth=2.0)
    ax_shift.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_shift.set_title("Stage-Transition Shift", fontweight="bold")
    ax_shift.set_xlabel("Dynamic Segment")
    ax_shift.set_ylabel("Delta evidence")

    ax_coop = fig.add_subplot(gs[1, 1])
    seg_axis = np.arange(len(mean_temporal_net))
    ax_coop.plot(seg_axis, mean_temporal_net, color="#3B5BA9", linewidth=2.0, label="Net")
    ax_coop.fill_between(seg_axis, 0, mean_cooperation, color="#55A868", alpha=0.25, label="Cooperation")
    ax_coop.fill_between(seg_axis, 0, mean_conflict, color="#C44E52", alpha=0.2, label="Conflict")
    ax_coop.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_coop.set_title("Transition Cooperation / Conflict", fontweight="bold")
    ax_coop.set_xlabel("Dynamic Segment")
    ax_coop.legend(fontsize=9)

    ax_align = fig.add_subplot(gs[0, 2])
    ax_align.plot(prior_alignment_trend, marker="o", color="#8172B2", linewidth=2.0)
    ax_align.set_ylim(0, 1.05)
    ax_align.set_title(
        f"Clinical Prior Alignment | transition={transition_prior_alignment:.3f}",
        fontweight="bold",
    )
    ax_align.set_xlabel("Epoch in transition window")
    ax_align.set_ylabel("Alignment")

    ax_channel = fig.add_subplot(gs[1, 2])
    ax_channel.bar(channel_names, channel_shift, color=["#4C72B0", "#DD8452", "#55A868"][: len(channel_names)])
    ax_channel.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_channel.set_title("Dominant Channel Shift", fontweight="bold")
    ax_channel.set_ylabel("Change in absolute evidence")

    ax_concept = fig.add_subplot(gs[:, 3])
    if concept_shift is not None and len(concept_names) == len(concept_shift):
        prior_pct = _as_percentage(concept_prior) if concept_prior is not None else np.zeros_like(concept_shift)
        top_strength = np.abs(concept_shift) + prior_pct / 100.0
        top_count = min(8, len(concept_names))
        top_idx = np.argsort(top_strength)[-top_count:][::-1]
        y_pos = np.arange(top_count)
        colors = ["#C44E52" if concept_shift[idx] >= 0 else "#4C72B0" for idx in top_idx]
        ax_concept.barh(y_pos, concept_shift[top_idx], color=colors, alpha=0.85, label="Observed shift")
        ax_concept.plot(prior_pct[top_idx] / 100.0, y_pos, "o", color="#55A868", label="Transition prior")
        ax_concept.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax_concept.set_yticks(y_pos)
        ax_concept.set_yticklabels([concept_names[idx] for idx in top_idx])
        ax_concept.invert_yaxis()
        ax_concept.set_title(
            f"Clinical Concept Shift\nPlausibility={transition_plausibility:.3f}",
            fontweight="bold",
        )
        ax_concept.set_xlabel("Change in concept DCCS")
        ax_concept.legend(fontsize=8)
    else:
        ax_concept.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Transition XAI dashboard saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_segmented_signals(signal, segments, channel_names=None, save_path=None):
    channel_names = channel_names or ["EEG", "EOG", "EMG"]
    plt.style.use("seaborn-v0_8-whitegrid")
    num_channels = signal.shape[1]

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    time_axis = np.arange(signal.shape[0])
    for channel_idx in range(num_channels):
        ax = axes[channel_idx]
        ax.plot(time_axis, signal[:, channel_idx], color="black", linewidth=0.8)
        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            ax.axvline(x=start_idx, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            if seg_idx % 2 == 0:
                ax.axvspan(start_idx, end_idx, facecolor="gray", alpha=0.1)
        ax.set_ylabel(channel_names[channel_idx], fontweight="bold")

    axes[0].set_title("Physiological Smoothness Segmentation Results", fontweight="bold", fontsize=14)
    axes[-1].set_xlabel("Time (Samples)", fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Segment debug plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_raw_signals(signal, title="Multichannel Physiological Signal", save_path=None):
    plt.figure(figsize=(12, 4))
    num_channels = signal.shape[1] if len(signal.shape) > 1 else 1

    if num_channels > 1:
        max_offset = np.max(np.abs(signal)) * 2
        for channel_idx in range(num_channels):
            offset = max_offset * channel_idx
            plt.plot(signal[:, channel_idx] - offset, linewidth=0.8, label=f"Channel {channel_idx}")
        plt.yticks([])
    else:
        plt.plot(signal, color="black", linewidth=0.8)

    plt.title(title, fontweight="bold")
    plt.xlabel("Time (Samples)", fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()