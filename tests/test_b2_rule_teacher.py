import numpy as np

from tools.b2_physiology_rule_teacher_disagreement import (
    compute_disagreement_risk,
    compute_rule_teacher,
    summarize_disagreement,
)


def test_rule_teacher_maps_basic_physiology_to_expected_stages():
    concepts = np.zeros((5, 9), dtype=np.float32)
    concepts[0, 2] = 1.0  # alpha wake
    concepts[0, 4] = 0.8  # beta arousal
    concepts[0, 8] = 0.9  # active EMG
    concepts[1, 3] = 1.0  # sigma spindle
    concepts[2, 0] = 1.0  # slow wave
    concepts[3, 6] = 1.0  # rapid EOG
    concepts[3, 7] = 1.0  # EMG atonia
    concepts[4, 1] = 0.9  # theta
    concepts[4, 5] = 0.8  # slow eye
    recording_id = np.asarray(["r0"] * len(concepts), dtype=object)

    result = compute_rule_teacher(
        concepts,
        recording_id,
        adaptive_concepts=False,
        rule_temperature=0.65,
        n1_window=3,
        transition_radius=1,
    )

    assert result["p_rule"].shape == (5, 5)
    assert result["rule_pred"][0] == 0
    assert result["rule_pred"][1] == 2
    assert result["rule_pred"][2] == 3
    assert result["rule_pred"][3] == 4
    assert result["rule_pred"][4] == 1


def test_disagreement_summary_high_conflict_error_rate_is_reported():
    y_true = np.asarray([0, 1, 2, 3], dtype=np.int32)
    probs = np.asarray(
        [
            [0.90, 0.03, 0.03, 0.02, 0.02],
            [0.02, 0.85, 0.05, 0.03, 0.05],
            [0.02, 0.04, 0.88, 0.04, 0.02],
            [0.10, 0.02, 0.02, 0.84, 0.02],
        ],
        dtype=np.float32,
    )
    model_pred = np.asarray([0, 1, 2, 3], dtype=np.int32)
    rule_pred = np.asarray([0, 2, 2, 4], dtype=np.int32)
    rule_conf = np.asarray([0.80, 0.70, 0.75, 0.70], dtype=np.float32)

    summary = summarize_disagreement(
        y_true,
        probs,
        model_pred,
        rule_pred,
        rule_conf,
        ("W", "N1", "N2", "N3", "REM"),
        model_conf_threshold=0.70,
        rule_conf_threshold=0.60,
    )

    assert summary["n_high_conflict"] == 2
    assert summary["model_error_rate_when_high_conflict"] == 0.0
    assert summary["rule_error_rate_when_high_conflict"] == 1.0


def test_disagreement_risk_prioritizes_low_model_margin_disagreements():
    probs = np.asarray(
        [
            [0.95, 0.02, 0.01, 0.01, 0.01],
            [0.34, 0.33, 0.31, 0.01, 0.01],
            [0.10, 0.05, 0.80, 0.03, 0.02],
        ],
        dtype=np.float32,
    )
    p_rule = np.asarray(
        [
            [0.90, 0.03, 0.03, 0.02, 0.02],
            [0.05, 0.70, 0.20, 0.03, 0.02],
            [0.05, 0.04, 0.08, 0.80, 0.03],
        ],
        dtype=np.float32,
    )
    model_pred = np.asarray([0, 0, 2], dtype=np.int32)
    rule_pred = np.asarray([0, 1, 3], dtype=np.int32)
    rule_conf = np.asarray([0.90, 0.70, 0.80], dtype=np.float32)

    risk = compute_disagreement_risk(probs, p_rule, model_pred, rule_pred, rule_conf)

    assert risk["risk_score"][0] == 0.0
    assert risk["risk_score"][1] > risk["risk_score"][2]
