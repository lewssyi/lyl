#!/usr/bin/env python
"""Autonomous A1 zero core-training experiment loop.

This supervisor manages Git experiment branches, waits for GPU availability,
runs the fixed-split launcher, parses metrics, builds hard-example weights,
and commits lightweight metadata records after each completed run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiment_configs" / "a1zero_autonomous.json"
STATE_DIR = ROOT / "outputs" / "diagnostics" / "a1zero_autonomous"
RECORD_DIR = ROOT / "experiment_records"
METRIC_RE = re.compile(
    r"Primary metric summary \| .*?val_selected: .*?macro_F1=(?P<macro_f1>[0-9.]+).*?"
    r"N1_F1=(?P<n1_f1>[0-9.]+).*?(?:REM_F1=(?P<rem_f1>[0-9.]+).*?)?"
    r"kappa=(?P<kappa>[0-9.]+).*?accuracy=(?P<accuracy>[0-9.]+).*?"
    r"N1_P=(?P<n1_p>[0-9.]+).*?N1_R=(?P<n1_r>[0-9.]+)",
    re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous A1 zero experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_checked(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        handle = log_path.open("a", encoding="utf-8") if log_path is not None else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                if handle is not None:
                    handle.write(line)
                    handle.flush()
        finally:
            if handle is not None:
                handle.close()
        return proc.wait()


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def branch_exists(branch: str) -> bool:
    return git(["rev-parse", "--verify", branch], check=False).returncode == 0


def checkout_experiment_branch(branch: str, base_branch: str | None) -> None:
    if branch_exists(branch):
        git(["checkout", branch])
        return
    if base_branch:
        git(["checkout", base_branch])
    git(["checkout", "-b", branch])


def nvidia_snapshot() -> dict[str, dict[str, int]]:
    gpu_proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if gpu_proc.returncode != 0:
        raise RuntimeError(gpu_proc.stderr or gpu_proc.stdout)
    snapshot: dict[str, dict[str, int]] = {}
    uuid_to_index: dict[str, str] = {}
    for line in gpu_proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        idx, uuid, mem, util = parts[:4]
        uuid_to_index[uuid] = idx
        snapshot[idx] = {"memory": int(float(mem)), "utilization": int(float(util)), "processes": 0}

    app_proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if app_proc.returncode == 0:
        for line in app_proc.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            idx = uuid_to_index.get(parts[0])
            if idx in snapshot:
                snapshot[idx]["processes"] += 1
    return snapshot

def select_gpus(policy: dict[str, Any], requested: str | None, no_wait: bool) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.append(str(policy.get("preferred_gpus", "6,7")))
    candidates.extend(str(item) for item in policy.get("fallback_gpu_sets", []))
    candidates = list(dict.fromkeys(item for item in candidates if item))
    max_used = int(policy.get("max_used_memory_mb_per_gpu", 6000))
    max_util = int(policy.get("max_utilization_pct_per_gpu", 10))
    max_processes = int(policy.get("max_compute_processes_per_gpu", 0))
    poll_seconds = int(policy.get("poll_seconds", 600))
    while True:
        snapshot = nvidia_snapshot()
        for candidate in candidates:
            ids = [item.strip() for item in candidate.split(",") if item.strip()]
            if ids and all(
                gpu in snapshot
                and snapshot[gpu]["memory"] <= max_used
                and snapshot[gpu]["utilization"] <= max_util
                and snapshot[gpu]["processes"] <= max_processes
                for gpu in ids
            ):
                print(f"Selected GPUs {candidate}; snapshot={snapshot}", flush=True)
                return candidate
        print(
            "No candidate GPU set satisfies "
            f"memory<={max_used}MiB, util<={max_util}%, processes<={max_processes}; "
            f"snapshot={snapshot}",
            flush=True,
        )
        if no_wait:
            raise RuntimeError("No GPU set available and --no-wait was set.")
        time.sleep(max(30, poll_seconds))


def latest_matching(pattern: str, started_at: float | None = None) -> Path | None:
    matches = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if started_at is not None:
        matches = [p for p in matches if p.stat().st_mtime >= started_at - 5]
    return matches[0] if matches else None


def parse_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(errors="ignore")
    matches = list(METRIC_RE.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse primary metrics from {log_path}")
    groups = matches[-1].groupdict()
    return {key: float(value) for key, value in groups.items() if value is not None}


def build_command(python: str, defaults: dict[str, Any], exp: dict[str, Any], gpus: str, env_items: dict[str, str]) -> list[str]:
    cmd = [
        python,
        "tools/run_main_with_protocol_split.py",
        "--split-json",
        str(exp.get("split_json") or defaults["split_json"]),
        "--dataset-preset",
        str(exp["preset"]),
        "--gpus",
        gpus,
        "--per-replica-batch",
        str(exp.get("per_replica_batch", defaults.get("per_replica_batch", 16))),
        "--predict-per-replica-batch",
        str(exp.get("predict_per_replica_batch", defaults.get("predict_per_replica_batch", 16))),
        "--seed",
        str(exp.get("seed", defaults.get("seed", 42))),
        "--memory-profile",
        str(exp.get("memory_profile", defaults.get("memory_profile", "balanced"))),
        "--no-preset-batch-clamp",
    ]
    for name, value in sorted(env_items.items()):
        cmd.extend(["--set-env", f"{name}={value}"])
    return cmd


def state_lookup(state: dict[str, Any], name: str) -> dict[str, Any] | None:
    for run in state.get("runs", []):
        if run.get("name") == name and run.get("status") == "complete":
            return run
    return None


def build_hard_weights(python: str, dep_record: dict[str, Any], exp_name: str) -> tuple[str, str]:
    split_pred = dep_record.get("split_prediction")
    if not split_pred:
        raise ValueError(f"Dependency {dep_record.get('name')} has no split_prediction path.")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "diagnostics" / "a1zero_hard_weights" / f"{exp_name}_{timestamp}"
    cmd = [
        python,
        "tools/build_hard_example_weights.py",
        "--prediction-npz",
        split_pred,
        "--output-dir",
        str(out_dir),
    ]
    code = run_checked(cmd, log_path=STATE_DIR / "hard_weight_builder.log")
    if code != 0:
        raise RuntimeError(f"Hard weight builder failed with exit code {code}")
    summary = load_json(out_dir / "hard_example_summary.json")
    return summary["output_npz"], summary["class_weights_env"]


def commit_record(record_path: Path, metrics: dict[str, float], exp: dict[str, Any]) -> None:
    git(["add", str(record_path.relative_to(ROOT))])
    filter_profile = exp.get("env", {}).get("LFTS_PREPROCESS_FILTER_PROFILE", "unknown")
    suffix = "hard_tuning" if "hard" in exp.get("name", "") else "probs"
    msg = (
        f"perf: A1_zero + {suffix} + filter[{filter_profile}], "
        f"overall_acc={metrics.get('accuracy', 0.0):.4f}, "
        f"REM_f1={metrics.get('rem_f1', 0.0):.4f}"
    )
    if git(["diff", "--cached", "--quiet"], check=False).returncode != 0:
        git(["commit", "-m", msg])


def run_experiment(cfg: dict[str, Any], exp: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    python = str(cfg["python"])
    policy = cfg.get("gpu_policy", {})
    defaults = cfg.get("default_args", {})
    branch = str(exp["branch"])
    base_branch = exp.get("base_branch") or cfg.get("base_branch")
    checkout_experiment_branch(branch, base_branch)

    env_items = {str(k): str(v) for k, v in exp.get("env", {}).items()}
    dep_record = None
    if exp.get("depends_on"):
        dep_record = state_lookup(state, str(exp["depends_on"]))
        if dep_record is None:
            raise RuntimeError(f"Experiment {exp['name']} depends on unfinished {exp['depends_on']}")
    if exp.get("build_risk_weights"):
        risk_path, class_weights_env = build_hard_weights(python, dep_record, str(exp["name"]))
        env_items["LFTS_RISK_WEIGHT_PATH"] = risk_path
        env_items["LFTS_RISK_CLASS_WEIGHTS"] = class_weights_env
    elif exp.get("reuse_best_risk_weights"):
        best_risk = state.get("best_risk_weight_path")
        best_class_weights = state.get("best_risk_class_weights")
        if best_risk:
            env_items["LFTS_RISK_WEIGHT_PATH"] = str(best_risk)
        if best_class_weights:
            env_items["LFTS_RISK_CLASS_WEIGHTS"] = str(best_class_weights)

    gpus = select_gpus(policy, exp.get("gpus"), args.no_wait)
    env = os.environ.copy()
    env["LFTS_REQUIRE_REQUESTED_GPUS"] = "1" if policy.get("require_requested_gpus", True) else "0"
    deps = ROOT / ".deps"
    if deps.exists():
        env["PYTHONPATH"] = f"{deps}:{env.get('PYTHONPATH', '')}"

    cmd = build_command(python, defaults, exp, gpus, env_items)
    if args.dry_run:
        print("DRY RUN: " + " ".join(cmd))
        return {"name": exp["name"], "status": "dry_run", "branch": branch}

    started = time.time()
    supervisor_log = STATE_DIR / f"{exp['name']}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    code = run_checked(cmd, env=env, log_path=supervisor_log)
    run_tag = env_items.get("LFTS_RUN_TAG") or exp["preset"]
    log_path = latest_matching(f"outputs/logs/*_{run_tag}.txt", started_at=started)
    pred_path = latest_matching(f"outputs/predictions/*_{run_tag}_seed*_predictions.npz", started_at=started)
    split_pred = latest_matching(f"outputs/predictions/*_{run_tag}_seed*_split_predictions.npz", started_at=started)

    record: dict[str, Any] = {
        "name": exp["name"],
        "status": "complete" if code == 0 else "failed",
        "branch": branch,
        "base_branch": base_branch,
        "hypothesis": exp.get("hypothesis"),
        "command": cmd,
        "gpus": gpus,
        "run_tag": run_tag,
        "started_at": dt.datetime.fromtimestamp(started).isoformat(),
        "finished_at": dt.datetime.now().isoformat(),
        "exit_code": code,
        "log": str(log_path) if log_path else None,
        "prediction": str(pred_path) if pred_path else None,
        "split_prediction": str(split_pred) if split_pred else None,
        "env": env_items,
        "supervisor_log": str(supervisor_log),
    }
    if code == 0 and log_path is not None:
        record["metrics"] = parse_metrics(log_path)
    else:
        record["metrics"] = {}

    record_path = RECORD_DIR / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp['name']}.json"
    write_json(record_path, record)
    if code == 0:
        commit_record(record_path, record["metrics"], exp)
        if env_items.get("LFTS_RISK_WEIGHT_PATH"):
            state["best_risk_weight_path"] = env_items.get("LFTS_RISK_WEIGHT_PATH")
            state["best_risk_class_weights"] = env_items.get("LFTS_RISK_CLASS_WEIGHTS", "")
    return record


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_json(cfg_path)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RECORD_DIR.mkdir(exist_ok=True)
    state_path = STATE_DIR / "state.json"
    state = load_json(state_path) if state_path.exists() else {"runs": [], "best": None}
    experiments = list(cfg.get("experiments", []))
    if args.start_at:
        names = [exp["name"] for exp in experiments]
        if args.start_at not in names:
            raise ValueError(f"Unknown --start-at {args.start_at!r}; choices={names}")
        experiments = experiments[names.index(args.start_at):]
    if args.max_runs is not None:
        experiments = experiments[: max(0, args.max_runs)]

    target = float(cfg.get("target_accuracy", 0.85))
    for exp in experiments:
        existing = state_lookup(state, str(exp["name"]))
        if existing is not None:
            print(f"Skipping completed experiment {exp['name']}")
            continue
        record = run_experiment(cfg, exp, state, args)
        state.setdefault("runs", []).append(record)
        metrics = record.get("metrics", {})
        acc = float(metrics.get("accuracy", 0.0))
        best = state.get("best")
        if record.get("status") == "complete" and (best is None or acc > float(best.get("metrics", {}).get("accuracy", 0.0))):
            state["best"] = record
        elif record.get("status") == "complete" and best is not None and acc < float(best.get("metrics", {}).get("accuracy", 0.0)):
            record["regressed_from"] = best.get("branch")
            git(["checkout", str(best.get("branch"))])
        write_json(state_path, state)
        if acc >= target:
            print(f"Target reached: accuracy={acc:.4f} >= {target:.4f}")
            break


if __name__ == "__main__":
    main()
