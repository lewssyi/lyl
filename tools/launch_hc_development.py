#!/usr/bin/env python
"""One-serial-worker-per-GPU launcher for HC-SeqLIME development."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"
PYTHON = Path("/home/b502/app/anaconda3/envs/l_venv/bin/python")
GPU_COUNT = 8
GROUP_SCHEMES = ("G15", "G9", "G25")
FEATURE_COUNTS = (30, 45)
LEVELS = ("R0", "R3", "R5", "R7")
SEED = 3101
lock = threading.Lock()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_id(task: dict) -> str:
    return (
        f"development_development_sample{task['sample_id']}_{task['group_scheme']}_"
        f"f{task['feature_count']}_{task['level']}_seed{SEED}_zero"
    )


def is_complete(task: dict) -> bool:
    directory = ROOT / "runs" / run_id(task)
    manifest = directory / "run_manifest.json"
    required = (
        directory / "hierarchical_sample_level.csv",
        directory / "context_group_importance.csv",
        directory / "within_epoch_attribution.csv",
        directory / "runtime_query_summary.csv",
    )
    if not manifest.exists() or any(not path.exists() or path.stat().st_size <= 1 for path in required):
        return False
    payload = read_json(manifest)
    return bool(
        payload.get("status") == "COMPLETE"
        and payload.get("all_outputs_nonempty")
        and int(payload.get("logical_query_budget_per_candidate", -1)) == 1530
    )


def tasks() -> list[dict]:
    manifest = read_json(ROOT / "development_samples.json")
    sample_ids = [int(row["global_epoch_index"]) for row in manifest["samples"]]
    result = []
    for sample_id in sample_ids:
        for group_scheme in GROUP_SCHEMES:
            for feature_count in FEATURE_COUNTS:
                for level in LEVELS:
                    result.append({
                        "sample_id": sample_id,
                        "group_scheme": group_scheme,
                        "feature_count": feature_count,
                        "level": level,
                    })
    return result


def worker(gpu: int, assigned: list[dict], state: dict) -> None:
    worker_status = ROOT / f"development_worker_{gpu}_status.json"
    for task in assigned:
        identifier = run_id(task)
        if is_complete(task):
            with lock:
                state["skipped_complete"] += 1
                write_json(worker_status, {"status": "RUNNING", "gpu": gpu, "last": identifier, "result": "SKIPPED_COMPLETE"})
            continue
        log = ROOT / "logs/development" / f"{identifier}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), "tools/hierarchical_context_xai_feasibility.py",
            "--stage", "development", "--data-split", "development",
            "--sample-id", str(task["sample_id"]), "--group-scheme", task["group_scheme"],
            "--feature-count", str(task["feature_count"]), "--candidates", "H0,H1,H2,H3",
            "--replacement", "zero", "--level", task["level"], "--seed", str(SEED),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with lock:
            state["active"][str(gpu)] = identifier
            write_json(ROOT / "development_launcher_status.json", state)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=PROJECT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        with lock:
            state["active"].pop(str(gpu), None)
            if process.returncode == 0 and is_complete(task):
                state["completed"] += 1
                result = "COMPLETE"
            else:
                state["failed"].append({"run_id": identifier, "returncode": process.returncode, "log": str(log.relative_to(PROJECT))})
                result = "FAILED"
            write_json(worker_status, {"status": "RUNNING", "gpu": gpu, "last": identifier, "result": result})
            write_json(ROOT / "development_launcher_status.json", state)
    write_json(worker_status, {"status": "COMPLETE", "gpu": gpu})


def main() -> None:
    all_tasks = tasks()
    state = {
        "schema_version": "hc_seqlime_development_launcher_v1",
        "status": "RUNNING",
        "expected": len(all_tasks),
        "completed": 0,
        "skipped_complete": 0,
        "failed": [],
        "active": {},
        "gpu_workers": GPU_COUNT,
        "automatic_retry": False,
    }
    assignments = [all_tasks[gpu::GPU_COUNT] for gpu in range(GPU_COUNT)]
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        futures = [executor.submit(worker, gpu, assignments[gpu], state) for gpu in range(GPU_COUNT)]
        for future in futures:
            future.result()
    state["status"] = "COMPLETE" if not state["failed"] else "COMPLETE_WITH_FAILURES"
    state["active"] = {}
    write_json(ROOT / "development_launcher_status.json", state)
    print(json.dumps(state, indent=2))
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
