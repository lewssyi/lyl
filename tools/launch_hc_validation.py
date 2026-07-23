#!/usr/bin/env python
"""Guarded one-serial-worker-per-GPU HC validation launcher."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
ROOT = PROJECT / "outputs/hierarchical_context_xai_feasibility/isruc_seed42"
PYTHON = Path("/home/b502/app/anaconda3/envs/l_venv/bin/python")
GPU_COUNT = 8
LEVELS = ("R0", "R3", "R5", "R7")
SEEDS = (3101, 3102, 3103)
lock = threading.Lock()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def all_tasks() -> list[dict]:
    samples = [
        int(row["global_epoch_index"])
        for row in read_json(ROOT / "validation_samples.json")["samples"]
    ]
    survivors = read_json(ROOT / "development_survivors.json")[
        "selected_for_validation"
    ]
    tasks = []
    for sample_id in samples:
        for level in LEVELS:
            for seed in SEEDS:
                tasks.append(
                    {
                        "kind": "classic",
                        "sample_id": sample_id,
                        "level": level,
                        "seed": seed,
                    }
                )
                for survivor in survivors:
                    tasks.append(
                        {
                            "kind": "hc",
                            "sample_id": sample_id,
                            "level": level,
                            "seed": seed,
                            **survivor,
                        }
                    )
    return tasks


def run_id(task: dict) -> str:
    if task["kind"] == "classic":
        return (
            f"validation_classic_sample{task['sample_id']}_{task['level']}_"
            f"seed{task['seed']}_zero"
        )
    return (
        f"validation_validation_sample{task['sample_id']}_"
        f"{task['group_scheme']}_f{task['feature_count']}_{task['level']}_"
        f"seed{task['seed']}_zero"
    )


def required_files(task: dict) -> tuple[str, ...]:
    if task["kind"] == "classic":
        return (
            "classic_sample_level.csv",
            "classic_attribution.csv",
            "runtime_query_summary.csv",
        )
    return (
        "hierarchical_sample_level.csv",
        "context_group_importance.csv",
        "within_epoch_attribution.csv",
        "runtime_query_summary.csv",
    )


def is_complete(task: dict) -> bool:
    directory = ROOT / "runs" / run_id(task)
    manifest = directory / "run_manifest.json"
    required = tuple(directory / name for name in required_files(task))
    if (
        not manifest.exists()
        or any(not path.exists() or path.stat().st_size <= 1 for path in required)
    ):
        return False
    payload = read_json(manifest)
    return bool(
        payload.get("status") == "COMPLETE"
        and payload.get("all_outputs_nonempty")
        and int(payload.get("logical_query_budget_per_candidate", -1)) == 1530
    )


def command(task: dict) -> list[str]:
    common = [
        "--stage",
        "validation",
        "--data-split",
        "validation",
        "--sample-id",
        str(task["sample_id"]),
        "--level",
        task["level"],
        "--seed",
        str(task["seed"]),
    ]
    if task["kind"] == "classic":
        return [str(PYTHON), "tools/hc_classic_baseline.py", *common]
    return [
        str(PYTHON),
        "tools/hierarchical_context_xai_feasibility.py",
        *common,
        "--group-scheme",
        task["group_scheme"],
        "--feature-count",
        str(task["feature_count"]),
        "--candidates",
        task["candidate"],
        "--replacement",
        "zero",
    ]


def worker(gpu: int, assigned: list[dict], state: dict) -> None:
    status_path = ROOT / f"validation_worker_{gpu}_status.json"
    for task in assigned:
        identifier = run_id(task)
        if is_complete(task):
            with lock:
                state["skipped_complete"] += 1
                write_json(
                    status_path,
                    {
                        "status": "RUNNING",
                        "gpu": gpu,
                        "last": identifier,
                        "result": "SKIPPED_COMPLETE",
                    },
                )
            continue
        log = ROOT / "logs/validation" / f"{identifier}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with lock:
            state["active"][str(gpu)] = identifier
            write_json(ROOT / "validation_launcher_status.json", state)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command(task),
                cwd=PROJECT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        with lock:
            state["active"].pop(str(gpu), None)
            if process.returncode == 0 and is_complete(task):
                state["completed"] += 1
                result = "COMPLETE"
            else:
                state["failed"].append(
                    {
                        "run_id": identifier,
                        "returncode": process.returncode,
                        "log": str(log.relative_to(PROJECT)),
                    }
                )
                result = "FAILED"
            write_json(
                status_path,
                {
                    "status": "RUNNING",
                    "gpu": gpu,
                    "last": identifier,
                    "result": result,
                },
            )
            write_json(ROOT / "validation_launcher_status.json", state)
    write_json(status_path, {"status": "COMPLETE", "gpu": gpu})


def main() -> None:
    tasks = all_tasks()
    expected = 30 * 4 * 3 * 3
    if len(tasks) != expected:
        raise RuntimeError(f"validation task coverage drift: {len(tasks)} != {expected}")
    state = {
        "schema_version": "hc_seqlime_validation_launcher_v1",
        "status": "RUNNING",
        "expected": expected,
        "completed": 0,
        "skipped_complete": 0,
        "failed": [],
        "active": {},
        "gpu_workers": GPU_COUNT,
        "automatic_retry": False,
        "test_execution_count": 0,
        "selection_split": "validation",
    }
    write_json(ROOT / "validation_launcher_status.json", state)
    assignments = [tasks[gpu::GPU_COUNT] for gpu in range(GPU_COUNT)]
    with ThreadPoolExecutor(max_workers=GPU_COUNT) as executor:
        futures = [
            executor.submit(worker, gpu, assignments[gpu], state)
            for gpu in range(GPU_COUNT)
        ]
        for future in futures:
            future.result()
    state["status"] = (
        "COMPLETE" if not state["failed"] else "COMPLETE_WITH_FAILURES"
    )
    state["active"] = {}
    write_json(ROOT / "validation_launcher_status.json", state)
    print(json.dumps(state, indent=2))
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
