import argparse
import os
import subprocess
import sys


DEFAULT_PRESETS = [
    "baseline",
    "mixstyle",
    "mixstyle_coral",
    "mixstyle_coral_tent",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run paired cross-dataset generalization ablations for run_main.py."
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=DEFAULT_PRESETS,
        help="Preset names to run in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without launching training.",
    )
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    run_main_path = os.path.join(project_dir, "run_main.py")

    for preset in args.presets:
        env = os.environ.copy()
        env["LFTS_GENERALIZATION_PRESET"] = preset
        command = [sys.executable, run_main_path]
        print(f"\n=== Generalization preset: {preset} ===")
        print(" ".join(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=project_dir, env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
