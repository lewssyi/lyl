#!/usr/bin/env python
"""Build a source-to-target cross-dataset protocol without touching raw data.

The generated protocol is compatible with tools/run_main_with_protocol_split.py.
It creates an isolated NPZ view directory containing prefixed links such as
``sleepedf__SC4001E0.npz`` and ``isruc__subject01.npz``. By default, the script
uses hardlinks first and then symlinks, so the PSG arrays are not duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT.parent / "SleepEDF" / "npz_epoch3"
DEFAULT_TARGET_DIR = PROJECT_ROOT.parent / "ISRUC-Sleep" / "npz_epoch3"
DEFAULT_PROTOCOL_ROOT = PROJECT_ROOT / "dataset_protocols"
DEFAULT_VIEW_ROOT = PROJECT_ROOT / "tmp"
CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "dataset"


def infer_subject_id_from_npz(filename: str) -> str:
    stem = Path(filename).stem
    upper_stem = stem.upper()
    if len(upper_stem) >= 5 and upper_stem[:2] in {"SC", "ST"} and upper_stem[2:5].isdigit():
        return upper_stem[:5]
    return stem


def list_npz_files(npz_dir: Path, max_files: int | None = None) -> List[str]:
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    files = sorted(path.name for path in npz_dir.glob("*.npz"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise ValueError(f"No .npz files found in {npz_dir}")
    return files


def group_files_by_subject(files: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for filename in sorted(files):
        groups.setdefault(infer_subject_id_from_npz(filename), []).append(filename)
    return {subject: sorted(subject_files) for subject, subject_files in sorted(groups.items())}


def split_source_subjects(
    source_files: Sequence[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    groups = group_files_by_subject(source_files)
    subjects = np.array(sorted(groups), dtype=object)
    if len(subjects) == 1:
        return sorted(groups[str(subjects[0])]), []

    rng = np.random.default_rng(int(seed))
    shuffled = subjects[rng.permutation(len(subjects))].tolist()
    n_val = int(round(len(shuffled) * float(val_ratio)))
    if val_ratio > 0 and len(shuffled) >= 2:
        n_val = max(1, n_val)
    n_val = min(n_val, len(shuffled) - 1)

    val_subjects = set(str(subject) for subject in shuffled[:n_val])
    train_files: List[str] = []
    val_files: List[str] = []
    for subject, subject_files in groups.items():
        if subject in val_subjects:
            val_files.extend(subject_files)
        else:
            train_files.extend(subject_files)
    return sorted(train_files), sorted(val_files)


def prefixed_filename(prefix: str, filename: str) -> str:
    return f"{sanitize_token(prefix)}__{Path(filename).name}"


def ensure_clean_child(parent: Path, child: Path) -> None:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve(strict=False)
    if parent_resolved != child_resolved and parent_resolved not in child_resolved.parents:
        raise ValueError(f"Refusing to write outside {parent_resolved}: {child_resolved}")


def link_or_copy_npz(
    source_path: Path,
    dest_path: Path,
    mode: str,
    overwrite: bool,
) -> str:
    ensure_clean_child(dest_path.parent, dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() or dest_path.is_symlink():
        if not overwrite:
            return "existing"
        dest_path.unlink()

    if mode == "hardlink":
        os.link(source_path, dest_path)
        return "hardlink"
    if mode == "symlink":
        os.symlink(source_path, dest_path)
        return "symlink"
    if mode == "copy":
        shutil.copy2(source_path, dest_path)
        return "copy"

    errors: List[str] = []
    for candidate in ("hardlink", "symlink"):
        try:
            return link_or_copy_npz(source_path, dest_path, candidate, overwrite=False)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    raise OSError(
        "Could not create hardlink or symlink. Re-run with --link-mode copy if you "
        f"intentionally want to duplicate data. Details: {'; '.join(errors)}"
    )


def inspect_npz(npz_dir: Path, files: Sequence[str]) -> Dict[str, object]:
    total_epochs = 0
    class_counts = np.zeros(len(CLASS_LABELS), dtype=np.int64)
    sample_shapes = {}
    label_min = None
    label_max = None
    for filename in files:
        with np.load(npz_dir / filename, mmap_mode="r") as data:
            x_shape = tuple(int(dim) for dim in data["x"].shape)
            labels = np.asarray(data["y"], dtype=np.int32).reshape(-1)
        total_epochs += int(labels.size)
        sample_shapes[str(x_shape[1:])] = sample_shapes.get(str(x_shape[1:]), 0) + 1
        if labels.size:
            label_min = int(labels.min()) if label_min is None else min(label_min, int(labels.min()))
            label_max = int(labels.max()) if label_max is None else max(label_max, int(labels.max()))
            bincount = np.bincount(labels, minlength=len(CLASS_LABELS))
            class_counts += bincount[: len(CLASS_LABELS)]
    return {
        "files": int(len(files)),
        "epochs": int(total_epochs),
        "sample_shapes": sample_shapes,
        "label_min": label_min,
        "label_max": label_max,
        "class_counts": {
            label: int(count)
            for label, count in zip(CLASS_LABELS, class_counts.tolist())
        },
    }


def build_protocol(args: argparse.Namespace) -> Mapping[str, object]:
    source_name = str(args.source_name)
    target_name = str(args.target_name)
    protocol_name = args.protocol_name or f"cross_{sanitize_token(source_name)}_to_{sanitize_token(target_name)}"

    source_dir = Path(args.source_npz_dir).expanduser().resolve()
    target_dir = Path(args.target_npz_dir).expanduser().resolve()
    protocol_dir = Path(args.output_dir).expanduser().resolve() / protocol_name
    view_dir = Path(args.view_dir).expanduser().resolve() if args.view_dir else (
        DEFAULT_VIEW_ROOT / f"{protocol_name}_npz"
    ).resolve()

    source_files = list_npz_files(source_dir, args.max_source_files)
    target_files = list_npz_files(target_dir, args.max_target_files)
    source_train, source_val = split_source_subjects(
        source_files,
        val_ratio=float(args.source_val_ratio),
        seed=int(args.seed),
    )

    split_original = {
        "train": source_train,
        "val": source_val,
        "test": target_files,
    }
    split_prefixed = {
        "train": [prefixed_filename(source_name, filename) for filename in source_train],
        "val": [prefixed_filename(source_name, filename) for filename in source_val],
        "test": [prefixed_filename(target_name, filename) for filename in target_files],
    }

    link_counts: Dict[str, int] = {}
    source_mapping: Dict[str, str] = {}
    for split, filenames in split_original.items():
        raw_dir = target_dir if split == "test" else source_dir
        raw_name = target_name if split == "test" else source_name
        for filename in filenames:
            destination_name = prefixed_filename(raw_name, filename)
            method = link_or_copy_npz(
                raw_dir / filename,
                view_dir / destination_name,
                mode=str(args.link_mode),
                overwrite=bool(args.overwrite),
            )
            link_counts[method] = link_counts.get(method, 0) + 1
            source_mapping[destination_name] = str(raw_dir / filename)

    source_train_report = inspect_npz(source_dir, source_train)
    source_val_report = inspect_npz(source_dir, source_val) if source_val else {
        "files": 0,
        "epochs": 0,
        "sample_shapes": {},
        "label_min": None,
        "label_max": None,
        "class_counts": {label: 0 for label in CLASS_LABELS},
    }
    target_test_report = inspect_npz(target_dir, target_files)

    protocol_dir.mkdir(parents=True, exist_ok=True)
    split_json = {
        "protocol": "cross_dataset_source_to_target_v1",
        "source_npz_dir": str(view_dir),
        "split_files": split_prefixed,
        "cross_dataset": {
            "source_name": source_name,
            "target_name": target_name,
            "source_npz_dir": str(source_dir),
            "target_npz_dir": str(target_dir),
            "source_val_ratio": float(args.source_val_ratio),
            "seed": int(args.seed),
            "test_policy": "all_target_files",
            "view_dir": str(view_dir),
            "link_mode": str(args.link_mode),
        },
    }
    report = {
        "protocol_name": protocol_name,
        "protocol_dir": str(protocol_dir),
        "view_dir": str(view_dir),
        "link_counts": link_counts,
        "split_original_files": split_original,
        "split_prefixed_files": split_prefixed,
        "source_mapping": source_mapping,
        "split_reports": {
            "train": source_train_report,
            "val": source_val_report,
            "test": target_test_report,
        },
    }

    with (protocol_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_json, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (protocol_dir / "protocol_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (protocol_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            f"# {source_name} -> {target_name} Cross-Dataset Protocol\n\n"
            "This protocol is isolated from the original single-dataset protocols.\n\n"
            f"- NPZ view: `{view_dir}`\n"
            f"- Source train/val: `{source_dir}`\n"
            f"- Target test: `{target_dir}`\n"
            "- Target files are used only as test data.\n\n"
            "Run with:\n\n"
            "```powershell\n"
            f"python tools\\run_main_with_protocol_split.py --split-json {protocol_dir / 'split.json'} --dataset-preset edf_c0_no_concept --seed {int(args.seed)}\n"
            "```\n"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated SleepEDF-to-ISRUC style cross-dataset protocol."
    )
    parser.add_argument("--source-name", default="SleepEDF")
    parser.add_argument("--target-name", default="ISRUC")
    parser.add_argument("--source-npz-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--target-npz-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--protocol-name", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_PROTOCOL_ROOT))
    parser.add_argument("--view-dir", default=None)
    parser.add_argument("--source-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-files", type=int, default=None)
    parser.add_argument("--max-target-files", type=int, default=None)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help="auto tries hardlink then symlink; it will not copy unless --link-mode copy is used.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files in the generated view.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_protocol(args)
    split_reports = report["split_reports"]
    print("Cross-dataset protocol built.")
    print(f"protocol_dir={report['protocol_dir']}")
    print(f"view_dir={report['view_dir']}")
    print(f"link_counts={report['link_counts']}")
    for split_name in ("train", "val", "test"):
        split_report = split_reports[split_name]
        print(
            f"{split_name}: files={split_report['files']}, "
            f"epochs={split_report['epochs']}, "
            f"shapes={split_report['sample_shapes']}, "
            f"labels=({split_report['label_min']}, {split_report['label_max']}), "
            f"class_counts={split_report['class_counts']}"
        )


if __name__ == "__main__":
    main()
