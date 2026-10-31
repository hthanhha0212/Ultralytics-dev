#!/usr/bin/env python3
"""
Update YOLO label files by changing class id 1 to 0.

Usage:
  python tools/update_label_ids.py --labels-dir path/to/labels
  python tools/update_label_ids.py --labels-dir path/to/labels --no-recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Change class id 1 to 0 in .txt label files.")
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Folder containing .txt label files.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process .txt files in the top-level folder.",
    )
    return parser.parse_args()


def update_file(path: Path) -> tuple[int, bool]:
    """Return (lines_updated, changed)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    changed = False
    updated_lines = 0

    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        parts = line.split()
        if parts and parts[0] == "1":
            parts[0] = "0"
            changed = True
            updated_lines += 1
        updated.append(" ".join(parts))

    if changed:
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return updated_lines, changed


def main() -> int:
    args = parse_args()
    labels_dir = args.labels_dir

    if not labels_dir.is_dir():
        raise SystemExit(f"Not a directory: {labels_dir}")

    pattern = "*.txt"
    files = labels_dir.glob(pattern) if args.no_recursive else labels_dir.rglob(pattern)

    total_files = 0
    changed_files = 0
    total_lines = 0

    for path in files:
        if not path.is_file():
            continue
        total_files += 1
        lines_updated, changed = update_file(path)
        total_lines += lines_updated
        if changed:
            changed_files += 1

    print(f"Processed {total_files} files; updated {changed_files} files; changed {total_lines} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
