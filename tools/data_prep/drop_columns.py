# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Drop per-frame columns from a converted LeRobot v3.0 dataset, in place.

Used for columns the training loader cannot tensorize or LeRobot cannot load, e.g. the
string tool-name columns of Hamlyn (``observation.meta.{left,right}_arm_tool``) and
rob_surgical (``observation.meta.{left,right,aux}_tool``), and Stanford's
``observation.meta.tool_type`` (dtype ``text``, rejected by LeRobot).

Each column is removed from:
  * ``data/**/*.parquet``
  * ``meta/info.json`` ``features``
  * ``meta/episodes/**/*.parquet`` ``stats/<column>/*``
  * ``meta/stats.json``

``--keep-features`` removes the column from the data files only (for leftover columns that
duplicate a video feature, e.g. v2.1 image-path strings). Idempotent: absent columns are skipped.

    python tools/data_prep/drop_columns.py --root <v30 dataset root> observation.meta.tool_type
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def _write_inplace(table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def drop_data_columns(root: Path, columns: list[str]) -> int:
    changed = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        present = [c for c in columns if c in table.column_names]
        if present:
            _write_inplace(table.drop_columns(present), path)
            changed += 1
    return changed


def drop_metadata(root: Path, columns: list[str]) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if any(c in info["features"] for c in columns):
        info["features"] = {k: v for k, v in info["features"].items() if k not in columns}
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        if any(c in stats for c in columns):
            stats_path.write_text(json.dumps({k: v for k, v in stats.items() if k not in columns}, indent=4) + "\n")

    prefixes = tuple(f"stats/{c}/" for c in columns)
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        present = [c for c in table.column_names if c.startswith(prefixes)]
        if present:
            _write_inplace(table.drop_columns(present), path)


def drop_columns(root: Path, columns: list[str], keep_features: bool = False) -> None:
    n = drop_data_columns(root, columns)
    if not keep_features:
        drop_metadata(root, columns)
    print(f"{root}: dropped {columns} ({n} data files changed)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, action="append", help="v3.0 dataset root (repeatable)")
    ap.add_argument("--keep-features", action="store_true", help="drop from the data files only")
    ap.add_argument("columns", nargs="+")
    args = ap.parse_args()
    for root in args.root:
        drop_columns(root, args.columns, args.keep_features)


if __name__ == "__main__":
    main()
