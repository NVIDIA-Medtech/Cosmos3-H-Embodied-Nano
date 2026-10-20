# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Add a LeRobot-standard ``tasks`` (list<string>) column to smarts v3.0 episode
metadata, derived from the existing singular ``task`` column.

The smarts source stores prompts under the singular ``task`` key; the preprocess
carried that through, so the converted v3.0 episodes parquet has a ``task``
column but not the plural ``tasks`` list that LeRobot's meta loader and
``_filter_valid_episodes`` read (every other Open-H leaf has ``tasks``). Without
it, every smarts episode looks prompt-less and the whole leaf is dropped as
"empty prompt". This patches the (tiny, KB-scale) episodes parquet in place —
no video recopy.

Idempotent: skips leaves that already have a ``tasks`` column.

    python tools/data_prep/smarts_patch_tasks_column.py --root <v30 smarts dir>
"""
from __future__ import annotations

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq


def _patch_parquet(p: str) -> str:
    t = pq.read_table(p)
    if "tasks" in t.column_names:
        return "already-has-tasks"
    if "task" not in t.column_names:
        return "no-task-column"
    task_vals = t.column("task").to_pylist()
    tasks_col = pa.array([[v] if v else [] for v in task_vals], type=pa.list_(pa.string()))
    t = t.append_column("tasks", tasks_col)
    tmp = p + ".tmp"
    pq.write_table(t, tmp)
    os.replace(tmp, p)
    n_empty = sum(1 for v in task_vals if not v)
    return f"patched {len(task_vals)} eps ({n_empty} empty)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="v30 SMARTS directory (searched recursively)")
    args = ap.parse_args()
    pfs = sorted(glob.glob(os.path.join(args.root, "**", "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not pfs:
        raise SystemExit(f"no episodes parquet under {args.root}")
    for p in pfs:
        leaf = p.split("/meta/")[0].replace(args.root.rstrip("/") + "/", "")
        print(f"{leaf}: {_patch_parquet(p)}")


if __name__ == "__main__":
    main()
