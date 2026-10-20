# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fix the smarts off-by-one prompt bug: rewrite ``task_index`` to be 0-based
positional (the LeRobot invariant) instead of the sparse source task IDs the
conversion carried through.

ROOT CAUSE. smarts leaves store the source's original
GLOBAL task ids in ``task_index`` (e.g. P2 = {1, 2, 6}) and in
``meta/tasks.parquet`` (row order [1, 2, 6]). But lerobot resolves a frame's
prompt POSITIONALLY::

    # lerobot/datasets/lerobot_dataset.py:1076
    task_idx = item["task_index"].item()
    item["task"] = self.meta.tasks.iloc[task_idx].name   # iloc == 0-based row

With 1-based / sparse ids, ``iloc[task_idx]`` returns the WRONG string (every
stitch frame, task_index=1, gets tasks.iloc[1] = the cold-cut prompt) and the
high ids (task_index=6, only 3 rows) raise IndexError -> the frame is silently
dropped by the bad-sample skip. Net: consistently wrong / missing prompts.

FIX. For each leaf build ``remap = {old_task_index: row_position}`` from the
tasks.parquet ROW ORDER, then rewrite:
  * meta/tasks.parquet    ``task_index`` -> 0..N-1 (row position); strings/order kept
  * data/**.parquet       per-frame ``task_index`` -> remap[value]
  * meta/episodes/**       ``task_index`` + ``stats/task_index/{min,max,mean}`` -> remap
The ``task`` / ``tasks`` STRING columns are already correct and untouched.

Idempotent: a leaf already 0-based-positional (remap is identity) is skipped.
NOTHING else keys on task_index (normalizer = action stats; latent cache =
(dataset,idx)), so this needs no recompute of either. Read/writes only tiny int
columns + KB metadata — NO video recopy.

    python tools/data_prep/smarts_fix_task_index_positional.py --root <v30 dir containing the SMARTS leaves>
    python tools/data_prep/smarts_fix_task_index_positional.py --root ... --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq


def build_remap(task_index_row_values) -> dict[int, int]:
    """Map each stored ``task_index`` value to its 0-based ROW POSITION in
    tasks.parquet — the identity lerobot's positional ``iloc`` lookup assumes."""
    return {int(v): i for i, v in enumerate(task_index_row_values)}


def is_identity(remap: dict[int, int]) -> bool:
    """True iff task_index already equals row position for every task (0-based)."""
    return all(k == v for k, v in remap.items())


def _remap_int_column(table: pa.Table, name: str, remap: dict[int, int]) -> pa.Table:
    """Return ``table`` with integer column ``name`` mapped through ``remap``.

    Handles scalar-int and 1-element-list storage. Values absent from ``remap``
    are left unchanged (defensive; should not happen for a self-consistent leaf).
    """
    if name not in table.column_names:
        return table
    col = table.column(name)
    is_list = pa.types.is_list(col.type)
    out = []
    for v in col.to_pylist():
        if isinstance(v, list):
            out.append([remap.get(int(x), x) for x in v])
        elif v is None:
            out.append(v)
        else:
            out.append(remap.get(int(v), v))
    new_col = pa.array(out, type=col.type)
    i = table.column_names.index(name)
    return table.set_column(i, name, new_col)


def _write_inplace(table: pa.Table, path: str) -> None:
    tmp = path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _tasks_row_values(tasks_path: str):
    """The ``task_index`` column of tasks.parquet, IN ROW ORDER."""
    t = pq.read_table(tasks_path)
    if "task_index" not in t.column_names:
        raise ValueError(f"{tasks_path}: no task_index column (cols={t.column_names})")
    return t.column("task_index").to_pylist(), t


def fix_leaf(leaf: str, dry_run: bool = False) -> str:
    tasks_path = os.path.join(leaf, "meta", "tasks.parquet")
    if not os.path.exists(tasks_path):
        return "no-tasks.parquet"
    row_vals, tasks_tbl = _tasks_row_values(tasks_path)
    remap = build_remap(row_vals)
    if is_identity(remap):
        return f"already-positional (task_index={row_vals})"
    if dry_run:
        return f"WOULD remap {remap} (data+episodes+tasks)"

    # 1) tasks.parquet: task_index -> 0..N-1 (row position), strings/order kept.
    ti_i = tasks_tbl.column_names.index("task_index")
    tasks_fixed = tasks_tbl.set_column(
        ti_i, "task_index", pa.array(list(range(tasks_tbl.num_rows)), type=tasks_tbl.column("task_index").type))
    _write_inplace(tasks_fixed, tasks_path)

    # 2) data/**.parquet: per-frame task_index -> remap
    n_data = 0
    for p in sorted(glob.glob(os.path.join(leaf, "data", "**", "*.parquet"), recursive=True)):
        t = pq.read_table(p)
        _write_inplace(_remap_int_column(t, "task_index", remap), p)
        n_data += 1

    # 3) meta/episodes/**.parquet: task_index + stats/task_index/{min,max,mean}
    n_ep = 0
    for p in sorted(glob.glob(os.path.join(leaf, "meta", "episodes", "**", "*.parquet"), recursive=True)):
        t = pq.read_table(p)
        for c in ("task_index", "stats/task_index/min", "stats/task_index/max", "stats/task_index/mean"):
            t = _remap_int_column(t, c, remap)
        _write_inplace(t, p)
        n_ep += 1

    return f"remapped {remap}: tasks + {n_data} data + {n_ep} episodes parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir containing leaves (each with meta/tasks.parquet)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    leaves = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)
                    and os.path.exists(os.path.join(d, "meta", "tasks.parquet")))
    if not leaves:
        raise SystemExit(f"no leaves with meta/tasks.parquet under {args.root}")
    for leaf in leaves:
        print(f"{os.path.basename(leaf)}: {fix_leaf(leaf, dry_run=args.dry_run)}")


if __name__ == "__main__":
    main()
