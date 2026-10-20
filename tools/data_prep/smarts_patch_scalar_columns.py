# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scalarize list-typed scalar columns in smarts v3.0 data parquets.

preprocess_smarts_v21.py copied the source's ``timestamp`` / ``frame_index`` /
``task_index`` columns as-is; in the smarts source these are 1-element
list<double>/list<int> arrays, but info.json (and LeRobot) declare them as
scalar float32[1] / int64[1]. LeRobot's ``cast_table_to_schema`` then throws
``Couldn't cast list<double> to float`` on the first sample fetch -> the leaf is
unloadable. This rewrites each data parquet, replacing any column that info.json
declares scalar (shape [1]) but is stored as a list, with the list's first
element cast to the declared dtype. Idempotent (skips already-scalar columns).

    python tools/data_prep/smarts_patch_scalar_columns.py --root <v30 smarts dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_NP = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32}
_PA = {"float32": pa.float32(), "float64": pa.float64(), "int64": pa.int64(), "int32": pa.int32()}


def _patch(parquet: str, info: dict) -> str:
    t = pq.read_table(parquet)
    scalar_feats = {n: f for n, f in info["features"].items()
                    if f.get("shape") == [1] and f.get("dtype") in _NP}
    changed = []
    cols = {}
    for name in t.column_names:
        col = t.column(name)
        if name in scalar_feats and pa.types.is_list(col.type):
            vals = col.to_pylist()  # each is a 1-element list
            flat = np.array([v[0] if isinstance(v, list) and v else v for v in vals],
                            dtype=_NP[scalar_feats[name]["dtype"]])
            cols[name] = pa.array(flat, type=_PA[scalar_feats[name]["dtype"]])
            changed.append(name)
        else:
            cols[name] = col
    if not changed:
        return "already-scalar"
    tmp = parquet + ".tmp"
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, parquet)
    return "scalarized " + ",".join(changed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    leaves = sorted({p.split("/data/")[0] for p in glob.glob(os.path.join(args.root, "**", "data", "**", "*.parquet"), recursive=True)})
    if not leaves:
        raise SystemExit(f"no data parquets under {args.root}")
    for d in leaves:
        info = json.load(open(os.path.join(d, "meta", "info.json")))
        for pqf in glob.glob(os.path.join(d, "data", "**", "*.parquet"), recursive=True):
            leaf = d.replace(args.root.rstrip("/") + "/", "")
            print(f"{leaf}: {_patch(pqf, info)}")


if __name__ == "__main__":
    main()
