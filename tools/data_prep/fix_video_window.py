# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Clip video windows that overshoot the kinematics length (JHU nephfat, applied after step T).

Fix episodes whose video window overshoots the kinematics length: set
``to_timestamp = from_timestamp + length / native_fps`` for EVERY video key of
each offending episode (all cameras share one timeline).

Motivating case: nephfat ep 1504 has length=337 but endoscope window [11.567,
35.367] = 23.8s = 714 frames @30fps. Decoding that window yields 714 frames for a
337-row episode -> the loader's frame_index->timestamp map lands on the wrong
frames (or the sample is skipped). The kinematics ``length`` is ground truth (it's
the data-parquet row count), so we clip the VIDEO window to match it. Episodes
already consistent (|expected-length|<=1) are untouched. Idempotent.

Read/writes only the tiny episodes parquet (float timestamp columns) — NO video
recopy, NO data-parquet change, NO normalizer recompute.

    python tools/data_prep/fix_video_window.py --root <leaf> [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def find_video_keys(column_names) -> list[str]:
    keys = []
    for c in column_names:
        if c.startswith("videos/") and c.endswith("/from_timestamp"):
            keys.append(c[len("videos/"):-len("/from_timestamp")])
    return keys


def corrected_to_timestamp(from_ts: float, length: int, fps: float) -> float:
    """Video window end that matches the kinematics length exactly."""
    return from_ts + length / fps


def needs_fix(from_ts: float, to_ts: float, length: int, fps: float, tol_frames: float = 1.0) -> bool:
    if from_ts is None or to_ts is None or not length:
        return False
    expected = (to_ts - from_ts) * fps
    return abs(expected - length) > tol_frames


def _scalar(v):
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def fix_leaf(leaf: str, dry_run: bool = False) -> str:
    fps = json.load(open(os.path.join(leaf, "meta", "info.json"))).get("fps")
    if not fps:
        return "no-fps"
    parquets = sorted(glob.glob(os.path.join(leaf, "meta", "episodes", "**", "*.parquet"), recursive=True))
    total_fixed = 0
    fixed_eps = []
    for p in parquets:
        t = pq.read_table(p)
        cn = t.column_names
        vkeys = find_video_keys(cn)
        if "length" not in cn or not vkeys:
            continue
        lengths = [_scalar(v) for v in t.column("length").to_pylist()]
        eis = t.column("episode_index").to_pylist() if "episode_index" in cn else list(range(t.num_rows))
        # Use the first video key to decide which rows are broken (shared timeline).
        k0 = vkeys[0]
        f0 = [_scalar(v) for v in t.column(f"videos/{k0}/from_timestamp").to_pylist()]
        to0 = [_scalar(v) for v in t.column(f"videos/{k0}/to_timestamp").to_pylist()]
        broken_rows = [i for i in range(t.num_rows)
                       if needs_fix(f0[i], to0[i], lengths[i], fps)]
        if not broken_rows:
            continue
        fixed_eps += [int(eis[i]) for i in broken_rows]
        total_fixed += len(broken_rows)
        if dry_run:
            continue
        # Rewrite each video key's to_timestamp for the broken rows.
        for k in vkeys:
            col_name = f"videos/{k}/to_timestamp"
            fcol = [_scalar(v) for v in t.column(f"videos/{k}/from_timestamp").to_pylist()]
            tcol = t.column(col_name).to_pylist()
            new = list(tcol)
            for i in broken_rows:
                val = corrected_to_timestamp(fcol[i], lengths[i], fps)
                new[i] = [val] if isinstance(tcol[i], list) else val
            idx = cn.index(col_name)
            t = t.set_column(idx, col_name, pa.array(new, type=t.column(col_name).type))
        tmp = p + ".tmp"
        pq.write_table(t, tmp)
        os.replace(tmp, p)

    if not total_fixed:
        return "clean (no video-window overshoot)"
    verb = "WOULD fix" if dry_run else "fixed"
    return f"{verb} {total_fixed} episode(s): {sorted(fixed_eps)[:10]}{'...' if total_fixed > 10 else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"{os.path.basename(args.root.rstrip('/'))}: {fix_leaf(args.root, dry_run=args.dry_run)}")


if __name__ == "__main__":
    main()
