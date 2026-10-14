# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a v2.1 shim keeping only episodes that have a video for every view (step VC).

Some Open-H v2.1 roots (Turin ex_vivo / plastic_pad / plastic_pad_3dmed) ship more data
episodes than videos. The v3.0 converter rejects that mismatch. This shim drops the
video-incomplete episodes and renumbers the rest contiguously 0..K-1 (episode IDs, the
total episode count and therefore the seed-based train/val split change accordingly).

Data and videos are symlinked, not copied. ``meta/episodes.jsonl``,
``meta/episodes_stats.jsonl`` and ``info.json`` totals are rewritten; ``tasks.jsonl``,
``modality.json`` and ``stats.json`` are symlinked. As in the training data, the parquet
``episode_index``/``index`` columns are NOT rewritten (the v3.0 converter assigns episode
indices by file order). Run ``episodes_stats.py --mode exact`` on the shim afterwards,
as was done for training.

Usage::

    python tools/data_prep/video_complete_shim.py --src <v21 root> --out <shim dir>
"""

from __future__ import annotations

import argparse
import json
import os


def _views(src: str) -> list[str]:
    base = os.path.join(src, "videos", "chunk-000")
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))) if os.path.isdir(base) else []


def build_shim(src: str, out: str) -> tuple[int, int]:
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    views = _views(src)
    eps = [json.loads(line) for line in open(os.path.join(src, "meta", "episodes.jsonl")) if line.strip()]
    eps.sort(key=lambda e: e["episode_index"])

    def complete(i: int) -> bool:
        if not os.path.exists(os.path.join(src, "data", "chunk-000", f"episode_{i:06d}.parquet")):
            return False
        return all(os.path.exists(os.path.join(src, "videos", "chunk-000", v, f"episode_{i:06d}.mp4")) for v in views)

    keep = [e for e in eps if complete(int(e["episode_index"]))]
    print(f"{src}: {len(eps)} episodes, {len(views)} views -> keeping {len(keep)} (dropped {len(eps) - len(keep)})")
    remap = {int(e["episode_index"]): n for n, e in enumerate(keep)}

    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out, "data", "chunk-000"), exist_ok=True)
    for v in views:
        os.makedirs(os.path.join(out, "videos", "chunk-000", v), exist_ok=True)

    stats = {}
    stats_path = os.path.join(src, "meta", "episodes_stats.jsonl")
    if os.path.exists(stats_path):
        for line in open(stats_path):
            if line.strip():
                r = json.loads(line)
                stats[int(r["episode_index"])] = r

    new_eps, new_stats, total_frames = [], [], 0
    for old, new in remap.items():
        os.symlink(
            os.path.join(src, "data", "chunk-000", f"episode_{old:06d}.parquet"),
            os.path.join(out, "data", "chunk-000", f"episode_{new:06d}.parquet"),
        )
        for v in views:
            os.symlink(
                os.path.join(src, "videos", "chunk-000", v, f"episode_{old:06d}.mp4"),
                os.path.join(out, "videos", "chunk-000", v, f"episode_{new:06d}.mp4"),
            )
        e = dict(next(x for x in keep if int(x["episode_index"]) == old))
        e["episode_index"] = new
        new_eps.append(e)
        total_frames += int(e.get("length", 0))
        if old in stats:
            s = dict(stats[old])
            s["episode_index"] = new
            new_stats.append(s)

    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in new_eps)
    if new_stats:
        with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in new_stats)
    for name in ("tasks.jsonl", "modality.json", "stats.json"):
        p = os.path.join(src, "meta", name)
        if os.path.exists(p):
            os.symlink(p, os.path.join(out, "meta", name))
    info["total_episodes"] = len(new_eps)
    info["total_frames"] = total_frames
    # No "splits" field: an ad-hoc "0:N" string confuses the HF datasets builder.
    info.pop("splits", None)
    json.dump(info, open(os.path.join(out, "meta", "info.json"), "w"))
    return len(new_eps), total_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n, frames = build_shim(args.src.rstrip("/"), args.out.rstrip("/"))
    print(f"shim ready: {n} episodes, {frames} frames")


if __name__ == "__main__":
    main()
