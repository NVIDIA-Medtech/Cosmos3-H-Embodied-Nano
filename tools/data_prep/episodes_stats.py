# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Write ``meta/episodes_stats.jsonl`` for a LeRobot v2.1 dataset that lacks it (step BF).

The v2.1 -> v3.0 converter needs per-episode stats. Two modes, matching how the training
data was built:

``exact``
    LeRobot's own ``compute_episode_stats`` over each episode's parquet, for all
    non-image/video/string features (no video decode). Used for roots whose source has
    no or NaN-containing stats (srth, Turin after the video-complete shim, TUD, ...).
``synthesize``
    Real min/max/mean/std/count over numeric features as flat per-dimension lists, plus
    placeholder per-channel stats (min 0, max 1, mean 0.5, std 0.25) for video features;
    string features are skipped. Used for rob_surgical and SMARTS.

Stats only feed metadata; the loaders do not read them.

Usage::

    python tools/data_prep/episodes_stats.py --mode exact --src <v21 root> [--out PATH]
    python tools/data_prep/episodes_stats.py --mode synthesize --src <v21 root> [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SKIP_DTYPES = {"image", "video", "string"}


def _episode_parquet_path(src: Path, info: dict, ep_idx: int) -> Path:
    chunk = ep_idx // int(info["chunks_size"])
    return src / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)


def _episode_data(table, stat_keys: list[str]) -> dict[str, np.ndarray]:
    """Column -> numpy array. Vector columns -> (T, D); scalar columns -> (T,)."""
    data = {}
    for key in stat_keys:
        col = table.column(key).to_pylist()
        if len(col) and isinstance(col[0], list):
            data[key] = np.asarray(col, dtype=np.float32)
        else:
            data[key] = np.asarray(col, dtype=np.float64)
    return data


def write_exact(src: Path, out: Path) -> int:
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import serialize_dict

    info = json.loads((src / "meta" / "info.json").read_text())
    features = info["features"]
    stat_keys = [k for k, ft in features.items() if ft["dtype"] not in SKIP_DTYPES]
    feat_subset = {k: features[k] for k in stat_keys}
    episodes = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    episodes.sort(key=lambda e: e["episode_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ep in episodes:
            ep_idx = ep["episode_index"]
            table = pq.read_table(_episode_parquet_path(src, info, ep_idx), columns=stat_keys)
            ep_stats = compute_episode_stats(_episode_data(table, stat_keys), feat_subset)
            f.write(json.dumps({"episode_index": ep_idx, "stats": serialize_dict(ep_stats)}) + "\n")
    return len(episodes)


def write_synthesized(src: Path, out: Path) -> int:
    info = json.loads((src / "meta" / "info.json").read_text())
    feats = info["features"]
    video_feats = [k for k, v in feats.items() if v.get("dtype") == "video"]
    string_feats = {k for k, v in feats.items() if v.get("dtype") == "string"}
    numeric_feats = [k for k in feats if k not in set(video_feats) and k not in string_feats]

    cols: dict[str, list[np.ndarray]] = {k: [] for k in numeric_feats}
    ep_col = []
    for path in sorted((src / "data").rglob("*.parquet")):
        avail = set(pq.ParquetFile(path).schema_arrow.names)
        need = [c for c in numeric_feats if c in avail]
        if "episode_index" not in numeric_feats:
            need.append("episode_index")
        t = pq.read_table(path, columns=need)
        for k in numeric_feats:
            if k in t.column_names:
                cols[k].append(np.asarray(t.column(k).to_pylist()))
        ep_col.append(np.asarray(t.column("episode_index").to_pylist()).reshape(-1))
    ep = np.concatenate(ep_col)
    data = {k: (np.concatenate(v) if v else None) for k, v in cols.items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w") as fh:
        for e in sorted(np.unique(ep).tolist()):
            mask = ep == e
            n = int(mask.sum())
            stats: dict[str, dict] = {}
            for k in numeric_feats:
                arr = data[k]
                if arr is None:
                    continue
                block = arr[mask]
                if block.ndim == 1:
                    block = block.reshape(-1, 1)
                block = block.astype(np.float64)
                stats[k] = {
                    "min": block.min(0).tolist(),
                    "max": block.max(0).tolist(),
                    "mean": block.mean(0).tolist(),
                    "std": block.std(0).tolist(),
                    "count": [n],
                }
            for k in video_feats:
                c = int(feats[k]["shape"][-1]) if feats[k].get("shape") else 3
                stats[k] = {
                    "min": [[[0.0]] for _ in range(c)],
                    "max": [[[1.0]] for _ in range(c)],
                    "mean": [[[0.5]] for _ in range(c)],
                    "std": [[[0.25]] for _ in range(c)],
                    "count": [n],
                }
            fh.write(json.dumps({"episode_index": int(e), "stats": stats}) + "\n")
            n_written += 1
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("exact", "synthesize"), required=True)
    ap.add_argument("--src", type=Path, required=True, help="LeRobot v2.1 dataset root")
    ap.add_argument("--out", type=Path, help="output path (default: <src>/meta/episodes_stats.jsonl)")
    args = ap.parse_args()
    src = args.src.resolve()
    out = args.out or src / "meta" / "episodes_stats.jsonl"
    n = write_exact(src, out) if args.mode == "exact" else write_synthesized(src, out)
    print(f"wrote {out}: {n} episodes ({args.mode})")


if __name__ == "__main__":
    main()
