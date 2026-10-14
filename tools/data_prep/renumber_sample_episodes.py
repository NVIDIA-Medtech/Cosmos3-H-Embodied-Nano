# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Copy a trimmed v2.1 sample root with its episodes renumbered 0..K-1 (test fixtures only).

Sample roots made by ``tools/download_openh_samples.py`` keep a few episodes of an
upstream root under their original IDs (e.g. 52 and 53). The v3.0 converter assigns
episode indices by file order and rejects IDs that are not 0..K-1. This tool writes a
copy with:

* data parquet files renamed and their ``episode_index`` (and, if present, ``index``)
  columns rewritten (``index`` becomes contiguous over the kept episodes);
* videos hard-linked under the new names;
* ``meta/episodes.jsonl`` / ``meta/episodes_stats.jsonl`` renumbered (stats of
  ``episode_index`` / ``index`` recomputed), ``info.json`` totals updated and its
  ``splits`` ranges mapped onto the new IDs;
* the remaining ``meta`` files copied unchanged;
* ``renumber_map.json`` (old -> new episode index) next to ``meta/``.

Not part of the training-data rebuild: full upstream roots are already contiguous.

Usage::

    python tools/data_prep/renumber_sample_episodes.py --src <trimmed v21 root> --dst <new dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REWRITTEN_META = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}


def _parse_ranges(spec: object) -> set[int]:
    out: set[int] = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if part:
            a, _, b = part.partition(":")
            out.update(range(int(a), (int(b) if b else int(a)) + 1))
    return out


def _format_ranges(ids: list[int]) -> str:
    ids = sorted(ids)
    runs, start = [], None
    for i, v in enumerate(ids):
        if start is None:
            start = v
        if i + 1 == len(ids) or ids[i + 1] != v + 1:
            runs.append(f"{start}:{v}")
            start = None
    return ", ".join(runs)


def _flat_stats(values: np.ndarray) -> dict:
    v = values.astype(np.float64).reshape(-1, 1)
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(), "count": [int(len(v))]}  # fmt: skip


def renumber(src: Path, dst: Path) -> dict[int, int]:
    if dst.exists():
        raise FileExistsError(f"destination exists: {dst}")
    info = json.loads((src / "meta" / "info.json").read_text())
    chunks = int(info.get("chunks_size", 1000))
    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    episodes.sort(key=lambda e: int(e["episode_index"]))
    remap = {int(e["episode_index"]): new for new, e in enumerate(episodes)}
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]

    (dst / "meta").mkdir(parents=True)
    global_index = 0
    new_stats_extra: dict[int, dict] = {}
    for ep in episodes:
        old = int(ep["episode_index"])
        new = remap[old]
        old_chunk = int(ep.get("episode_chunk", old // chunks))
        new_chunk = new // chunks
        src_data = src / info["data_path"].format(episode_chunk=old_chunk, episode_index=old)
        dst_data = dst / info["data_path"].format(episode_chunk=new_chunk, episode_index=new)
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(src_data)
        n = table.num_rows
        extra = {}
        for name, values in (("episode_index", np.full(n, new)), ("index", np.arange(global_index, global_index + n))):
            if name not in table.column_names:
                continue
            col = table.column(name)
            if pa.types.is_list(col.type):  # 1-element list storage
                arr = pa.array([[int(v)] for v in values], type=col.type)
            else:
                arr = pa.array(values, type=col.type)
            table = table.set_column(table.column_names.index(name), name, arr)
            extra[name] = _flat_stats(values)
        global_index += n
        pq.write_table(table, dst_data)
        new_stats_extra[new] = extra
        for key in video_keys:
            kwargs = dict(video_key=key)
            src_vid = src / info["video_path"].format(episode_chunk=old_chunk, episode_index=old, **kwargs)
            if src_vid.exists():
                dst_vid = dst / info["video_path"].format(episode_chunk=new_chunk, episode_index=new, **kwargs)
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                os.link(src_vid.resolve(), dst_vid)

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for ep in episodes:
            ep = dict(ep, episode_index=remap[int(ep["episode_index"])])
            if "episode_chunk" in ep:
                ep["episode_chunk"] = ep["episode_index"] // chunks
            f.write(json.dumps(ep) + "\n")
    stats_path = src / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
            rows = [json.loads(l) for l in stats_path.read_text().splitlines() if l.strip()]
            for row in sorted(rows, key=lambda r: int(r["episode_index"])):
                new = remap[int(row["episode_index"])]
                stats = dict(row["stats"])
                for name, value in new_stats_extra[new].items():
                    if name in stats:
                        stats[name] = value
                f.write(json.dumps({"episode_index": new, "stats": stats}) + "\n")
    for path in (src / "meta").iterdir():
        if path.is_file() and path.name not in REWRITTEN_META:
            shutil.copy2(path, dst / "meta" / path.name)

    info["total_episodes"] = len(episodes)
    info["total_frames"] = global_index
    if "total_chunks" in info:
        info["total_chunks"] = (len(episodes) - 1) // chunks + 1
    if "total_videos" in info:
        info["total_videos"] = len(episodes) * len(video_keys)
    if info.get("splits"):
        splits = {}
        for name, spec in info["splits"].items():
            kept = [remap[i] for i in _parse_ranges(spec) if i in remap]
            if kept:
                splits[name] = _format_ranges(kept)
        info["splits"] = splits
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (dst / "renumber_map.json").write_text(json.dumps({str(k): v for k, v in remap.items()}, indent=1))
    return remap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    remap = renumber(args.src.resolve(), args.dst.absolute())
    print(f"{args.src} -> {args.dst}: {remap}")


if __name__ == "__main__":
    main()
