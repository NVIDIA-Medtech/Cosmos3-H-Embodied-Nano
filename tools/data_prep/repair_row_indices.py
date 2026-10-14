# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Out-of-place row-index repair for a LeRobot v3.0 dataset (step I).

Rewrites the global ``index`` column to the physical row position and recomputes
``stats/index/*`` in ``meta/episodes`` and ``meta/stats.json``. Episode membership,
frame indices, row order, timestamps, actions, states and videos must already agree
with the episode metadata and are verified unchanged; ``videos/`` is symlinked.

Usage::

    python tools/data_prep/repair_row_indices.py --source <v30 root> --destination <new root>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_timestamps import tree_hash  # noqa: E402

MANIFEST_NAME = "row_index_repair.json"


def inspect_coordinates(root: Path):
    """Check that physical rows, episode ranges and frame indices agree; count stale indices."""
    paths = sorted((root / "data").rglob("*.parquet"))
    meta_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not paths or not meta_paths:
        raise ValueError("row repair requires v3 data and episode parquet files")
    episodes = pa.concat_tables([pq.read_table(p) for p in meta_paths])
    ids, starts, ends, lengths = [
        episodes[k].to_numpy(zero_copy_only=False).astype(np.int64)
        for k in ("episode_index", "dataset_from_index", "dataset_to_index", "length")
    ]
    if (
        not np.array_equal(ids, np.arange(len(ids)))
        or not len(ids)
        or starts[0] != 0
        or not np.array_equal(starts[1:], ends[:-1])
        or not np.array_equal(ends - starts, lengths)
        or np.any(lengths <= 0)
    ):
        raise ValueError("episode metadata must describe contiguous ordered positive spans")
    offset = changed = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["index", "episode_index", "frame_index"]):
            values = {k: batch.column(k).to_numpy() for k in ("index", "episode_index", "frame_index")}
            if any(v.dtype.kind not in "iu" for v in values.values()):
                raise ValueError("row coordinate columns must be integers")
            ep = values["episode_index"].astype(np.int64)
            pos = np.arange(offset, offset + len(batch), dtype=np.int64)
            if np.any(ep < 0) or np.any(ep >= len(ids)):
                raise ValueError("row episode identity is outside metadata")
            if np.any(pos < starts[ep]) or np.any(pos >= ends[ep]):
                raise ValueError("physical row membership disagrees with episode metadata")
            if not np.array_equal(values["frame_index"], pos - starts[ep]):
                raise ValueError("physical frame identity disagrees with episode metadata")
            changed += int(np.count_nonzero(values["index"] != pos))
            offset += len(batch)
    if offset != ends[-1]:
        raise ValueError("physical row count disagrees with episode metadata")
    return {"rows": offset, "episodes": len(ids), "changed_indices": changed}, paths, meta_paths


def index_stat(name: str, start: int, stop: int) -> list:
    n = stop - start
    values = {
        "min": start,
        "max": stop - 1,
        "mean": (start + stop - 1) / 2,
        "std": np.sqrt((n * n - 1) / 12),
        "count": n,
        "q01": start + 0.01 * (n - 1),
        "q99": start + 0.99 * (n - 1),
    }
    if name not in values:
        raise ValueError(f"unrecognized index statistic: {name}")
    return [float(values[name]) if name != "count" else int(values[name])]


def verify_preserved_columns(original: pa.Table, repaired: pa.Table) -> None:
    left, right = original.drop(["index"]), repaired.drop(["index"])
    if not left.schema.equals(right.schema, check_metadata=True):
        raise ValueError("repair changed a non-index schema")
    for name in left.column_names:
        if left[name].equals(right[name]):
            continue
        # Arrow may compare NaNs as unequal; NaN positions must be retained.
        a, b = np.asarray(left[name].to_pylist()), np.asarray(right[name].to_pylist())
        if a.dtype.kind not in "fiu" or b.dtype != a.dtype or not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"repair changed non-index values: {name}")


def repair(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.absolute()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists() or source == destination.resolve(strict=False):
        raise FileExistsError("repair requires an unused destination distinct from the source")
    result, paths, meta_paths = inspect_coordinates(source)
    if not result["changed_indices"]:
        raise ValueError("source indices are already correct")
    info = json.loads((source / "meta/info.json").read_text())
    if int(info["total_frames"]) != result["rows"]:
        raise ValueError("info total_frames disagrees with physical row count")
    source_hash = tree_hash(paths, source)
    partial.mkdir(parents=True)
    shutil.copytree(source / "meta", partial / "meta")
    (partial / "videos").symlink_to(source / "videos", target_is_directory=True)
    offset = 0
    for path in paths:
        table = pq.read_table(path)
        field = table.schema.field("index")
        values = pa.array(np.arange(offset, offset + len(table)), type=field.type)
        corrected = table.set_column(table.schema.get_field_index("index"), field, values)
        target = partial / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(corrected, target, compression="zstd")
        verify_preserved_columns(table, pq.read_table(target))
        offset += len(table)
    for path in meta_paths:
        target = partial / path.relative_to(source)
        table = pq.read_table(target)
        starts, ends = table["dataset_from_index"].to_pylist(), table["dataset_to_index"].to_pylist()
        for name in table.column_names:
            if name.startswith("stats/index/"):
                field = table.schema.field(name)
                values = [index_stat(name.rsplit("/", 1)[1], start, stop) for start, stop in zip(starts, ends)]
                table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
        pq.write_table(table, target, compression="zstd")
    stats_path = partial / "meta/stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if "index" in stats:
            stats["index"] = {name: index_stat(name, 0, offset) for name in stats["index"]}
            stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    verified, repaired_paths, _ = inspect_coordinates(partial)
    if verified["changed_indices"] or tree_hash(paths, source) != source_hash:
        raise ValueError("repaired coordinates failed verification or source changed during repair")
    result.update(
        method="index = physical row position; episode/frame coordinates independently verified",
        source_data_sha256=source_hash,
        repaired_data_sha256=tree_hash(repaired_paths, partial),
        non_index_data_columns_verified_unchanged=True,
        verified=True,
    )
    (partial / "meta" / MANIFEST_NAME).write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
