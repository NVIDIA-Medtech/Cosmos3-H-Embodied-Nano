# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Out-of-place timestamp repair for a LeRobot v3.0 dataset (step T).

The source is never modified. Metadata and data parquet files are copied, ``timestamp``
is replaced with ``frame_index / fps`` (episode-relative, the v3.0 convention),
``meta/episodes`` and ``meta/stats.json`` timestamp statistics are recomputed, and
``videos/`` is symlinked to the source. Data parquet files are written with zstd.

``--verify-root`` checks a repaired root on values: every row must satisfy
``|timestamp - frame_index / fps| <= 1e-4`` and ``videos/`` must resolve to a directory.
``--expected-data-sha256`` additionally pins the byte-level tree hash of
``data/**.parquet`` (relative path + bytes); note that byte hashes depend on the
parquet writer version and need not match across environments.

Usage::

    python tools/data_prep/repair_timestamps.py --source <v30 root> --destination <new root>
    python tools/data_prep/repair_timestamps.py --verify-root <repaired root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TIMESTAMP_SEMANTIC_TOLERANCE_S = 1e-4
MANIFEST_NAME = "timestamp_repair.json"


def tree_hash(paths: list[Path], root: Path) -> str:
    """SHA-256 over (relative path, bytes) of ``paths``, in sorted order."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"missing required column {name!r}")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def _timestamp_stats(lengths: np.ndarray, fps: float) -> dict[str, list[float | int]]:
    counts = lengths.astype(np.float64)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("dataset contains no rows")
    # For each episode, timestamps are 0/fps, ..., (length-1)/fps.
    sums = (counts * (counts - 1.0) / 2.0) / fps
    sum_squares = (counts * (counts - 1.0) * (2.0 * counts - 1.0) / 6.0) / (fps * fps)
    mean = float(sums.sum() / total)
    variance = max(0.0, float(sum_squares.sum() / total) - mean * mean)
    return {
        "min": [0.0],
        "max": [float((counts.max() - 1.0) / fps)],
        "mean": [mean],
        "std": [math.sqrt(variance)],
        "count": [total],
    }


def repair(source: Path, destination: Path, *, expected_fps: float | None = None) -> dict:
    source = source.resolve()
    destination = destination.absolute()
    if source == destination.resolve(strict=False):
        raise ValueError("source and destination must differ")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if expected_fps is not None and not math.isclose(fps, expected_fps):
        raise ValueError(f"source fps {fps} does not match expected {expected_fps}")
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")

    source_data = sorted((source / "data").glob("**/*.parquet"))
    if not source_data:
        raise ValueError(f"no data parquet files under {source / 'data'}")
    parts = [pq.read_table(path, columns=["episode_index", "frame_index", "timestamp"]) for path in source_data]
    index_table = pa.concat_tables(parts)
    episode = index_table["episode_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    frame = index_table["frame_index"].to_numpy(zero_copy_only=False).astype(np.int64)
    timestamp = index_table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.lexsort((frame, episode))
    episode_ordered, frame_ordered, timestamp_ordered = episode[order], frame[order], timestamp[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(episode_ordered)) + 1, len(episode_ordered)]
    lengths: list[int] = []
    bad_frame_sequences: list[int] = []
    timestamp_violations = 0
    semantic_error = np.abs(timestamp_ordered - frame_ordered.astype(np.float64) / fps)
    timestamp_semantic_violations = int(
        np.count_nonzero(~np.isfinite(timestamp_ordered) | (semantic_error > TIMESTAMP_SEMANTIC_TOLERANCE_S))
    )
    source_max_semantic_error_s = float(np.nanmax(semantic_error))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        frames = frame_ordered[start:end]
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            bad_frame_sequences.append(int(episode_ordered[start]))
        timestamp_violations += int(np.count_nonzero(np.diff(timestamp_ordered[start:end]) <= 0))
        lengths.append(end - start)
    if bad_frame_sequences:
        raise ValueError(f"non-contiguous frame_index in episodes {bad_frame_sequences[:20]}")
    if timestamp_violations == 0 and timestamp_semantic_violations == 0:
        raise ValueError("source timestamps are already episode-relative and strictly increasing; refusing repair")

    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale partial destination exists: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, partial, ignore=shutil.ignore_patterns("videos"), copy_function=shutil.copy2)
    os.symlink(source / "videos", partial / "videos", target_is_directory=True)

    for source_path in source_data:
        table = pq.read_table(source_path)
        repaired = table["frame_index"].to_numpy(zero_copy_only=False).astype(np.float64) / fps
        table = _replace_column(table, "timestamp", repaired)
        pq.write_table(table, partial / source_path.relative_to(source), compression="zstd")

    episode_lengths = {int(ep): int(length) for ep, length in zip(episode_ordered[boundaries[:-1]], lengths)}
    for path in sorted((partial / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        episodes = table["episode_index"].to_pylist()
        for stat_name in ("min", "max", "mean", "std", "count"):
            column = f"stats/timestamp/{stat_name}"
            if table.schema.get_field_index(column) < 0:
                continue
            values = []
            for ep in episodes:
                length = episode_lengths[int(ep)]
                if stat_name == "min":
                    value = 0.0
                elif stat_name == "max":
                    value = (length - 1) / fps
                elif stat_name == "mean":
                    value = (length - 1) / (2.0 * fps)
                elif stat_name == "std":
                    value = math.sqrt((length * length - 1) / 12.0) / fps
                else:
                    value = length
                values.append([value])
            index = table.schema.get_field_index(column)
            table = table.set_column(index, column, pa.array(values, type=table.schema.field(index).type))
        pq.write_table(table, path, compression="zstd")

    stats_path = partial / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["timestamp"] = _timestamp_stats(np.asarray(lengths), fps)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    repaired_data = sorted((partial / "data").glob("**/*.parquet"))
    manifest = {
        "format_version": 2,
        "method": "timestamp = frame_index / fps",
        "fps": fps,
        "episodes": len(lengths),
        "rows": int(len(episode)),
        "source_timestamp_violations": timestamp_violations,
        "source_timestamp_semantic_violations": timestamp_semantic_violations,
        "source_timestamp_max_semantic_error_s": source_max_semantic_error_s,
        "timestamp_semantic_tolerance_s": TIMESTAMP_SEMANTIC_TOLERANCE_S,
        "source_data_sha256": tree_hash(source_data, source),
        "repaired_data_sha256": tree_hash(repaired_data, partial),
    }
    (partial / "meta" / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(partial, destination)
    return manifest


def verify_repair(root: Path, expected_data_sha256: str | None = None) -> dict:
    root = root.resolve()
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    result: dict = {"root": str(root)}
    if expected_data_sha256:
        actual = tree_hash(data_paths, root)
        if actual != expected_data_sha256:
            raise ValueError(f"repaired data hash {actual} does not match expected {expected_data_sha256}")
        result["repaired_data_sha256"] = actual
    videos = root / "videos"
    if not videos.resolve().is_dir():
        raise ValueError(f"videos directory missing: {videos}")
    fps = float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    violating_rows = rows = 0
    max_error_s = 0.0
    for path in data_paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["frame_index", "timestamp"]):
            frame = batch.column("frame_index").to_numpy().astype(np.float64)
            timestamp = batch.column("timestamp").to_numpy().astype(np.float64)
            error = np.abs(timestamp - frame / fps)
            violating_rows += int(np.count_nonzero(~np.isfinite(timestamp) | (error > TIMESTAMP_SEMANTIC_TOLERANCE_S)))
            rows += len(error)
            if len(error):
                max_error_s = max(max_error_s, float(np.nanmax(error)))
    if violating_rows:
        raise ValueError(
            f"timestamps are not episode-relative: violating_rows={violating_rows} max_error_s={max_error_s}"
        )
    result.update(rows=rows, timestamp_max_abs_error_s=max_error_s, passed=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--verify-root", type=Path)
    parser.add_argument("--expected-data-sha256", help="with --verify-root: also pin the data tree hash")
    args = parser.parse_args()
    if args.verify_root is not None:
        if args.source is not None or args.destination is not None:
            parser.error("--verify-root cannot be combined with --source/--destination")
        print(json.dumps(verify_repair(args.verify_root, args.expected_data_sha256), indent=2, sort_keys=True))
        return 0
    if args.source is None or args.destination is None:
        parser.error("repair mode requires --source and --destination")
    print(json.dumps(repair(args.source, args.destination, expected_fps=args.expected_fps), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
