# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a compact, action-only valid-window index for CMR clinical roots.

The live loader rejects arm-swap, clutch-out, and camera-pan windows only
after fetching a sample. On the raw 60 Hz clinical corpus that can trigger a
random cross-video retry before every cache write. This script evaluates the
same predicates directly from Arrow, records contiguous valid candidate runs,
and never decodes video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices

from cosmos3_h_embodied.data.cmr_clinical_dataset import (
    _L_THUMBSTICK_COLS,
    _R_THUMBSTICK_COLS,
    _ST_ARMLINK_L,
    _ST_ARMLINK_R,
    _ST_HAPTIC_L,
    _ST_HAPTIC_R,
    _stable_root_key,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projected_components(root: Path) -> tuple[dict[str, np.ndarray], int]:
    """Read only the eight scalar action/state components used by the gate.

    ``LeRobotDataset`` routes ``Dataset.from_parquet`` through the Hugging Face
    builder cache.  A single CMR procedure can then materialize tens of GB in
    node-local ``/tmp`` before the action-only scan starts.  Arrow projection
    evaluates ``list_element`` while scanning the source Parquet fragments, so
    neither the full 100-D state nor a duplicate HF table is written locally.
    """

    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{root}: no data/**/*.parquet files")
    source = pads.dataset([str(path) for path in paths], format="parquet")
    projections = {
        "haptic_l": pc.list_element(pads.field("observation.state"), _ST_HAPTIC_L),
        "haptic_r": pc.list_element(pads.field("observation.state"), _ST_HAPTIC_R),
        "armlink_l": pc.list_element(pads.field("observation.state"), _ST_ARMLINK_L),
        "armlink_r": pc.list_element(pads.field("observation.state"), _ST_ARMLINK_R),
        "thumb_lx": pc.list_element(pads.field("action"), _L_THUMBSTICK_COLS[0]),
        "thumb_ly": pc.list_element(pads.field("action"), _L_THUMBSTICK_COLS[1]),
        "thumb_rx": pc.list_element(pads.field("action"), _R_THUMBSTICK_COLS[0]),
        "thumb_ry": pc.list_element(pads.field("action"), _R_THUMBSTICK_COLS[1]),
        "episode_index": pads.field("episode_index"),
        "frame_index": pads.field("frame_index"),
    }
    table = source.scanner(
        columns=projections,
        batch_size=262144,
        batch_readahead=8,
        fragment_readahead=8,
        use_threads=True,
    ).to_table()
    components = {
        name: table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
        for name in projections
    }
    return components, table.num_rows


def validate_source_row_order(
    components: dict[str, np.ndarray], ep_starts: np.ndarray, ep_stops: np.ndarray
) -> None:
    """Prove the projected fragment order still matches LeRobot metadata."""

    episode_index = components.pop("episode_index")
    frame_index = components.pop("frame_index")
    for episode_id, (start, stop) in enumerate(zip(ep_starts, ep_stops, strict=True)):
        start = int(start)
        stop = int(stop)
        if stop <= start:
            raise RuntimeError(f"episode {episode_id}: empty metadata span [{start}, {stop})")
        if (
            int(episode_index[start]) != episode_id
            or int(episode_index[stop - 1]) != episode_id
            or int(frame_index[start]) != 0
            or int(frame_index[stop - 1]) != stop - start - 1
        ):
            raise RuntimeError(
                f"episode {episode_id}: projected Parquet order disagrees with metadata"
            )


def contiguous_runs(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.flatnonzero(valid)
    if not len(offsets):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    split = np.flatnonzero(np.diff(offsets) != 1) + 1
    groups = np.split(offsets, split)
    starts = np.asarray([group[0] for group in groups], dtype=np.int64)
    lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
    return starts, lengths


def build_root(
    root: Path,
    *,
    fps: float,
    chunk_length: int,
    sample_stride: int,
    camera_motion_threshold: float | None,
    clutch_policy: str,
    block_windows: int,
) -> tuple[dict, list[tuple[int, int, int]]]:
    delta = [i / fps for i in range(chunk_length + 1)]
    meta = LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")
    delta_timestamps = {"action": delta, "observation.state": delta}
    check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s=0.008)
    delta_indices = get_delta_indices(delta_timestamps, meta.fps)
    state_offsets = np.asarray(delta_indices["observation.state"], dtype=np.int64)
    action_offsets = np.asarray(delta_indices["action"], dtype=np.int64)
    if not np.array_equal(state_offsets, action_offsets):
        raise RuntimeError(f"{root}: action/state delta indices differ")
    offsets = state_offsets

    components, total_rows = projected_components(root)
    ep_starts = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
    ep_stops = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)
    if len(ep_starts) != int(meta.total_episodes) or len(ep_stops) != int(meta.total_episodes):
        raise RuntimeError(f"{root}: episode metadata count mismatch")
    if not len(ep_stops) or int(ep_stops[-1]) != total_rows:
        raise RuntimeError(
            f"{root}: metadata ends at {ep_stops[-1] if len(ep_stops) else None}, "
            f"but Parquet projection has {total_rows} rows"
        )
    validate_source_row_order(components, ep_starts, ep_stops)
    state = {
        "haptic_l": components["haptic_l"],
        "haptic_r": components["haptic_r"],
        "armlink_l": components["armlink_l"],
        "armlink_r": components["armlink_r"],
    }
    action = {
        "thumb_lx": components["thumb_lx"],
        "thumb_ly": components["thumb_ly"],
        "thumb_rx": components["thumb_rx"],
        "thumb_ry": components["thumb_ry"],
    }

    runs: list[tuple[int, int, int]] = []
    counters = {
        "base_candidates": 0,
        "evaluated_candidates": 0,
        "valid_candidates": 0,
        "tail_excluded": 0,
        "arm_swap": 0,
        "fully_disengaged": 0,
        "any_disengaged": 0,
        "camera_pan": 0,
    }
    started = time.perf_counter()
    max_offset = int(offsets.max())

    for episode_id, (ep_start, ep_stop) in enumerate(zip(ep_starts, ep_stops)):
        base_raw = max(0, int(ep_stop - ep_start - chunk_length))
        base_count = (base_raw + sample_stride - 1) // sample_stride if base_raw else 0
        exact_last = int(ep_stop - 1 - max_offset)
        exact_count = (
            (exact_last - int(ep_start)) // sample_stride + 1
            if exact_last >= int(ep_start)
            else 0
        )
        counters["base_candidates"] += base_count
        counters["evaluated_candidates"] += exact_count
        counters["tail_excluded"] += base_count - exact_count
        valid = np.empty(exact_count, dtype=bool)

        for first in range(0, exact_count, block_windows):
            stop = min(exact_count, first + block_windows)
            starts = int(ep_start) + np.arange(first, stop, dtype=np.int64) * sample_stride
            query = starts[:, None] + offsets[None, :]

            armlink_l = state["armlink_l"][query]
            armlink_r = state["armlink_r"][query]
            arm_swap = np.any(armlink_l != armlink_l[:, :1], axis=1) | np.any(
                armlink_r != armlink_r[:, :1], axis=1
            )
            fully_disengaged = np.all(state["haptic_l"][query] < 0.5, axis=1) & np.all(
                state["haptic_r"][query] < 0.5, axis=1
            )
            any_disengaged = np.any(state["haptic_l"][query] < 0.5, axis=1) | np.any(
                state["haptic_r"][query] < 0.5, axis=1
            )
            clutch_rejected = (
                any_disengaged if clutch_policy == "reject_any" else fully_disengaged
            )
            if camera_motion_threshold is None:
                camera_pan = np.zeros(stop - first, dtype=bool)
            else:
                lmag = np.hypot(action["thumb_lx"][query], action["thumb_ly"][query])
                rmag = np.hypot(action["thumb_rx"][query], action["thumb_ry"][query])
                camera_pan = np.any(
                    np.maximum(lmag, rmag) > float(camera_motion_threshold), axis=1
                )

            valid[first:stop] = ~(arm_swap | clutch_rejected | camera_pan)
            counters["arm_swap"] += int(arm_swap.sum())
            counters["fully_disengaged"] += int(fully_disengaged.sum())
            counters["any_disengaged"] += int(any_disengaged.sum())
            counters["camera_pan"] += int(camera_pan.sum())

        starts, lengths = contiguous_runs(valid)
        runs.extend(
            (episode_id, int(run_start), int(run_length))
            for run_start, run_length in zip(starts, lengths)
        )
        counters["valid_candidates"] += int(valid.sum())

    report = {
        "root": str(root),
        "root_key": _stable_root_key(root),
        "total_rows": total_rows,
        "total_episodes": int(meta.total_episodes),
        "delta_indices": offsets.tolist(),
        "clutch_policy": clutch_policy,
        "run_count": len(runs),
        "seconds": time.perf_counter() - started,
        **counters,
    }
    return report, runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--chunk-length", type=int, default=24)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--camera-motion-threshold", type=float, default=5000.0)
    parser.add_argument(
        "--clutch-policy",
        choices=("legacy_mask", "reject_any"),
        default="reject_any",
        help="reject_any (any disengaged frame rejects the window) was used for training",
    )
    parser.add_argument("--block-windows", type=int, default=65536)
    args = parser.parse_args()

    root_keys: list[str] = []
    root_rows: list[int] = []
    root_episodes: list[int] = []
    run_root: list[int] = []
    run_episode: list[int] = []
    run_start: list[int] = []
    run_length: list[int] = []
    reports: list[dict] = []

    for root_idx, root in enumerate(args.roots):
        report, runs = build_root(
            root,
            fps=args.fps,
            chunk_length=args.chunk_length,
            sample_stride=args.sample_stride,
            camera_motion_threshold=args.camera_motion_threshold,
            clutch_policy=args.clutch_policy,
            block_windows=args.block_windows,
        )
        reports.append(report)
        root_keys.append(report["root_key"])
        root_rows.append(report["total_rows"])
        root_episodes.append(report["total_episodes"])
        for episode, start, length in runs:
            run_root.append(root_idx)
            run_episode.append(episode)
            run_start.append(start)
            run_length.append(length)
        print(json.dumps(report, sort_keys=True), flush=True)

    if len(set(root_keys)) != len(root_keys):
        raise RuntimeError(f"duplicate stable root keys: {root_keys}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        schema_version=np.asarray(2, dtype=np.int64),
        chunk_length=np.asarray(args.chunk_length, dtype=np.int64),
        sample_stride=np.asarray(args.sample_stride, dtype=np.int64),
        camera_motion_threshold=np.asarray(args.camera_motion_threshold, dtype=np.float64),
        clutch_policy=np.asarray(args.clutch_policy),
        root_keys=np.asarray(root_keys),
        root_total_rows=np.asarray(root_rows, dtype=np.int64),
        root_total_episodes=np.asarray(root_episodes, dtype=np.int64),
        run_root=np.asarray(run_root, dtype=np.uint16),
        run_episode=np.asarray(run_episode, dtype=np.uint32),
        run_start=np.asarray(run_start, dtype=np.uint32),
        run_length=np.asarray(run_length, dtype=np.uint32),
    )
    os.replace(tmp, args.out)
    checksum = sha256_file(args.out)
    summary = {
        "schema_version": 2,
        "artifact": str(args.out),
        "sha256": checksum,
        "fps": args.fps,
        "chunk_length": args.chunk_length,
        "sample_stride": args.sample_stride,
        "camera_motion_threshold": args.camera_motion_threshold,
        "clutch_policy": args.clutch_policy,
        "roots": reports,
        "total_base_candidates": sum(item["base_candidates"] for item in reports),
        "total_valid_candidates": sum(item["valid_candidates"] for item in reports),
        "total_runs": len(run_root),
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
