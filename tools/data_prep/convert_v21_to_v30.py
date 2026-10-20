# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert Open-H LeRobot v2.1 datasets to v3.0, out of place.

Wraps ``lerobot.datasets.v30.convert_dataset_v21_to_v30`` (the lerobot revision pinned by
cosmos-framework) with the adjustments used to build the training data of the released
checkpoint:

* data parquet files are concatenated with pyarrow (lerobot's pandas path fails on
  parquet files written with HuggingFace ``datasets`` pandas metadata);
* videos are concatenated with the ffmpeg CLI (``-fflags +genpts``, stream copy), which
  accepts per-episode files whose DTS does not rebase monotonically; decoded frames are
  identical to the source;
* ``info.json`` conversion tolerates missing ``total_chunks`` / ``total_videos``;
* parquet timestamps stored as absolute wall-clock time are rewritten to episode-relative
  ``frame_index / fps`` (the LeRobot v3.0 convention), as float64 like the training data;
  the per-episode timestamp statistics are left as converted (see ``repair_timestamps.py``);
* ``meta/modality.json``, ``meta/episodes.*_vlm_qwen3.jsonl`` and ``meta/annotations.jsonl``
  are carried over (the lerobot converter drops them).

Usage::

    # one dataset
    python tools/data_prep/convert_v21_to_v30.py --src <v21 root> --dst <v30 root>
    # every v2.1 dataset below SRC_ROOT, mirrored below DST_ROOT
    python tools/data_prep/convert_v21_to_v30.py --src-root <dir> --dst-root <dir> [SUBDIR ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.datasets.v30 import convert_dataset_v21_to_v30 as _v30
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    V21,
    V30,
    convert_data,
    convert_episodes_metadata,
    convert_tasks,
    convert_videos,
    load_info,
    write_info,
)

# Absolute (wall-clock) timestamps are recognised by their magnitude.
ABSOLUTE_TIMESTAMP_MIN = 1e6
CARRIED_META_GLOBS = ("modality.json", "episodes.*_vlm_qwen3.jsonl", "annotations.jsonl")


def _concat_data_files_arrow(paths_to_cat, new_root, chunk_idx, file_idx, image_keys):
    tables = [pq.read_table(p) for p in paths_to_cat]
    table = pa.concat_tables(tables, promote_options="default")
    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_keys:
        from datasets import Features, Image

        features = Features.from_arrow_schema(table.schema)
        for key in image_keys:
            features[key] = Image()
        table = table.cast(features.arrow_schema)
    pq.write_table(table, path)


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_concat_video_files(input_video_paths: list, output_video_path: Path, overwrite: bool = True) -> None:
    output_video_path = Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return
    if not input_video_paths:
        raise FileNotFoundError("No input video paths provided.")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as f:
        f.write("ffconcat version 1.0\n")
        for p in input_video_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = f.name
    try:
        cmd = [
            _ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart", "-y", str(output_video_path),
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
    finally:
        Path(list_path).unlink(missing_ok=True)


_v30.concat_data_files = _concat_data_files_arrow
_v30.concatenate_video_files = _ffmpeg_concat_video_files


def _convert_info(root: Path, new_root: Path, data_mb: int, video_mb: int) -> None:
    info = load_info(root)
    info["codebase_version"] = V30
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = data_mb
    info["video_files_size_in_mb"] = video_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") is not None else None
    info["fps"] = int(info["fps"])
    for key in info["features"]:
        if info["features"][key]["dtype"] != "video":
            info["features"][key]["fps"] = info["fps"]
    write_info(info, new_root)


def relativize_absolute_timestamps(dst: Path) -> int:
    """Rewrite absolute parquet timestamps to episode-relative ``frame_index / fps``.

    Returns the number of rewritten parquet files. Files whose timestamps are already
    relative are left untouched.
    """
    fps = float(json.loads((dst / "meta" / "info.json").read_text())["fps"])
    n_fixed = 0
    for path in sorted((dst / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        if "timestamp" not in table.column_names or table.num_rows == 0:
            continue
        if min(table.column("timestamp").to_pylist()) < ABSOLUTE_TIMESTAMP_MIN:
            continue
        frame_index = table.column("frame_index").to_numpy()
        # float64, as in the training data (the absolute float32 source values carry no
        # sub-second precision, and float32 would round frame_index / fps).
        timestamps = pa.array(frame_index / fps, type=pa.float64())
        table = table.set_column(table.column_names.index("timestamp"), "timestamp", timestamps)
        pq.write_table(table, path)
        n_fixed += 1
    if n_fixed:
        logging.info(f"Rewrote absolute timestamps to episode-relative in {n_fixed} parquet file(s) at {dst}")
    return n_fixed


def convert(src: Path, dst: Path, data_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB, video_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB) -> None:
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{src} does not look like a LeRobot dataset (no meta/info.json)")
    version = json.loads(info_path.read_text()).get("codebase_version")
    if version != V21:
        raise ValueError(f"Expected codebase_version={V21!r} at {src}, got {version!r}")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Destination {dst} is not empty")
    dst.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting {src} -> {dst}")
    _convert_info(src, dst, data_mb, video_mb)
    convert_tasks(src, dst)
    episodes_metadata = convert_data(src, dst, data_mb)
    episodes_video_metadata = convert_videos(src, dst, video_mb)
    convert_episodes_metadata(src, dst, episodes_metadata, episodes_video_metadata)
    relativize_absolute_timestamps(dst)
    for pattern in CARRIED_META_GLOBS:
        for extra in (src / "meta").glob(pattern):
            shutil.copy2(extra, dst / "meta" / extra.name)


def find_v21_roots(src_root: Path, subdirs: list[str]) -> list[Path]:
    roots = []
    for sub in subdirs or ["."]:
        for info in sorted((src_root / sub).rglob("meta/info.json")):
            if json.loads(info.read_text()).get("codebase_version") == V21:
                roots.append(info.parent.parent)
    return roots


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, help="v2.1 dataset root")
    parser.add_argument("--dst", type=Path, help="output v3.0 dataset root")
    parser.add_argument("--src-root", type=Path, help="convert every v2.1 dataset below this directory")
    parser.add_argument("--dst-root", type=Path, help="mirror converted datasets below this directory")
    parser.add_argument("subdirs", nargs="*", help="with --src-root: only these subdirectories")
    parser.add_argument("--data-file-size-in-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    parser.add_argument("--video-file-size-in-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)
    args = parser.parse_args()
    sizes = (args.data_file_size_in_mb, args.video_file_size_in_mb)

    if args.src is not None:
        if args.dst is None or args.src_root is not None:
            parser.error("--src needs --dst (and excludes --src-root)")
        convert(args.src.resolve(), args.dst.resolve(), *sizes)
        return
    if args.src_root is None or args.dst_root is None:
        parser.error("pass --src/--dst or --src-root/--dst-root")
    failed = []
    for src in find_v21_roots(args.src_root.resolve(), args.subdirs):
        dst = args.dst_root.resolve() / src.relative_to(args.src_root.resolve())
        if (dst / "meta" / "info.json").is_file():
            logging.info(f"skip (already converted): {dst}")
            continue
        try:
            convert(src, dst, *sizes)
        except Exception:
            logging.exception(f"conversion failed: {src}")
            failed.append(src)
    if failed:
        raise SystemExit(f"{len(failed)} conversion(s) failed: {[str(p) for p in failed]}")


if __name__ == "__main__":
    main()
