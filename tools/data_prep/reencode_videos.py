# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Re-encode the videos of one view of a converted v3.0 root to H.264, in place (steps E and DS).

Training read H.264 for roots whose upstream video is AV1 or MPEG-4 (the AV1 decoder was too slow
in the data workers). The settings below are those recorded in the training videos' x264 headers:

  * E  (CAO cautery, SRT-H, ARCADE, rob_surgical, TUD): ``libx264 -preset fast -crf 20``, default GOP;
  * DS (Turin): ``-vf scale=854:480:flags=lanczos libx264 -preset fast -crf 18``.

Both drop audio and keep every frame (``-fps_mode passthrough``); the frame count of each file is
checked before it replaces the original. ``meta/info.json`` is left unchanged, as in training (it
still describes the upstream codec and size; the loaders read the decoded frames).

Re-encoding is not bit-reproducible: other ffmpeg/x264 builds give frames that differ at codec-noise
level. Needs ``ffmpeg`` and ``ffprobe`` on PATH.

    python tools/data_prep/reencode_videos.py --root <v30 root> --view observation.images.endoscope.left
    python tools/data_prep/reencode_videos.py --root <turin root> --view observation.images.endoscope.left --crf 18 --scale 854x480
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out.split(",")[0])


def codec(path: Path) -> tuple[str, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,height",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    name, height = out.split(",")[:2]
    return name, int(height)


def reencode(src: Path, crf: int, preset: str, scale: tuple[int, int] | None, gop: int | None, threads: int) -> str:
    name, height = codec(src)
    if name == "h264" and (scale is None or height <= scale[1]):
        return "skip (already H.264)"
    tmp = src.with_name(src.stem + ".reencode.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-fps_mode", "passthrough"]
    if scale:
        cmd += ["-vf", f"scale={scale[0]}:{scale[1]}:flags=lanczos"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", str(threads)]
    if gop:
        cmd += ["-g", str(gop)]
    subprocess.run(cmd + [str(tmp)], check=True)
    n_src, n_out = frame_count(src), frame_count(tmp)
    if n_src != n_out:
        tmp.unlink()
        raise RuntimeError(f"{src}: frame count changed {n_src} -> {n_out}; original kept")
    os.replace(tmp, src)
    return f"{name} -> h264, {n_out} frames"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="converted v3.0 root")
    ap.add_argument("--view", required=True, action="append", help="video key, e.g. observation.images.endoscope.left")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--scale", help="WxH, Lanczos (Turin: 854x480)")
    ap.add_argument("--gop", type=int, help="keyframe interval (default: x264's 250)")
    ap.add_argument("--threads", type=int, default=0, help="x264 threads (0 = auto)")
    args = ap.parse_args()
    scale = tuple(int(x) for x in args.scale.lower().split("x")) if args.scale else None
    for view in args.view:
        files = sorted((args.root / "videos" / view).rglob("*.mp4"))
        if not files:
            raise SystemExit(f"no videos under {args.root / 'videos' / view}")
        for path in files:
            print(f"{path.relative_to(args.root)}: {reencode(path, args.crf, args.preset, scale, args.gop, args.threads)}")


if __name__ == "__main__":
    main()
