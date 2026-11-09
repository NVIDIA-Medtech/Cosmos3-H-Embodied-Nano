# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Check media libraries and BF16 on each selected CUDA device before downloads."""

import argparse
import os
import subprocess
from pathlib import Path

import torch
from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

from cosmos3_h_embodied.dreams.contract import write_json


def check_media() -> dict:
    """Check both the executable and the shared-library loader used for scoring."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "FFmpeg is unavailable in this worker/container. Install the tutorial's "
            "system packages in the training image before downloading model/data assets."
        ) from exc
    try:
        codec = load_torchcodec()
    except Exception as exc:
        raise RuntimeError(
            "TorchCodec cannot load in this worker/container. The ffmpeg executable "
            "alone is insufficient: install the tutorial's FFmpeg shared libraries "
            "and preserve the CUDA/NPP runtime. See the chained loader error."
        ) from exc
    return {"ffmpeg": result.stdout.splitlines()[0], "torchcodec": codec.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        media = check_media()
    except Exception as exc:
        write_json(args.output, {"status": "failed", "phase": "media", "error": str(exc)})
        raise
    if args.gpus < 1 or not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Select a positive GPU count within the visible devices and a compatible CUDA runtime")
    result = {
        "status": "passed",
        "media": media,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [],
    }
    for index in range(args.gpus):
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(index)
            x = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.bfloat16)
            (x @ x).sum().item()
            torch.cuda.synchronize()
            row = dict(
                index=index,
                name=props.name,
                uuid=str(getattr(props, "uuid", "unavailable")),
                free_bytes=free,
                total_bytes=total,
                bf16="passed",
            )
            result["devices"].append(row)
            print(row, flush=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
