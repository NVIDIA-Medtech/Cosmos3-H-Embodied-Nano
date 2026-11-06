# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert a pinned released student to DCP, or verify an existing conversion."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cosmos3_h_embodied.dreams.contract import file_sha256, write_json
from cosmos3_h_embodied.dreams.prepare import assets


def convert(workspace: Path, steps: int, output: Path) -> None:
    pins = assets()["students"]
    filename = next(f for f in pins["files"] if f"_{steps}-step_" in f)
    source = workspace / "assets/students" / filename
    download = json.loads((workspace / "assets/receipt.json").read_text())["students"]
    if download["revision"] != pins["revision"] or download["repo"] != pins["repository"]:
        raise ValueError("Released student download does not match pinned repository/revision")
    identity = next(f for f in download["files"] if f["path"] == str(source.relative_to(workspace)))
    source_hash = file_sha256(source)
    if source_hash != identity["sha256"]:
        raise ValueError("Released student source hash mismatch")
    if output.exists():
        receipt = json.loads((output / "conversion.json").read_text())
        if receipt["source_sha256"] != source_hash or not (output / "model/.metadata").is_file():
            raise ValueError("Existing conversion is incomplete or from different weights; choose a new output")
        for name, digest in receipt["files"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Converted file changed: {name}")
        print(f"Verified existing conversion: {output}")
        return
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    state = load_file(source)
    if not state or any(k.startswith(("net.", "net_ema.")) for k in state):
        raise ValueError("Expected released unprefixed regular network weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "converted"
        dcp.save({f"net.{k}": v for k, v in state.items()}, checkpoint_id=staging / "model", no_dist=True)
        files = {
            str(p.relative_to(staging)): file_sha256(p) for p in sorted((staging / "model").iterdir()) if p.is_file()
        }
        write_json(
            staging / "conversion.json",
            {
                "source_sha256": source_hash,
                "revision": pins["revision"],
                "steps": steps,
                "tensor_count": len(state),
                "regular_weights": True,
                "files": files,
            },
        )
        staging.rename(output)
    print(f"Converted released SF{steps}: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=[4, 2], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    convert(args.workspace, args.steps, args.output or args.workspace / f"released_sf{args.steps}.dcp")


if __name__ == "__main__":
    main()
