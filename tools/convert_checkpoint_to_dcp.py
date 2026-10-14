# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a released network state dict (``.pt``) into a DCP checkpoint for training.

The released checkpoint holds the network weights only (keys without the ``net.``
prefix). cosmos-framework's checkpointer warm-starts from a DCP directory whose ``model``
component holds ``net.*`` and ``net_ema.*``; both are written from the same weights, so the
EMA starts at the released weights. Point ``checkpoint.load_path`` at the output directory
(``load_training_state=False``). Tensors keep their dtype; DCP casts them into the
training dtype on load.

Usage::

    python tools/convert_checkpoint_to_dcp.py --checkpoint <model.pt> -o <dcp_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def convert(checkpoint: Path, output: Path, *, with_ema: bool = True) -> int:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{checkpoint} is not a state dict")
    if any(k.startswith(("net.", "net_ema.")) for k in state):
        raise ValueError(f"{checkpoint} already has net./net_ema. prefixes")
    model_state = {f"net.{k}": v for k, v in state.items()}
    if with_ema:
        model_state.update({f"net_ema.{k}": v for k, v in state.items()})
    output.mkdir(parents=True, exist_ok=False)
    dcp.save(model_state, checkpoint_id=str(output / "model"), no_dist=True)
    return len(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="released network state dict (.pt)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output DCP directory (must not exist)")
    parser.add_argument("--no-ema", action="store_true", help="write net.* only (then skip net_ema. when loading)")
    args = parser.parse_args()
    n = convert(args.checkpoint, args.output, with_ema=not args.no_ema)
    print(f"wrote {n} tensors{' (+ EMA copy)' if not args.no_ema else ''} to {args.output / 'model'}")


if __name__ == "__main__":
    main()
