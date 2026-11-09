# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Migrate public H16 20D CMR projectors to 28D, preserving all other weights."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from cosmos3_h_embodied.dreams.contract import DOMAIN, OLD_TO_NEW, file_sha256, stamp, write_json


def remap(state: dict[str, torch.Tensor]) -> None:
    """Remap only domain 25's flattened [input,output] boundary tensors in place."""
    wi = state["action2llm.fc.weight"]  # [32,64*C]
    wo = state["llm2action.fc.weight"]  # [32,C*64]
    bo = state["llm2action.bias.weight"]  # [32,64]
    if wi.shape != wo.shape or wi.ndim != 2 or wi.shape[0] != 32 or wi.shape[1] % 64 or bo.shape != (32, 64):
        raise ValueError("Expected 32-domain, padded-64D H16 action projectors")
    width = wi.shape[1] // 64
    if state["action2llm.bias.weight"].shape != (32, width):
        raise ValueError("Unexpected input-projector bias")
    source_i = wi[DOMAIN].reshape(64, width).clone()  # [64,C]
    source_o = wo[DOMAIN].reshape(width, 64).clone()  # [C,64]
    source_b = bo[DOMAIN].clone()  # [64]
    target_i = wi[DOMAIN].reshape(64, width)  # [64,C]
    target_o = wo[DOMAIN].reshape(width, 64)  # [C,64]
    target_i.zero_()
    target_o.zero_()
    bo[DOMAIN].zero_()
    target_i[list(OLD_TO_NEW)] = source_i[:20]  # [20,C]
    target_o[:, list(OLD_TO_NEW)] = source_o[:, :20]  # [C,20]
    bo[DOMAIN, list(OLD_TO_NEW)] = source_b[:20]  # [20]


def _migrate_into(source: Path, output: Path) -> None:
    """Write a new regular-net-only DCP, round-trip projections, then stamp it."""
    torch.set_num_threads(1)
    expected = json.loads((Path(__file__).parent / "assets.json").read_text())["initializer"]["sha256"]
    source_hash = file_sha256(source)
    if source_hash != expected:
        raise ValueError(
            "Migration requires the pinned public 20D H16 initializer; refusing an unknown or already mapped checkpoint"
        )
    state = (
        load_file(source)
        if source.suffix == ".safetensors"
        else torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    )
    if not isinstance(state, dict) or not state or any(k.startswith(("net.", "net_ema.")) for k in state):
        raise ValueError("Expected released unprefixed network weights")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid tensor: {key}")
    remap(state)
    output.mkdir(parents=True, exist_ok=True)
    dcp.save({f"net.{k}": v for k, v in state.items()}, checkpoint_id=output / "model", no_dist=True)
    keys = [k for k in state if k.startswith(("action2llm.", "llm2action."))]
    restored = {f"net.{k}": torch.empty_like(state[k]) for k in keys}  # dict[tensor]
    dcp.load(restored, checkpoint_id=output / "model", no_dist=True)
    for key in keys:
        torch.testing.assert_close(restored[f"net.{key}"], state[key], rtol=0, atol=0)
    receipt = {
        "source_sha256": source_hash,
        "source_filename": source.name,
        "old_to_new": list(OLD_TO_NEW),
        "domain": DOMAIN,
        "regular_net_only": True,
        "new_controls_zero": True,
        "projection_roundtrip": True,
        "optimizer_copied": False,
    }
    write_json(output / "migration.json", receipt)
    stamp(output, receipt)


def migrate(source: Path, output: Path) -> None:
    """Publish only a fully written and verified migration; leave the source intact."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite migration: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-migration-", dir=output.parent) as temporary:
        staging = Path(temporary)
        _migrate_into(source, staging)
        staging.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == "__main__":
    main()
